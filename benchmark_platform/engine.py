from __future__ import annotations

import json
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .catalog import Benchmark, Catalog
from .store import CaseStore, TERMINAL_STATUSES
from .util import atomic_json, command_exists, slug, stream_process, utc_now


def docker_socket_source(context: dict[str, Any]) -> str:
    override = os.environ.get("BENCHMARK_DOCKER_SOCKET_SOURCE")
    if override:
        return override
    if context.get("Name") == "colima":
        return "/var/run/docker.sock"
    host = context["Endpoints"]["docker"]["Host"]
    prefix = "unix://"
    if not host.startswith(prefix):
        raise RuntimeError(f"Docker context does not expose a Unix socket: {host}")
    return host[len(prefix) :]


def terminal_agent_command(command_override: list[str] | None, smoke: bool) -> str | None:
    if command_override:
        return shlex.join(command_override)
    if smoke:
        return "bash /solution/solve.sh"
    return None


def _container_reachable_proxy(proxy: str) -> str:
    """Translate a macOS loopback proxy URL into Docker's host gateway name."""
    parsed = urlsplit(proxy)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return proxy
    credentials = ""
    if parsed.username is not None:
        credentials = parsed.username
        if parsed.password is not None:
            credentials += f":{parsed.password}"
        credentials += "@"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(
        (
            parsed.scheme,
            f"{credentials}host.docker.internal{port}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def local_proxy_url() -> str | None:
    """Return the active host proxy suitable for container use, if one exists."""
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        if value := os.environ.get(name, "").strip():
            return _container_reachable_proxy(value)
    if sys.platform != "darwin" or shutil.which("scutil") is None:
        return None
    completed = subprocess.run(
        ["scutil", "--proxy"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(" : ")
        if separator:
            values[key.strip()] = value.strip()
    for prefix, scheme in (("HTTPS", "http"), ("HTTP", "http"), ("SOCKS", "socks5")):
        if values.get(f"{prefix}Enable") != "1":
            continue
        host = values.get(f"{prefix}Proxy", "")
        port = values.get(f"{prefix}Port", "")
        if host and port.isdigit():
            return _container_reachable_proxy(f"{scheme}://{host}:{port}")
    return None


class Platform:
    def __init__(self, root: Path, orch_root: Path, catalog_path: Path):
        self.root = root.resolve()
        self.orch_root = orch_root.resolve()
        self.catalog = Catalog(catalog_path, self.root, self.orch_root)

    def _docker(self, *args: str) -> list[str]:
        return ["docker", *args]

    def _egress_env(self, network: str) -> list[str]:
        """Docker-run flags that pin container egress explicitly.

        Docker Desktop and colima inject their daemon-level proxy settings into every
        container. When that proxy is stale or host-local it is unreachable from the
        container namespace, and urllib honours it, so every API call fails with
        ECONNREFUSED before a request leaves the VM. `--network host` does not help on
        macOS: it joins the Linux VM namespace, not the macOS host stack.

        BENCHMARK_RUN_PROXY supports reproducible overrides around the default auto mode:
          unset or "auto" -> mirror the host proxy, or direct when the host has none
          "inherit"       -> preserve Docker's injected proxy settings unchanged
          "direct"        -> neutralise injected proxies and egress directly
          "<url>"         -> route container egress through <url>
        Containers on `--network none` get nothing; they have no egress to configure.
        """
        if network == "none":
            return []
        configured = os.environ.get("BENCHMARK_RUN_PROXY")
        mode = configured.strip() if configured is not None else "auto"
        if not mode:
            mode = "auto"
        if mode.lower() == "inherit":
            return []
        if mode.lower() == "auto":
            proxy = local_proxy_url() or ""
        elif mode.lower() == "direct":
            proxy = ""
        else:
            proxy = _container_reachable_proxy(mode)
            parsed = urlsplit(proxy)
            if parsed.scheme not in {"http", "https", "socks5", "socks5h"} or not parsed.hostname:
                raise ValueError(
                    "BENCHMARK_RUN_PROXY must be 'auto', 'inherit', 'direct', or a complete proxy URL"
                )
        flags: list[str] = []
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            flags.extend(["-e", f"{name}={proxy}"])
        no_proxy = (
            (
                os.environ.get("BENCHMARK_RUN_NO_PROXY")
                or os.environ.get("NO_PROXY")
                or os.environ.get("no_proxy")
                or "localhost,127.0.0.1,::1,host.docker.internal"
            ).strip()
            if proxy
            else "*"
        )
        flags.extend(["-e", f"NO_PROXY={no_proxy}", "-e", f"no_proxy={no_proxy}"])
        return flags

    def image_exists(self, image: str) -> bool:
        return subprocess.run(
            self._docker("image", "inspect", image),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0

    def adapter_fingerprint(self, adapter: dict[str, Any]) -> str | None:
        configured = adapter.get("fingerprint_paths", [])
        dockerfile = adapter.get("dockerfile")
        raw_paths = [dockerfile, *configured] if dockerfile else list(configured)
        paths = [Path(value).resolve() for value in raw_paths if value]
        if not paths:
            return None
        digest = hashlib.sha256()
        ignored_parts = {".git", ".venv", ".sources", "node_modules", "__pycache__", "runs", "build", "dist"}
        for index, root in enumerate(paths):
            files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
            for path in files:
                relative = Path(path.name) if root.is_file() else path.relative_to(root)
                if (
                    any(part in ignored_parts or part.endswith(".egg-info") for part in relative.parts)
                    or path.suffix in {".pyc", ".pyo"}
                ):
                    continue
                label = f"{index}:{relative.as_posix()}"
                digest.update(label.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
        return digest.hexdigest()

    def expected_image_labels(self, adapter: dict[str, Any]) -> dict[str, str]:
        expected = dict(adapter.get("image_labels", {}))
        if fingerprint := self.adapter_fingerprint(adapter):
            expected["org.harnesseval.adapter-fingerprint"] = fingerprint
        return expected

    def image_is_current(self, adapter: dict[str, Any]) -> bool:
        image = adapter.get("image")
        if not image or not self.image_exists(image):
            return False
        expected = self.expected_image_labels(adapter)
        if not expected:
            return True
        inspect = subprocess.run(
            self._docker("image", "inspect", image),
            text=True,
            capture_output=True,
            check=False,
        )
        if inspect.returncode != 0:
            return False
        observed = json.loads(inspect.stdout)[0].get("Config", {}).get("Labels", {}) or {}
        return all(observed.get(name) == value for name, value in expected.items())

    def _ensure_base_images(self, adapter: dict[str, Any], *, pull: bool) -> dict[str, Any] | None:
        """Pull declared build bases only when missing or explicitly refreshed."""
        for base_image in adapter.get("pre_pull", []):
            if not pull and self.image_exists(base_image):
                continue
            pull_result = subprocess.run(self._docker("pull", base_image), check=False).returncode
            if pull_result != 0:
                return {
                    "status": "failed",
                    "returncode": pull_result,
                    "reason": f"Unable to pull base image {base_image}",
                }
        return None

    def build(self, benchmark: Benchmark, *, pull: bool = False) -> dict[str, Any]:
        adapter = benchmark.adapter
        if adapter["kind"] == "terminal-task":
            return self._build_terminal_task(benchmark, pull=pull)
        if adapter["kind"] != "docker-image":
            return {"benchmark": benchmark.id, "status": "not_buildable", "kind": adapter["kind"]}
        if source := adapter.get("source_checkout"):
            source_result = self._ensure_source_checkout(source)
            if source_result["status"] != "completed":
                return {"benchmark": benchmark.id, "status": "failed", "source_checkout": source_result}
        if base_failure := self._ensure_base_images(adapter, pull=pull):
            return {"benchmark": benchmark.id, **base_failure}
        command = self._docker("build")
        if pull:
            command.append("--pull")
        if platform := adapter.get("platform"):
            command.extend(["--platform", platform])
        if target := adapter.get("target"):
            command.extend(["--target", target])
        if fingerprint := self.adapter_fingerprint(adapter):
            command.extend(["--label", f"org.harnesseval.adapter-fingerprint={fingerprint}"])
        if "BENCHMARK_BUILD_PROXY" in os.environ:
            build_proxy = os.environ["BENCHMARK_BUILD_PROXY"]
            build_proxy = "" if build_proxy == "direct" else build_proxy
            command.extend(["--build-arg", f"HTTP_PROXY={build_proxy}"])
            command.extend(["--build-arg", f"HTTPS_PROXY={build_proxy}"])
        if pip_index := os.environ.get("BENCHMARK_PIP_INDEX_URL"):
            command.extend(["--build-arg", f"PIP_INDEX_URL={pip_index}"])
        if apt_mirror := os.environ.get("BENCHMARK_APT_MIRROR"):
            command.extend(["--build-arg", f"APT_MIRROR={apt_mirror}"])
        for name, path in sorted(adapter.get("build_contexts", {}).items()):
            command.extend(["--build-context", f"{name}={path}"])
        command.extend(["-f", adapter["dockerfile"], "-t", adapter["image"]])
        for name, value in sorted(adapter.get("build_args", {}).items()):
            command.extend(["--build-arg", f"{name}={value}"])
        command.append(adapter.get("build_context", str(self.root)))
        started = time.perf_counter()
        try:
            returncode = subprocess.run(command, check=False).returncode
        except KeyboardInterrupt:
            return {
                "benchmark": benchmark.id,
                "image": adapter["image"],
                "status": "interrupted",
                "elapsed_seconds": time.perf_counter() - started,
            }
        return {
            "benchmark": benchmark.id,
            "image": adapter["image"],
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def _source_revision(self, path: Path) -> str | None:
        if not path.is_dir():
            return None
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _ensure_source_checkout(self, source: dict[str, str]) -> dict[str, Any]:
        path = Path(source["path"])
        revision = source["revision"]
        actual = self._source_revision(path)
        if actual:
            if actual == revision:
                return {"status": "completed", "path": str(path), "revision": actual, "fetched": False}
            return {
                "status": "failed",
                "path": str(path),
                "reason": "Existing source checkout has a different revision",
                "expected_revision": revision,
                "actual_revision": actual,
            }
        if path.exists():
            return {"status": "failed", "path": str(path), "reason": "Source checkout path exists but is not Git"}
        if not command_exists("git"):
            return {"status": "failed", "path": str(path), "reason": "Host git is unavailable"}
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.fetch-{os.getpid()}")
        shutil.rmtree(pending, ignore_errors=True)
        commands = [
            ["git", "init", str(pending)],
            ["git", "-C", str(pending), "remote", "add", "origin", source["url"]],
            ["git", "-C", str(pending), "fetch", "--depth", "1", "origin", revision],
            ["git", "-C", str(pending), "checkout", "--detach", "FETCH_HEAD"],
        ]
        started = time.perf_counter()
        try:
            for command in commands:
                if subprocess.run(command, check=False).returncode != 0:
                    return {
                        "status": "failed",
                        "path": str(path),
                        "reason": f"Source fetch command failed: {shlex.join(command)}",
                    }
            actual = self._source_revision(pending)
            if actual != revision:
                return {
                    "status": "failed",
                    "path": str(path),
                    "reason": "Fetched source revision mismatch",
                    "expected_revision": revision,
                    "actual_revision": actual,
                }
            os.replace(pending, path)
            return {
                "status": "completed",
                "path": str(path),
                "revision": actual,
                "fetched": True,
                "elapsed_seconds": time.perf_counter() - started,
            }
        finally:
            if pending.exists():
                shutil.rmtree(pending)

    def _terminal_metadata(self, benchmark: Benchmark) -> tuple[Path, dict[str, Any]]:
        try:
            import tomllib
        except ModuleNotFoundError as exc:
            raise RuntimeError("Terminal task metadata requires Python 3.11 or newer") from exc
        task_dir = Path(benchmark.adapter["task_dir"])
        with (task_dir / "task.toml").open("rb") as stream:
            metadata = tomllib.load(stream)
        return task_dir, metadata

    def _build_terminal_task(self, benchmark: Benchmark, *, pull: bool) -> dict[str, Any]:
        task_dir, metadata = self._terminal_metadata(benchmark)
        adapter = benchmark.adapter
        task_image = metadata.get("environment", {}).get("docker_image")
        if task_image:
            if self.image_exists(task_image) and not pull:
                return {
                    "benchmark": benchmark.id,
                    "image": task_image,
                    "status": "completed",
                    "source": "official_task_metadata",
                    "elapsed_seconds": 0.0,
                }
            started = time.perf_counter()
            returncode = subprocess.run(self._docker("pull", task_image), check=False).returncode
            return {
                "benchmark": benchmark.id,
                "image": task_image,
                "status": "completed" if returncode == 0 else "failed",
                "source": "official_task_metadata",
                "returncode": returncode,
                "elapsed_seconds": time.perf_counter() - started,
            }
        if base_failure := self._ensure_base_images(adapter, pull=pull):
            return {"benchmark": benchmark.id, **base_failure}
        command = self._docker("build")
        if pull:
            command.append("--pull")
        if platform := adapter.get("platform"):
            command.extend(["--platform", platform])
        command.extend(
            ["-f", str(task_dir / "environment" / "Dockerfile"), "-t", adapter["image"], str(task_dir / "environment")]
        )
        started = time.perf_counter()
        returncode = subprocess.run(command, check=False).returncode
        return {
            "benchmark": benchmark.id,
            "image": adapter["image"],
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def doctor(self, benchmark: Benchmark) -> dict[str, Any]:
        adapter = benchmark.adapter
        checks: list[dict[str, Any]] = []
        docker_ok = command_exists("docker") and subprocess.run(
            self._docker("info"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        ).returncode == 0
        checks.append({"name": "docker", "ok": docker_ok})
        source_checkout_missing = False
        if source := adapter.get("source_checkout"):
            path = Path(source["path"])
            actual = self._source_revision(path)
            source_checkout_missing = not path.exists()
            checks.append(
                {
                    "name": "source checkout",
                    "ok": actual == source["revision"],
                    "path": str(path),
                    "fetchable": source_checkout_missing,
                    "expected_revision": source["revision"],
                    "actual_revision": actual,
                }
            )
        for required in benchmark.raw.get("required_paths", []):
            path = Path(required["path"])
            if required.get("type") == "file":
                ok = path.is_file()
            else:
                ok = path.is_dir()
            check = {"name": required.get("name", str(path)), "ok": ok, "path": str(path)}
            if ok and required.get("git_revision"):
                actual = subprocess.run(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                actual_revision = actual.stdout.strip()
                check.update(
                    {
                        "ok": actual.returncode == 0 and actual_revision == required["git_revision"],
                        "expected_revision": required["git_revision"],
                        "actual_revision": actual_revision or None,
                    }
                )
            checks.append(check)
        runtime_images_ok = True
        for required in adapter.get("runtime_images", []):
            inspect = subprocess.run(
                self._docker("image", "inspect", required["image"]),
                text=True,
                capture_output=True,
                check=False,
            )
            digests: list[str] = []
            if inspect.returncode == 0:
                digests = json.loads(inspect.stdout)[0].get("RepoDigests", [])
            ok = inspect.returncode == 0 and (not required.get("digest") or required["digest"] in digests)
            runtime_images_ok = runtime_images_ok and ok
            checks.append(
                {
                    "name": "runtime image",
                    "ok": ok,
                    "image": required["image"],
                    "expected_digest": required.get("digest"),
                    "observed_digests": digests,
                }
            )
        prerequisites_ok = docker_ok and all(check["ok"] or check.get("fetchable", False) for check in checks)
        image = adapter.get("image")
        if adapter["kind"] == "terminal-task":
            try:
                _, metadata = self._terminal_metadata(benchmark)
                image = metadata.get("environment", {}).get("docker_image") or image
            except (FileNotFoundError, RuntimeError):
                image = adapter.get("image")
        image_ok = False
        if image:
            image_ok = self.image_exists(image)
            expected_labels = self.expected_image_labels(adapter)
            observed_labels: dict[str, str] = {}
            if image_ok and expected_labels:
                inspect = subprocess.run(
                    self._docker("image", "inspect", image),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if inspect.returncode == 0:
                    observed_labels = json.loads(inspect.stdout)[0].get("Config", {}).get("Labels", {}) or {}
                image_ok = all(observed_labels.get(name) == value for name, value in expected_labels.items())
            checks.append(
                {
                    "name": "image",
                    "ok": image_ok,
                    "image": image,
                    "expected_labels": expected_labels,
                    "observed_labels": {name: observed_labels.get(name) for name in expected_labels},
                }
            )
        if adapter["kind"] == "external-vm":
            checks.append({"name": "local_provider", "ok": False, "reason": adapter["blocker"]})
        buildable = prerequisites_ok and adapter["kind"] in {"docker-image", "terminal-task"}
        ready = prerequisites_ok and runtime_images_ok and bool(image and image_ok)
        if adapter["kind"] == "external-vm":
            ready = False
            buildable = False
        return {
            "benchmark": benchmark.id,
            "name": benchmark.name,
            "adapter": adapter["kind"],
            "runnable": ready,
            "ready": ready,
            "buildable": buildable,
            "checks": checks,
            "scoring": benchmark.raw["scoring"],
        }

    def run(
        self,
        benchmark: Benchmark,
        *,
        case_id: str,
        run_dir: Path,
        command_override: list[str] | None = None,
        smoke: bool = False,
        resume: bool = True,
        retry_failed: bool = False,
        build_missing: bool = True,
        pass_env: list[str] | None = None,
        extra_mounts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        store = CaseStore(run_dir.resolve(), benchmark.id, case_id)
        with store.lock():
            return self._run_locked(
                store,
                benchmark,
                command_override=command_override,
                smoke=smoke,
                resume=resume,
                retry_failed=retry_failed,
                build_missing=build_missing,
                pass_env=pass_env,
                extra_mounts=extra_mounts,
            )

    def run_harness(
        self,
        *,
        profile: dict[str, Any],
        request_path: Path,
        case_id: str,
        run_dir: Path,
        image: str,
        network: str,
        resume: bool,
        retry_failed: bool,
        pull_missing: bool,
        pass_env: list[str],
        extra_mounts: list[dict[str, str]],
    ) -> dict[str, Any]:
        request_path = request_path.resolve()
        if not request_path.is_file():
            raise FileNotFoundError(f"Harness request does not exist: {request_path}")
        if not self.image_exists(image):
            if not pull_missing:
                raise RuntimeError(f"Harness runtime image is unavailable: {image}")
            if subprocess.run(self._docker("pull", image), check=False).returncode != 0:
                raise RuntimeError(f"Unable to pull harness runtime image: {image}")
        harness_env = [
            "HARNESS_API_BASE",
            "HARNESS_API_KEY",
            "HARNESS_MODEL",
            "HARNESS_TEMPERATURE",
            "HARNESS_API_TIMEOUT_S",
            "HARNESS_API_RETRIES",
            "HARNESS_MAX_OUTPUT_TOKENS",
            "HARNESS_REASONING_EFFORT",
        ]
        api_base = os.environ.get("HARNESS_API_BASE", "")
        resume_identity = {
            "profile": profile["id"],
            "profile_revision": profile.get("revision"),
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "image": image,
            "network": network,
            "model": os.environ.get("HARNESS_MODEL") or None,
            "api_base_sha256": hashlib.sha256(api_base.encode("utf-8")).hexdigest() if api_base else None,
        }
        benchmark = Benchmark(
            id=f"harness-{profile['id']}",
            name=profile["name"],
            raw={
                "id": f"harness-{profile['id']}",
                "name": profile["name"],
                "source": {
                    "provenance": profile["provenance"],
                    "reference": profile.get("source"),
                    "revision": profile.get("revision"),
                },
                "adapter": {
                    "kind": "docker-image",
                    "image": image,
                    "read_only": True,
                    "working_dir": "/opt/harnesseval",
                    "network": network,
                },
                "run": {
                    "network": network,
                    "command": [
                        "python",
                        "-m",
                        "benchmark_platform.harnesses.runner",
                        "--profile",
                        profile["id"],
                        "--request",
                        "/input/request.json",
                    ],
                },
                "mounts": [
                    {"host": str(self.root), "container": "/opt/harnesseval", "mode": "ro"},
                    {"host": str(request_path), "container": "/input/request.json", "mode": "ro"},
                ],
                "scoring": {
                    "authority": "Request-provided benchmark finalizer",
                    "comparability": "user-supplied",
                    "note": "HarnessEval records the harness trajectory; benchmark authority remains external.",
                },
                "env_allowlist": harness_env,
                "resume_identity": resume_identity,
            },
        )
        unknown = sorted(set(pass_env) - set(harness_env))
        if unknown:
            raise ValueError(f"Unsupported harness environment variable(s): {', '.join(unknown)}")
        return self.run(
            benchmark,
            case_id=case_id,
            run_dir=run_dir,
            resume=resume,
            retry_failed=retry_failed,
            build_missing=False,
            pass_env=pass_env,
            extra_mounts=extra_mounts,
        )

    def run_bridge_harness(
        self,
        *,
        benchmark: Benchmark,
        profile: dict[str, Any],
        case_id: str,
        run_dir: Path,
        network: str,
        policy: dict[str, Any],
        resume: bool,
        retry_failed: bool,
        build_missing: bool,
        pass_env: list[str],
    ) -> dict[str, Any]:
        supported = {
            "gaia",
            "gdpval",
            "trajectory-bench",
            "bfcl",
            "vitabench",
            "tau2",
            "terminal-bench-2",
            "swe-bench-verified",
        }
        if benchmark.id not in supported:
            raise ValueError(
                f"{benchmark.id} requires a conversation/task-container bridge; "
                "no built-in baseline lifecycle adapter is available"
            )
        if benchmark.id == "terminal-bench-2":
            return self._run_terminal_bridge_harness(
                benchmark=benchmark,
                profile=profile,
                case_id=case_id,
                run_dir=run_dir,
                policy=policy,
                resume=resume,
                retry_failed=retry_failed,
                build_missing=build_missing,
                pass_env=pass_env,
            )
        if benchmark.id == "swe-bench-verified":
            return self._run_swe_bridge_harness(
                benchmark=benchmark,
                profile=profile,
                case_id=case_id,
                run_dir=run_dir,
                policy=policy,
                resume=resume,
                retry_failed=retry_failed,
                build_missing=build_missing,
                pass_env=pass_env,
            )
        if benchmark.adapter["kind"] != "docker-image":
            raise ValueError(f"Bridge harness requires a Docker image benchmark: {benchmark.id}")
        if not self.image_is_current(benchmark.adapter):
            if not build_missing:
                raise RuntimeError("Benchmark image is missing or stale")
            built = self.build(benchmark)
            if built["status"] != "completed":
                raise RuntimeError(f"Benchmark image build failed: {built}")
        native_episode = benchmark.id in {"vitabench", "tau2"}
        prepared = None if native_episode else self._prepare_bridge_case(benchmark, case_id, run_dir)
        harness_env = {
            "HARNESS_API_BASE",
            "HARNESS_API_KEY",
            "HARNESS_MODEL",
            "HARNESS_TEMPERATURE",
            "HARNESS_API_TIMEOUT_S",
            "HARNESS_API_RETRIES",
            "HARNESS_MAX_OUTPUT_TOKENS",
            "HARNESS_REASONING_EFFORT",
        }
        allowed = harness_env | set(benchmark.raw.get("env_allowlist", []))
        unknown = sorted(set(pass_env) - allowed)
        if unknown:
            raise ValueError(f"Unsupported bridge environment variable(s): {', '.join(unknown)}")
        adapter = dict(benchmark.adapter)
        adapter.update({"working_dir": "/opt/harnesseval", "network": network})
        if not native_episode:
            adapter.update({"read_only": True, "run_as_root": True})
        adapter.pop("docker_socket", None)
        api_base = os.environ.get("HARNESS_API_BASE", "")
        request_identity = {
            "benchmark": benchmark.id,
            "case": case_id,
            "profile": profile["id"],
            "profile_revision": profile.get("revision"),
            "policy": policy,
            "image": adapter["image"],
            "model": os.environ.get("HARNESS_MODEL") or None,
            "api_base_sha256": hashlib.sha256(api_base.encode("utf-8")).hexdigest() if api_base else None,
            "prepared_case_sha256": (
                hashlib.sha256((prepared / "input" / "case.json").read_bytes()).hexdigest()
                if prepared is not None
                else None
            ),
        }
        if native_episode:
            episode_module = (
                "benchmark_platform.bridges.vita_episode"
                if benchmark.id == "vitabench"
                else "benchmark_platform.bridges.tau_episode"
            )
            inner_command = [
                "python",
                "-m",
                episode_module,
                "--profile",
                profile["id"],
                "--case",
                case_id,
                "--policy",
                json.dumps(policy, ensure_ascii=False, sort_keys=True),
            ]
            mounts = [{"host": str(self.root), "container": "/opt/harnesseval", "mode": "ro"}]
            comparability = "native_episode"
            scoring_note = (
                "The official hidden-user and environment lifecycle is preserved. Native scoring behavior is "
                "recorded in harness_result.native_score_status."
            )
        else:
            inner_command = [
                "python",
                "-m",
                "benchmark_platform.bridges.runner",
                "--benchmark",
                benchmark.id,
                "--profile",
                profile["id"],
                "--case",
                case_id,
                "--policy",
                json.dumps(policy, ensure_ascii=False, sort_keys=True),
            ]
            mounts = [
                {"host": str(self.root), "container": "/opt/harnesseval", "mode": "ro"},
                {"host": str(prepared / "input"), "container": "/bridge", "mode": "ro"},
            ]
            comparability = "unscored_bridge_trajectory"
            scoring_note = "The bridge result must be finalized by the benchmark-native scorer before publication."
        synthetic = Benchmark(
            id=f"{benchmark.id}-{profile['id']}",
            name=f"{benchmark.name} / {profile['name']}",
            raw={
                "id": f"{benchmark.id}-{profile['id']}",
                "name": f"{benchmark.name} / {profile['name']}",
                "source": benchmark.source,
                "adapter": adapter,
                "run": {
                    "network": network,
                    "command": inner_command,
                },
                "mounts": mounts,
                "scoring": {
                    "authority": benchmark.raw["scoring"]["authority"],
                    "comparability": comparability,
                    "note": scoring_note,
                },
                "env_allowlist": sorted(allowed),
                "resume_identity": request_identity,
            },
        )
        return self.run(
            synthetic,
            case_id=case_id,
            run_dir=run_dir,
            resume=resume,
            retry_failed=retry_failed,
            build_missing=build_missing,
            pass_env=pass_env,
        )

    def _run_terminal_bridge_harness(
        self,
        *,
        benchmark: Benchmark,
        profile: dict[str, Any],
        case_id: str,
        run_dir: Path,
        policy: dict[str, Any],
        resume: bool,
        retry_failed: bool,
        build_missing: bool,
        pass_env: list[str],
    ) -> dict[str, Any]:
        harness_env = {
            "HARNESS_API_BASE",
            "HARNESS_API_KEY",
            "HARNESS_MODEL",
            "HARNESS_TEMPERATURE",
            "HARNESS_API_TIMEOUT_S",
            "HARNESS_API_RETRIES",
            "HARNESS_MAX_OUTPUT_TOKENS",
            "HARNESS_REASONING_EFFORT",
        }
        unknown = sorted(set(pass_env) - harness_env)
        if unknown:
            raise ValueError(f"Unsupported bridge environment variable(s): {', '.join(unknown)}")
        task_dir, metadata = self._terminal_metadata(benchmark)
        image = metadata.get("environment", {}).get("docker_image") or benchmark.adapter["image"]
        if not self.image_exists(image):
            if not build_missing:
                raise RuntimeError("Terminal task image is unavailable")
            built = self.build(benchmark)
            if built["status"] != "completed":
                raise RuntimeError(f"Terminal task image build failed: {built}")
        api_base = os.environ.get("HARNESS_API_BASE", "")
        resume_identity = {
            "benchmark": benchmark.id,
            "case": case_id,
            "profile": profile["id"],
            "profile_revision": profile.get("revision"),
            "policy": policy,
            "image": image,
            "model": os.environ.get("HARNESS_MODEL") or None,
            "api_base_sha256": hashlib.sha256(api_base.encode("utf-8")).hexdigest() if api_base else None,
        }
        synthetic = Benchmark(
            id=f"{benchmark.id}-{profile['id']}",
            name=f"{benchmark.name} / {profile['name']}",
            raw={
                "id": f"{benchmark.id}-{profile['id']}",
                "name": f"{benchmark.name} / {profile['name']}",
                "source": benchmark.source,
                "adapter": {
                    "kind": "terminal-profile",
                    "image": image,
                    "platform": benchmark.adapter.get("platform"),
                    "task_dir": str(task_dir),
                    "profile": profile,
                    "policy": policy,
                    "allow_internet": bool(metadata["environment"].get("allow_internet")),
                },
                "scoring": benchmark.raw["scoring"],
                "resume_identity": resume_identity,
            },
        )
        return self.run(
            synthetic,
            case_id=case_id,
            run_dir=run_dir,
            resume=resume,
            retry_failed=retry_failed,
            build_missing=False,
            pass_env=pass_env,
        )

    def _run_swe_bridge_harness(
        self,
        *,
        benchmark: Benchmark,
        profile: dict[str, Any],
        case_id: str,
        run_dir: Path,
        policy: dict[str, Any],
        resume: bool,
        retry_failed: bool,
        build_missing: bool,
        pass_env: list[str],
    ) -> dict[str, Any]:
        harness_env = {
            "HARNESS_API_BASE",
            "HARNESS_API_KEY",
            "HARNESS_MODEL",
            "HARNESS_TEMPERATURE",
            "HARNESS_API_TIMEOUT_S",
            "HARNESS_API_RETRIES",
            "HARNESS_MAX_OUTPUT_TOKENS",
            "HARNESS_REASONING_EFFORT",
        }
        allowed = harness_env | {"HF_TOKEN"}
        unknown = sorted(set(pass_env) - allowed)
        if unknown:
            raise ValueError(f"Unsupported bridge environment variable(s): {', '.join(unknown)}")
        if not self.image_is_current(benchmark.adapter):
            if not build_missing:
                raise RuntimeError("SWE-bench controller image is missing or stale")
            built = self.build(benchmark)
            if built["status"] != "completed":
                raise RuntimeError(f"SWE-bench controller image build failed: {built}")
        api_base = os.environ.get("HARNESS_API_BASE", "")
        resume_identity = {
            "benchmark": benchmark.id,
            "case": case_id,
            "profile": profile["id"],
            "profile_revision": profile.get("revision"),
            "policy": policy,
            "controller_image": benchmark.adapter["image"],
            "model": os.environ.get("HARNESS_MODEL") or None,
            "api_base_sha256": hashlib.sha256(api_base.encode("utf-8")).hexdigest() if api_base else None,
        }
        synthetic = Benchmark(
            id=f"{benchmark.id}-{profile['id']}",
            name=f"{benchmark.name} / {profile['name']}",
            raw={
                "id": f"{benchmark.id}-{profile['id']}",
                "name": f"{benchmark.name} / {profile['name']}",
                "source": benchmark.source,
                "adapter": {
                    "kind": "swe-profile",
                    "image": benchmark.adapter["image"],
                    "profile": profile,
                    "policy": policy,
                    "pass_env": sorted(name for name in pass_env if name == "HF_TOKEN" and name in os.environ),
                },
                "scoring": benchmark.raw["scoring"],
                "resume_identity": resume_identity,
            },
        )
        return self.run(
            synthetic,
            case_id=case_id,
            run_dir=run_dir,
            resume=resume,
            retry_failed=retry_failed,
            build_missing=False,
            pass_env=pass_env,
        )

    def _prepare_bridge_case(self, benchmark: Benchmark, case_id: str, run_dir: Path) -> Path:
        prepared = run_dir.resolve() / ".bridge_inputs" / slug(benchmark.id) / slug(case_id)
        prepare_source = self.root / "benchmark_platform" / "bridges" / "prepare.py"
        preparation_identity = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": case_id,
            "prepare_sha256": hashlib.sha256(prepare_source.read_bytes()).hexdigest(),
            "source": benchmark.source,
        }
        identity_path = prepared / "preparation_identity.json"
        if (
            (prepared / "input" / "case.json").is_file()
            and (prepared / "authority" / "gold.json").is_file()
            and identity_path.is_file()
            and json.loads(identity_path.read_text(encoding="utf-8")) == preparation_identity
        ):
            return prepared
        staging = prepared.with_name(f".{prepared.name}.staging-{os.getpid()}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        adapter = benchmark.adapter
        command = self._docker("run", "--rm", "--init", "--network", "none")
        if adapter.get("platform"):
            command.extend(["--platform", adapter["platform"]])
        command.extend(["--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=1g"])
        if not adapter.get("run_as_root", False):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend(["-e", "HOME=/tmp", "-w", "/opt/harnesseval"])
        for host, container, mode in self._validated_mounts(benchmark.raw.get("mounts", [])):
            command.extend(["-v", f"{host}:{container}:{mode}"])
        command.extend(["-v", f"{self.root}:/opt/harnesseval:ro", "-v", f"{staging}:/prepared:rw"])
        command.extend(
            [
                adapter["image"],
                "python",
                "-m",
                "benchmark_platform.bridges.prepare",
                benchmark.id,
                "--case",
                case_id,
            ]
        )
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"Bridge case preparation failed: {completed.stderr or completed.stdout}")
        atomic_json(staging / "preparation_identity.json", preparation_identity)
        prepared.parent.mkdir(parents=True, exist_ok=True)
        if prepared.exists():
            shutil.rmtree(prepared)
        staging.replace(prepared)
        return prepared

    def _run_locked(
        self,
        store: CaseStore,
        benchmark: Benchmark,
        *,
        command_override: list[str] | None,
        smoke: bool,
        resume: bool,
        retry_failed: bool,
        build_missing: bool,
        pass_env: list[str] | None,
        extra_mounts: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        existing = store.existing()
        resume_identity = benchmark.raw.get("resume_identity")
        if existing and resume and resume_identity and existing.get("resume_identity") != resume_identity:
            raise RuntimeError(
                "The existing case was produced by a different harness request, image, model, or provider. "
                "Use a new case id/run directory, or pass --no-resume to append a deliberate new attempt."
            )
        if existing and resume and existing.get("status") in TERMINAL_STATUSES:
            if existing.get("status") != "failed" or not retry_failed:
                return {**existing, "resume_skipped": True}
        if benchmark.adapter["kind"] == "external-vm":
            return self._record_blocked(store, benchmark, benchmark.adapter["blocker"])
        if benchmark.adapter["kind"] == "swe-profile":
            return self._run_swe_profile(store, benchmark)
        if benchmark.adapter["kind"] == "terminal-profile":
            return self._run_terminal_profile(store, benchmark)
        if benchmark.adapter["kind"] == "terminal-task":
            return self._run_terminal_task(store, benchmark, smoke=smoke, command_override=command_override)
        if benchmark.adapter["kind"] != "docker-image":
            return self._record_blocked(store, benchmark, f"Unsupported adapter: {benchmark.adapter['kind']}")
        if not self.image_is_current(benchmark.adapter):
            if not build_missing:
                return self._record_blocked(store, benchmark, "Image is missing or its source revision label is stale")
            build_result = self.build(benchmark)
            if build_result["status"] != "completed":
                return self._record_blocked(store, benchmark, "Image build failed", extra={"build": build_result})
        return self._run_image(
            store,
            benchmark,
            smoke=smoke,
            command_override=command_override,
            pass_env=pass_env or [],
            extra_mounts=extra_mounts or [],
        )

    def _record_blocked(
        self, store: CaseStore, benchmark: Benchmark, reason: str, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        _, attempt = store.next_attempt()
        now = utc_now()
        request = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": store.case_id,
            "started_at": now,
            "adapter": benchmark.adapter["kind"],
        }
        store.start(attempt, request)
        result = {
            **request,
            "status": "blocked",
            "finished_at": utc_now(),
            "reason": reason,
            **(extra or {}),
        }
        store.finish(attempt, result)
        return result

    def _allowed_env(self, benchmark: Benchmark, requested: list[str]) -> list[str]:
        allowed = set(benchmark.raw.get("env_allowlist", []))
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(f"Environment variable(s) not allow-listed for {benchmark.id}: {', '.join(unknown)}")
        return sorted(name for name in set(requested) if name in os.environ)

    def _validated_mounts(self, mounts: list[dict[str, str]]) -> list[tuple[Path, str, str]]:
        validated = []
        for mount in mounts:
            host = Path(mount["host"]).resolve()
            container = mount["container"]
            mode = mount.get("mode", "ro")
            if not host.exists():
                raise FileNotFoundError(f"Mount source does not exist: {host}")
            if not container.startswith("/"):
                raise ValueError(f"Container mount path must be absolute: {container}")
            if mode not in {"ro", "rw"}:
                raise ValueError(f"Unsupported mount mode: {mode}")
            validated.append((host, container, mode))
        return validated

    def _run_image(
        self,
        store: CaseStore,
        benchmark: Benchmark,
        *,
        smoke: bool,
        command_override: list[str] | None,
        pass_env: list[str],
        extra_mounts: list[dict[str, str]],
    ) -> dict[str, Any]:
        adapter = benchmark.adapter
        spec = benchmark.smoke if smoke else benchmark.raw.get("run")
        if command_override:
            inner_command = command_override
        elif spec and spec.get("command"):
            inner_command = spec["command"]
        else:
            return self._record_blocked(store, benchmark, "No run command was supplied")
        attempt_number, attempt = store.next_attempt()
        env_names = self._allowed_env(benchmark, pass_env)
        container_name = f"bench-{benchmark.id}-{os.getpid()}-{attempt_number}"
        command = self._docker("run", "--rm", "--init", "--name", container_name)
        command.extend(["--label", "orch.benchmark-platform=1", "--label", f"orch.benchmark={benchmark.id}"])
        network = (spec or {}).get("network", adapter.get("network", "none"))
        command.extend(["--network", network])
        if adapter.get("platform"):
            command.extend(["--platform", adapter["platform"]])
        if adapter.get("read_only", True):
            command.extend(["--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=1g"])
        if not adapter.get("run_as_root", False):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend(["-e", "HOME=/tmp", "-e", "PYTHONUNBUFFERED=1"])
        command.extend(self._egress_env(network))
        if working_dir := adapter.get("working_dir"):
            command.extend(["-w", working_dir])
        for name in env_names:
            command.extend(["-e", name])
        for host, container, mode in self._validated_mounts([*benchmark.raw.get("mounts", []), *extra_mounts]):
            command.extend(["-v", f"{host}:{container}:{mode}"])
        command.extend(["-v", f"{attempt.resolve()}:/job:rw"])
        if adapter.get("docker_socket"):
            socket_path = self._docker_socket()
            command.extend(["-v", f"{socket_path}:/var/run/docker.sock:rw"])
            command.extend(["-e", "DOCKER_HOST=unix:///var/run/docker.sock"])
        command.append(adapter["image"])
        command.extend(inner_command)
        started_at = utc_now()
        request = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": store.case_id,
            "attempt": attempt_number,
            "started_at": started_at,
            "adapter": adapter["kind"],
            "image": adapter["image"],
            "source": benchmark.source,
            "smoke": smoke,
            "environment_names": env_names,
            "inner_command": inner_command,
            **({"resume_identity": benchmark.raw["resume_identity"]} if benchmark.raw.get("resume_identity") else {}),
        }
        store.start(attempt, request)
        store.event(attempt, "container_starting", image=adapter["image"])
        started = time.perf_counter()
        with (attempt / "terminal.log").open("a", encoding="utf-8") as terminal:
            try:
                returncode = stream_process(command, terminal, prefix=f"[{benchmark.id}] ")
            except KeyboardInterrupt:
                subprocess.run(
                    self._docker("rm", "-f", container_name),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                result = {
                    **request,
                    "status": "cancelled",
                    "finished_at": utc_now(),
                    "execution_seconds": time.perf_counter() - started,
                }
                store.finish(attempt, result)
                raise
        payload_path = attempt / "payload.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.is_file() else {}
        harness_path = attempt / "harness_result.json"
        harness_result = json.loads(harness_path.read_text(encoding="utf-8")) if harness_path.is_file() else None
        result = {
            **request,
            "status": "completed" if returncode == 0 else "failed",
            "finished_at": utc_now(),
            "execution_seconds": time.perf_counter() - started,
            "returncode": returncode,
            "payload": payload,
            **({"harness": harness_result} if harness_result is not None else {}),
        }
        store.finish(attempt, result)
        return result

    def _docker_socket(self) -> str:
        output = subprocess.check_output(self._docker("context", "inspect"), text=True)
        contexts = json.loads(output)
        return docker_socket_source(contexts[0])

    def _run_swe_profile(self, store: CaseStore, benchmark: Benchmark) -> dict[str, Any]:
        from .bridges.terminal_episode import run as run_task_profile

        adapter = benchmark.adapter
        attempt_number, attempt = store.next_attempt()
        request = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": store.case_id,
            "attempt": attempt_number,
            "started_at": utc_now(),
            "adapter": "swe-profile",
            "controller_image": adapter["image"],
            "source": benchmark.source,
            "environment_names": [],
            "profile": adapter["profile"]["id"],
            "policy": adapter["policy"],
            "resume_identity": benchmark.raw["resume_identity"],
        }
        store.start(attempt, request)
        socket_path = self._docker_socket()

        def controller_command(action: str) -> list[str]:
            command = self._docker("run", "--rm", "--init", "--network", "bridge")
            command.extend(self._egress_env("bridge"))
            command.extend(["-v", f"{socket_path}:/var/run/docker.sock:rw"])
            command.extend(["-e", "DOCKER_HOST=unix:///var/run/docker.sock"])
            for name in adapter.get("pass_env", []):
                command.extend(["-e", name])
            command.extend(
                [
                    "-v",
                    f"{attempt.resolve()}:/job:rw",
                    adapter["image"],
                    "python",
                    "/opt/platform/swebench_bridge.py",
                    action,
                    "--case",
                    store.case_id,
                ]
            )
            return command

        started = time.perf_counter()
        task_container: str | None = None
        harness_result: dict[str, Any] = {"status": "failed", "error": "Harness did not start"}
        evaluator_returncode = 1
        with (attempt / "terminal.log").open("a", encoding="utf-8") as terminal:
            try:
                prepared = subprocess.run(
                    controller_command("prepare"),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                terminal.write(prepared.stdout)
                terminal.write(prepared.stderr)
                terminal.flush()
                public_path = attempt / "public_case.json"
                if prepared.returncode != 0 or not public_path.is_file():
                    raise RuntimeError("SWE-bench public case preparation failed")
                public_case = json.loads(public_path.read_text(encoding="utf-8"))
                if public_case.get("hidden_fields_exposed_to_agent") != []:
                    raise RuntimeError("SWE-bench preparation attempted to expose hidden authority fields")
                task_container = f"bench-{benchmark.id}-{os.getpid()}-{attempt_number}"
                create = self._docker("create", "--init", "--name", task_container)
                create.extend(["--label", "orch.benchmark-platform=1", "--label", "orch.task-container-agent=1"])
                create.extend(["--platform", public_case["task_image"]["platform"]])
                create.extend(["--network", "bridge"])
                create.extend(self._egress_env("bridge"))
                create.extend(
                    [
                        "-w",
                        public_case["workspace_root"],
                        public_case["task_image"]["name"],
                        "sh",
                        "-lc",
                        "while :; do sleep 3600; done",
                    ]
                )
                created = subprocess.run(create, text=True, capture_output=True, check=False)
                terminal.write(created.stdout)
                terminal.write(created.stderr)
                terminal.flush()
                if created.returncode != 0:
                    raise RuntimeError("SWE-bench task container creation failed")
                started_container = subprocess.run(
                    self._docker("start", task_container),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if started_container.returncode != 0:
                    terminal.write(started_container.stdout)
                    terminal.write(started_container.stderr)
                    terminal.flush()
                    raise RuntimeError("SWE-bench task container start failed")
                harness_result = run_task_profile(
                    profile_id=adapter["profile"]["id"],
                    prompt=public_case["prompt"],
                    policy=adapter["policy"],
                    container=task_container,
                    trace_path=attempt / "harness_trace.jsonl",
                    workspace_root=public_case["workspace_root"],
                )
                atomic_json(attempt / "harness_result.json", harness_result)
                staged = subprocess.run(
                    self._docker("exec", "-w", public_case["workspace_root"], task_container, "git", "add", "-A"),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if staged.returncode != 0:
                    raise RuntimeError(f"Unable to stage SWE-bench workspace changes: {staged.stderr}")
                patch_result = subprocess.run(
                    self._docker(
                        "exec",
                        "-w",
                        public_case["workspace_root"],
                        task_container,
                        "git",
                        "-c",
                        "core.fileMode=false",
                        "diff",
                        "--cached",
                        "--binary",
                    ),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if patch_result.returncode != 0:
                    raise RuntimeError(f"Unable to extract SWE-bench model patch: {patch_result.stderr}")
                (attempt / "model.patch").write_text(patch_result.stdout, encoding="utf-8")
                subprocess.run(
                    self._docker("rm", "-f", task_container),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                task_container = None
                evaluator_returncode = stream_process(
                    controller_command("evaluate"),
                    terminal,
                    prefix=f"[{benchmark.id}:evaluator] ",
                )
            except KeyboardInterrupt:
                result = {
                    **request,
                    "status": "cancelled",
                    "finished_at": utc_now(),
                    "execution_seconds": time.perf_counter() - started,
                }
                store.finish(attempt, result)
                raise
            except Exception as exc:
                terminal.write(f"{type(exc).__name__}: {exc}\n")
                terminal.flush()
            finally:
                if task_container:
                    subprocess.run(
                        self._docker("rm", "-f", task_container),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
        payload_path = attempt / "payload.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.is_file() else {}
        resolved = payload.get("scores", {}).get("resolved") == 1.0
        result = {
            **request,
            "status": (
                "completed"
                if harness_result.get("status") == "completed" and evaluator_returncode == 0 and resolved
                else "failed"
            ),
            "finished_at": utc_now(),
            "execution_seconds": time.perf_counter() - started,
            "returncode": 0 if harness_result.get("status") == "completed" else 1,
            "evaluator_returncode": evaluator_returncode,
            "payload": payload,
            "harness": harness_result,
        }
        store.finish(attempt, result)
        return result

    def _run_terminal_profile(self, store: CaseStore, benchmark: Benchmark) -> dict[str, Any]:
        from .bridges.terminal_episode import run as run_terminal_profile

        adapter = benchmark.adapter
        task_dir = Path(adapter["task_dir"])
        with (task_dir / "task.toml").open("rb") as stream:
            import tomllib

            metadata = tomllib.load(stream)
        prompt = (task_dir / "instruction.md").read_text(encoding="utf-8")
        image = adapter["image"]
        attempt_number, attempt = store.next_attempt()
        logs = attempt / "verifier"
        workspace = attempt / "workspace"
        logs.mkdir()
        workspace.mkdir()
        container_name = f"bench-{benchmark.id}-{os.getpid()}-{attempt_number}"
        create = self._docker("create", "--init", "--name", container_name)
        create.extend(["--label", "orch.benchmark-platform=1", "--label", "orch.task-container-agent=1"])
        if platform := adapter.get("platform"):
            create.extend(["--platform", platform])
        create.extend(
            [
                "--network",
                "bridge" if adapter.get("allow_internet") else "none",
                *self._egress_env("bridge" if adapter.get("allow_internet") else "none"),
                "-w",
                "/app",
                image,
                "sh",
                "-lc",
                "while :; do sleep 3600; done",
            ]
        )
        request = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": store.case_id,
            "attempt": attempt_number,
            "started_at": utc_now(),
            "adapter": "terminal-profile",
            "image": image,
            "source": benchmark.source,
            "environment_names": [],
            "task_name": metadata["task"]["name"],
            "profile": adapter["profile"]["id"],
            "policy": adapter["policy"],
            "resume_identity": benchmark.raw["resume_identity"],
        }
        store.start(attempt, request)
        started = time.perf_counter()
        harness_result: dict[str, Any] = {"status": "failed", "error": "Harness did not start"}
        verifier_returncode = 1
        verifier_name: str | None = None
        with (attempt / "terminal.log").open("a", encoding="utf-8") as terminal:
            try:
                created = subprocess.run(create, text=True, capture_output=True, check=False)
                if created.returncode != 0:
                    terminal.write(created.stdout)
                    terminal.write(created.stderr)
                    terminal.flush()
                else:
                    started_container = subprocess.run(
                        self._docker("start", container_name),
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if started_container.returncode != 0:
                        terminal.write(started_container.stdout)
                        terminal.write(started_container.stderr)
                        terminal.flush()
                    else:
                        harness_result = run_terminal_profile(
                            profile_id=adapter["profile"]["id"],
                            prompt=prompt,
                            policy=adapter["policy"],
                            container=container_name,
                            trace_path=attempt / "harness_trace.jsonl",
                        )
                        atomic_json(attempt / "harness_result.json", harness_result)
                    copied = subprocess.run(
                        self._docker("cp", f"{container_name}:/app/.", str(workspace)),
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if copied.returncode != 0:
                        terminal.write(copied.stdout)
                        terminal.write(copied.stderr)
                        terminal.flush()
                    verifier_name = f"{container_name}-verifier"
                    verify = self._docker("run", "--rm", "--init", "--name", verifier_name)
                    verify.extend(["--label", "orch.benchmark-platform=1", "--label", "orch.verifier-only=1"])
                    if platform := adapter.get("platform"):
                        verify.extend(["--platform", platform])
                    verify.extend(self._egress_env("bridge"))
                    verify.extend(
                        [
                            "--network",
                            "bridge",
                            "-v",
                            f"{workspace.resolve()}:/app:rw",
                            "-v",
                            f"{task_dir / 'tests'}:/tests:ro",
                            "-v",
                            f"{logs.resolve()}:/logs/verifier:rw",
                            "-w",
                            "/app",
                            image,
                            "bash",
                            "/tests/test.sh",
                        ]
                    )
                    verifier_returncode = stream_process(
                        verify,
                        terminal,
                        prefix=f"[{benchmark.id}:verifier] ",
                    )
            except KeyboardInterrupt:
                result = {
                    **request,
                    "status": "cancelled",
                    "finished_at": utc_now(),
                    "execution_seconds": time.perf_counter() - started,
                }
                store.finish(attempt, result)
                raise
            finally:
                if verifier_name:
                    subprocess.run(
                        self._docker("rm", "-f", verifier_name),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                subprocess.run(
                    self._docker("rm", "-f", container_name),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        reward_path = logs / "reward.txt"
        reward = reward_path.read_text(encoding="utf-8").strip() if reward_path.is_file() else None
        payload = {
            "scores": {"reward": float(reward)} if reward is not None else {},
            "verifier": str(logs),
            "oracle_smoke": False,
        }
        atomic_json(attempt / "payload.json", payload)
        result = {
            **request,
            "status": (
                "completed"
                if harness_result.get("status") == "completed"
                and verifier_returncode == 0
                and reward == "1"
                else "failed"
            ),
            "finished_at": utc_now(),
            "execution_seconds": time.perf_counter() - started,
            "returncode": 0 if harness_result.get("status") == "completed" else 1,
            "verifier_returncode": verifier_returncode,
            "payload": payload,
            "harness": harness_result,
        }
        store.finish(attempt, result)
        return result

    def _run_terminal_task(
        self,
        store: CaseStore,
        benchmark: Benchmark,
        *,
        smoke: bool,
        command_override: list[str] | None,
    ) -> dict[str, Any]:
        adapter = benchmark.adapter
        task_dir, metadata = self._terminal_metadata(benchmark)
        image = metadata.get("environment", {}).get("docker_image") or adapter["image"]
        if not self.image_exists(image):
            built = self.build(benchmark)
            if built["status"] != "completed":
                return self._record_blocked(store, benchmark, "Terminal task image build failed", extra={"build": built})
        shell_command = terminal_agent_command(command_override, smoke)
        if shell_command is None:
            return self._record_blocked(store, benchmark, "Terminal task run requires an agent command after --")
        attempt_number, attempt = store.next_attempt()
        logs = attempt / "verifier"
        workspace = attempt / "workspace"
        logs.mkdir()
        workspace.mkdir()
        container_name = f"bench-{benchmark.id}-{os.getpid()}-{attempt_number}"
        create = self._docker("create", "--init", "--name", container_name)
        create.extend(["--label", "orch.benchmark-platform=1", "--label", f"orch.benchmark={benchmark.id}"])
        if platform := adapter.get("platform"):
            create.extend(["--platform", platform])
        create.extend(["--network", "bridge" if metadata["environment"].get("allow_internet") else "none"])
        create.extend(self._egress_env("bridge" if metadata["environment"].get("allow_internet") else "none"))
        if smoke:
            create.extend(["-v", f"{task_dir / 'solution'}:/solution:ro"])
        create.extend(["-w", "/app", image, "bash", "-lc", shell_command])
        request = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": store.case_id,
            "attempt": attempt_number,
            "started_at": utc_now(),
            "adapter": "terminal-task",
            "image": image,
            "source": benchmark.source,
            "smoke": smoke,
            "environment_names": [],
            "task_name": metadata["task"]["name"],
        }
        store.start(attempt, request)
        started = time.perf_counter()
        agent_returncode = 1
        verifier_returncode = 1
        verifier_name: str | None = None
        with (attempt / "terminal.log").open("a", encoding="utf-8") as terminal:
            try:
                created = subprocess.run(create, text=True, capture_output=True, check=False)
                if created.returncode != 0:
                    terminal.write(created.stdout)
                    terminal.write(created.stderr)
                    terminal.flush()
                else:
                    agent_returncode = stream_process(
                        self._docker("start", "-a", container_name),
                        terminal,
                        prefix=f"[{benchmark.id}:agent] ",
                    )
                    copied = subprocess.run(
                        self._docker("cp", f"{container_name}:/app/.", str(workspace)),
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if copied.returncode != 0:
                        terminal.write(copied.stdout)
                        terminal.write(copied.stderr)
                        terminal.flush()
                    verifier_name = f"{container_name}-verifier"
                    verify = self._docker("run", "--rm", "--init", "--name", verifier_name)
                    verify.extend(["--label", "orch.benchmark-platform=1", "--label", f"orch.benchmark={benchmark.id}"])
                    if platform := adapter.get("platform"):
                        verify.extend(["--platform", platform])
                    verify.extend(self._egress_env("bridge"))
                    verify.extend(
                        [
                            "--network",
                            "bridge",
                            "-v",
                            f"{workspace.resolve()}:/app:rw",
                            "-v",
                            f"{task_dir / 'tests'}:/tests:ro",
                            "-v",
                            f"{logs.resolve()}:/logs/verifier:rw",
                            "-w",
                            "/app",
                            image,
                            "bash",
                            "/tests/test.sh",
                        ]
                    )
                    verifier_returncode = stream_process(
                        verify,
                        terminal,
                        prefix=f"[{benchmark.id}:verifier] ",
                    )
            except KeyboardInterrupt:
                if verifier_name:
                    subprocess.run(
                        self._docker("rm", "-f", verifier_name),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                subprocess.run(
                    self._docker("rm", "-f", container_name),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                result = {
                    **request,
                    "status": "cancelled",
                    "finished_at": utc_now(),
                    "execution_seconds": time.perf_counter() - started,
                }
                store.finish(attempt, result)
                raise
            finally:
                if verifier_name:
                    subprocess.run(
                        self._docker("rm", "-f", verifier_name),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                subprocess.run(
                    self._docker("rm", "-f", container_name),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        reward_path = logs / "reward.txt"
        reward = reward_path.read_text(encoding="utf-8").strip() if reward_path.is_file() else None
        payload = {
            "scores": {"reward": float(reward)} if reward is not None else {},
            "verifier": str(logs),
            "oracle_smoke": smoke,
        }
        atomic_json(attempt / "payload.json", payload)
        result = {
            **request,
            "status": "completed" if agent_returncode == 0 and verifier_returncode == 0 and reward == "1" else "failed",
            "finished_at": utc_now(),
            "execution_seconds": time.perf_counter() - started,
            "returncode": agent_returncode,
            "verifier_returncode": verifier_returncode,
            "payload": payload,
        }
        store.finish(attempt, result)
        return result
