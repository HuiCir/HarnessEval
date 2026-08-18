from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .core import RunContext, extract_json


ACTION_SYSTEM = """You are a tool-using agent. Work only from the task and complete tool observations.
Available tools: {tools}
Return exactly one JSON object per turn, either:
{{"tool":"tool_name","arguments":{{...}}}}
or {{"final":"answer"}}.
Do not invent a tool result."""


def _normalize_action(action: dict[str, Any], names: list[str]) -> dict[str, Any]:
    if "tool" in action or "final" in action:
        return action
    if len(action) == 1:
        name, arguments = next(iter(action.items()))
        if name in names and isinstance(arguments, dict):
            return {"tool": name, "arguments": arguments}
    return action


async def _json_tool_loop(ctx: RunContext, role: str) -> str:
    messages = [
        {"role": "system", "content": ACTION_SYSTEM.format(tools=ctx.environment.schema)},
        {"role": "user", "content": ctx.prompt},
    ]
    for _ in range(ctx.max_turns):
        raw = await ctx.complete(role, messages, json_mode=True)
        try:
            action = _normalize_action(extract_json(raw, expected_type=dict), ctx.environment.names)
        except ValueError as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Protocol error: {exc}. Return one complete action object."},
                ]
            )
            continue
        if "final" in action:
            return str(action["final"])
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Protocol error: arguments must be one JSON object."},
                ]
            )
            continue
        result = await ctx.environment.call(str(action.get("tool", "")), arguments)
        canonical_action = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        messages.extend(
            [
                {"role": "assistant", "content": canonical_action},
                {"role": "user", "content": "Observation: " + json.dumps(result, ensure_ascii=False)},
            ]
        )
    raise RuntimeError("Agent-loop turn budget exhausted without a final answer")


def _parse_react(text: str) -> dict[str, Any]:
    final = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    action = re.search(r"Action\s*:\s*([\w.-]+)", text, flags=re.IGNORECASE)
    if final and (action is None or final.start() < action.start()):
        return {"final": final.group(1).strip()}
    if not action:
        raise ValueError("Response is neither a ReAct action nor a final answer")
    # The arguments are the first JSON object after the action name. The literal
    # "Action Input:" label the parser used to demand is only described in prose by the
    # system prompt, so a model that names the tool and then emits its JSON has followed
    # the protocol as stated; requiring the label turned every such turn into a retry
    # until the budget was gone. Every other profile parses arguments with extract_json,
    # which never demanded a label either, so ReAct was the only stricter contract here.
    decoder = json.JSONDecoder()
    tail = text[action.end() :]
    for start, character in enumerate(tail):
        if character != "{":
            continue
        try:
            arguments, _ = decoder.raw_decode(tail[start:])
        except json.JSONDecodeError:
            continue
        return {"tool": action.group(1), "arguments": arguments}
    raise ValueError("ReAct action names a tool but supplies no JSON action input")


async def run_react(ctx: RunContext) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Solve the task by interleaving Thought, Action, and Observation, as in ReAct.\n"
                f"Available tools: {ctx.environment.schema}\n"
                "For a tool turn emit Thought, Action, and JSON Action Input. When complete emit Thought and Final Answer. "
                "Never invent an observation."
            ),
        },
        {"role": "user", "content": ctx.prompt},
    ]
    for _ in range(ctx.max_turns):
        raw = await ctx.complete("react", messages)
        try:
            action = _parse_react(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Protocol error: {exc}. Emit one complete ReAct step."},
                ]
            )
            continue
        if "final" in action:
            return str(action["final"])
        result = await ctx.environment.call(action["tool"], action["arguments"])
        messages.extend(
            [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Observation: " + json.dumps(result, ensure_ascii=False)},
            ]
        )
    raise RuntimeError("ReAct turn budget exhausted without a final answer")


def _substitute(value: Any, observations: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        step, _, field = value[1:].partition(".")
        selected = observations[step]
        for part in field.split(".") if field else []:
            selected = selected[int(part)] if isinstance(selected, list) else selected[part]
        return selected
    if isinstance(value, list):
        return [_substitute(item, observations) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, observations) for key, item in value.items()}
    return value


async def run_plan_execute(ctx: RunContext) -> str:
    plan = await ctx.complete_json(
        "planner",
        [
            {
                "role": "user",
                "content": (
                    "Create a complete ordered execution plan. Do not execute or answer.\n"
                    f"Available tools: {ctx.environment.schema}\n"
                    'Return JSON only: {"steps":[{"id":"s1","tool":"tool_name","arguments":{}}]}. '
                    "Later arguments may reference $step_id.field.\n"
                    f"Task: {ctx.prompt}"
                ),
            }
        ],
    )
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Plan-and-Execute planner omitted steps")
    observations: dict[str, Any] = {}
    for step in steps:
        arguments = _substitute(step["arguments"], observations)
        observations[str(step["id"])] = await ctx.environment.call(str(step["tool"]), arguments)
    return await ctx.complete(
        "executor_final",
        [
            {
                "role": "user",
                "content": (
                    f"Return the final answer to this task: {ctx.prompt}\n"
                    f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
                    f"Observations: {json.dumps(observations, ensure_ascii=False)}"
                ),
            }
        ],
    )


async def run_cmws(ctx: RunContext) -> str:
    plan = await ctx.complete_json(
        "manager",
        [
            {
                "role": "user",
                "content": (
                    "Decompose the task into independent worker assignments. Do not execute or answer.\n"
                    f"Available tools: {ctx.environment.schema}\n"
                    'Return JSON only: {"assignments":[{"id":"w1","instruction":"...","tool":"tool_name","arguments":{}}]}.\n'
                    f"Task: {ctx.prompt}"
                ),
            }
        ],
    )
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("CMWS manager omitted assignments")
    semaphore = asyncio.Semaphore(ctx.max_parallel) if ctx.max_parallel is not None else None

    async def worker(assignment: dict[str, Any]) -> dict[str, Any]:
        async def execute() -> dict[str, Any]:
            result = await ctx.environment.call(str(assignment["tool"]), assignment["arguments"])
            return {
                "id": assignment["id"],
                "instruction": assignment.get("instruction", ""),
                "result": result,
            }

        if semaphore is None:
            return await execute()
        async with semaphore:
            return await execute()

    reports = await asyncio.gather(*(worker(assignment) for assignment in assignments))
    decision = await ctx.complete_json(
        "manager_synthesis",
        [
            {
                "role": "user",
                "content": (
                    "Synthesize the independent reports. Return JSON only, either one final tool action or a final answer.\n"
                    f"Available tools: {ctx.environment.schema}\n"
                    f"Task: {ctx.prompt}\nReports: {json.dumps(reports, ensure_ascii=False)}\n"
                    'Schema: {"tool":"tool_name","arguments":{}} or {"final":"answer"}'
                ),
            }
        ],
    )
    if "final" in decision:
        return str(decision["final"])
    result = await ctx.environment.call(str(decision["tool"]), decision["arguments"])
    return await ctx.complete(
        "manager_delivery",
        [
            {
                "role": "user",
                "content": f"Task: {ctx.prompt}\nFinal tool result: {json.dumps(result, ensure_ascii=False)}\nReturn the answer.",
            }
        ],
    )


async def run_profile(ctx: RunContext) -> str:
    if ctx.profile == "actor-only":
        return await _json_tool_loop(ctx, "actor")
    if ctx.profile == "react":
        return await run_react(ctx)
    if ctx.profile == "plan-execute":
        return await run_plan_execute(ctx)
    if ctx.profile == "cmws":
        return await run_cmws(ctx)
    from .paper_methods import (
        run_aflow,
        run_dylan,
        run_llmcompiler,
        run_magentic_one,
        run_multi_persona,
        run_rewoo,
        run_sa,
    )

    extended = {
        "aflow": run_aflow,
        "dylan": run_dylan,
        "magentic-one": run_magentic_one,
        "multi-persona": run_multi_persona,
        "llmcompiler": run_llmcompiler,
        "rewoo": run_rewoo,
        "sa": run_sa,
    }
    if runner := extended.get(ctx.profile):
        return await runner(ctx)
    raise ValueError(f"Unknown harness profile: {ctx.profile}")
