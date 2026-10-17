"""A short-lived researcher: reads memory and the best train.py, runs experiments, reports."""

from dataclasses import dataclass, field
from typing import Any

from app import events
from app.agents.base import Tool, run_agent
from app.config import ResearchConfig, get_settings
from app.lab import Lab
from app.services import search


@dataclass
class ResearcherOutcome:
    name: str
    focus: str
    report: str
    run_log: list[dict[str, Any]]
    usage: dict[str, int]
    turns: int
    messages: list[dict[str, Any]] = field(repr=False)

    def best_run(self) -> dict[str, Any] | None:
        ok = [r for r in self.run_log if r["val_loss"] is not None]
        return min(ok, key=lambda r: r["val_loss"]) if ok else None

    def run_log_text(self) -> str:
        lines = [
            f"- {r['run_id']}: {r['status']}, val "
            + (f"{r['val_loss']:.5f}" if r["val_loss"] is not None else "-")
            + f" | {r['note']}"
            for r in self.run_log
        ]
        return "\n".join(lines) or "(no runs)"


RESEARCHER_PARAMS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 8192,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True}
    },
}


def researcher_tools(
    *,
    lab: Lab,
    name: str,
    gen: int,
    best_version: int,
    runs: int,
    memory: str | None,
    cfg: ResearchConfig,
    run_log: list[dict[str, Any]],
) -> list[Tool]:
    state = {"runs_left": runs, "searches_left": cfg.web_searches}

    async def run_code(code: str, note: str = "") -> dict[str, Any]:
        if state["runs_left"] <= 0:
            return {"error": "no runs left; write your report now", "runs_left": 0}
        state["runs_left"] -= 1
        exp = await lab.run(code, cfg.research_seed, name, note, gen, best_version)
        run_log.append(
            {"run_id": exp.id, "note": note, "status": exp.status, "val_loss": exp.val_loss}
        )
        out: dict[str, Any] = {
            "run_id": exp.id,
            "status": exp.status,
            "val_loss": exp.val_loss,
            "output": exp.output_tail,
            "runs_left": state["runs_left"],
        }
        if state["runs_left"] == 0:
            out["next"] = "No runs left. Write your report now."
        return out

    tools = [
        Tool(
            "run_code",
            "Run a whole train.py once (seed 0) and return its val loss. Uses one run.",
            {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The whole train.py file."},
                    "note": {"type": "string", "description": "One line: what changed."},
                },
                "required": ["code", "note"],
            },
            run_code,
        )
    ]

    if memory is not None:

        async def get_memory() -> dict[str, Any]:
            return {"memory": memory or "(empty so far)"}

        tools.append(
            Tool(
                "get_memory",
                "Return the team memory.",
                {"type": "object", "properties": {}},
                get_memory,
            )
        )

    if get_settings().ydc_api_key:

        async def web_search(query: str) -> dict[str, Any]:
            if state["searches_left"] <= 0:
                return {"error": "no web searches left"}
            state["searches_left"] -= 1
            try:
                results = await search.web_search(query, cfg.web_results)
            except Exception as e:  # search is best effort; never crash the researcher
                return {"error": f"search failed: {e}"}
            return {"results": results, "searches_left": state["searches_left"]}

        tools.append(
            Tool(
                "web_search",
                "Search the web. Results are hypotheses, not facts.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                web_search,
            )
        )
    return tools


async def run_researcher(
    *,
    lab: Lab,
    name: str,
    focus: str,
    system: str,
    task: str,
    gen: int,
    best_version: int,
    runs: int,
    memory: str | None,
    cfg: ResearchConfig,
) -> ResearcherOutcome:
    run_log: list[dict[str, Any]] = []
    tools = researcher_tools(
        lab=lab,
        name=name,
        gen=gen,
        best_version=best_version,
        runs=runs,
        memory=memory,
        cfg=cfg,
        run_log=run_log,
    )
    with events.scope(agent=name, role="researcher", gen=gen):
        result = await run_agent(
            model=get_settings().researcher_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": task}],
            tools=tools,
            params=RESEARCHER_PARAMS,
            max_turns=cfg.max_turns,
            wrap_up="You are out of turns. Write your report now, in the required format.",
            public_info={"focus": focus, "run_budget": runs, "best_version": best_version},
        )
    return ResearcherOutcome(
        name=name,
        focus=focus,
        report=result.final,
        run_log=run_log,
        usage=result.usage,
        turns=result.turns,
        messages=result.messages,
    )
