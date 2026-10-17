"""The curator: verifies the top claim, then keeps memory current, lean and conflict-free."""

import logging
from dataclasses import dataclass

from app.agents import prompts
from app.agents.base import AgentResult, run_agent
from app.agents.researcher import ResearcherOutcome
from app.agents.tools import (
    Budget,
    Verification,
    curator_run_code,
    get_memory,
    remove_memory,
    upsert_memory,
)
from app.config import get_settings
from app.lab import Lab
from app.schemas import Usage
from app.scoring import Ranked

PARAMS = {
    "max_tokens": 8192,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True}
    },
}

log = logging.getLogger(__name__)


@dataclass
class CuratorOutcome:
    report: str
    verifications: list[Verification]
    runs_used: int
    usage: Usage


async def run_curator(
    lab: Lab, *, generation: int, outcomes: list[ResearcherOutcome], ranked: list[Ranked]
) -> CuratorOutcome:
    budget = Budget(lab.config.curator_runs)
    verifications: list[Verification] = []
    tools = [
        get_memory(lab),
        curator_run_code(lab, generation=generation, budget=budget, verifications=verifications),
        upsert_memory(lab, generation=generation),
        remove_memory(lab, generation=generation),
    ]
    by_agent = {o.agent: o for o in outcomes}
    reports = [
        (r.agent, by_agent[r.agent].focus, by_agent[r.agent].report, by_agent[r.agent].runs)
        for r in ranked
    ]
    model = get_settings().curator_model
    messages = [
        {"role": "system", "content": prompts.curator_system(lab)},
        {"role": "user", "content": prompts.curator_brief(lab, generation, reports, ranked)},
    ]
    try:
        result = await run_agent(
            model=model,
            messages=messages,
            tools=tools,
            max_turns=lab.config.curator_turns,
            params=PARAMS,
        )
    except Exception as e:  # keep the run going; memory keeps whatever was written so far
        log.exception("g%d curator crashed", generation)
        report = f"(curator crashed: {type(e).__name__}: {e})"[:1000]
        result = AgentResult(report, messages, Usage(), 0)
    runs_used = lab.config.curator_runs - budget.left
    lab.log_agent(
        {
            "agent": "curator",
            "role": "curator",
            "generation": generation,
            "focus": None,
            "model": model,
            "report": result.report,
            "run_ids": [rid for v in verifications for rid in v.run_ids],
            "turns": result.turns,
            "usage": result.usage.model_dump(),
            "transcript": result.transcript,
        }
    )
    return CuratorOutcome(result.report, verifications, runs_used, result.usage)
