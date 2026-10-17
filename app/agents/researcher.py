"""A researcher: edits train.py with run_code, then reports what it learned."""

import logging
from dataclasses import dataclass

from app.agents import prompts
from app.agents.base import AgentResult, run_agent
from app.agents.tools import Budget, get_memory, researcher_run_code
from app.config import get_settings
from app.lab import Lab
from app.schemas import Experiment, Usage

PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 8192,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True}
    },
}


log = logging.getLogger(__name__)


@dataclass
class ResearcherOutcome:
    agent: str
    focus: str
    report: str
    runs: list[Experiment]
    usage: Usage


async def run_researcher(
    lab: Lab, *, generation: int, index: int, focus: str, system: str
) -> ResearcherOutcome:
    agent = f"r{index}"
    budget = Budget(lab.researcher_runs)
    runs: list[Experiment] = []
    tools = [researcher_run_code(lab, agent=agent, generation=generation, budget=budget, runs=runs)]
    if lab.setup == "shared_brain":
        tools.insert(0, get_memory(lab))
    model = get_settings().researcher_model
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.researcher_task(focus)},
    ]
    try:
        result = await run_agent(
            model=model,
            messages=messages,
            tools=tools,
            max_turns=lab.config.researcher_turns,
            params=PARAMS,
        )
    except Exception as e:  # one researcher failing must not end the research run
        log.exception("g%d %s crashed", generation, agent)
        report = f"(researcher crashed: {type(e).__name__}: {e})"[:1000]
        result = AgentResult(report, messages, Usage(), 0)
    lab.log_agent(
        {
            "agent": agent,
            "role": "researcher",
            "generation": generation,
            "focus": focus,
            "model": model,
            "report": result.report,
            "run_ids": [r.run_id for r in runs],
            "turns": result.turns,
            "usage": result.usage.model_dump(),
            "transcript": result.transcript,
        }
    )
    return ResearcherOutcome(agent, focus, result.report, runs, result.usage)
