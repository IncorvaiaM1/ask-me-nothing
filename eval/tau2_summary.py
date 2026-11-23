"""Summarize τ²-bench text runs: pass rate, and how much the agent asked the user.

    python -m eval.tau2_summary path/to/results.json [more results.json ...]

τ²-bench scores task success by final database state. We add two counts from the
transcripts: agent messages to the user that contain a question, and agent tool calls.
On ``intake`` (τ-Entity, scaffolded text) the reward is exact capture of a missing value,
which is the outside check that training did not suppress necessary questions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize_run(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    sims = data.get("simulations", [])
    rows = []
    for sim in sims:
        reward = (sim.get("reward_info") or {}).get("reward")
        msgs = sim.get("messages") or []
        to_user = [m for m in msgs if m.get("role") == "assistant" and (m.get("content") or "").strip()
                   and not m.get("tool_calls")]
        questions = sum("?" in (m.get("content") or "") for m in to_user)
        tool_calls = sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant")
        rows.append({"task_id": sim.get("task_id"), "reward": reward, "questions": questions,
                     "agent_messages": len(to_user), "tool_calls": tool_calls})
    scored = [r for r in rows if r["reward"] is not None]
    n = max(1, len(scored))
    return {
        "file": str(path),
        "simulations": len(rows),
        "pass_rate": sum(r["reward"] >= 1.0 - 1e-9 for r in scored) / n,
        "mean_reward": sum(r["reward"] for r in scored) / n,
        "questions_per_conversation": sum(r["questions"] for r in rows) / max(1, len(rows)),
        "agent_messages_per_conversation": sum(r["agent_messages"] for r in rows) / max(1, len(rows)),
        "tool_calls_per_conversation": sum(r["tool_calls"] for r in rows) / max(1, len(rows)),
    }


if __name__ == "__main__":
    out = [summarize_run(p) for p in sys.argv[1:]]
    print("| run | sims | pass | mean reward | questions/conv | agent msgs/conv | tool calls/conv |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for s in out:
        print(f"| {Path(s['file']).parent.name} | {s['simulations']} | {s['pass_rate']:.2f} | {s['mean_reward']:.2f} | "
              f"{s['questions_per_conversation']:.2f} | {s['agent_messages_per_conversation']:.2f} | "
              f"{s['tool_calls_per_conversation']:.2f} |")
    Path("results").mkdir(exist_ok=True)
    Path("results/tau2_summary.json").write_text(json.dumps(out, indent=2))
