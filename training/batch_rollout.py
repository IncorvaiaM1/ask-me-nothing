"""Batched multi-turn rollouts: every active dialogue's next turn in one generate call.

The engine is anything with ``generate(conversations, temperature, max_tokens) ->
list[str]``. ``VLLMEngine`` wraps an in-process ``vllm.LLM`` and swaps LoRA adapters
between RL steps; tests use a scripted engine. The model's reply is kept verbatim as
the assistant message (reasoning stripped, tool call as text), so the training
sequence is built from what the model actually wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from baselines.prompted import parse_text_tool_call, strip_thinking
from brokerdesk.env import MAX_TURNS, BrokerDeskEnv, RewardWeights
from brokerdesk.policy import Policy, default_policy
from brokerdesk.tasks import Task


@dataclass
class Job:
    task: Task
    system: str
    seed: int = 0
    weights: RewardWeights | None = None
    injection: object = None
    group_id: str = ""
    index: int = 0
    client_factory: object = None


@dataclass
class Episode:
    job: Job
    messages: list[dict]
    transcript: list[dict]
    score: dict
    advantage: float = 0.0
    turn_advantages: list[float] = field(default_factory=list)


class VLLMEngine:
    """In-process vLLM with optional LoRA. Imported lazily: vLLM only exists on the GPU node."""

    def __init__(self, model: str, max_lora_rank: int = 16, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 16384, thinking: bool = False, seed: int = 0):
        from vllm import LLM
        self.llm = LLM(model=model, enable_lora=True, max_lora_rank=max_lora_rank, max_loras=2,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       seed=seed, enable_prefix_caching=True)
        self.thinking = thinking
        self.lora = None
        self._lora_id = 0

    def set_adapter(self, path: str | None) -> None:
        from vllm.lora.request import LoRARequest
        if path is None:
            self.lora = None
            return
        self._lora_id += 1
        self.lora = LoRARequest(f"adapter{self._lora_id}", self._lora_id, path)

    def generate(self, conversations: list[list[dict]], temperature: float, max_tokens: int) -> list[str]:
        from vllm import SamplingParams

        from brokerdesk.tools import openai_tools
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                top_p=1.0 if temperature > 0 else 1.0)
        outs = self.llm.chat(conversations, params, tools=openai_tools(), lora_request=self.lora,
                             chat_template_kwargs={"enable_thinking": self.thinking}, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def run_jobs(engine, jobs: list[Job], policy: Policy | None = None, max_turns: int = MAX_TURNS,
             temperature: float = 1.0, max_tokens: int = 512) -> list[Episode]:
    policy = policy or default_policy()
    envs, msgs, transcripts, done = [], [], [], []
    for job in jobs:
        env = BrokerDeskEnv(job.task, policy, job.seed, job.weights, max_turns, job.injection, job.client_factory)
        opening = env.reset()
        envs.append(env)
        msgs.append([{"role": "system", "content": job.system}, {"role": "user", "content": opening}])
        transcripts.append([{"role": "client", "content": opening}])
        done.append(False)
    while not all(done):
        active = [i for i, d in enumerate(done) if not d]
        texts = engine.generate([msgs[i] for i in active], temperature, max_tokens)
        for i, text in zip(active, texts):
            tool, args = parse_text_tool_call(text)
            msgs[i].append({"role": "assistant", "content": strip_thinking(text)})
            obs, done[i] = envs[i].step(tool, args)
            msgs[i].append({"role": "tool", "content": obs})
            entry = {"role": "agent", "tool": tool, "arguments": args}
            if obs.startswith('{"error"'):
                entry["raw"] = text[:2000]          # what the model actually wrote
            transcripts[i] += [entry, {"role": "tool", "content": obs}]
    return [Episode(job=j, messages=m, transcript=t, score=e.score())
            for j, m, t, e in zip(jobs, msgs, transcripts, envs)]


class ScriptedEngine:
    """Test engine: each conversation is driven by a scripted agent, rendered as tool-call text."""

    def __init__(self, make_agent):
        self.make_agent = make_agent
        self.agents: dict[int, object] = {}

    def generate(self, conversations, temperature, max_tokens):
        out = []
        for conv in conversations:
            key = id(conv)
            if key not in self.agents:
                self.agents[key] = self.make_agent(len(self.agents))
                self.agents[key].reset(conv[1]["content"])
                tool, args = self.agents[key].act(None)
            else:
                tool, args = self.agents[key].act(conv[-1]["content"])
            out.append("<tool_call>\n" + json.dumps({"name": tool, "arguments": args}) + "\n</tool_call>")
        return out
