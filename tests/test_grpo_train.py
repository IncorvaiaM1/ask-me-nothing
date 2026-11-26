import json

import pytest

from baselines.prompted import system_prompt
from baselines.scripted import HeuristicAgent
from brokerdesk.policy import default_policy
from brokerdesk.tasks import generate_tasks
from training.batch_rollout import Job, ScriptedEngine, run_jobs
from training.config import REWARD_CONFIGS
from training.grpo_train import add_reference, build_batch, compute_advantages, make_jobs, train_step

POLICY = default_policy()
TASKS = [t for t in generate_tasks(30, seed=21) if t.underspecified][:4]
MODES = ("need_to_know", "never_ask", "always_ask", "need_to_know")


def _episodes(weights=None):
    jobs = [Job(task=t, system=system_prompt(POLICY), seed=1, weights=weights or REWARD_CONFIGS["tiered"],
                group_id=t.task_id, index=i) for t in TASKS for i in range(4)]
    return run_jobs(ScriptedEngine(lambda i: HeuristicAgent(MODES[i % 4], POLICY)), jobs, POLICY)


class SmallTokenizer:
    """Character-level tokenizer with a Qwen3-like chat template and a vocabulary built on
    the fly, small enough for a tiny test model."""

    SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
               "<tool_response>", "</tool_response>"]

    def __init__(self):
        self.vocab = {s: i for i, s in enumerate(self.SPECIAL)}
        self.inv = dict(enumerate(self.SPECIAL))

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False,
                            enable_thinking=True, **kw):
        text = "<|im_start|>system\ntools<|im_end|>\n" if tools else ""
        last_query = max((n for n, m in enumerate(messages) if m["role"] == "user"), default=-1)
        for n, m in enumerate(messages):
            body = m.get("content") or ""
            if m["role"] == "tool":
                text += f"<|im_start|>user\n<tool_response>\n{body}\n</tool_response><|im_end|>\n"
                continue
            if m["role"] == "assistant" and n == len(messages) - 1 and n > last_query:
                body = "<think>\n\n</think>\n\n" + body
            text += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        import re
        ids = []
        for piece in re.split("(" + "|".join(map(re.escape, self.SPECIAL)) + ")", text):
            if piece in self.vocab and piece in self.SPECIAL:
                ids.append(self.vocab[piece])
            else:
                ids += [self._id(c) for c in piece]
        return ids

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)


def test_dialogue_advantages_rank_the_behaviours():
    eps = _episodes()
    compute_advantages(eps, "dialogue")
    for t in TASKS:
        g = [e for e in eps if e.job.group_id == t.task_id]
        assert sum(e.advantage for e in g) == pytest.approx(0, abs=1e-6)
        best = max(g, key=lambda e: e.advantage)
        assert best.score["reward"] == max(e.score["reward"] for e in g)


def test_turn_advantages_charge_the_turns_that_asked():
    eps = _episodes()
    compute_advantages(eps, "turn")
    over = next(e for e in eps if e.job.index == 2 and any(e.turn_advantages))
    asks = [k for k, x in enumerate([x for x in over.transcript if x["role"] == "agent"]) if x["tool"] == "ask_client"]
    after = len(over.turn_advantages) - 1
    # cost-to-go shrinks as the questions are behind it, so later turns look better
    assert over.turn_advantages[asks[0]] < over.turn_advantages[after]
    assert len(over.turn_advantages) == over.score["turns"]


def test_batch_and_a_real_gradient_step_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    tok = SmallTokenizer()
    eps = _episodes()
    for mode in ("dialogue", "turn"):
        compute_advantages(eps, mode)
        batch, skipped = build_batch(tok, eps, mode)
        assert batch and skipped == 0
        for ids, mask, adv in batch:
            assert len(ids) == len(mask) == len(adv) and sum(mask) > 0
            trained = tok.decode([i for i, m in zip(ids, mask) if m])
            assert "<tool_call>" in trained and "book_value" not in trained
    compute_advantages(eps, "dialogue")
    batch, _ = build_batch(tok, eps, "dialogue")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok.vocab) + 8, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                   head_dim=16, max_position_embeddings=16384)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                       task_type="CAUSAL_LM", lora_dropout=0.0))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-2)
    small = batch[:6]
    losses = [train_step(model, opt, small, "cpu")["loss"] for _ in range(4)]
    assert all(l == l for l in losses)                       # no NaN
    assert losses[-1] < losses[0]                            # the surrogate falls on a fixed batch
    stats = train_step(model, opt, [], "cpu")
    assert stats["dialogues"] == 0


def test_kl_to_the_sft_reference_starts_at_zero_and_grows_with_training(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    tok = SmallTokenizer()
    eps = _episodes()
    compute_advantages(eps, "dialogue")
    batch, _ = build_batch(tok, eps, "dialogue")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=len(tok.vocab) + 8, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                   head_dim=16, max_position_embeddings=16384)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                       task_type="CAUSAL_LM", lora_dropout=0.0,
                                                       init_lora_weights=False))
    model.save_pretrained(tmp_path / "sft")
    add_reference(model, str(tmp_path / "sft"))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and not any(".ref." in n for n in trainable)
    ref_before = {n: p.detach().clone() for n, p in model.named_parameters() if ".ref." in n}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-2)
    small = batch[:6]
    first = train_step(model, opt, small, "cpu", kl_beta=0.1)
    assert abs(first["kl"]) < 1e-5                           # policy == reference at the start
    for _ in range(3):
        last = train_step(model, opt, small, "cpu", kl_beta=0.1)
    assert last["kl"] > first["kl"]
    assert all(torch.equal(p, ref_before[n]) for n, p in model.named_parameters() if ".ref." in n)
    model.save_pretrained(tmp_path / "out", selected_adapters=["default"])
    assert not (tmp_path / "out" / "ref").exists()


def test_jobs_share_price_paths_within_groups_and_steerable_costs_reach_the_prompt():
    import random
    jobs = make_jobs(TASKS[:2], 3, step=5, arm="tiered", steerable=True, rng=random.Random(0), base_system="SYS")
    assert len(jobs) == 6 and len({(j.group_id, j.seed) for j in jobs}) == 2
    g = [j for j in jobs if j.group_id == jobs[0].group_id]
    assert len({id(j.weights) for j in g}) == 1 and "Collection costs this session" in g[0].system
    c = g[0].weights.privacy_cost
    assert c[0] <= c[1] <= c[2]
    plain = make_jobs(TASKS[:1], 2, 1, "flat", False, random.Random(0), "SYS")
    assert plain[0].system == "SYS" and plain[0].weights is REWARD_CONFIGS["flat"]


def test_eval_script_end_to_end_with_scripted_engine_and_stub_llm_client(tmp_path, monkeypatch):
    import brokerdesk.clients as clients
    import training.batch_rollout as br
    from eval import eval_adapter

    class Engine(br.ScriptedEngine):
        def __init__(self, *a, **k):
            super().__init__(lambda i: HeuristicAgent("need_to_know", POLICY))

        def set_adapter(self, path):
            self.adapter = path

    def stub_chat(model, url, seed=0, **k):
        def chat(messages):
            facts = dict(l[2:].split(": ", 1) for l in messages[0]["content"].splitlines() if l.startswith("- ")
                         and ": " in l)
            q = messages[-1]["content"].lower()
            for key, value in facts.items():
                if key.split()[0] in q:
                    return f"It's {value}."
            return "I'd rather not say."
        return chat

    monkeypatch.setattr(br, "VLLMEngine", Engine)
    monkeypatch.setattr(clients, "openai_chat", stub_chat)
    res = eval_adapter.main(["--label", "ntk", "--split", "val", "--n", "9", "--injection", "paired",
                             "--attack-set", "heldout", "--out", str(tmp_path)])
    assert res["clean"]["ntk"]["success"] == 1.0 and res["injected"]["ntk"]["attack_success_rate"] == 0.0
    assert (tmp_path / "ntk__val.jsonl").exists() and (tmp_path / "ntk+inj-heldout__val.summary.json").exists()
    res = eval_adapter.main(["--label", "ntk", "--split", "val", "--n", "9", "--client", "llm",
                             "--persona", "terse", "--out", str(tmp_path)])
    s = res["clean"]["ntk+llm-terse"]
    assert s["client_answers"] > 0 and s["client_correct_rate"] == 1.0 and s["success"] == 1.0
    res = eval_adapter.main(["--label", "steer", "--split", "val", "--n", "3", "--steer-costs", "0.001,0.002,0.5",
                             "--out", str(tmp_path)])
    cfg = json.loads((tmp_path / "steer__val.summary.json").read_text())["config"]
    assert cfg["steer_costs"] == "0.001,0.002,0.5"
