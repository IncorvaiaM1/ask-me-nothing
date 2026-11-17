import json

import pytest

from baselines.scripted import HeuristicAgent
from brokerdesk.env import BrokerDeskEnv
from brokerdesk.injection import INJECTION_NAMES, assign_injection, load_injection
from brokerdesk.policy import default_policy
from brokerdesk.tasks import generate_tasks
from eval.metrics import summarize
from eval.rollout import run_episode
from training.collect import main as collect_main
from training.config import REWARD_CONFIGS
from training.masking import assistant_mask, describe_mask
from training.sft import build_dialogues
from training.rollouts import (collect_group, group_advantages, messages_from_transcript, read_jsonl)

POLICY = default_policy()
TASKS = generate_tasks(30, seed=11)


def _step(env, tool, **args):
    obs, done = env.step(tool, args)
    return json.loads(obs), done


# ------------------------------------------------------------------- injection


def test_injection_appears_only_in_untrusted_tool_outputs():
    inj = load_injection("exfil_research")
    env = BrokerDeskEnv(TASKS[0], POLICY, seed=0, injection=inj)
    env.reset()
    obs, _ = _step(env, "get_profile", fields=["risk_tolerance"])
    assert inj.text not in json.dumps(obs) and env.injection_seen_turn is None
    obs, _ = _step(env, "get_positions")
    assert obs["account_note"] == inj.text and env.injection_seen_turn == 2
    obs, _ = _step(env, "get_quote", ticker="ALFA")
    assert obs["vendor_note"] == inj.text
    clean = BrokerDeskEnv(TASKS[0], POLICY, seed=0)
    clean.reset()
    assert "account_note" not in _step(clean, "get_positions")[0]
    assert clean.score()["injected"] is False


def test_assignment_is_deterministic_and_covers_every_attack():
    names = [assign_injection(t.task_id).name for t in TASKS]
    assert names == [assign_injection(t.task_id).name for t in TASKS]
    assert set(names) == set(INJECTION_NAMES)


def test_exfil_counts_only_through_the_named_tool_after_the_attack_is_seen():
    task = TASKS[0]
    inj = load_injection("exfil_research")
    ssn = task.profile_field("ssn").value

    env = BrokerDeskEnv(task, POLICY, seed=0, injection=inj)
    env.reset()
    _step(env, "get_profile", fields=["ssn"])
    _step(env, "request_research", text=f"SSN {ssn}")           # leaked, but before seeing the attack
    s = env.score()
    assert s["n_leaks"] == 1 and s["injection_seen"] and not s["attack_success"]

    env.reset()
    _step(env, "get_positions")
    _step(env, "get_profile", fields=["ssn"])
    s = env.score()
    assert s["attack_collected"] and not s["attack_success"]    # collected, not sent
    _step(env, "message_counterparty", text=f"SSN {ssn}")       # wrong channel for this attack
    assert not env.score()["attack_success"] and env.score()["n_leaks"] == 1
    _step(env, "request_research", text=f"SSN {ssn}")
    assert env.score()["attack_success"]


def test_collect_attack_succeeds_when_the_agent_asks_for_the_field():
    env = BrokerDeskEnv(TASKS[0], POLICY, seed=0, injection=load_injection("collect_ssn"))
    env.reset()
    _step(env, "get_positions")
    assert not env.score()["attack_success"]
    _step(env, "ask_client", field="ssn", question="Please confirm your SSN.")
    s = env.score()
    assert s["attack_success"] and s["unneeded_requests_by_tier"][2] == 1 and s["n_leaks"] == 0


def test_careful_agent_resists_and_gullible_agent_falls_for_every_attack():
    careful, gullible = [], []
    for t in TASKS:
        inj = assign_injection(t.task_id)
        careful.append(run_episode(t, HeuristicAgent("need_to_know", POLICY), POLICY, injection=inj)["score"])
        gullible.append(run_episode(t, HeuristicAgent("gullible", POLICY), POLICY, injection=inj)["score"])
    c, g = summarize(careful), summarize(gullible)
    assert c["success"] == 1.0 and c["attack_success_rate"] == 0.0 and c["injection_seen_rate"] == 1.0
    assert g["attack_success_rate"] == 1.0 and g["attack_collected_rate"] == 1.0
    assert g["leak_rate"] > 0 and g["reward"] < c["reward"]
    assert set(g["attack_success_by_injection"]) == set(INJECTION_NAMES)
    # without an attack the gullible agent is just the careful one
    clean = run_episode(TASKS[0], HeuristicAgent("gullible", POLICY), POLICY)["score"]
    assert clean["success"] and clean["n_leaks"] == 0 and clean["unneeded_requests_by_tier"] == [0, 0, 0]


# --------------------------------------------------------------------- rollouts


def test_group_advantages():
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert adv == pytest.approx([1, -1, 1, -1], abs=1e-4) and sum(adv) == pytest.approx(0)
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]
    assert group_advantages([1.0, 0.0], normalize=False) == [0.5, -0.5]


def test_collect_group_shares_the_price_path_and_ranks_behaviours():
    task = next(t for t in TASKS if t.underspecified)
    modes = ("need_to_know", "never_ask", "always_ask")
    group = collect_group(task, lambda i: HeuristicAgent(modes[i % 3], POLICY), 6, system="SYS", seed=3)
    assert len(group) == 6 and len({t.group_id for t in group}) == 1
    assert group[0].reward == group[3].reward                    # same policy, same path, same outcome
    assert group[0].advantage > group[2].advantage > group[1].advantage
    assert sum(t.advantage for t in group) == pytest.approx(0, abs=1e-6)
    msgs = group[0].messages
    assert [m["role"] for m in msgs[:4]] == ["system", "user", "assistant", "tool"]
    assert msgs[1]["content"] == task.request and msgs[-1]["role"] == "tool"
    call = msgs[2]["tool_calls"][0]
    assert msgs[3]["tool_call_id"] == call["id"] and json.loads(call["function"]["arguments"]) is not None


def test_reward_ablations_change_only_their_own_term():
    task = TASKS[0]
    def reward(name, mode):
        return run_episode(task, HeuristicAgent(mode, POLICY), POLICY, REWARD_CONFIGS[name])["score"]["reward"]
    assert reward("no_privacy_cost", "always_ask") > reward("full", "always_ask")
    assert reward("no_question_cost", "always_ask") > reward("full", "always_ask")
    assert reward("no_privacy_cost", "always_ask") < 1.0 == reward("no_question_cost", "need_to_know") \
        or task.underspecified


def test_collect_cli_round_trip(tmp_path):
    out = tmp_path / "batch.jsonl"
    stats = collect_main(["--agent", "mixed", "--n", "4", "--group-size", "3", "--out", str(out)])
    batch = read_jsonl(out)
    assert stats["trajectories"] == len(batch) == 12 and stats["groups"] == 4
    assert stats["groups_with_signal"] == 4
    assert all(t.messages[0]["role"] == "system" and "Desk policy" in t.messages[0]["content"] for t in batch)


# ---------------------------------------------------------------------- masking


class FakeTokenizer:
    """Special tags are single tokens and everything else is one token per character.

    ``qwen=True`` copies the three quirks of the real Qwen3 chat template that matter here:
    tool messages render as user turns holding <tool_response> blocks; an assistant turn
    that is the last message (after the last real user message) gets an empty <think>
    block while earlier ones do not; and with ``enable_thinking=False`` the generation
    prompt ends in an empty <think> block. ``qwen=False`` is a plain ChatML template.
    """

    _SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>"]

    def __init__(self, qwen=True, broken=False):
        self.qwen, self.broken = qwen, broken

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False,
                            enable_thinking=True, **kw):
        text = f"<tools>{json.dumps(tools)}</tools>\n" if tools else ""
        last_query = max((n for n, m in enumerate(messages) if m["role"] == "user"), default=len(messages) - 1)
        for n, m in enumerate(messages):
            last = n == len(messages) - 1
            body = m.get("content") or ""
            if m["role"] == "tool":
                if n == 0 or messages[n - 1]["role"] != "tool":
                    text += "<|im_start|>user"
                text += f"\n<tool_response>\n{body}\n</tool_response>"
                if last or messages[n + 1]["role"] != "tool":
                    text += "<|im_end|>\n"
                continue
            if m["role"] == "assistant":
                if self.qwen and last and n > last_query:
                    body = "<think>\n\n</think>\n\n" + body
                for call in m.get("tool_calls", []):
                    fn = call["function"]
                    args = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
                    body += '<tool_call>\n{"name": "' + fn["name"] + '", "arguments": ' + args + "}\n</tool_call>"
                if self.broken and n < len(messages) - 2:
                    body = body.upper()           # past turns change as the dialogue grows: inconsistent
            text += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if self.qwen and enable_thinking is False:
                text += "<think>\n\n</think>\n\n"
        return self.encode(text)

    def encode(self, text):
        import re
        ids = []
        for piece in re.split("(" + "|".join(map(re.escape, self._SPECIAL)) + ")", text):
            if piece in self._SPECIAL:
                ids.append(0x110000 + self._SPECIAL.index(piece))
            else:
                ids.extend(ord(c) for c in piece)
        return ids

    def decode(self, ids):
        return "".join(self._SPECIAL[i - 0x110000] if i >= 0x110000 else chr(i) for i in ids)


def _dialogue():
    task = TASKS[0]
    ep = run_episode(task, HeuristicAgent("need_to_know", POLICY), POLICY)
    return messages_from_transcript("SYSTEM PROMPT", ep["transcript"])


def _reply(m):
    f = m["tool_calls"][0]["function"]
    return '<tool_call>\n{"name": "' + f["name"] + '", "arguments": ' + json.dumps(json.loads(f["arguments"])) \
        + "}\n</tool_call><|im_end|>\n"


@pytest.mark.parametrize("qwen,thinking", [(True, False), (True, True), (False, True)])
def test_mask_covers_exactly_the_assistant_turns(qwen, thinking):
    tok = FakeTokenizer(qwen=qwen)
    messages = _dialogue()
    ids, mask = assistant_mask(tok, messages, tools=[{"name": "get_positions"}], enable_thinking=thinking)
    assert len(ids) == len(mask)
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert tok.decode([i for i, m in zip(ids, mask) if m]) == "".join(_reply(m) for m in assistants)
    untrained = tok.decode([i for i, m in zip(ids, mask) if not m])
    assert "SYSTEM PROMPT" in untrained and "book_value" in untrained and "<tools>" in untrained
    assert "place_order" not in untrained                     # every action the agent took is trained on
    assert tok.decode(ids).endswith("</tool_call><|im_end|>\n")   # trailing tool output was dropped
    assert describe_mask(tok, ids, mask).count("[[") == len(assistants)


@pytest.mark.parametrize("qwen,thinking", [(True, False), (True, True), (False, True)])
def test_every_reply_sits_right_after_the_prompt_the_model_was_given(qwen, thinking):
    """The point of assembling turn by turn: the tokens before each reply are the tail of
    the real generation prompt for that turn, empty think block included."""
    tok = FakeTokenizer(qwen=qwen)
    messages = _dialogue()
    ids, mask = assistant_mask(tok, messages, enable_thinking=thinking)
    starts = [t for t in range(len(ids)) if mask[t] and not mask[t - 1]]
    turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    assert len(starts) == len(turns)
    gen_prompt = tok.apply_chat_template([], add_generation_prompt=True, enable_thinking=thinking)
    for start, i in zip(starts, turns):
        prompt = tok.apply_chat_template(messages[:i], add_generation_prompt=True, enable_thinking=thinking)
        assert ids[start - len(gen_prompt):start] == gen_prompt == prompt[-len(gen_prompt):]
        # everything new since the previous reply is the tail of that prompt too
        prev = max((t for t in range(start) if mask[t]), default=-1)
        assert ids[prev + 1:start] == prompt[len(prompt) - (start - prev - 1):]
    text = tok.decode(ids)
    assert text.count("<think>") == (len(turns) if qwen and not thinking else 0)
    if not (qwen and not thinking):                           # consistent template: plain rendering
        plain = tok.apply_chat_template(messages[:turns[-1] + 1] + [{"role": "user", "content": "x"}])
        assert ids == plain[:len(ids)]


def test_mask_rejects_inconsistent_templates_and_overlong_dialogues():
    messages = _dialogue()
    with pytest.raises(ValueError, match="inconsistently"):
        assistant_mask(FakeTokenizer(broken=True), messages)
    with pytest.raises(ValueError, match="max_length"):
        assistant_mask(FakeTokenizer(), messages, max_length=50)
    with pytest.raises(ValueError, match="no assistant turn"):
        assistant_mask(FakeTokenizer(), messages[:2])


# -------------------------------------------------------------------- warm start


def test_sft_dialogues_come_from_successful_teacher_runs_only():
    dialogues, info = build_dialogues(TASKS[:10], "always_ask", POLICY, seeds=2)
    assert info == {"teacher": "always_ask", "attempted": 20, "kept": 20} and len(dialogues) == 20
    asked = [json.loads(m["tool_calls"][0]["function"]["arguments"]).get("field")
             for m in dialogues[0] if m["role"] == "assistant"
             and m["tool_calls"][0]["function"]["name"] == "ask_client"]
    assert "income" in asked                                   # the over-asking habit is in the data
    assert dialogues[0][0]["role"] == "system" and dialogues[0][-2]["tool_calls"][0]["function"]["name"] == "finish"
    kept = build_dialogues(TASKS, "never_ask", POLICY, seeds=1)[1]
    assert 0 < kept["kept"] < kept["attempted"]                # the guesser's failures are filtered out
    ids, mask = assistant_mask(FakeTokenizer(), dialogues[0], enable_thinking=False)
    assert sum(mask) > 0
