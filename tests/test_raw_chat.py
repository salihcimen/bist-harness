"""RawOllamaChat: the prompt is rendered with the training chat template (HF/Qwen), and <tool_call> blocks are parsed back."""
import json

from bist.finetune.export import to_hf_messages, tokenizer
from bist.harness.llm_agent import LLMAgent, RawOllamaChat, parse_tool_calls, tool_defs

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "State LOAD_MARKET_DATA."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"index": 0, "name": "get_stock_history", "arguments": {"ticker": "EREGL"}}}]},
        {"role": "tool", "tool_name": "get_stock_history", "content": "get_stock_history: OK"},
        {"role": "user", "content": "State RUN_BACKTEST."}]


class FakePost:
    def __init__(self, response):
        self.response, self.bodies = response, []

    def __call__(self, path, body):
        self.bodies.append((path, body))
        return {"response": self.response}


def test_parse_tool_calls_multiple_and_empty_args():
    text = ('<tool_call>\n{"name": "run_backtest", "arguments": {"variant": "D", "hold_days": 5}}\n</tool_call>\n'
            '<tool_call>\n{"name": "next_state", "arguments": {}}\n</tool_call>')
    content, calls = parse_tool_calls(text)
    assert content == ""
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in calls] == [("run_backtest", {"variant": "D", "hold_days": 5}), ("next_state", {})]


def test_parse_plain_text_and_malformed_call():
    assert parse_tool_calls('{"action": "WATCH"}') == ('{"action": "WATCH"}', [])
    content, calls = parse_tool_calls('<tool_call>\n{"name": "run_backtest", "arguments": {"hold_days": 5\n</tool_call>')
    assert calls[0]["function"]["name"] == "run_backtest"
    assert isinstance(calls[0]["function"]["arguments"], str)            # LLMAgent reports it as MALFORMED_ARGUMENTS
    agent = LLMAgent(chat=None)
    try:
        agent._args(calls[0])
        assert False, "malformed arguments must raise"
    except ValueError:
        pass


def test_prompt_is_the_training_rendering():
    tools = tool_defs(["run_backtest"])
    post = FakePost('<tool_call>\n{"name": "next_state", "arguments": {}}\n</tool_call>')
    chat = RawOllamaChat(model="m", post=post)
    out = chat(MSGS, tools=tools)
    tok = tokenizer()
    want = tok.apply_chat_template(to_hf_messages(MSGS), tools=tools, tokenize=False, add_generation_prompt=True)
    path, body = post.bodies[0]
    assert path == "/api/generate" and body["raw"] is True and body["prompt"] == want
    assert '"minimum": 1' in body["prompt"] and '"default": 5' in body["prompt"]   # full schema, unlike Ollama's /api/chat
    assert body["options"]["temperature"] == 0.0 and "<|im_end|>" in body["options"]["stop"]
    assert out == {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "next_state", "arguments": {}}}]}


def test_json_format_passed_through():
    post = FakePost('{"action": "WATCH", "explanation": "x"}')
    out = RawOllamaChat(model="m", post=post)(MSGS, json_format=True)
    assert post.bodies[0][1]["format"] == "json" and json.loads(out["content"])["action"] == "WATCH" and "tool_calls" not in out


def test_no_output_token_cap_by_default():
    """A long explanation (Q4: 9 quarters of evidence, >2k tokens) must not be cut mid-JSON; /api/chat has no cap either."""
    post = FakePost("{}")
    RawOllamaChat(model="m", post=post)(MSGS, json_format=True)
    assert post.bodies[0][1]["options"].get("num_predict", -1) == -1
