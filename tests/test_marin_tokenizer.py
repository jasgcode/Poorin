# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest
from levanter.data.text.formats import ChatProcessor
from levanter.data.text.trace_chat import (
    TRACE_LABEL_ASSISTANT_TEXT,
    TRACE_LABEL_ASSISTANT_TOOL_CALL,
    TRACE_LABEL_FINAL_ASSISTANT,
    TRACE_LABEL_OBSERVATION,
    TraceChatProcessor,
)
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from transformers import AutoTokenizer, PreTrainedTokenizer

from experiments.marin_tokenizer import (
    MARIN_CUSTOM_SPECIAL_TOKENS,
    create_marin_tokenizer,
)

REASONING_TRACE = (
    "<|start_think|>User is asking how am I doing. This should be straightforward. I should reply politely.<|end_think|>"
)

CONVERSATION = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": REASONING_TRACE + "I'm doing well, thanks!"},
    {"role": "user", "content": "That's good to hear!"},
    {"role": "assistant", "content": "Great!"},
]

QUESTION = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "The answer is 4."},
]

_RESERVED_SPECIAL_TOKENS = ("<|reserved_special_token_0|>", "<|reserved_special_token_1|>")


@dataclass(frozen=True)
class MarinTokenizerFixture:
    path: str
    token_renames: dict[int, str]


@pytest.fixture(scope="module")
def marin_tokenizer_fixture(gpt2_tokenizer_path, tmp_path_factory) -> MarinTokenizerFixture:
    base = AutoTokenizer.from_pretrained(gpt2_tokenizer_path, local_files_only=True)
    base.add_special_tokens({"additional_special_tokens": list(_RESERVED_SPECIAL_TOKENS)})
    reserved_ids = base.convert_tokens_to_ids(list(_RESERVED_SPECIAL_TOKENS))
    token_renames = dict(zip(reserved_ids, MARIN_CUSTOM_SPECIAL_TOKENS.values(), strict=True))

    tokenizer = create_marin_tokenizer(base, token_renames)
    output_dir = tmp_path_factory.mktemp("marin_tokenizer")
    tokenizer.save_pretrained(output_dir)
    return MarinTokenizerFixture(path=str(output_dir), token_renames=token_renames)


@pytest.fixture(scope="module")
def marin_tokenizer(marin_tokenizer_fixture) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(marin_tokenizer_fixture.path, local_files_only=True)


@pytest.fixture(scope="module")
def marin_chat_tokenizer(marin_tokenizer_fixture) -> MarinTokenizer:
    return load_tokenizer(marin_tokenizer_fixture.path)


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def test_create_marin_tokenizer_preserves_base_tokens_and_renames_slots(
    gpt2_tokenizer_path,
    marin_tokenizer_fixture,
    marin_tokenizer,
):
    base = AutoTokenizer.from_pretrained(gpt2_tokenizer_path, local_files_only=True)
    plain_text = "Hello, how are you?"

    assert marin_tokenizer.encode(plain_text, add_special_tokens=False) == base.encode(
        plain_text, add_special_tokens=False
    )
    assert marin_tokenizer.chat_template == MARIN_CHAT_TEMPLATE
    for token_id, token_str in marin_tokenizer_fixture.token_renames.items():
        assert marin_tokenizer.encode(token_str, add_special_tokens=False) == [token_id]
        assert marin_tokenizer.decode([token_id]) == token_str


def test_assistant_mask_covers_only_assistant_turns(marin_chat_tokenizer: MarinTokenizer):
    result = marin_chat_tokenizer.apply_chat_template_with_masks([CONVERSATION])
    input_ids = np.array(result["input_ids"][0])
    assistant_mask = np.array(result["assistant_masks"][0]).astype(bool)

    masked = marin_chat_tokenizer.decode(input_ids[assistant_mask].tolist())
    assert REASONING_TRACE + "I'm doing well, thanks!" in masked
    assert "Great!" in masked
    assert "Hello, how are you?" not in masked
    assert "That's good to hear!" not in masked


def test_message_spans_cover_each_marin_chat_turn(marin_chat_tokenizer: MarinTokenizer):
    result = marin_chat_tokenizer.apply_chat_template_with_masks([CONVERSATION], return_message_spans=True)

    input_ids = result["input_ids"][0]
    spans = result["message_spans"][0]
    assert len(spans) == len(CONVERSATION)
    assert all(start < end for start, end in spans)
    assert all(left[1] <= right[0] for left, right in pairwise(spans))
    for message, (start, end) in zip(CONVERSATION, spans, strict=True):
        rendered_turn = marin_chat_tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
        assert message["content"] in rendered_turn


def test_trace_chat_processor_labels_marin_tool_trace(marin_chat_tokenizer: MarinTokenizer):
    processor = TraceChatProcessor(
        marin_chat_tokenizer,
        loss_tags=("assistant", "tool_call", "observation", "final_assistant"),
    )
    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Call the lookup tool."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_lookup",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": {"key": "marin"}},
                            }
                        ],
                    },
                    {"role": "tool", "content": '{"result": 3}'},
                    {"role": "assistant", "content": "The final answer is done."},
                ]
            }
        ]
    )[0]

    labels = result["loss_labels"]
    input_ids = result["input_ids"]
    tool_call_text = _decode(marin_chat_tokenizer, input_ids[labels == TRACE_LABEL_ASSISTANT_TOOL_CALL])
    observation_text = _decode(marin_chat_tokenizer, input_ids[labels == TRACE_LABEL_OBSERVATION])
    final_text = _decode(marin_chat_tokenizer, input_ids[labels == TRACE_LABEL_FINAL_ASSISTANT])

    assert "lookup" in tool_call_text
    assert "marin" in tool_call_text
    assert "result" in observation_text
    assert "3" in observation_text
    assert "The final answer is done." in final_text


def test_generation_prompt(marin_chat_tokenizer: MarinTokenizer):
    rendered = marin_chat_tokenizer.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=True)
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n")


@pytest.mark.parametrize(
    "enable_thinking,expected",
    [(True, "Reasoning: /think"), (False, "Reasoning: /nothink"), ("experimental", "Reasoning: experimental")],
)
def test_reasoning_mode(marin_chat_tokenizer: MarinTokenizer, enable_thinking, expected):
    rendered = marin_chat_tokenizer.apply_chat_template(QUESTION, tokenize=False, enable_thinking=enable_thinking)
    assert expected in rendered


def test_tool_definitions_rendered(marin_chat_tokenizer: MarinTokenizer):
    rendered = marin_chat_tokenizer.apply_chat_template(
        QUESTION,
        tokenize=False,
        xml_tools=[
            '{"type": "function", "function": {"name": "final_answer", "description": "Provides final answers."}}',
        ],
        python_tools=[
            '{"type": "function", "function": {"name": "python_exec", "description": "Execute Python code."}}',
        ],
        enable_thinking=True,
    )
    assert "### Tools" in rendered
    assert "<tools>" in rendered
    assert "final_answer" in rendered
    assert "When you send a message containing Python code" in rendered
    assert "python_exec" in rendered


def test_chat_processor_renders_tool_calls(marin_chat_tokenizer: MarinTokenizer):
    processor = ChatProcessor(marin_chat_tokenizer, mask_user_turns=True)
    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Run the VIN check."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "check_valid_vin", "arguments": {"vin": "1FMXK92W8YPA12345"}},
                            }
                        ],
                    },
                    {"role": "tool", "name": "check_valid_vin", "tool_call_id": "call_abc", "content": {"valid": True}},
                    {"role": "assistant", "content": "VIN 1FMXK92W8YPA12345 is valid."},
                ]
            }
        ]
    )[0]

    rendered = _decode(marin_chat_tokenizer, result["input_ids"])
    assert '{"name": "check_valid_vin", "arguments": {"vin": "1FMXK92W8YPA12345"}}' in rendered
    assert '<tool_response name="check_valid_vin">' in rendered
    assert result["assistant_masks"].sum() > 0


def test_chat_processor_renders_ipython_output(marin_chat_tokenizer: MarinTokenizer):
    processor = ChatProcessor(marin_chat_tokenizer, mask_user_turns=True)
    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Show me the result."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_output",
                                "type": "function",
                                "function": {"name": "python_exec", "arguments": {"code": "print(1+1)"}},
                            }
                        ],
                    },
                    {"role": "ipython", "content": [{"type": "text", "text": "4\n"}]},
                    {"role": "assistant", "content": "The result is 4."},
                ]
            }
        ]
    )[0]

    rendered = _decode(marin_chat_tokenizer, result["input_ids"])
    assert '{"name": "python_exec", "arguments": {"code": "print(1+1)"}}' in rendered
    assert "<|start_header_id|>ipython<|end_header_id|>" in rendered
    assert '{"output": "4\\n"}' in rendered
    assert result["assistant_masks"].sum() > 0


@pytest.mark.parametrize("arguments", [{"query": "café <>&"}, '{"query": "café <>&"}'])
def test_agent_turn_retains_reasoning_content_and_parallel_calls(marin_chat_tokenizer, arguments):
    messages = [
        {"role": "user", "content": "Look in both places."},
        {
            "role": "assistant",
            "reasoning_content": "Search both indexes.",
            "content": "Looking now.",
            "tool_calls": [
                {"function": {"name": "first", "arguments": arguments}},
                {"function": {"name": "second", "arguments": {}}},
            ],
        },
        {"role": "tool", "name": "first", "content": "One result."},
        {"role": "tool", "name": "second", "content": "No results."},
        {"role": "assistant", "content": "Found one result."},
    ]
    result = marin_chat_tokenizer.apply_chat_template_with_masks([messages])
    ids = np.array(result["input_ids"][0])
    mask = np.array(result["assistant_masks"][0]).astype(bool)
    rendered = marin_chat_tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n"
    first_turn = rendered.split(assistant_start)[1].split("<|eot_id|>")[0]
    assert first_turn.startswith("<|start_think|>Search both indexes.<|end_think|>Looking now.")
    calls = [json.loads(payload) for payload in re.findall(r"<tool_call>(.*?)</tool_call>", first_turn, re.DOTALL)]
    assert calls == [
        {"name": "first", "arguments": {"query": "café <>&"}},
        {"name": "second", "arguments": {}},
    ]
    trained = marin_chat_tokenizer.decode(ids[mask].tolist(), skip_special_tokens=False)
    assert first_turn in trained
    assert "Found one result." in trained
    assert "One result." not in trained
    assert "No results." not in trained


def test_structured_tool_definitions_are_json(marin_chat_tokenizer):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    rendered = marin_chat_tokenizer.apply_chat_template(QUESTION, tools=tools, tokenize=False)
    definitions = rendered.split("<tools>\n")[1].split("</tools>")[0]
    assert [json.loads(line) for line in definitions.splitlines() if line.strip()] == tools


def test_tool_reply_ids_render_like_named_harmony_observations(marin_chat_tokenizer):
    calls = [
        {"id": "call_a", "function": {"name": "first", "arguments": {}}},
        {"id": "call_b", "function": {"name": "second", "arguments": {}}},
    ]
    prefix = [{"role": "user", "content": "Run both."}, {"role": "assistant", "tool_calls": calls}]
    api_replies = [
        {"role": "tool", "tool_call_id": "call_a", "content": "A"},
        {"role": "tool", "tool_call_id": "call_b", "content": "B"},
    ]
    named_replies = [
        {"role": "tool", "name": "first", "content": "A"},
        {"role": "tool", "name": "second", "content": "B"},
    ]
    assert marin_chat_tokenizer.apply_chat_template(prefix + api_replies) == marin_chat_tokenizer.apply_chat_template(
        prefix + named_replies
    )


def test_trace_inline_calls_preserve_assistant_turn(marin_chat_tokenizer):
    calls = [
        {"name": "first", "arguments": {"key": "a"}},
        {"name": "second", "arguments": {}},
    ]
    user = {"role": "user", "content": "Look in both places."}
    prose = "Looking now."
    inline = prose + "".join(f"<tool_call>{json.dumps(call)}</tool_call>" for call in calls)
    processor = TraceChatProcessor(marin_chat_tokenizer, loss_tags=("assistant", "assistant_text", "tool_call"))
    actual = processor([{"messages": [user, {"role": "assistant", "content": inline}]}])[0]
    expected = marin_chat_tokenizer.apply_chat_template(
        [
            user,
            {
                "role": "assistant",
                "content": prose,
                "tool_calls": [{"type": "function", "function": call} for call in calls],
            },
        ]
    )
    assert actual["input_ids"].tolist() == expected
    labels = actual["loss_labels"]
    assert prose in _decode(marin_chat_tokenizer, actual["input_ids"][labels == TRACE_LABEL_ASSISTANT_TEXT])
    tool_text = _decode(marin_chat_tokenizer, actual["input_ids"][labels == TRACE_LABEL_ASSISTANT_TOOL_CALL])
    assert [
        json.loads(payload) for payload in re.findall(r"<tool_call>(.*?)</tool_call>", tool_text, re.DOTALL)
    ] == calls


@pytest.mark.parametrize("reply_role", [None, "tool", "ipython"], ids=["plain-chat", "tool-reply", "python-output"])
def test_text_content_representations_have_identical_prefixes(marin_tokenizer, reply_role):
    messages = [
        {"role": "system", "content": "  Be helpful.\n"},
        {"role": "developer", "content": "Keep responses short."},
        {"role": "user", "content": "  Find café records.\n"},
        {"role": "assistant", "content": "  Looking now.\n"},
    ]
    if reply_role is not None:
        messages[-1]["tool_calls"] = [
            {"id": "call_lookup", "type": "function", "function": {"name": "lookup", "arguments": {"key": "café"}}},
        ]
        messages.append({"role": reply_role, "tool_call_id": "call_lookup", "content": "  Found one record.\n"})
        messages.append({"role": "assistant", "content": "Here it is."})
    messages.append({"role": "user", "content": "  Tell me more.\n"})

    expected_text = marin_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected_tokens = marin_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    for representation in ("mapping", "parts", "split-parts"):
        converted = []
        for message in messages:
            content = message["content"]
            if representation == "mapping":
                content = {"text": content}
            elif representation == "parts":
                content = [{"type": "text", "text": content}]
            else:
                # Split at a word boundary: trimming individual parts loses the space.
                split = content.index(" ", len(content) - len(content.lstrip()) + 1) + 1
                content = [{"type": "text", "text": content[:split]}, {"type": "text", "text": content[split:]}]
            converted.append({**message, "content": content})

        assert (
            marin_tokenizer.apply_chat_template(converted, tokenize=False, add_generation_prompt=True) == expected_text
        ), representation
        assert (
            marin_tokenizer.apply_chat_template(converted, tokenize=True, add_generation_prompt=True) == expected_tokens
        ), representation
