# encoding:utf-8
"""A user message whose content list holds a non-dict block must still convert.

``_convert_messages_to_openai_format`` looks for a tool result with::

    if role == "user" and any(block.get("type") == "tool_result" for block in content):

``block.get`` assumes every element of a content list is an object. The very
next loop in the same branch, and the assistant branch below it, both guard with
``isinstance(block, dict)``, as do the mimo, doubao, deepseek and zhipuai
converters -- this one call site was the exception.

Anything that puts a non-object in a user content list therefore raised
``AttributeError: 'str' object has no attribute 'get'`` here: a provider adapter
that appends a bare string, a compaction pass that trims a block down to its
text, or a hand-built message. It failed inside message conversion, so the whole
turn died before a single request was made -- and the traceback pointed at the
converter rather than at the message that was malformed.

Guarding the membership test is not enough on its own: once it passes, the loop
that follows reads the same elements unguarded, so a list holding both a
tool_result and a bare string would only have moved the crash one line down.
Both loops now skip what is not a block, which is what the other converters
already do.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.openai_compatible_bot import OpenAICompatibleBot


def _user_text_with_extra(extra):
    """A user message whose content list is ``extra`` plus one text block."""
    return {"role": "user", "content": [extra, {"type": "text", "text": "hello"}]}


def _tool_result(tool_use_id="call_1", content="ok"):
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def _assistant_tool_call(tool_use_id="call_1"):
    """The assistant turn a tool result has to answer.

    ``drop_orphaned_tool_results_openai`` runs at the end of the conversion and
    removes a tool message that follows no assistant tool call, so a test that
    wants to see the converted result has to supply the call it belongs to --
    which is also the only shape a real history ever has.
    """
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_use_id,
                     "name": "read", "input": {"path": "a.py"}}],
    }


class NonObjectContentBlockTest(unittest.TestCase):
    """Conversion tolerates a non-dict element in a user content list."""

    def setUp(self):
        self.bot = OpenAICompatibleBot()

    def _convert(self, messages):
        return self.bot._convert_messages_to_openai_format(messages)

    def test_a_bare_string_in_a_user_content_list_does_not_raise(self):
        """The reported case: a string where a content block was expected."""
        converted = self._convert([_user_text_with_extra("plain string")])

        self.assertEqual(1, len(converted))
        self.assertEqual("user", converted[0]["role"])

    def test_a_null_block_in_a_user_content_list_does_not_raise(self):
        """``None`` reaches the same line and has to be skipped just the same."""
        converted = self._convert([_user_text_with_extra(None)])

        self.assertEqual(1, len(converted))

    def test_a_string_next_to_a_tool_result_still_converts_the_tool_result(self):
        """Skipping the non-block must not cost the tool result next to it.

        This is the case a guard on the membership test alone would still get
        wrong: the test now passes, and the loop reading the same list raises.
        """
        converted = self._convert([
            _assistant_tool_call(),
            {"role": "user", "content": ["plain string", _tool_result()]},
        ])

        self.assertEqual(["assistant", "tool"], [m["role"] for m in converted])
        self.assertEqual("call_1", converted[1]["tool_call_id"])
        self.assertEqual("ok", converted[1]["content"])

    def test_text_beside_a_non_block_is_kept_when_a_tool_result_is_present(self):
        """The text that can be read is still collected and sent."""
        converted = self._convert([
            _assistant_tool_call(),
            {"role": "user", "content": [
                None,
                _tool_result(),
                {"type": "text", "text": "and this"},
            ]},
        ])

        self.assertEqual(["assistant", "tool", "user"], [m["role"] for m in converted])
        self.assertEqual("and this", converted[2]["content"])

    def test_a_well_formed_tool_result_message_is_unchanged(self):
        """The happy path is untouched by the guard.

        Every provider adapter that has no reason to emit a non-object block must
        keep converting exactly as before, so this pins the shape rather than
        only the absence of an exception.
        """
        converted = self._convert([
            {"role": "user", "content": [{"type": "text", "text": "run it"}]},
            _assistant_tool_call(),
            {"role": "user", "content": [_tool_result("call_1", "print(1)")]},
        ])

        self.assertEqual(["user", "assistant", "tool"], [m["role"] for m in converted])
        self.assertEqual("print(1)", converted[2]["content"])

    def test_a_non_block_in_an_assistant_content_list_is_still_skipped(self):
        """The assistant branch already guarded this; keep it that way."""
        converted = self._convert([{
            "role": "assistant",
            "content": ["plain string", {"type": "text", "text": "done"}],
        }])

        self.assertEqual("done", converted[0]["content"])


if __name__ == "__main__":
    unittest.main()
