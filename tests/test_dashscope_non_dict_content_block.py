# encoding:utf-8

"""A non-dict element in a user message's content list crashes the DashScope converter.

``DashscopeBot._convert_messages_to_dashscope_format`` decides whether a user
message carries tool results with::

    any(block.get("type") == "tool_result" for block in content)

``block`` is only assumed to be a dict. A content list is not guaranteed to hold
dicts: a provider adapter or a history-compaction pass can leave a bare string
(or a null) among the blocks. ``str`` has no ``.get``, so the ``any()`` raises
``AttributeError`` and takes down the whole turn *before* any HTTP request is
made -- the user sees an internal error instead of an answer, and no request
ever reaches DashScope.

The same unguarded ``any()`` sits in the ZhipuAI converter, and
``mimo_bot.py`` already shows the intended shape: guard the ``any()`` *and* the
loop that follows it. This file asserts that a mixed list -- a plain text string
sitting next to a real ``tool_result`` block -- converts successfully and that
the ``tool_result`` is still recognised, i.e. the guard must not "fix" the crash
by skipping the tool-result detection.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.dashscope.dashscope_bot import DashscopeBot


def _convert(messages):
    """Run the real converter without constructing a configured bot.

    The method needs nothing from ``self``, so an instance built via ``__new__``
    avoids the network/config setup in ``__init__``.
    """
    bot = DashscopeBot.__new__(DashscopeBot)
    return bot._convert_messages_to_dashscope_format(messages)


class TestNonDictContentBlockDoesNotCrash(unittest.TestCase):
    """The crash and the tool_result that must survive it."""

    def test_string_element_next_to_tool_result_still_converts(self):
        """The reported defect: a bare string beside a tool_result.

        ``"here is the result"`` is the shape a compaction pass or an adapter
        leaves behind. Before the fix the ``any()`` calls ``.get`` on that
        ``str`` and raises, killing the turn before any request.
        """
        messages = [{
            "role": "user",
            "content": [
                "here is the result",
                {
                    "type": "tool_result",
                    "tool_use_id": "call_abc123",
                    "content": "42",
                },
            ],
        }]

        converted = _convert(messages)

        # It must convert, and the tool_result must still be recognised --
        # a guard that swallowed the whole list would hide the tool output
        # from the model and make the model look like it never called a tool.
        tool_messages = [m for m in converted if m.get("role") == "tool"]
        self.assertEqual(
            len(tool_messages),
            1,
            "the tool_result block was lost instead of converted",
        )
        self.assertEqual(tool_messages[0].get("tool_call_id"), "call_abc123")
        self.assertEqual(tool_messages[0].get("content"), "42")

    def test_none_element_does_not_crash(self):
        """A null element is the other way a non-dict block shows up."""
        messages = [{
            "role": "user",
            "content": [
                None,
                {
                    "type": "tool_result",
                    "tool_use_id": "call_none_case",
                    "content": "ok",
                },
            ],
        }]

        converted = _convert(messages)

        tool_messages = [m for m in converted if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].get("tool_call_id"), "call_none_case")

    def test_list_with_only_non_dict_elements_does_not_crash(self):
        """No tool_result at all, only junk: nothing to convert, no crash."""
        messages = [{"role": "user", "content": ["just a string", None]}]

        converted = _convert(messages)

        # No tool_result is present, so the message passes through untouched
        # rather than being shredded into an empty tool message.
        self.assertEqual(len(converted), 1)
        self.assertIs(converted[0], messages[0])

    def test_integer_element_does_not_crash(self):
        """An int also has no ``.get``; same failure, same guard."""
        messages = [{
            "role": "user",
            "content": [
                123,
                {"type": "tool_result", "tool_use_id": "call_int", "content": "x"},
            ],
        }]

        converted = _convert(messages)

        tool_messages = [m for m in converted if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)

    def test_well_formed_tool_result_is_unaffected(self):
        """The existing dict-only behaviour must not change."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_ok", "content": "hi"},
            ],
        }]

        converted = _convert(messages)

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["role"], "tool")
        self.assertEqual(converted[0]["tool_call_id"], "call_ok")

    def test_string_only_user_message_is_unaffected(self):
        """Plain string content short-circuits before the block loop."""
        messages = [{"role": "user", "content": "hello"}]

        converted = _convert(messages)

        self.assertEqual(converted, messages)


if __name__ == "__main__":
    unittest.main()
