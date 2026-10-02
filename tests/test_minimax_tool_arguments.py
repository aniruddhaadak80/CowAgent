# encoding:utf-8
"""A MiniMax tool call whose argument string never became valid JSON.

``MinimaxBot._handle_sync_response`` built each ``tool_use`` block with::

    "input": json.loads(tool_call["function"]["arguments"])

bare. ``arguments`` is a string the model wrote, so it is the one field in that
response neither MiniMax nor the caller controls, and a partial one is an
ordinary event rather than an exotic one: a long argument object cut off by the
output token limit arrives exactly like that.

The consequence is bigger than the one broken call. ``json.loads`` raises
``JSONDecodeError``, that propagates out of the ``for`` loop into the
``except Exception`` at the bottom of the method, and what the agent receives is
a ``{"error": True, ..., "status_code": 500}`` chunk. The whole turn -- the
assistant text produced alongside the call, its ``reasoning_details`` blocks, its
``stop_reason`` -- is discarded and reported as a server error, so the model
never gets to see that its call was malformed and retry it. The user watching
the chat sees the bot stop mid-task and report a failure that never happened.

The fix is the one ``models/deepseek/deepseek_bot.py`` already applies to this
exact field, and the one ``models/mimo/mimo_bot.py`` applies to the field beside
it: fall back to an empty argument object. An empty input is a case the executor
already reports properly -- "no arguments at all", naming the required
parameters -- which is the message that gets the model to resend the call
correctly. A raised exception is not.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _bot():
    """A MinimaxBot holding only the state ``_handle_sync_response`` reads.

    ``__init__`` builds a SessionManager out of the developer's config, which a
    test should not touch, so this goes through ``__new__`` the way the existing
    minimax tests do.
    """
    from models.minimax.minimax_bot import MinimaxBot

    return MinimaxBot.__new__(MinimaxBot)


class _Response:
    """A 200 response carrying one assistant message."""

    def __init__(self, message):
        self.status_code = 200
        self.text = json.dumps({"choices": [{"message": message}]})
        self._message = message

    def json(self):
        return {"choices": [{"message": self._message, "finish_reason": "tool_calls"}]}


def _tool_call(arguments, name="read_file", call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _sync_chunks(message):
    """Run the sync response handler over one assistant message.

    The handler is a generator, and what matters here is what it yields, so the
    chunks are collected rather than the call being read for a single value.
    """
    conf = {"minimax_api_key": "test-key", "minimax_api_base": "https://api.minimaxi.com/v1"}
    with patch("models.minimax.minimax_bot.conf", return_value=conf):
        with patch("models.minimax.minimax_bot.requests.post",
                   return_value=_Response(message)):
            return list(_bot()._handle_sync_response({"model": "MiniMax-M3"}))


class TestMinimaxToolArgumentsThatAreNotJson(unittest.TestCase):
    """A malformed argument string must cost one tool call, not the whole turn."""

    def test_a_truncated_argument_string_still_produces_the_tool_call(self):
        """The cut-off case: valid JSON opening, nothing closing it.

        This is what a call that ran into the output token limit looks like, and
        it is the case the un-guarded ``json.loads`` turned into a 500.
        """
        chunks = _sync_chunks({
            "content": "Let me look at that file.",
            "tool_calls": [_tool_call('{"path": "notes/todo')],
        })

        self.assertEqual(len(chunks), 1)
        # Not an error chunk: the turn survived.
        self.assertNotIn("error", chunks[0])
        self.assertEqual(chunks[0]["role"], "assistant")
        self.assertEqual(chunks[0]["stop_reason"], "tool_use")

    def test_the_tool_call_is_still_delivered_with_an_empty_input(self):
        """The call itself is the point, so it has to reach the executor.

        Dropping the arguments keeps the model in the conversation, where the
        executor's "no arguments at all" reply can send it round again. Raising
        drops the model out of the conversation entirely.
        """
        chunks = _sync_chunks({"tool_calls": [_tool_call('{"path": "notes/todo')]})

        tool_uses = [c for c in chunks[0]["content"] if c["type"] == "tool_use"]
        self.assertEqual(len(tool_uses), 1)
        self.assertEqual(tool_uses[0]["id"], "call_1")
        self.assertEqual(tool_uses[0]["name"], "read_file")
        self.assertEqual(tool_uses[0]["input"], {})

    def test_the_text_the_model_already_wrote_survives_a_bad_tool_call(self):
        """A malformed call must not discard the rest of the assistant turn.

        Content blocks are accumulated into one list before being yielded, so
        raising partway through the tool-call loop used to throw away the text
        and reasoning blocks that had already been appended.
        """
        chunks = _sync_chunks({
            "content": "Let me look at that file.",
            "reasoning_details": [{"text": "the user mentioned a todo file"}],
            "tool_calls": [_tool_call("not json at all")],
        })

        self.assertNotIn("error", chunks[0])
        kinds = [c["type"] for c in chunks[0]["content"]]
        self.assertEqual(kinds, ["thinking", "text", "tool_use"])
        thinking = [c for c in chunks[0]["content"] if c["type"] == "thinking"][0]
        self.assertEqual(thinking["thinking"], "the user mentioned a todo file")

    def test_a_missing_arguments_field_is_treated_the_same_way(self):
        """``None`` reaches ``json.loads`` as a ``TypeError``, not a decode error.

        Some OpenAI-compatible servers omit ``arguments`` entirely for a call
        that takes no input. Catching only ``JSONDecodeError`` would leave that
        path raising, which is why the sibling providers catch ``TypeError`` too.
        """
        chunks = _sync_chunks({"tool_calls": [_tool_call(None)]})

        self.assertNotIn("error", chunks[0])
        tool_uses = [c for c in chunks[0]["content"] if c["type"] == "tool_use"]
        self.assertEqual(tool_uses[0]["input"], {})

    def test_an_empty_argument_string_becomes_an_empty_input_too(self):
        """``json.loads("")`` raises as well, so this path needed the guard too.

        A model that emitted a call with no arguments at all -- common when the
        output token limit lands between the call and its arguments -- arrives
        here, and it used to be turned into a 500 rather than an empty input.
        """
        chunks = _sync_chunks({"tool_calls": [_tool_call("")]})

        self.assertNotIn("error", chunks[0])
        tool_uses = [c for c in chunks[0]["content"] if c["type"] == "tool_use"]
        self.assertEqual(tool_uses[0]["input"], {})

    def test_one_bad_call_does_not_stop_the_good_calls_after_it(self):
        """Each call is parsed on its own, so one bad argument is local.

        A single ``try`` around the loop would have handed back a partial list
        with no way to tell which call was the problem.
        """
        chunks = _sync_chunks({
            "tool_calls": [
                _tool_call('{"path": "a"}', name="read_file", call_id="call_ok"),
                _tool_call('{"path": ', name="write_file", call_id="call_bad"),
                _tool_call('{"path": "c"}', name="read_file", call_id="call_ok2"),
            ],
        })

        self.assertNotIn("error", chunks[0])
        tool_uses = [c for c in chunks[0]["content"] if c["type"] == "tool_use"]
        self.assertEqual([t["id"] for t in tool_uses], ["call_ok", "call_bad", "call_ok2"])
        self.assertEqual(tool_uses[0]["input"], {"path": "a"})
        self.assertEqual(tool_uses[1]["input"], {})
        self.assertEqual(tool_uses[2]["input"], {"path": "c"})

    def test_well_formed_arguments_are_unchanged(self):
        """The guard is a fallback, not a replacement: valid JSON still parses."""
        chunks = _sync_chunks({
            "tool_calls": [_tool_call('{"path": "notes/todo.md", "limit": 20}')],
        })

        tool_uses = [c for c in chunks[0]["content"] if c["type"] == "tool_use"]
        self.assertEqual(
            tool_uses[0]["input"], {"path": "notes/todo.md", "limit": 20}
        )

    def test_a_non_200_response_still_reports_an_error(self):
        """The fallback does not swallow a genuine API failure.

        An error chunk is still an error chunk; what changed is only that a
        malformed tool argument is no longer misreported as one.
        """
        conf = {"minimax_api_key": "test-key", "minimax_api_base": "https://api.minimaxi.com/v1"}
        response = MagicMock(status_code=429, text="too many requests")
        with patch("models.minimax.minimax_bot.conf", return_value=conf):
            with patch("models.minimax.minimax_bot.requests.post", return_value=response):
                chunks = list(_bot()._handle_sync_response({"model": "MiniMax-M3"}))

        self.assertEqual(chunks, [
            {"error": True, "message": "too many requests", "status_code": 429}
        ])


if __name__ == "__main__":
    unittest.main()
