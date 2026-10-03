# encoding:utf-8
"""
Unit tests for the DashScope streaming error path.

DashScope reports a mid-stream failure as a chunk of its own: a chunk whose
``status_code`` is not 200, carrying the code and message, with no ``output``
at all. ``_handle_stream_response`` recognises that chunk, logs it, and yields
it to the caller as ``{"error": True, ...}`` -- but then it ``continue``s to the
next chunk instead of stopping.

A stream that fails part-way through is not a clean cut, though: the SDK keeps
handing over the chunks the provider had already produced, and those arrive
after the failure. Because the loop keeps going, the generator yields the error
chunk and then happily appends the leftover tail of a response the provider
had already abandoned. The agent renders "upstream stream failed" and then
keeps streaming the orphan text that followed it, so the user reads a failure
notice followed by half an answer as if both were intended.

Every sibling provider already treats a failed stream as the end of the stream:
DeepSeek, MiniMax and LinkAI each yield the error chunk and ``return``. This
covers DashScope doing the same.
"""
import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "dashscope" not in sys.modules:
    _fake_dashscope = types.ModuleType("dashscope")
    _fake_dashscope.Generation = types.SimpleNamespace(
        Models=types.SimpleNamespace(
            qwen_turbo="qwen-turbo",
            qwen_plus="qwen-plus",
            qwen_max="qwen-max",
            bailian_v1="qwen-bailian-v1",
        )
    )
    _fake_dashscope.MultiModalConversation = object
    sys.modules["dashscope"] = _fake_dashscope


def _error_chunk(code="InternalError", message="upstream stream failed", status_code=500):
    """A DashScope mid-stream failure chunk: a status, and no output at all."""
    return {"status_code": status_code, "code": code, "message": message}


def _content_chunk(text, request_id="req-1", finish_reason=None):
    """A normal DashScope content chunk."""
    return {
        "request_id": request_id,
        "status_code": 200,
        "output": {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": text},
                }
            ]
        },
    }


class TestDashscopeStreamErrorStopsTheStream(unittest.TestCase):
    """A failed chunk ends the stream: nothing after it may be yielded."""

    def _stream(self, chunks, model_name="qwen-plus"):
        from models.dashscope import dashscope_bot

        bot = dashscope_bot.DashscopeBot.__new__(dashscope_bot.DashscopeBot)

        with patch.object(dashscope_bot.DashscopeBot, "_apply_base_url",
                          staticmethod(lambda: None)), \
                patch.object(dashscope_bot.dashscope.Generation, "call",
                             return_value=chunks, create=True):
            return list(bot._handle_stream_response(model_name, [], {}))

    def test_iteration_stops_after_a_mid_stream_error_chunk(self):
        # The provider streamed a partial answer, failed, then produced two more
        # chunks. The caller must see the partial answer, then the error, and
        # then nothing at all.
        chunks = self._stream([
            _content_chunk("partial answer"),
            _error_chunk(),
            _content_chunk("orphan tail"),
            _content_chunk("more orphan tail"),
        ])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "partial answer")
        self.assertTrue(chunks[1].get("error"))

    def test_error_chunk_still_reports_the_provider_reason(self):
        # Stopping the stream must not cost us the diagnosis.
        chunks = self._stream([_error_chunk()])
        self.assertEqual(chunks[0]["message"], "upstream stream failed")
        self.assertEqual(chunks[0]["status_code"], 500)

    def test_no_orphan_content_chunk_is_yielded_after_the_error(self):
        chunks = self._stream([_error_chunk(), _content_chunk("orphan tail")])
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("orphan tail", str(chunks))

    def test_a_healthy_stream_is_unaffected(self):
        # Content still streams, in order, when nothing fails.
        chunks = self._stream([_content_chunk("hello "), _content_chunk("world")])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "hello ")
        self.assertEqual(chunks[1]["choices"][0]["delta"]["content"], "world")


if __name__ == "__main__":
    unittest.main()
