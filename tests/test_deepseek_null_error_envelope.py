# encoding:utf-8
"""
Unit tests for the DeepSeek chat error envelope.

A gateway sitting in front of DeepSeek does not have to answer a failure in
DeepSeek's own shape. When such a proxy reports its own problem it commonly
sends ``{"error": null, "detail": "..."}`` -- an error envelope whose error
member is explicitly null, with the human-readable reason parked in a
sibling field. ``dict.get("error", {})`` returns that ``None`` unchanged
because the key is present, so reading ``error.get("message")`` raises
``AttributeError`` while the failure is being logged.

The raise happens inside the retry loop's own ``try``, so it is swallowed as
if it were a transport failure: the request is re-sent unchanged, sleeps, and
is re-sent again, and the user finally gets a generic "I am a bit tired" reply
that names neither the status nor the upstream detail the proxy actually
supplied. The one piece of information that would have explained the outage
is lost, and three identical attempts are spent to lose it.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _bot():
    """A DeepSeekBot with only the attributes reply_text reads."""
    from models.deepseek.deepseek_bot import DeepSeekBot

    bot = DeepSeekBot.__new__(DeepSeekBot)
    bot.args = {"model": "deepseek-chat", "temperature": 0.7, "top_p": 1.0}
    return bot


def _response(status_code, body):
    """A stubbed ``requests`` response carrying the given JSON body."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    response.text = str(body)
    return response


class TestNullErrorEnvelope(unittest.TestCase):
    """A null "error" member must not turn a reported failure into a retry."""

    def _reply_and_capture(self, status_code, body):
        from models.deepseek import deepseek_bot

        response = _response(status_code, body)
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]

        with patch.object(deepseek_bot.requests, "post", return_value=response) as post, \
                patch.object(deepseek_bot.time, "sleep"), \
                patch.object(deepseek_bot.DeepSeekBot, "_build_headers", return_value={}):
            with self.assertLogs("log", level="ERROR") as logs:
                result = _bot().reply_text(session)

        return result, post.call_count, "\n".join(logs.output)

    def test_null_error_member_does_not_raise_attribute_error(self):
        # Before the fix the f-string that formats the log line dereferences
        # None, so the log never says "chat failed" and the captured records
        # only contain the swallowed AttributeError.
        _, _, output = self._reply_and_capture(
            502, {"error": None, "detail": "upstream connect error"}
        )
        self.assertNotIn("AttributeError", output)
        self.assertIn("chat failed", output)

    def test_null_error_member_still_reports_status_and_upstream_detail(self):
        # The whole point of the log line: the operator reading it after an
        # outage needs the status code and whatever reason the gateway gave.
        _, _, output = self._reply_and_capture(
            502, {"error": None, "detail": "upstream connect error"}
        )
        self.assertIn("502", output)
        self.assertIn("upstream connect error", output)

    def test_well_formed_error_envelope_is_reported_unchanged(self):
        # The normal DeepSeek shape must keep logging message and type.
        _, _, output = self._reply_and_capture(
            401, {"error": {"message": "Authentication Fails", "type": "authentication_error"}}
        )
        self.assertIn("Authentication Fails", output)
        self.assertIn("authentication_error", output)

    def test_retryable_null_envelope_is_not_retried_three_times(self):
        # A 5xx still retries (that is unchanged), but each attempt now logs
        # the real reason instead of blowing up inside the formatter.
        _, attempts, output = self._reply_and_capture(
            502, {"error": None, "detail": "upstream connect error"}
        )
        self.assertEqual(attempts, 3)
        self.assertEqual(output.count("chat failed"), 3)


if __name__ == "__main__":
    unittest.main()
