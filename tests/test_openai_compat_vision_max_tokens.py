# encoding:utf-8
"""
Unit tests for ``call_vision`` on the OpenAI-compatible provider.

``call_vision`` takes a ``max_tokens`` argument and every caller passes one --
it is how a caller asks for a shorter or longer image description. The argument
is then dropped on the floor: the request body carries only ``model`` and
``messages``, so the provider applies whatever its own default is instead.

The failure is silent and it always points the wrong way. A caller that asked
for a 64-token caption gets the provider's full-length default and no error, so
the returned description is a paragraph of commentary instead of the short
label the caller needed -- and a caller that asked for a large budget can be
cut off mid-sentence by a default it never asked for. Every sibling vision
provider (Moonshot, Doubao, ZhipuAI, MiMo) forwards the value.

These cover the body that actually goes out.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCallVisionForwardsMaxTokens(unittest.TestCase):
    """The requested token budget must reach the endpoint."""

    def _vision_request(self, **kwargs):
        from models import openai_compatible_bot
        from models.openai_compatible_bot import OpenAICompatibleBot

        class _Bot(OpenAICompatibleBot):
            def get_api_config(self):
                return {
                    "model": "gpt-4o",
                    "api_key": "test-key",
                    "api_base": "https://example.invalid/v1",
                }

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "a cat"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        with patch.object(openai_compatible_bot.requests, "post",
                          return_value=response) as post:
            result = _Bot().call_vision(
                "https://example.invalid/cat.png", "what is this?", **kwargs
            )

        self.assertTrue(post.call_count, "no request was sent")
        self.assertNotIn("error", result)
        return post.call_args.kwargs["json"]

    def test_requested_max_tokens_reaches_the_endpoint(self):
        payload = self._vision_request(max_tokens=64)
        self.assertEqual(payload["max_tokens"], 64)

    def test_a_different_budget_is_forwarded_verbatim(self):
        # Not a hardcoded default: whatever the caller asked for goes out.
        payload = self._vision_request(max_tokens=8)
        self.assertEqual(payload["max_tokens"], 8)

    def test_the_default_argument_is_forwarded_too(self):
        # A caller that says nothing still gets its budget honoured, rather
        # than whatever the provider would have picked.
        payload = self._vision_request()
        self.assertEqual(payload["max_tokens"], 1000)

    def test_the_rest_of_the_request_is_unchanged(self):
        payload = self._vision_request(max_tokens=64)
        self.assertEqual(payload["model"], "gpt-4o")
        self.assertEqual(len(payload["messages"]), 1)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["image_url"]["url"], "https://example.invalid/cat.png")


if __name__ == "__main__":
    unittest.main()
