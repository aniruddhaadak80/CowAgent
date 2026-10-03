# encoding:utf-8
"""
Unit tests for reading ``Content-Type`` in ``text_to_speech_aliyun``.

The success test indexed the header directly::

    if response.status_code == 200 and response.headers['Content-Type'] == 'audio/mpeg':

``response.headers`` is a dict-like, so a response that simply does not carry the
header raises ``KeyError`` -- there is no default. That is not an exotic body: any
intermediary that answers 200 without inventing a ``Content-Type`` (some reverse
proxies and CDN edges, some error pages that still carry a 200) produces it.

The index sits in the ``if`` condition, so the ``else`` branch below it -- the one
that logs the status code and calls ``response.close()`` -- never ran. The
``KeyError`` therefore did two things at once: it escaped
``text_to_speech_aliyun`` instead of returning ``None``, and it abandoned a
response opened with ``stream=True``, so the socket stayed open. Every subsequent
message hit the same path, which turned text-to-speech into a permanent ERROR
reply for the whole session rather than one degraded message.

The fix is ``headers.get(...)``, which is what ``common.media_download.save_response``
already does on the very response it is handed two lines below.
"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.media_download import DownloadResult
from voice.ali import ali_api

_URL = "https://nls-gateway-cn-shanghai.aliyuncs.com/tts"


class _FakeResponse:
    """A stand-in for the streamed ``requests`` response.

    ``headers`` is a plain dict, so a missing ``Content-Type`` behaves the way a
    real ``CaseInsensitiveDict`` does: ``[]`` raises ``KeyError`` and ``.get()``
    returns ``None``.
    """

    def __init__(self, status_code=200, headers=None, text="", body=b"audio"):
        self.status_code = status_code
        self.headers = {} if headers is None else headers
        self.text = text
        self.body = body
        self.close_calls = 0

    def iter_content(self, chunk_size=1):
        yield self.body

    def close(self):
        self.close_calls += 1


def _synthesise(response):
    """Run the real function against *response*, patching only the POST."""
    with unittest.mock.patch.object(ali_api.requests, "post", return_value=response):
        return ali_api.text_to_speech_aliyun(_URL, "你好", "appkey", "token")


class TestAliTtsContentTypeLookup(unittest.TestCase):
    def test_a_200_without_a_content_type_degrades_instead_of_raising(self):
        """The defect: KeyError escaped and the streamed response was abandoned."""
        response = _FakeResponse(status_code=200, headers={}, text="ok")

        try:
            output_file = _synthesise(response)
        except KeyError as exc:
            self.fail("text_to_speech_aliyun raised KeyError on a 200 with no "
                      "Content-Type: {}".format(exc))

        self.assertIsNone(output_file)
        self.assertEqual(response.close_calls, 1)

    def test_a_200_with_an_unrelated_content_type_is_still_refused(self):
        """A body that is not audio must not be written out as a wav file."""
        response = _FakeResponse(status_code=200, headers={"Content-Type": "text/html"},
                                 text="<html>login</html>")

        output_file = _synthesise(response)

        self.assertIsNone(output_file)
        self.assertEqual(response.close_calls, 1)

    def test_an_audio_response_is_still_saved(self):
        """The control: the working path must survive the lookup change."""
        response = _FakeResponse(status_code=200, headers={"Content-Type": "audio/mpeg"})
        saved = {}

        def fake_save_response(resp, path, max_bytes, max_seconds=None):
            saved["path"] = path
            with open(path, "wb") as out:
                out.write(resp.body)
            return DownloadResult(len(resp.body), resp.headers.get("Content-Type", ""))

        with unittest.mock.patch.object(ali_api, "save_response", fake_save_response):
            output_file = _synthesise(response)

        self.assertIsNotNone(output_file)
        self.assertEqual(output_file, saved["path"])
        self.assertTrue(output_file.endswith(".wav"))
        self.assertTrue(os.path.exists(output_file))
        os.remove(output_file)

    def test_a_non_200_is_still_refused_and_closed(self):
        """The control for the pre-existing else branch."""
        response = _FakeResponse(status_code=500, headers={"Content-Type": "application/json"},
                                 text='{"Code": "InternalError"}')

        output_file = _synthesise(response)

        self.assertIsNone(output_file)
        self.assertEqual(response.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
