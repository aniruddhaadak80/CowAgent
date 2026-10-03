# encoding:utf-8
"""
Unit tests for connection handling in ``speech_to_text_aliyun``.

The call opens one TLS connection by hand -- ``http.client.HTTPSConnection`` --
and closed it on only two of its exits: the recognised branch and the fall-through
at the end of the function. There was no ``try``/``finally`` around it, so every
other exit leaked the socket: a raise from ``conn.request`` or
``conn.getresponse``, and a response body that parsed as JSON but carried no
``status`` field.

That last one is the reachable case. The inner handler catches ``ValueError``,
which covers unparseable bodies, but ``body['status']`` raises ``KeyError`` --
not a ``ValueError`` -- so a body such as ``{"Message": "..."}`` skipped straight
past the handler, past both ``conn.close()`` calls and out of the function.
Aliyun answers that shape on a revoked key or an exhausted quota, i.e. exactly
when speech recognition is being retried, so each attempt left a TLS socket and
its file descriptor open until the garbage collector reclaimed it.

Rule 3 for this suite: a socket counts as leaked only when nothing on any path
closes it. Here both ``conn.close()`` calls sit after the only two exits that
return, so the failure paths had no closer at all.
"""
import json
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice.ali import ali_api

_URL = "https://nls-gateway-cn-shanghai.aliyuncs.com/recognition"


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body


class _FakeConnection:
    """A stand-in for ``http.client.HTTPSConnection`` that counts ``close()``.

    ``fail_on`` selects which call blows up, so the test drives the real control
    flow instead of asserting on the shape of the source.
    """

    def __init__(self, body=b"{}", fail_on=None):
        self.body = body
        self.fail_on = fail_on
        self.close_calls = 0

    def request(self, **kwargs):
        if self.fail_on == "request":
            raise ConnectionResetError("connection reset by peer")

    def getresponse(self):
        if self.fail_on == "getresponse":
            raise ConnectionResetError("connection reset by peer")
        return _FakeResponse(self.body)

    def close(self):
        self.close_calls += 1


def _recognise(conn):
    """Run the real function against *conn*, patching only the constructor."""
    with unittest.mock.patch.object(ali_api.http.client, "HTTPSConnection", return_value=conn):
        return ali_api.speech_to_text_aliyun(_URL, b"pcm-bytes", "appkey", "token")


class TestAliAsrConnectionCleanup(unittest.TestCase):
    def test_a_body_without_a_status_field_returns_none_and_closes(self):
        """The defect: KeyError escaped and the connection was never closed."""
        conn = _FakeConnection(body=json.dumps({"Message": "InvalidToken"}).encode())

        try:
            result = _recognise(conn)
        except KeyError as exc:
            self.fail("speech_to_text_aliyun raised KeyError on a body with no "
                      "status field: {}".format(exc))

        self.assertIsNone(result)
        self.assertEqual(conn.close_calls, 1)

    def test_a_failure_in_conn_request_still_closes(self):
        """A reset before the request lands used to skip both close() calls."""
        conn = _FakeConnection(fail_on="request")

        with self.assertRaises(ConnectionResetError):
            _recognise(conn)

        self.assertEqual(conn.close_calls, 1)

    def test_a_failure_in_getresponse_still_closes(self):
        conn = _FakeConnection(fail_on="getresponse")

        with self.assertRaises(ConnectionResetError):
            _recognise(conn)

        self.assertEqual(conn.close_calls, 1)

    def test_a_recognised_request_returns_the_result_and_closes(self):
        """The control: the success path must keep returning the transcript."""
        conn = _FakeConnection(body=json.dumps({
            "status": 20000000,
            "result": "hello there",
        }).encode())

        result = _recognise(conn)

        self.assertEqual(result, "hello there")
        self.assertEqual(conn.close_calls, 1)

    def test_a_body_that_is_not_json_returns_none_and_closes(self):
        """The control for the pre-existing ValueError branch."""
        conn = _FakeConnection(body=b"<html><body>502 Bad Gateway</body></html>")

        result = _recognise(conn)

        self.assertIsNone(result)
        self.assertEqual(conn.close_calls, 1)

    def test_a_rejected_status_code_returns_none_and_closes(self):
        conn = _FakeConnection(body=json.dumps({"status": 40000000}).encode())

        result = _recognise(conn)

        self.assertIsNone(result)
        self.assertEqual(conn.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
