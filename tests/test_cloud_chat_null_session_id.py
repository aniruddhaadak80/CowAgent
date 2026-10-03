# encoding:utf-8
"""
Tests that a null session_id in a chat push reaches the agent turn.

Defect: ``CloudClient.on_chat`` defaulted the session id with
``payload.get("session_id", "cloud_console")``, which only applies when the
key is *absent*. An explicit JSON ``null`` is present, so ``.get`` returned
None and ``session_id.startswith("session_")`` raised ``AttributeError``.

User-visible consequence: the console's chat turn died before any agent
work started, so the user saw the request fail instead of an answer, and the
session was never opened. ``_query_history`` in the same class already
rejects an empty session id up front; this path had no such check.

The session id the turn actually receives is recorded here, so the tests
assert the documented default was applied and not merely that nothing
raised.
"""

import contextlib
import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# The remote client module imports an optional runtime SDK that is not present
# in the test environment. Stub the few names it binds at import time, the same
# way the other cloud_client tests do.
if "linkai" not in sys.modules:
    _stub = types.ModuleType("linkai")

    class _LinkAIClient:  # minimal base so CloudClient can subclass it
        def __init__(self, *a, **k):
            pass

    class _PushMsg:
        pass

    _stub.LinkAIClient = _LinkAIClient
    _stub.PushMsg = _PushMsg
    sys.modules["linkai"] = _stub

import common.cloud_client as cloud_client  # noqa: E402
from common.cloud_client import CloudClient  # noqa: E402

CONSOLE_SESSION = "session_cloud_console"


class RecordingChatService:
    """Stands in for ChatService and remembers the session id it was handed."""

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)

    @property
    def last_session_id(self):
        return self.calls[-1]["session_id"]


def run_chat(payload):
    """Drive on_chat as far as the agent turn and return the recording service.

    ``__new__`` skips ``__init__`` so no websocket or cloud connection is
    opened. The agent registry, workspace and identity lookups are replaced
    because a unit test has no agents registered; the session-id handling
    under test is untouched.
    """
    client = CloudClient.__new__(CloudClient)
    client.client_id = "test-client"
    client._peer_transport = None
    service = RecordingChatService()
    client._chat_service = service
    client._resolve_chat_agent_id = lambda agent_id: None
    client._resolve_member_id = lambda agent_id: None
    client._chat_identity = lambda *a, **k: contextlib.nullcontext()
    with patch.object(cloud_client, "_acting_user",
                      lambda user_id: contextlib.nullcontext()):
        client.on_chat({"action": "chat", "payload": payload},
                       send_chunk_fn=lambda chunk: None)
    return service


class TestNullSessionIdFallsBackToConsoleDefault(unittest.TestCase):
    """An explicit null means "no session given", not a crash."""

    def test_null_session_id_reaches_the_agent_turn(self):
        """The reported defect: AttributeError before any agent work."""
        service = run_chat({"query": "hello", "session_id": None})
        self.assertEqual(len(service.calls), 1)

    def test_null_session_id_uses_the_console_default(self):
        """The documented default applies, so the console session is used."""
        self.assertEqual(run_chat({"query": "hi", "session_id": None}).last_session_id,
                         CONSOLE_SESSION)

    def test_empty_session_id_uses_the_console_default(self):
        """An empty string is the same "not supplied" signal as null."""
        self.assertEqual(run_chat({"query": "hi", "session_id": ""}).last_session_id,
                         CONSOLE_SESSION)


class TestSessionIdPrefixingIsUnchanged(unittest.TestCase):
    """The prefix rule the method exists to enforce must still hold."""

    def test_absent_session_id_uses_the_console_default(self):
        """A missing key keeps its original behaviour."""
        self.assertEqual(run_chat({"query": "hi"}).last_session_id, CONSOLE_SESSION)

    def test_unprefixed_id_is_prefixed(self):
        """A bare id gains the web console's prefix, as before."""
        self.assertEqual(run_chat({"query": "hi", "session_id": "abc"}).last_session_id,
                         "session_abc")

    def test_already_prefixed_id_is_left_alone(self):
        """An already-prefixed id must not be prefixed twice."""
        self.assertEqual(run_chat({"query": "hi", "session_id": "session_abc"}).last_session_id,
                         "session_abc")


if __name__ == "__main__":
    unittest.main()
