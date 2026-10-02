# encoding:utf-8
"""shutdown_all() must stop the shared MCP subprocess pool, not just _clients.

Agents that resolve to the same ``mcp.json`` reuse one subprocess through
``get_or_boot_shared``, and the loader then also registers that client in
``_clients`` once its tools are live. ``shutdown_all()`` walked only
``_clients``, so it never touched ``_shared_pool``: after a shutdown every
shared server still had its child process and both reader threads running, and
because the pool entry survived, the next reload handed the *old* subprocess
back -- or, once the config changed, forked a new one and orphaned the old one
for good. A user who reloads MCP config repeatedly ends up with a growing pile
of invisible ``node`` processes holding their pipes open.

These tests pin that shutdown drains the pool through the same ``shutdown()``
helper the registry already uses, and that a client reachable both ways is torn
down exactly once.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tools.mcp.mcp_client import McpClient, McpClientRegistry


class _FakePipe:
    """The child's pipes, already drained, so the reader threads end at once."""

    def __init__(self):
        self.written = []
        self.closed = False

    def __iter__(self):
        return iter(())

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _FakeProc:
    """Stands in for the ``subprocess.Popen`` handle of a live stdio server."""

    def __init__(self):
        self.pid = 5150
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self.stderr = _FakePipe()
        self.terminate_calls = 0
        self.kill_calls = 0
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminate_calls += 1
        self._alive = False

    def kill(self):
        self.kill_calls += 1
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


def _live_client(name):
    """A stdio client with a running child, the way the pool hands them out."""
    client = McpClient({"name": name, "type": "stdio", "command": "node"})
    proc = _FakeProc()
    client._proc = proc
    client._initialized = True
    return client, proc


class ShutdownStopsTheSharedPoolTest(unittest.TestCase):
    """The pool is part of the registry's lifecycle, so shutdown covers it."""

    def setUp(self):
        # McpClientRegistry is a process-wide singleton; take it over for the
        # duration of one test so pool entries don't leak between tests.
        self._saved_instance = McpClientRegistry._instance
        McpClientRegistry._instance = None
        self.registry = McpClientRegistry()

    def tearDown(self):
        McpClientRegistry._instance = self._saved_instance

    def test_pooled_child_process_is_terminated(self):
        """A client reachable only through the pool is still a running
        subprocess, so shutdown has to stop it."""
        client, proc = _live_client("shared-only")
        key = self.registry.shared_key("/mcp.json", "shared-only", {"command": "node"})
        self.registry.put_shared_client(key, client)

        self.registry.shutdown_all()

        self.assertEqual(proc.terminate_calls, 1)
        self.assertIsNotNone(
            proc.poll(),
            "the pooled child survived shutdown_all; nothing else ever stops "
            "it, so its process and two reader threads linger until the "
            "machine is rebooted",
        )

    def test_pool_is_emptied(self):
        """Leaving the entry behind is half the bug: the next reload would be
        handed the process shutdown just stopped."""
        client, _proc = _live_client("shared-only")
        key = self.registry.shared_key("/mcp.json", "shared-only", {"command": "node"})
        self.registry.put_shared_client(key, client)

        self.registry.shutdown_all()

        self.assertEqual(
            self.registry._shared_pool,
            {},
            "the pool still holds the client shutdown_all just terminated",
        )

    def test_client_in_both_pool_and_clients_is_stopped_once(self):
        """The loader registers a pooled client in _clients too (once its tools
        are live), so the two lists overlap and teardown must not double up."""
        client, proc = _live_client("shared")
        key = self.registry.shared_key("/mcp.json", "shared", {"command": "node"})
        self.registry.put_shared_client(key, client)
        self.registry._clients["shared"] = client

        self.registry.shutdown_all()

        self.assertEqual(proc.terminate_calls, 1)
        self.assertEqual(proc.kill_calls, 0)

    def test_reload_after_shutdown_boots_a_fresh_client(self):
        """What the user actually notices: after a shutdown the pool must not
        hand the stopped subprocess back to the next Agent that loads."""
        client, _proc = _live_client("shared")
        key = self.registry.shared_key("/mcp.json", "shared", {"command": "node"})
        self.registry.put_shared_client(key, client)
        self.registry.shutdown_all()

        fresh, proc = _live_client("fresh")
        handed, reused = self.registry.get_or_boot_shared(key, lambda: fresh)

        self.assertIs(handed, fresh)
        self.assertFalse(reused)
        self.assertIsNone(proc.poll(), "the factory's own client must be untouched")
        self.assertEqual(self.registry._shared_pool, {key: fresh})

    def test_plain_registry_clients_still_shut_down(self):
        """The guard on the guard: the pre-existing _clients path must keep
        working unchanged."""
        client, proc = _live_client("plain")
        self.registry._clients["plain"] = client

        self.registry.shutdown_all()

        self.assertEqual(proc.terminate_calls, 1)
        self.assertEqual(self.registry._clients, {})

    def test_pool_is_cleared_even_when_one_shutdown_raises(self):
        """One misbehaving client must not strand the rest: the pool has to be
        emptied before any teardown runs, or the entries that were not reached
        stay live forever."""
        bad, bad_proc = _live_client("bad")
        bad.shutdown = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        good, good_proc = _live_client("good")
        self.registry.put_shared_client(("p", "bad", "s"), bad)
        self.registry.put_shared_client(("p", "good", "s"), good)

        self.registry.shutdown_all()

        self.assertEqual(self.registry._shared_pool, {})
        self.assertEqual(
            good_proc.terminate_calls,
            1,
            "a raising client must not stop the others from being stopped",
        )

    def test_shutdown_with_an_empty_pool_is_a_noop(self):
        """Nothing pooled is the common case for a single-Agent setup."""
        client, proc = _live_client("plain")
        self.registry._clients["plain"] = client

        self.registry.shutdown_all()

        self.assertEqual(proc.terminate_calls, 1)


if __name__ == "__main__":
    unittest.main()
