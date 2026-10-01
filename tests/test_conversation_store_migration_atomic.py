"""The composite-key schema migration must be all-or-nothing.

``_ensure_composite_key`` rebuilds ``sessions`` and ``messages`` with the
create-new / copy / drop / rename dance. That dance only makes sense as a unit:
between ``DROP TABLE sessions`` and ``ALTER TABLE sessions_new RENAME TO
sessions`` the copied rows live in a table nothing else in the codebase knows
about, and once the old table is gone the originals are unrecoverable. Its
docstring therefore promises the whole rebuild is "wrapped in one transaction so
an interruption rolls back to the old shape rather than a half-migrated one".

``sqlite3.Connection.executescript()`` breaks that promise: it issues an implicit
COMMIT before running the script and then runs every statement in autocommit
mode, so the surrounding ``with conn:`` has nothing left to roll back. A process
killed partway through -- or any later statement raising -- leaves the file
permanently half-migrated, and the next open silently recreates empty tables
over the top, losing the conversation history for good.

These tests inject a failure at each point in the rebuild that is unrecoverable
if the migration is not atomic: right after each old table is dropped. A real
interruption is not reproducible on demand, so the fault is injected through the
connection instead, deterministically and identically for either implementation
(whether the statements are handed to ``execute`` one at a time or bundled into
an ``executescript`` script).
"""

import re
import sqlite3

import pytest

from agent.memory.conversation_store import ConversationStore

_POISON = "THIS IS NOT VALID SQL;"

_ROWS = ("user-1", "user-2")


class _SimulatedInterruption(Exception):
    """Stands in for the process being killed mid-rebuild."""


def _drop_re(table):
    return re.compile(rf"DROP\s+TABLE\s+{table}\b", re.IGNORECASE)


def _poison_after_drop(script, table):
    """Insert a statement that cannot run, right after ``DROP TABLE <table>``.

    Locates the statement from the SQL rather than from the literal source text,
    so a reindented or one-statement-per-line rebuild still trips the same
    tripwire at the same point in the rebuild.
    """
    match = _drop_re(table).search(script)
    if match is None:
        return script
    end = script.find(";", match.end())
    if end == -1:
        return script
    return script[: end + 1] + f"\n{_POISON}\n" + script[end + 1 :]


class _InterruptingConnection(sqlite3.Connection):
    """A connection that dies right after the old ``<table>`` table is dropped.

    Overriding both entry points matters: the same test has to hold whether the
    migration runs its statements one by one or hands them over as a single
    ``executescript`` script, otherwise it would only prove the current
    implementation's particular shape is broken.
    """

    trigger = "sessions"

    def execute(self, sql, *args, **kwargs):
        result = super().execute(sql, *args, **kwargs)
        if _drop_re(self.trigger).search(sql or ""):
            raise _SimulatedInterruption("process killed mid-rebuild")
        return result

    def executescript(self, script, *args, **kwargs):
        return super().executescript(
            _poison_after_drop(script, self.trigger), *args, **kwargs
        )


def _interrupting_factory(trigger):
    return type(
        f"_Interrupting_{trigger}",
        (_InterruptingConnection,),
        {"trigger": trigger},
    )


def _seed_legacy_db(path, rows=_ROWS):
    """Write the exact shape the rebuild exists to fix: ``agent_id`` is already
    backfilled by ``_migrate``/``_merge_one_agent``, but ``sessions`` is still
    keyed by ``session_id`` alone, so a second Agent using the same id collides.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                agent_id          TEXT    NOT NULL DEFAULT '',
                session_id        TEXT    NOT NULL,
                channel_type      TEXT    NOT NULL DEFAULT '',
                title             TEXT    NOT NULL DEFAULT '',
                context_start_seq INTEGER NOT NULL DEFAULT 0,
                created_at        INTEGER NOT NULL,
                last_active       INTEGER NOT NULL,
                msg_count         INTEGER NOT NULL DEFAULT 0,
                pinned            INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (session_id)
            );
            CREATE TABLE messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id   TEXT    NOT NULL DEFAULT '',
                session_id TEXT    NOT NULL,
                seq        INTEGER NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at INTEGER NOT NULL,
                extras     TEXT    NOT NULL DEFAULT '',
                run_id     TEXT    NOT NULL DEFAULT '',
                UNIQUE (agent_id, session_id, seq)
            );
            """
        )
        for i, session_id in enumerate(rows, start=1):
            conn.execute(
                "INSERT INTO sessions (session_id, channel_type, created_at, "
                "last_active, msg_count) VALUES (?, 'feishu', ?, ?, 1)",
                (session_id, i, i),
            )
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content, created_at) "
                "VALUES (?, 0, 'user', ?, ?)",
                (session_id, f'[{{"type":"text","text":"kept-{session_id}"}}]', i),
            )
        conn.commit()
    finally:
        conn.close()


def _session_pk_columns(path):
    conn = sqlite3.connect(str(path))
    try:
        return [
            row[1]
            for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            if row[5]
        ]
    finally:
        conn.close()


def _table_names(path):
    conn = sqlite3.connect(str(path))
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()


def _run_interrupted_migration(store, db_path, trigger):
    conn = sqlite3.connect(str(db_path), factory=_interrupting_factory(trigger))
    try:
        with pytest.raises((_SimulatedInterruption, sqlite3.Error)):
            store._ensure_composite_key(conn)
    finally:
        conn.close()


@pytest.mark.parametrize("trigger", ["sessions", "messages"])
def test_interrupted_rebuild_leaves_schema_and_rows_intact(tmp_path, trigger):
    """A failure partway through the rebuild must roll the whole thing back."""
    db_path = tmp_path / "memory" / "long-term" / "index.db"
    _seed_legacy_db(db_path)
    store = ConversationStore(db_path)
    assert _session_pk_columns(db_path) == ["session_id"]  # needs the rebuild

    _run_interrupted_migration(store, db_path, trigger)

    # The rebuild is all-or-nothing: the old shape is still the live schema.
    tables = _table_names(db_path)
    assert "sessions" in tables, f"sessions lost when interrupted after {trigger}"
    assert "messages" in tables, f"messages lost when interrupted after {trigger}"
    assert _session_pk_columns(db_path) == ["session_id"]

    # ...and no orphaned copy was left stranded in a table nothing reads.
    assert "sessions_new" not in tables
    assert "messages_new" not in tables

    # Every row survived, and the store still serves them.
    reopened = ConversationStore(db_path)
    for session_id in _ROWS:
        msgs = reopened.load_messages(session_id)
        assert msgs, f"history lost for {session_id}"
        assert msgs[0]["content"][0]["text"] == f"kept-{session_id}"


def test_interrupted_rebuild_does_not_lose_history_to_empty_recreated_tables(tmp_path):
    """The failure mode that actually eats conversations: a half-migrated file
    whose dropped table is silently recreated empty on the next open, so the
    history reads back as an empty conversation instead of erroring. After an
    interrupted rebuild the reopened store must still return the original rows.
    """
    db_path = tmp_path / "memory" / "long-term" / "index.db"
    _seed_legacy_db(db_path)
    store = ConversationStore(db_path)

    _run_interrupted_migration(store, db_path, "messages")

    # A fresh handle, exactly as a restarted process would build.
    reopened = ConversationStore(db_path)
    for session_id in _ROWS:
        msgs = reopened.load_messages(session_id)
        assert msgs, f"history silently emptied for {session_id}"
        assert msgs[0]["content"][0]["text"] == f"kept-{session_id}"
    # The session list survived too, not just the message rows.
    assert {s["session_id"] for s in reopened.list_sessions()["sessions"]} == set(_ROWS)


def test_uninterrupted_rebuild_still_migrates_to_composite_key(tmp_path):
    """Guards the fix from trading atomicity for a working migration: a rebuild
    that is allowed to finish must still reach the composite key and keep rows.
    """
    db_path = tmp_path / "memory" / "long-term" / "index.db"
    _seed_legacy_db(db_path)
    store = ConversationStore(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        store._ensure_composite_key(conn)
        conn.commit()
    finally:
        conn.close()

    assert _session_pk_columns(db_path) == ["agent_id", "session_id"]
    tables = _table_names(db_path)
    assert "sessions_new" not in tables
    assert "messages_new" not in tables

    reopened = ConversationStore(db_path)
    for session_id in _ROWS:
        msgs = reopened.load_messages(session_id)
        assert msgs[0]["content"][0]["text"] == f"kept-{session_id}"


def test_rebuild_is_a_noop_when_keys_are_already_composite(tmp_path):
    """Preserves existing behaviour for databases that do not need migration:
    a store already on the composite key is left alone, not rebuilt.
    """
    db_path = tmp_path / "memory" / "long-term" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = ConversationStore(db_path)  # fresh store == composite keys
    store.append_messages(
        "already-composite",
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    )
    assert _session_pk_columns(db_path) == ["agent_id", "session_id"]

    conn = sqlite3.connect(str(db_path))
    try:
        store._ensure_composite_key(conn)
        conn.commit()
    finally:
        conn.close()

    assert _session_pk_columns(db_path) == ["agent_id", "session_id"]
    assert store.load_messages("already-composite")[0]["content"][0]["text"] == "hello"
