# encoding:utf-8
"""``ls`` must not report a populated directory as empty.

The ``limit`` argument is declared as an integer in the tool schema, but it is
model output rather than a value the runtime enforces. Handed to
``len(results) >= limit`` unchecked, a non-positive limit made that comparison
true on the first entry, so the loop stopped with nothing collected and the
empty-directory branch answered for a directory that was full. A non-numeric
limit raised ``TypeError`` inside the loop instead.
"""

import pytest

from agent.tools.ls.ls import DEFAULT_LIMIT, Ls

NAMES = ["alpha.txt", "beta.txt", "gamma.md"]


@pytest.fixture
def populated(tmp_path):
    for name in NAMES:
        (tmp_path / name).write_text("x", encoding="utf-8")
    return tmp_path


def _entries(tmp_path, **args):
    result = Ls().execute({"path": str(tmp_path), **args})
    assert result.status == "success", f"unexpected failure: {result.result}"
    return result.result


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_non_positive_limit_does_not_report_a_full_directory_as_empty(populated, limit):
    payload = _entries(populated, limit=limit)
    assert payload.get("message") != "(empty directory)"
    assert payload.get("entry_count", 0) > 0
    assert any(name in payload["output"] for name in NAMES)


@pytest.mark.parametrize("limit", [None, "not-a-number", "", "12abc", {}, []])
def test_non_numeric_limit_falls_back_to_the_default(populated, limit):
    payload = _entries(populated, limit=limit)
    assert payload.get("entry_count") == len(NAMES)
    assert "limit reached" not in payload["output"]


def test_numeric_string_limit_is_honoured(populated):
    payload = _entries(populated, limit="2")
    assert payload["entry_count"] == 2
    assert "2 entries limit reached" in payload["output"]


def test_float_limit_is_truncated_not_reported_as_a_fraction(populated):
    payload = _entries(populated, limit=1.5)
    assert payload["entry_count"] == 1
    assert "1.5 entries limit reached" not in payload["output"]
    assert "1 entries limit reached" in payload["output"]


def test_entry_limit_notice_points_at_the_next_value(populated):
    payload = _entries(populated, limit=1)
    assert payload["details"]["entry_limit_reached"] == 1
    assert "Use limit=2 for more" in payload["output"]


def test_omitted_limit_lists_everything(populated):
    payload = _entries(populated)
    assert payload["entry_count"] == len(NAMES)


def test_a_genuinely_empty_directory_still_reports_empty(tmp_path):
    payload = _entries(tmp_path)
    assert payload.get("message") == "(empty directory)"
    assert payload.get("entries") == []


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, DEFAULT_LIMIT),
        ("not-a-number", DEFAULT_LIMIT),
        ("", DEFAULT_LIMIT),
        ({}, DEFAULT_LIMIT),
        ([], DEFAULT_LIMIT),
        (object(), DEFAULT_LIMIT),
        ("20", 20),
        (7, 7),
        (1.5, 1),
        (2.9, 2),
        (0, 1),
        (-1, 1),
    ],
)
def test_resolve_limit(value, expected):
    # Imported here rather than at module scope so the behavioural tests above
    # still collect and run on a tree without the helper.
    from agent.tools.ls.ls import _resolve_limit

    assert _resolve_limit(value) == expected
