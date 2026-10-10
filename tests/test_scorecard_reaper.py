"""tests/test_scorecard_reaper.py — 404 失败分类与清理测试。"""
from __future__ import annotations

import json
from pathlib import Path

from tools.aop.scorecard_reaper import (
    ALIVE,
    COOKIE_MISSING,
    ERROR,
    NOT_FOUND,
    collect_scorecard_ids,
    probe_scorecard,
    reap,
)


class _FakeResponse:
    def __init__(self, code: int) -> None:
        self.status_code = code


class _Http404(Exception):
    def __init__(self) -> None:
        super().__init__("404 Not Found")
        self.response = _FakeResponse(404)


class _Http500(Exception):
    def __init__(self) -> None:
        super().__init__("500 Server Error")
        self.response = _FakeResponse(500)


# ── ID 提取 ───────────────────────────────────────────────────────────────
def test_collect_ids_from_json_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"scorecard_id": "card-A", "episode": 1}), encoding="utf-8"
    )
    rec = tmp_path / "rec.jsonl"
    rec.write_text(
        json.dumps({"card_id": "card-B"}) + "\n" + json.dumps({"scorecard_id": "card-A"}) + "\n",
        encoding="utf-8",
    )
    mapping = collect_scorecard_ids(tmp_path)
    assert set(mapping) == {"card-A", "card-B"}
    assert (tmp_path / "manifest.json") in mapping["card-A"]


def test_collect_ignores_non_json(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("scorecard_id: card-X", encoding="utf-8")
    mapping = collect_scorecard_ids(tmp_path)
    assert mapping == {}


# ── 探活分类 ──────────────────────────────────────────────────────────────
def test_probe_alive() -> None:
    r = probe_scorecard("c", lambda cid: None)
    assert r.status == ALIVE


def test_probe_not_found_no_retry() -> None:
    def get(cid: str) -> None:
        raise _Http404()
    assert probe_scorecard("c", get).status == NOT_FOUND


def test_probe_cookie_missing_recovers_on_retry() -> None:
    def get(cid: str) -> None:
        raise _Http404()
    def retry(cid: str) -> None:
        return None  # 换实例后成功 → cookie 丢失

    assert probe_scorecard("c", get, retry).status == COOKIE_MISSING


def test_probe_not_found_with_retry_still_404() -> None:
    def get(cid: str) -> None:
        raise _Http404()
    def retry(cid: str) -> None:
        raise _Http404()

    assert probe_scorecard("c", get, retry).status == NOT_FOUND


def test_probe_error_on_500() -> None:
    def get(cid: str) -> None:
        raise _Http500()
    assert probe_scorecard("c", get).status == ERROR


# ── 清理 ──────────────────────────────────────────────────────────────────
def test_reap_moves_not_found_to_quarantine(tmp_path: Path) -> None:
    (tmp_path / "ep1.json").write_text(json.dumps({"scorecard_id": "dead"}), encoding="utf-8")
    (tmp_path / "ep2.json").write_text(json.dumps({"scorecard_id": "live"}), encoding="utf-8")
    q = tmp_path / "quarantine"

    def get(cid: str) -> None:
        if cid == "dead":
            raise _Http404()

    rep = reap(tmp_path, get, quarantine=q)
    assert rep.not_found_404 == ["dead"]
    assert rep.alive == ["live"]
    assert (q / "ep1.json").exists()
    assert not (tmp_path / "ep1.json").exists()


def test_reap_does_not_move_on_error(tmp_path: Path) -> None:
    (tmp_path / "ep.json").write_text(json.dumps({"scorecard_id": "flaky"}), encoding="utf-8")
    q = tmp_path / "quarantine"

    def get(cid: str) -> None:
        raise _Http500()

    rep = reap(tmp_path, get, quarantine=q)
    assert rep.error == ["flaky"]
    assert rep.moved_files == []
    assert (tmp_path / "ep.json").exists()  # 保留供诊断


def test_reap_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "ep1.json").write_text(json.dumps({"scorecard_id": "dead"}), encoding="utf-8")
    q = tmp_path / "quarantine"

    def get(cid: str) -> None:
        raise _Http404()

    reap(tmp_path, get, quarantine=q)
    assert (q / "ep1.json").exists()
    # 第二次：quarantine 下的文件被排除，root 下已无引用
    rep2 = reap(tmp_path, get, quarantine=q)
    assert "dead" not in rep2.not_found_404
    assert rep2.moved_files == []


def test_reap_no_move_mode(tmp_path: Path) -> None:
    (tmp_path / "ep.json").write_text(json.dumps({"scorecard_id": "dead"}), encoding="utf-8")
    q = tmp_path / "quarantine"

    def get(cid: str) -> None:
        raise _Http404()

    rep = reap(tmp_path, get, quarantine=q, move=False)
    assert rep.not_found_404 == ["dead"]
    assert rep.moved_files == []
    assert (tmp_path / "ep.json").exists()  # 未移动
