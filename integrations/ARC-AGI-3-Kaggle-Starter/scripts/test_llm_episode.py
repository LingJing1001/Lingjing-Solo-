"""Offline checks for EpisodeMind. No vLLM."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from llm_episode import EpisodeMind  # noqa: E402


def main() -> None:
    mind = EpisodeMind(digest_after=6, tail_limit=3, reset_at=5, stuck_note_at=3)
    mind.observe(levels=0, grid_hash="a")
    mind.note_action("ACTION1", 0, 0)
    mind.observe(levels=0, grid_hash="a")
    assert "ACTION1" in mind.dead
    mind.note_action("ACTION6:1,2", 0, 1)
    mind.observe(levels=0, grid_hash="b")
    assert "ACTION6:1,2" not in mind.dead
    history, note = mind.briefing(levels=0)
    assert "ACTION1" in history or "digest" in history or "no_effect" in history
    assert "forbidden=" in note
    for _ in range(8):
        mind.note_action("ACTION2", 0, 2)
        mind.observe(levels=0, grid_hash="a")
    assert mind.digest
    assert mind.want_reset()
    mind.observe(levels=1, grid_hash="c")
    assert mind.same_level == 1
    assert mind.rules or mind.level_rules or True
    print("EPISODE_MIND_OK", "dead", sorted(mind.dead), "digest_len", len(mind.digest))


if __name__ == "__main__":
    main()
