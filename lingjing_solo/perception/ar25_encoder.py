"""AR25 Layer-0 encoder: parse engine source / structured sprites → Ar25Obs.

Primary path is offline source parsing of ``environment_files/ar25/*/ar25.py``
(tag set for engine ``0c556536``). Frame-only decoding is intentionally out of
scope here — R2 needs reliable object ontology for cover search.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.types import Ar25Axis, Ar25Obs, Ar25Piece

# Official obfuscated tags for ar25-0c556536
TAG_TARGET = "0001sruqbuvukh"
TAG_AXIS = "0003uqrdzdofso"
TAG_AXIS_V = "0054kgxrvfihgm"
TAG_AXIS_H = "0002nuguepuujf"
TAG_PIECE = "0006lxjtqggkmi"
TAG_FIXED = "0056icpryeujyf"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL = _REPO_ROOT / "environment_files" / "ar25" / "0c556536" / "ar25.py"
_REFERENCE = (
    _REPO_ROOT
    / "reference"
    / "arcsage-ar25"
    / "environment_files"
    / "ar25"
    / "0c556536"
    / "ar25.py"
)
DEFAULT_SOURCE = _CANONICAL if _CANONICAL.is_file() else _REFERENCE


class Ar25Encoder:
    """Parse AR25 level definitions into structured observations."""

    def __init__(self, source_path: Optional[str | Path] = None):
        self.source_path = Path(source_path) if source_path else DEFAULT_SOURCE
        self._source: Optional[str] = None
        self._tag_map: Dict[str, List[str]] = {}
        self._pixel_map: Dict[str, list] = {}
        self._levels: List[Ar25Obs] = []

    def load(self, source_path: Optional[str | Path] = None) -> "Ar25Encoder":
        path = Path(source_path) if source_path else self.source_path
        self.source_path = path
        text = path.read_text(encoding="utf-8")
        self._source = text
        self._tag_map = _extract_sprite_tags(text)
        self._pixel_map = _extract_sprite_pixels(text)
        self._levels = _extract_levels(text, self._tag_map, self._pixel_map)
        return self

    @property
    def levels(self) -> List[Ar25Obs]:
        if not self._levels:
            self.load()
        return self._levels

    def encode_level(self, level_index: int) -> Ar25Obs:
        """1-based level index as in source comments (`# Level 7`)."""
        for obs in self.levels:
            if obs.level_index == level_index:
                return obs
        raise KeyError(f"AR25 level {level_index} not found in {self.source_path}")

    def encode_from_parts(
        self,
        *,
        grid_w: int,
        grid_h: int,
        targets: List[Tuple[int, int]],
        pieces: List[Ar25Piece],
        axes: List[Ar25Axis],
        steps_left: int = 64,
        selected_id: Optional[str] = None,
        level_index: Optional[int] = None,
    ) -> Ar25Obs:
        return Ar25Obs(
            grid_w=grid_w,
            grid_h=grid_h,
            targets=list(targets),
            pieces=list(pieces),
            axes=list(axes),
            steps_left=steps_left,
            selected_id=selected_id,
            level_index=level_index,
        )


def _extract_sprite_pixels(source: str) -> Dict[str, list]:
    pixel_map: Dict[str, list] = {}
    pattern = re.compile(
        r'"(\w+)":\s*Sprite\(\s*pixels=(\[(?:\s*\[[\d\s,\-]+\]\s*,?\s*)+\])',
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        name = m.group(1)
        try:
            pixel_map[name] = eval(m.group(2), {"__builtins__": {}}, {})
        except Exception:
            continue
    return pixel_map


def _extract_sprite_tags(source: str) -> Dict[str, List[str]]:
    tag_map: Dict[str, List[str]] = {}
    pattern = re.compile(
        r'"(\w+)":\s*Sprite\([^)]*?tags=\[([^\]]*)\]',
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        name = m.group(1)
        raw = m.group(2)
        tags = [t.strip().strip("\"'") for t in raw.split(",") if t.strip()]
        tag_map[name] = tags
    return tag_map


def _extract_levels(
    source: str,
    tag_map: Dict[str, List[str]],
    pixel_map: Dict[str, list],
) -> List[Ar25Obs]:
    levels: List[Ar25Obs] = []
    # Split on "# Level N" comments that precede Level( blocks
    blocks = re.split(r"#\s*Level\s+(\d+)", source)
    for i in range(1, len(blocks), 2):
        level_num = int(blocks[i])
        block = blocks[i + 1]
        grid_m = re.search(r"grid_size=\((\d+),\s*(\d+)\)", block)
        if not grid_m:
            continue
        gw, gh = int(grid_m.group(1)), int(grid_m.group(2))
        # Only the Level(...) sprite list before grid_size
        level_block = block[: block.find("grid_size")]

        pieces: List[Ar25Piece] = []
        axes: List[Ar25Axis] = []
        targets: List[Tuple[int, int]] = []
        piece_idx = 0
        axis_idx = 0

        sp_pattern = re.compile(
            r'sprites\["(\w+)"\]\.clone\(\)((?:\.\w+\([^)]*\))*)'
        )
        for sp_m in sp_pattern.finditer(level_block):
            sname = sp_m.group(1)
            chain = sp_m.group(2)
            x, y = 0, 0
            pos_m = re.search(r"\.set_position\((-?\d+),\s*(-?\d+)\)", chain)
            if pos_m:
                x, y = int(pos_m.group(1)), int(pos_m.group(2))

            tags = tuple(tag_map.get(sname, []))
            name_tags = set(tags)
            # Name-as-template: axis/target sprites are defined under their tag id
            if sname == TAG_TARGET or TAG_TARGET in name_tags:
                targets.append((x, y))
                continue
            is_v = sname == TAG_AXIS_V or TAG_AXIS_V in name_tags
            is_h = sname == TAG_AXIS_H or TAG_AXIS_H in name_tags
            if is_v or (TAG_AXIS in name_tags and is_v):
                axis_idx += 1
                axes.append(
                    Ar25Axis(
                        id=f"axisV_{axis_idx}_{sname}",
                        kind="V",
                        x=x,
                        y=y,
                        tags=tags or (TAG_AXIS, TAG_AXIS_V),
                        fixed=TAG_FIXED in name_tags,
                    )
                )
                continue
            if is_h or (TAG_AXIS in name_tags and is_h):
                axis_idx += 1
                axes.append(
                    Ar25Axis(
                        id=f"axisH_{axis_idx}_{sname}",
                        kind="H",
                        x=x,
                        y=y,
                        tags=tags or (TAG_AXIS, TAG_AXIS_H),
                        fixed=TAG_FIXED in name_tags,
                    )
                )
                continue
            if TAG_PIECE in name_tags:
                pixels = pixel_map.get(sname)
                if not pixels:
                    continue
                piece_idx += 1
                pieces.append(
                    Ar25Piece(
                        id=f"piece_{piece_idx}_{sname}",
                        x=x,
                        y=y,
                        pixels=pixels,
                        tags=tags,
                        fixed=TAG_FIXED in name_tags,
                    )
                )

        steps = 64
        step_m = re.search(r'"StepCounter":\s*(\d+)', block)
        if step_m:
            steps = int(step_m.group(1))

        levels.append(
            Ar25Obs(
                grid_w=gw,
                grid_h=gh,
                targets=targets,
                pieces=pieces,
                axes=axes,
                steps_left=steps,
                level_index=level_num,
            )
        )

    levels.sort(key=lambda o: o.level_index or 0)
    return levels


def pixels_fingerprint(pixels: list) -> str:
    raw = repr(pixels).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]
