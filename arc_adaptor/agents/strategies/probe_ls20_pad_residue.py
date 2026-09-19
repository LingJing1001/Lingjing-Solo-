"""每关入场审计：三台检测器给出的台子，玩家到底踩不踩得到（残差类判定）。

用法：
    python probe_ls20_pad_residue.py [recording.jsonl]

LS20 玩家一步走 STEP=5 像素（ls20_solver.py:12-14 的 DIRS），所以 (x%5, y%5)
这一「残差类」在关卡内永远不变——只有和玩家同残差的格才踩得到。
检测器里 _find_rot_pad/_find_shape_pad 用 _reachable_from/_reachable_from_push 过滤了
（ls20_solver.py:191-196 / 211-214），_find_color_pad 只按曼哈顿距离取最近
（ls20_solver.py:228-229），没做同样的过滤。这里逐关把三台的
「同残差 / 可达」情况打出来，验证哪些关卡会因此定一个永远走不到的目标。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # arc_adaptor
from paths import add_to_sys_path, environments_dir, state_dir  # noqa: E402

add_to_sys_path()

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

import r2_ls20 as r2  # noqa: E402
from lingjing_solo.planning.ls20_solver import (  # noqa: E402
    STEP, _find_color_pad, _find_goal_markers, _find_player, _find_rot_pad,
    _find_shape_pad, _reachable_from,
)

ACT = {n: getattr(GameAction, f"ACTION{n}") for n in (1, 2, 3, 4)}


def default_recording() -> Path:
    """默认挑一个 7/7 通关的 recording（把每关都走一遍）。"""
    cands = []
    for p in sorted(state_dir().glob("ls20_r2r3_*/recording.jsonl")):
        lines = [json.loads(raw) for raw in p.open(encoding="utf-8") if raw.strip()]
        if lines and (lines[-1].get("data") or {}).get("state") == "WIN":
            cands.append(p)
    if not cands:
        raise SystemExit("找不到 WIN 的 state/ls20_r2r3_*/recording.jsonl")
    return cands[-1]


def audit(env, grid, lvl):
    player = _find_player(grid)
    if not player:
        print(f"  L{lvl + 1}: 玩家定位失败")
        return
    res = (player[0] % STEP, player[1] % STEP)
    reach = _reachable_from(grid, player)
    print(f"  L{lvl + 1}: 玩家={player} 残差={res} 可达格={len(reach)} "
          f"目标={_find_goal_markers(grid, player)}")
    for label, fn in (("shape", _find_shape_pad), ("rot", _find_rot_pad),
                      ("color", _find_color_pad)):
        pad = fn(grid, player)
        if pad is None:
            print(f"      {label:<5} = None")
            continue
        same = (pad[0] % STEP, pad[1] % STEP) == res
        ok = pad in reach
        print(f"      {label:<5} = {str(pad):<9} 同残差={same} 可达={ok}"
              f"{'' if same and ok else '   <<< 踩不到：这一段 cost 会是 None'}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_recording()
    lines = [json.loads(raw) for raw in path.open(encoding="utf-8") if raw.strip()]
    actions = [r["data"]["requested_action"]["name"] for r in lines[1:]]
    print(f"recording = {path}  ({len(actions)} 步)")

    arcade = Arcade(environments_dir=environments_dir("ls20"), operation_mode=OperationMode.OFFLINE)
    gid = [e for e in arcade.get_environments() if e.game_id.startswith("ls20")][0].game_id
    env = arcade.make(gid)
    env.reset()
    settle = 2                      # 换关后等几帧，等布局画完
    pending = (int(env._game._current_level_index), settle)
    audited = set()
    for tick, name in enumerate(actions, 1):
        grid = r2.to_grid(env.step(ACT[int(name[-1])], data=None))
        lvl = int(env._game._current_level_index)
        if lvl != pending[0]:
            pending = (lvl, settle)          # 刚换关：重新计时
            continue
        if lvl in audited or pending[1] > 0:
            pending = (lvl, pending[1] - 1)
            continue
        audited.add(lvl)
        audit(env, grid, lvl)
    print(f"\n覆盖关卡 {sorted(audited)}（共 {len(audited)} 关）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
