"""Route replay gate — 「本地 WIN 三遍才进路由」的机器闸门。

routes/ 下每条路线 JSON 必须在真机引擎上回放 N_REPLAYS 遍、逐遍过关且结果互相一致。
破新局 = 在 routes/ 新增一个 JSON;引擎/依赖锁/游戏文件的改动导致路线失效 = CI 直接红。

JSON schema:
  game_id                 例如 "bp35-0a0ad940"
  target_levels           路线必须达到的 levels_completed
  expected_state          结束时允许的 GameState 名(通常 NOT_FINISHED,通关局为 WIN)
  human_baseline_actions  可选,人类基准步数,仅作分数记录
  actions                 字面引擎动作:{"a": "ACTION4"} 或 {"a": "ACTION6", "x": 45, "y": 33}
  meta                    出处与验证证据(date/source/verification/score_note)

没有 arc_agi/arcengine 的环境(如核心 package 的 CI)自动跳过,与仓库既有 skip 边界一致。
"""
import json
import logging
from pathlib import Path

import pytest

pytest.importorskip('arc_agi')
pytest.importorskip('arcengine')

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

ROOT = Path(__file__).resolve().parents[1]
ROUTES_DIR = Path(__file__).parent / 'routes'
ROUTE_FILES = sorted(ROUTES_DIR.glob('*.json'))
N_REPLAYS = 3

_log = logging.getLogger('route_gate')
_log.setLevel(logging.WARNING)


def _replay(arc, route):
    env = arc.make(route['game_id'], seed=0, save_recording=False)
    frame = env.reset()
    for spec in route['actions']:
        action = getattr(GameAction, spec['a'])
        if spec['a'] == 'ACTION6':
            frame = env.step(action, data={'x': spec['x'], 'y': spec['y']})
        else:
            frame = env.step(action)
        if frame.state.name != 'NOT_FINISHED':
            break
    return int(frame.levels_completed), frame.state.name


@pytest.mark.parametrize('route_path', ROUTE_FILES, ids=[p.stem for p in ROUTE_FILES])
def test_route_replay(route_path):
    route = json.loads(route_path.read_text(encoding='utf-8'))
    arc = Arcade(environments_dir=str(ROOT / 'environment_files'),
                 logger=_log, operation_mode=OperationMode.OFFLINE)
    outcomes = [_replay(arc, route) for _ in range(N_REPLAYS)]

    target = route['target_levels']
    expected = route.get('expected_state', 'NOT_FINISHED')
    for levels, state in outcomes:
        assert levels >= target, (
            f'{route_path.stem}: levels_completed={levels} < target {target} '
            f'(state={state}); route no longer clears — engine/dependency drift?')
        assert state in (expected, 'WIN'), (
            f'{route_path.stem}: ended in {state}, expected {expected}')
    assert len(set(outcomes)) == 1, (
        f'{route_path.stem}: replays disagree {outcomes} — route is not deterministic')


def test_routes_exist():
    assert ROUTE_FILES, f'no route JSONs found in {ROUTES_DIR}'
