"""Opt-in real offline VC33 first-level acceptance test."""
import importlib.util
import os
from pathlib import Path
import unittest

COMBO_PATH = Path(__file__).resolve().parents[1] / 'r2r3r4_combo.py'
spec = importlib.util.spec_from_file_location('vc33_combo', COMBO_PATH)
combo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combo)


@unittest.skipUnless(os.environ.get('VC33_ENVIRONMENTS_DIR'), 'VC33 local environment not configured')
class VC33EndToEnd(unittest.TestCase):
    def test_first_level_advances_within_200_clicks(self):
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction
        arc = Arcade(environments_dir=os.environ['VC33_ENVIRONMENTS_DIR'],
                     operation_mode=OperationMode.OFFLINE)
        env = arc.make('vc33-5430563c', seed=0, save_recording=False)
        frame = env.reset()
        initial = frame.levels_completed
        combo.reset_smart_play()
        steps = 0
        for _ in range(200):
            action = combo.smart_play_step(
                frame.frame[-1].tolist(), game_state=frame.state,
                available_actions=frame.available_actions, game_id='vc33-5430563c')
            if action is None:
                break
            frame = env.step(GameAction.ACTION6, data=action['data'])
            steps += 1
            if frame.levels_completed > initial:
                break
        print(f'VC33 steps={steps} levels={frame.levels_completed} state={frame.state.name}')
        self.assertGreater(frame.levels_completed, initial)


if __name__ == '__main__':
    unittest.main()
