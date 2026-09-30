import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('combo', Path(__file__).resolve().parents[1] / 'r2r3r4_combo.py')
combo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combo)


class ClickTests(unittest.TestCase):
    def setUp(self):
        combo.reset_smart_play()

    def test_adjacent_colors_are_not_victory(self):
        self.assertFalse(combo.r2_detect_victory([[5, 11]]))
        self.assertFalse(combo.r2_perceive([[5, 11]], 'NOT_FINISHED')['victory'])
        self.assertTrue(combo.r2_detect_victory([[0]], SimpleNamespace(name='WIN')))

    def test_terminal_states_stop(self):
        for state in ('WIN', 'GAME_OVER'):
            self.assertIsNone(combo.smart_play_step([[9]], game_id='vc33', game_state=state, available_actions=[6]))

    def test_clicks_stay_on_components(self):
        grid = [[9,9,9,0,12], [9,0,0,0,12], [9,9,9,0,0]]
        for action in combo.vc33_click_candidates(grid):
            x, y = action['data']['x'], action['data']['y']
            self.assertIn(grid[y][x], (9, 12))
            self.assertEqual(action['action'], 6)

    def test_repeat_avoidance_and_reset(self):
        grid = [[0,0,0], [9,0,9]]
        kwargs = dict(game_id='vc33', game_state='NOT_FINISHED', available_actions=[6])
        first = combo.smart_play_step(grid, **kwargs)
        self.assertNotEqual(first, combo.smart_play_step(grid, **kwargs))
        combo.reset_smart_play()
        self.assertEqual(first, combo.smart_play_step(grid, **kwargs))

    def test_reject_unsupported_action_space(self):
        with self.assertRaises(ValueError):
            combo.smart_play_step([[9]], game_id='vc33', available_actions=[1])

    def test_legacy_integer_output(self):
        with patch.object(combo, '_record_transfer'):
            self.assertIsInstance(combo.smart_play_step([[5, 11]]), int)

    def test_search_uses_progress_not_picture(self):
        state = {'value': 0}
        def frame():
            return SimpleNamespace(state='NOT_FINISHED', levels_completed=int(state['value'] == 2), value=state['value'])
        def reset():
            state['value'] = 0
            return frame()
        def step(action):
            state['value'] += action
            return frame()
        result = combo.r3_search_click_path(reset, step, lambda f: [0,1], lambda f: f.value)
        self.assertEqual(result['path'], [1,1])
        self.assertEqual(result['reason'], 'solved')
        limited = combo.r3_search_click_path(reset, step, lambda f: [1], lambda f: f.value, max_depth=1)
        self.assertEqual(limited['reason'], 'depth_limit')


if __name__ == '__main__':
    unittest.main()
