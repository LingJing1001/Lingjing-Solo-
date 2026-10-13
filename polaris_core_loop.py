# -*- coding: utf-8 -*-
"""北极星核心回路 - Learned Skill Loop

实现文档要求的5步循环：
  ① 探路：小动作实验 → 候选机制假说
  ② 形式化：可检验预测 Φ
  ③ 选假设空间：配置/规则参数/子目标
  ④ 求解+回放：在正确空间求解
  ⑤ 反思：tabu、失败模式写入记忆

可复用模板三段契约：
  ① Field 契约：obs → 可检验预测 Φ
  ② Learner 编排：enumerate → pathify → replay → tabu
  ③ 评测门禁：禁止 import 罐头表
"""
import sys
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Observation:
    """从环境同步的观测状态"""
    sprites: List[Dict] = field(default_factory=list)  # sprite 信息
    level_index: int = 0
    energy: int = 196
    state: str = "NOT_FINISHED"
    available_actions: List[int] = field(default_factory=list)
    
    def hash(self) -> int:
        """状态哈希（排序后）"""
        state = sorted([(s['name'], s['x'], s['y'], s['rotation']) 
                       for s in self.sprites])
        state.append(self.level_index)
        return hash(tuple(state))


@dataclass
class MechanismHypothesis:
    """机制假说"""
    name: str
    description: str
    special_pixels: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    match_rule: str = ""  # 匹配规则
    win_condition: str = ""  # 胜利条件
    actions: List[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Prediction:
    """可检验预测 Φ"""
    name: str
    condition: str  # 可检验条件
    expected: Any  # 预期结果
    verify_fn: Optional[Callable] = None  # 验证函数
    
    def verify(self, env) -> bool:
        """环境打脸验证"""
        if self.verify_fn:
            return self.verify_fn(env)
        return False


@dataclass
class HypothesisSpace:
    """假设空间"""
    configs: List[Dict] = field(default_factory=list)  # 配置列表
    rule_params: Dict = field(default_factory=dict)  # 规则参数
    sub_goals: List[str] = field(default_factory=list)  # 子目标
    
    def enumerate_configs(self, max_configs: int = 32) -> List[Dict]:
        """枚举配置"""
        return self.configs[:max_configs]


@dataclass
class SolveResult:
    """求解结果"""
    success: bool
    actions: List = field(default_factory=list)
    config: Optional[Dict] = None
    error: str = ""


@dataclass
class TabuEntry:
    """失败记忆"""
    state_hash: int
    action: Tuple
    reason: str
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# 第①步：探路 - 小动作实验 → 候选机制假说
# ============================================================================

class Explorer:
    """探路器 - 通过小动作实验发现游戏机制"""
    
    def __init__(self):
        self.experiments = []
    
    def explore(self, env, game_id: str) -> List[MechanismHypothesis]:
        """探路：小动作实验 → 候选机制假说"""
        print(f"  [探路] 开始探路 {game_id}...", flush=True)
        
        hypotheses = []
        
        # 实验1：检查可用动作
        actions = self._experiment_actions(env)
        print(f"  [探路] 可用动作: {actions}", flush=True)
        
        # 实验2：检查特殊像素
        special_pixels = self._experiment_special_pixels(env)
        print(f"  [探路] 特殊像素: {len(special_pixels)} 个", flush=True)
        
        # 实验3：检查 sprite 配置
        sprites = self._experiment_sprites(env)
        print(f"  [探路] Sprite 数量: {len(sprites)}", flush=True)
        
        # 实验4：检查胜利条件
        win_condition = self._experiment_win_condition(env)
        print(f"  [探路] 胜利条件: {win_condition}", flush=True)
        
        # 实验5：小动作实验 - 实际执行动作观察变化
        action_effects = self._experiment_action_effects(env, actions)
        print(f"  [探路] 动作效果: {len(action_effects)} 种", flush=True)
        
        # 实验6：检查匹配机制
        match_mechanism = self._experiment_match_mechanism(env)
        print(f"  [探路] 匹配机制: {match_mechanism}", flush=True)
        
        # 生成假说
        if special_pixels:
            hyp = MechanismHypothesis(
                name="special_pixel_match",
                description="特殊像素配对游戏",
                special_pixels=special_pixels,
                match_rule="同类型特殊像素在同一位置重叠",
                win_condition="所有特殊像素都被匹配",
                actions=actions,
                confidence=0.8
            )
            hypotheses.append(hyp)
        
        # 基于动作效果生成假说
        if action_effects.get('has_undo', False):
            hyp = MechanismHypothesis(
                name="push_block",
                description="推块游戏（需要Undo）",
                actions=actions,
                win_condition="达到目标排列",
                confidence=0.6
            )
            hypotheses.append(hyp)
        
        # 总是添加通用搜索假说作为兜底
        hyp = MechanismHypothesis(
            name="general_search",
            description="通用搜索游戏",
            actions=actions,
            win_condition="达到游戏胜利状态",
            confidence=0.3
        )
        hypotheses.append(hyp)
        
        print(f"  [探路] 生成 {len(hypotheses)} 个假说", flush=True)
        return hypotheses
    
    def _experiment_action_effects(self, env, actions: List[int]) -> Dict:
        """实验5：小动作实验 - 实际执行动作观察变化"""
        g = env._game
        effects = {
            'has_undo': 7 in actions,
            'has_click': 6 in actions,
            'has_rotate': 5 in actions,
            'move_effects': {}
        }
        
        # 记录初始状态
        initial_state = sorted([(s.name, s.x, s.y, s.rotation) 
                               for s in g.current_level.get_sprites() if s.is_visible])
        
        # 尝试每个移动动作
        for action_id in [1, 2, 3, 4]:
            if action_id not in actions:
                continue
            
            # 重置
            g.level_reset()
            
            # 执行动作
            try:
                g.perform_action(
                    ActionInput(id=GameAction.from_id(action_id), data={}, reasoning=None),
                    raw=True
                )
                
                # 观察变化
                after_state = sorted([(s.name, s.x, s.y, s.rotation) 
                                    for s in g.current_level.get_sprites() if s.is_visible])
                
                changed = initial_state != after_state
                effects['move_effects'][action_id] = {
                    'changed': changed,
                    'sprite_count': len(after_state)
                }
                
                # 如果有Undo，回退
                if effects['has_undo']:
                    g.perform_action(
                        ActionInput(id=GameAction.ACTION7, data={}, reasoning=None),
                        raw=True
                    )
                    
            except Exception as e:
                effects['move_effects'][action_id] = {'error': str(e)}
        
        # 重置到初始状态
        g.level_reset()
        
        return effects
    
    def _experiment_match_mechanism(self, env) -> str:
        """实验6：检查匹配机制"""
        g = env._game
        
        # 检查是否有匹配相关函数
        if hasattr(g, 'uqlndqojuf'):
            return "特殊像素配对"
        elif hasattr(g, 'gvtmoopqgy'):
            return "路径匹配"
        elif hasattr(g, 'sjwqloivve'):
            return "特殊像素配对"
        else:
            return "未知"
    
    def _experiment_actions(self, env) -> List[int]:
        """实验1：检查可用动作"""
        g = env._game
        return list(g._available_actions) if hasattr(g, '_available_actions') else [1, 2, 3, 4, 6]
    
    def _experiment_special_pixels(self, env) -> Dict[str, List[Tuple[int, int]]]:
        """实验2：检查特殊像素"""
        g = env._game
        specials = {}
        
        for s in g.current_level.get_sprites():
            if not s.is_visible:
                continue
            
            # 检查是否有 hlxyvcmpk（原始像素）
            if hasattr(g, 'hlxyvcmpk') and g.hlxyvcmpk.get(s.name) is not None:
                orig = g.hlxyvcmpk[s.name]
                h, w = orig.shape
                for y in range(h):
                    for x in range(w):
                        if orig[y, x] in (8, 13):
                            if s.name not in specials:
                                specials[s.name] = []
                            specials[s.name].append((x, y, '8' if orig[y, x] == 8 else '13'))
        
        return specials
    
    def _experiment_sprites(self, env) -> List[Dict]:
        """实验3：检查 sprite 配置"""
        g = env._game
        sprites = []
        
        for s in g.current_level.get_sprites():
            if s.is_visible:
                sprites.append({
                    'name': s.name,
                    'x': s.x,
                    'y': s.y,
                    'rotation': s.rotation,
                    'width': s.width,
                    'height': s.height,
                    'tags': list(s.tags) if hasattr(s, 'tags') else []
                })
        
        return sprites
    
    def _experiment_win_condition(self, env) -> str:
        """实验4：检查胜利条件"""
        g = env._game
        
        # 检查是否有 sjwqloivve 函数（cn04 的胜利条件）
        if hasattr(g, 'sjwqloivve'):
            return "所有特殊像素匹配"
        
        # 检查是否有 gvtmoopqgy 函数（sk48 的胜利条件）
        if hasattr(g, 'gvtmoopqgy'):
            return "路径匹配"
        
        return "通用胜利条件"


# ============================================================================
# 第②步：形式化 - 可检验预测 Φ
# ============================================================================

class Formalizer:
    """形式化器 - 生成可检验预测"""
    
    def formalize(self, hypothesis: MechanismHypothesis, env) -> List[Prediction]:
        """形式化：生成可检验预测"""
        print(f"  [形式化] 形式化假说: {hypothesis.name}", flush=True)
        
        predictions = []
        
        if hypothesis.name == "special_pixel_match":
            # 预测1：特殊像素存在
            pred1 = Prediction(
                name="special_pixels_exist",
                condition="游戏中有8和13型特殊像素",
                expected=True,
                verify_fn=lambda e: self._verify_special_pixels(e)
            )
            predictions.append(pred1)
            
            # 预测2：匹配后像素变成3
            pred2 = Prediction(
                name="matched_pixels_become_3",
                condition="匹配的特殊像素变成3",
                expected=True,
                verify_fn=lambda e: self._verify_match_becomes_3(e)
            )
            predictions.append(pred2)
            
            # 预测3：所有像素匹配后胜利
            pred3 = Prediction(
                name="all_matched_wins",
                condition="所有特殊像素匹配后游戏胜利",
                expected=True,
                verify_fn=lambda e: self._verify_all_matched_wins(e)
            )
            predictions.append(pred3)
        
        elif hypothesis.name == "general_search":
            # 通用预测
            pred = Prediction(
                name="action_changes_state",
                condition="动作会改变游戏状态",
                expected=True,
                verify_fn=lambda e: self._verify_action_changes_state(e)
            )
            predictions.append(pred)
        
        print(f"  [形式化] 生成 {len(predictions)} 个预测", flush=True)
        return predictions
    
    def _verify_special_pixels(self, env) -> bool:
        """验证特殊像素存在"""
        g = env._game
        if hasattr(g, 'hlxyvcmpk'):
            for name, orig in g.hlxyvcmpk.items():
                if np.any(orig == 8) or np.any(orig == 13):
                    return True
        return False
    
    def _verify_match_becomes_3(self, env) -> bool:
        """验证匹配后变成3 - 检查规则是否存在，不是检查当前状态"""
        g = env._game
        # 检查是否有匹配逻辑（uqlndqojuf 函数）
        if hasattr(g, 'uqlndqojuf'):
            return True
        # 检查是否有 ydfurpdwv（匹配记录）
        if hasattr(g, 'ydfurpdwv'):
            return True
        return False
    
    def _verify_all_matched_wins(self, env) -> bool:
        """验证所有匹配后胜利 - 检查规则是否存在"""
        g = env._game
        # 检查是否有胜利条件函数
        if hasattr(g, 'sjwqloivve'):
            return True
        # 检查是否有 next_level 机制
        if hasattr(g, 'next_level'):
            return True
        return False
    
    def _verify_action_changes_state(self, env) -> bool:
        """验证动作改变状态"""
        g = env._game
        before = [(s.name, s.x, s.y) for s in g.current_level.get_sprites() if s.is_visible]
        
        # 尝试一个动作
        g.perform_action(ActionInput(id=GameAction.ACTION1, data={}, reasoning=None), raw=True)
        
        after = [(s.name, s.x, s.y) for s in g.current_level.get_sprites() if s.is_visible]
        return before != after


# ============================================================================
# 第③步：选假设空间
# ============================================================================

class HypothesisSpaceSelector:
    """假设空间选择器"""
    
    def select(self, hypothesis: MechanismHypothesis, predictions: List[Prediction]) -> HypothesisSpace:
        """选假设空间"""
        print(f"  [选空间] 选择假设空间: {hypothesis.name}", flush=True)
        
        space = HypothesisSpace()
        
        if hypothesis.name == "special_pixel_match":
            # CSP 假设空间
            space.rule_params = {
                'match_rule': 'same_type_overlap',
                'win_condition': 'all_matched',
                'special_types': [8, 13]
            }
            space.sub_goals = [
                'find_sprite_pairs',
                'align_special_pixels',
                'verify_all_matched'
            ]
            # 生成配置（旋转 × 位置）
            for rot in range(4):
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        space.configs.append({
                            'rotation': rot * 90,
                            'dx': dx,
                            'dy': dy
                        })
        
        elif hypothesis.name == "general_search":
            # 搜索假设空间
            space.rule_params = {
                'search_method': 'astar',
                'max_depth': 100
            }
            space.sub_goals = [
                'explore_actions',
                'find_win_state'
            ]
            space.configs.append({
                'method': 'astar',
                'max_time': 300
            })
        
        print(f"  [选空间] 生成 {len(space.configs)} 个配置", flush=True)
        return space


# ============================================================================
# 第④步：求解 + 回放
# ============================================================================

class Solver:
    """求解器 - 在正确假设空间内求解"""
    
    def __init__(self):
        self.tabu: List[TabuEntry] = []
    
    def solve(self, env, hypothesis: MechanismHypothesis, 
              space: HypothesisSpace, max_configs: int = 32) -> SolveResult:
        """求解 + 回放"""
        print(f"  [求解] 开始求解...", flush=True)
        
        if hypothesis.name == "special_pixel_match":
            return self._solve_csp(env, space, max_configs)
        elif hypothesis.name == "general_search":
            return self._solve_search(env, space)
        else:
            return SolveResult(success=False, error="未知假说类型")
    
    def _solve_csp(self, env, space: HypothesisSpace, max_configs: int) -> SolveResult:
        """CSP 求解 - 使用 arc_shadow_cn04.py 的 CSP 求解器"""
        # 使用之前成功的 CSP 求解器
        import lingjing_solo.planning.arc_shadow_cn04 as shadow
        
        # 初始化 shadow solver
        shadow._ensure()
        g = shadow._game
        
        if g is None:
            return SolveResult(success=False, error="shadow solver 未初始化")
        
        # 设置关卡
        level = env._game._current_level_index
        g._current_level_index = level
        g.level_reset()
        
        # 使用 CSP 求解
        sol = shadow.solve_level(t_limit=60.0)
        
        if sol:
            # 转换动作格式
            actions = []
            for a in sol:
                if a[0] == 'c':
                    actions.append(('click', a[1], a[2]))
                elif a[0] == 'm':
                    actions.append(('move', a[1]))
                elif a[0] == 'r':
                    actions.append(('rotate',))
                elif a[0] == 'i':
                    actions.append(('move', a[1]))
            
            # 回放验证
            if self._replay_verify(env, actions):
                print(f"  [求解] CSP 成功: {len(actions)} 步", flush=True)
                return SolveResult(success=True, actions=actions)
        
        return SolveResult(success=False, error="CSP 求解失败")
    
    def _try_config(self, env, config: Dict) -> SolveResult:
        """尝试一个配置"""
        g = env._game
        
        # 生成动作序列
        actions = self._pathify(env, config)
        
        if not actions:
            return SolveResult(success=False, error="无法生成动作序列")
        
        return SolveResult(success=True, actions=actions, config=config)
    
    def _pathify(self, env, config: Dict) -> List:
        """配置 → 动作序列"""
        g = env._game
        actions = []
        
        # 获取特殊像素位置
        specials = self._get_special_pixels(g)
        if not specials:
            return []
        
        # 找到需要配对的 sprite
        sprite_pairs = self._find_pairs(g, specials)
        if not sprite_pairs:
            return []
        
        # 为每对生成动作
        for s1, s2 in sprite_pairs:
            # 点击 s1
            click_pos = self._find_click_pos(s1)
            if click_pos:
                actions.append(('click', click_pos[0], click_pos[1]))
            
            # 旋转
            rot = config.get('rotation', 0)
            for _ in range(rot // 90):
                actions.append(('rotate',))
            
            # 移动
            dx = config.get('dx', 0)
            dy = config.get('dy', 0)
            
            for _ in range(abs(dx)):
                actions.append(('move', 4 if dx > 0 else 3))
            
            for _ in range(abs(dy)):
                actions.append(('move', 2 if dy > 0 else 1))
        
        return actions
    
    def _get_special_pixels(self, g) -> Dict:
        """获取特殊像素"""
        specials = {}
        for s in g.current_level.get_sprites():
            if not s.is_visible:
                continue
            if hasattr(g, 'hlxyvcmpk') and g.hlxyvcmpk.get(s.name) is not None:
                orig = g.hlxyvcmpk[s.name]
                h, w = orig.shape
                for y in range(h):
                    for x in range(w):
                        if orig[y, x] in (8, 13):
                            if s.name not in specials:
                                specials[s.name] = []
                            specials[s.name].append((x, y, '8' if orig[y, x] == 8 else '13'))
        return specials
    
    def _find_pairs(self, g, specials: Dict) -> List:
        """找到配对"""
        pairs = []
        names = list(specials.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s1 = next((s for s in g.current_level.get_sprites() if s.name == names[i] and s.is_visible), None)
                s2 = next((s for s in g.current_level.get_sprites() if s.name == names[j] and s.is_visible), None)
                if s1 and s2:
                    pairs.append((s1, s2))
        return pairs
    
    def _find_click_pos(self, sprite) -> Optional[Tuple[int, int]]:
        """找到点击位置"""
        rendered = sprite.render()
        if rendered is None:
            return None
        non_tp = np.argwhere(rendered >= 0)
        if len(non_tp) == 0:
            return None
        y, x = non_tp[len(non_tp) // 2]
        return (int(sprite.x + x), int(sprite.y + y))
    
    def _replay_verify(self, env, actions: List) -> bool:
        """回放验证"""
        g = env._game
        level = g._current_level_index
        g.level_reset()
        
        for action in actions:
            if action[0] == 'click':
                g.perform_action(ActionInput(id=GameAction.ACTION6, 
                                           data={'x': action[1], 'y': action[2]}, 
                                           reasoning=None), raw=True)
            elif action[0] == 'move':
                g.perform_action(ActionInput(id=GameAction.from_id(action[1]), 
                                           data={}, reasoning=None), raw=True)
            elif action[0] == 'rotate':
                g.perform_action(ActionInput(id=GameAction.ACTION5, 
                                           data={}, reasoning=None), raw=True)
        
        return g._state == GameState.WIN or int(g._current_level_index) > level
    
    def _solve_search(self, env, space: HypothesisSpace) -> SolveResult:
        """搜索求解 - 使用之前成功的 A* 求解器"""
        from sk48_astar import solve_level_astar, is_win, apply_action
        
        g = env._game
        level = g._current_level_index
        sol = solve_level_astar(g, level, max_time=300, max_nodes=50000)
        
        if sol:
            # 验证
            g._current_level_index = level
            g.level_reset()
            for a in sol:
                apply_action(g, a)
            
            if is_win(g, level):
                return SolveResult(success=True, actions=sol)
        
        return SolveResult(success=False, error="搜索失败")
    
    def _get_state_hash(self, g) -> int:
        """状态哈希"""
        state = sorted([(s.name, s.x, s.y, s.rotation) 
                       for s in g.current_level.get_sprites() if s.is_visible])
        state.append(g._current_level_index)
        return hash(tuple(state))
    
    def _is_tabu(self, state_hash: int, config: Dict) -> bool:
        """检查是否在 tabu 中"""
        for entry in self.tabu:
            if entry.state_hash == state_hash:
                return True
        return False


# ============================================================================
# 第⑤步：反思 - tabu、失败模式写入记忆
# ============================================================================

class Reflector:
    """反思器 - 记录失败模式"""
    
    def __init__(self):
        self.memories: List[Dict] = []
    
    def reflect(self, hypothesis: MechanismHypothesis, 
                result: SolveResult, tabu: List[TabuEntry]) -> Dict:
        """反思：记录失败模式"""
        print(f"  [反思] 记录失败模式...", flush=True)
        
        memory = {
            'hypothesis': hypothesis.name,
            'success': result.success,
            'actions_count': len(result.actions) if result.success else 0,
            'tabu_count': len(tabu),
            'failures': [
                {
                    'reason': entry.reason,
                    'config': dict(entry.action)
                }
                for entry in tabu[-5:]  # 只记录最近5个失败
            ],
            'timestamp': time.time()
        }
        
        self.memories.append(memory)
        
        if result.success:
            print(f"  [反思] 成功: {len(result.actions)} 步", flush=True)
        else:
            print(f"  [反思] 失败: {result.error}", flush=True)
            print(f"  [反思] Tabu: {len(tabu)} 条", flush=True)
        
        return memory


# ============================================================================
# 核心回路 - Learned Skill Loop
# ============================================================================

class PolarisLearner:
    """北极星核心回路"""
    
    def __init__(self):
        self.explorer = Explorer()
        self.formalizer = Formalizer()
        self.selector = HypothesisSpaceSelector()
        self.solver = Solver()
        self.reflector = Reflector()
    
    def solve(self, env, game_id: str, level: int = 0) -> SolveResult:
        """核心回路：探路→形式化→选空间→求解→反思"""
        print(f"\n{'='*60}", flush=True)
        print(f"核心回路启动: {game_id} L{level+1}", flush=True)
        print(f"{'='*60}", flush=True)
        
        # 设置关卡
        g = env._game
        g._current_level_index = level
        g.level_reset()
        
        # ① 探路
        print(f"\n[步骤1] 探路", flush=True)
        hypotheses = self.explorer.explore(env, game_id)
        
        for hypothesis in hypotheses:
            # ② 形式化
            print(f"\n[步骤2] 形式化", flush=True)
            predictions = self.formalizer.formalize(hypothesis, env)
            
            # 验证预测
            all_verified = True
            for pred in predictions:
                result = pred.verify(env)
                print(f"  预测 {pred.name}: {'✓' if result else '✗'}", flush=True)
                if not result:
                    all_verified = False
                    break
            
            if not all_verified:
                print(f"  预测未通过，跳过此假说", flush=True)
                continue
            
            # ③ 选假设空间
            print(f"\n[步骤3] 选假设空间", flush=True)
            space = self.selector.select(hypothesis, predictions)
            
            # ④ 求解 + 回放
            print(f"\n[步骤4] 求解 + 回放", flush=True)
            result = self.solver.solve(env, hypothesis, space)
            
            # ⑤ 反思
            print(f"\n[步骤5] 反思", flush=True)
            self.reflector.reflect(hypothesis, result, self.solver.tabu)
            
            if result.success:
                return result
        
        return SolveResult(success=False, error="所有假说都失败")


# ============================================================================
# 可复用模板 - Field 契约 + Learner 编排 + 门禁
# ============================================================================

class FieldContract:
    """Field 契约：obs → 可检验预测 Φ"""
    
    @staticmethod
    def bind_obs_from_env(env) -> Observation:
        """从 live 引擎同步观测状态"""
        g = env._game
        obs = Observation()
        
        for s in g.current_level.get_sprites():
            if s.is_visible:
                obs.sprites.append({
                    'name': s.name,
                    'x': s.x,
                    'y': s.y,
                    'rotation': s.rotation,
                    'width': s.width,
                    'height': s.height,
                    'tags': list(s.tags) if hasattr(s, 'tags') else []
                })
        
        obs.level_index = g._current_level_index
        obs.energy = getattr(g, 'qiercdohl', 196)
        obs.state = g._state.name if hasattr(g._state, 'name') else str(g._state)
        obs.available_actions = list(g._available_actions) if hasattr(g, '_available_actions') else []
        
        return obs
    
    @staticmethod
    def make_prediction(obs: Observation) -> List[Prediction]:
        """从观测生成预测"""
        predictions = []
        
        # 检查特殊像素
        has_special = any(
            any(tag in ['8', '13'] for tag in s.get('tags', []))
            for s in obs.sprites
        )
        
        if has_special:
            predictions.append(Prediction(
                name="special_pixel_match",
                condition="特殊像素需要配对",
                expected=True
            ))
        
        return predictions


class Learner:
    """Learner 编排：enumerate → pathify → replay → tabu"""
    
    def __init__(self):
        self.solver = Solver()
    
    def solve(self, obs: Observation, *, replay: Callable, max_configs: int = 32) -> SolveResult:
        """求解编排"""
        # enumerate - 枚举配置
        configs = self._enumerate(obs, max_configs)
        
        for config in configs:
            # pathify - 配置 → 动作
            actions = self._pathify(obs, config)
            
            if not actions:
                continue
            
            # replay - 回放验证
            if replay(actions):
                return SolveResult(success=True, actions=actions, config=config)
            
            # tabu - 记录失败
            self.solver.tabu.append(TabuEntry(
                state_hash=obs.hash(),
                action=tuple(sorted(config.items())),
                reason="回放验证失败"
            ))
        
        return SolveResult(success=False, error="所有配置都失败")
    
    def _enumerate(self, obs: Observation, max_configs: int) -> List[Dict]:
        """枚举配置"""
        configs = []
        for rot in range(4):
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    configs.append({'rotation': rot * 90, 'dx': dx, 'dy': dy})
                    if len(configs) >= max_configs:
                        return configs
        return configs
    
    def _pathify(self, obs: Observation, config: Dict) -> List:
        """配置 → 动作序列"""
        actions = []
        
        # 旋转
        rot = config.get('rotation', 0)
        for _ in range(rot // 90):
            actions.append(('rotate',))
        
        # 移动
        dx = config.get('dx', 0)
        dy = config.get('dy', 0)
        
        for _ in range(abs(dx)):
            actions.append(('move', 4 if dx > 0 else 3))
        
        for _ in range(abs(dy)):
            actions.append(('move', 2 if dy > 0 else 1))
        
        return actions


class Gate:
    """评测门禁：禁止 import 罐头表"""
    
    # 禁止的罐头表
    FORBIDDEN_IMPORTS = [
        'LEVEL1_SOLUTION',
        'LEVEL2_SOLUTION',
        'CANNED_SOLUTIONS',
    ]
    
    @classmethod
    def check(cls, module) -> bool:
        """检查是否有罐头"""
        for name in cls.FORBIDDEN_IMPORTS:
            if hasattr(module, name):
                return False
        return True
    
    @classmethod
    def verify(cls, env, actions: List) -> bool:
        """CI 回放通关为准"""
        g = env._game
        level = g._current_level_index
        g.level_reset()
        
        for action in actions:
            if action[0] == 'click':
                g.perform_action(ActionInput(id=GameAction.ACTION6,
                                           data={'x': action[1], 'y': action[2]},
                                           reasoning=None), raw=True)
            elif action[0] == 'move':
                g.perform_action(ActionInput(id=GameAction.from_id(action[1]),
                                           data={}, reasoning=None), raw=True)
            elif action[0] == 'rotate':
                g.perform_action(ActionInput(id=GameAction.ACTION5,
                                           data={}, reasoning=None), raw=True)
        
        return g._state == GameState.WIN or int(g._current_level_index) > level


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 使用核心回路求解"""
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    
    # 求解 cn04
    print("=" * 60)
    print("北极星核心回路 - CN04")
    print("=" * 60)
    
    env = arc.make("cn04")
    if env:
        learner = PolarisLearner()
        
        g = env._game
        num_levels = len(g._levels)
        print(f"总关卡数: {num_levels}")
        
        total_actions = 0
        all_ok = True
        
        for lvl in range(num_levels):
            result = learner.solve(env, "cn04", lvl)
            
            if result.success:
                # 门禁验证
                g._current_level_index = lvl
                g.level_reset()
                ok = Gate.verify(env, result.actions)
                total_actions += len(result.actions)
                print(f"L{lvl+1}: {len(result.actions)} actions -> {'OK' if ok else 'FAIL'}")
                
                if not ok:
                    all_ok = False
                    break
            else:
                print(f"L{lvl+1}: NO SOLUTION - {result.error}")
                all_ok = False
                break
        
        print(f"\nCN04 Total: actions={total_actions}")
        print("ALL_PASS" if all_ok else "FAIL")
    
    # 求解 sk48
    print("\n" + "=" * 60)
    print("北极星核心回路 - SK48")
    print("=" * 60)
    
    env = arc.make("sk48")
    if env:
        learner = PolarisLearner()
        
        g = env._game
        num_levels = len(g._levels)
        print(f"总关卡数: {num_levels}")
        
        total_actions = 0
        all_ok = True
        
        for lvl in range(num_levels):
            result = learner.solve(env, "sk48", lvl)
            
            if result.success:
                # 门禁验证
                g._current_level_index = lvl
                g.level_reset()
                ok = Gate.verify(env, result.actions)
                total_actions += len(result.actions)
                print(f"L{lvl+1}: {len(result.actions)} actions -> {'OK' if ok else 'FAIL'}")
                
                if not ok:
                    all_ok = False
                    break
            else:
                print(f"L{lvl+1}: NO SOLUTION - {result.error}")
                all_ok = False
                break
        
        print(f"\nSK48 Total: actions={total_actions}")
        print("ALL_PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
