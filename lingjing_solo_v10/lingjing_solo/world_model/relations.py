"""
relations.py — 关系型规则（v0.8）

突破 v0.7 的瓶颈：单对象响应无法描述「推动 / 拾取 / 门锁联动」。

核心思想：规则的 LHS 不再是 (obj, action)，而是「对象 + 关系谓词 + 动作」：
    push(box, dir)  ⇔  avatar 相邻(box) 且 box.前方(dir) 为空

关系谓词（relation predicate）是「泡壁局部高精度」的下沉：
    只在对象相邻泡壁内做关系推理，而非全局网格。
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum

from .symbols import SymbolTable, Predicate, GameObject

ObjectID = Union[str, int]


class Relation(Enum):
    """二元关系谓词。方向敏感。"""
    ADJACENT = "adjacent"          # 相邻（四邻域，无向）
    IN_FRONT_OF = "in_front_of"    # A 在 B 的 dir 方向（有向）
    BEHIND = "behind"              # A 在 B 的反方向
    ON = "on"                      # A 在 B 之上（用于拾取/堆叠）
    CARRIES = "carries"            # avatar 携带 A


@dataclass
class RelationalFact:
    """一个关系事实：relation(subj, obj, dir?)"""
    relation: Relation
    subj: ObjectID
    obj: ObjectID
    direction: Optional[str] = None  # IN_FRONT_OF / BEHIND 需要方向

    def __hash__(self):
        d = self.direction or ""
        return hash((self.relation.value, self.subj, self.obj, d))

    def __eq__(self, o):
        return (self.relation, self.subj, self.obj, self.direction) == \
               (o.relation, o.subj, o.obj, o.direction)


@dataclass
class RelationalRule:
    """
    关系型转移规则：
        ∀ dir:  IF  avatar 与 box 满足 ADJACENT
                 AND box 前方(dir) 为 FREE
                 AND action == move(dir)
              THEN box.position += dir        (push)
    """
    condition_relations: List[RelationalFact]   # LHS 关系集合
    condition_action: Optional[str] = None      # LHS 动作（None=任意）
    effect: str = ""                            # 文本描述（供 LLM / 人类阅读）
    effect_lambda: Optional[str] = None         # 可选：Python 表达式（CEGIS 用）
    confidence: float = 0.0
    support: int = 0
    rule_id: str = ""

    def matches(self, facts: set, action: str) -> bool:
        """当前关系集合 + 动作是否满足 LHS。"""
        if self.condition_action and self.condition_action != action:
            return False
        return all(r in facts for r in self.condition_relations)


@dataclass
class RelationGraph:
    """对象间关系图。每步由 ObjectTracker 的快照构建。"""
    facts: set = field(default_factory=set)

    def add(self, rel: RelationalFact):
        self.facts.add(rel)

    def has(self, rel: RelationalFact) -> bool:
        return rel in self.facts

    def relations_of(self, obj: ObjectID) -> List[RelationalFact]:
        return [f for f in self.facts if f.subj == obj or f.obj == obj]


def build_relation_graph(symbols: SymbolTable, avatar_dir: str = "right") -> RelationGraph:
    """
    从符号表构建关系图（泡壁局部：只算相邻 / 前后，不全局遍历）。
        - 两个对象相邻 → ADJACENT（双向）
        - avatar 在某个对象的前方方向 → IN_FRONT_OF（avatar 为 subj）
    """
    g = RelationGraph()
    objects = list(symbols.objects.values())
    avatar = next((o for o in objects if o.role == "avatar"), None)
    if avatar is None:
        return g

    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if not _are_adjacent(a, b):
                continue
            g.add(RelationalFact(Relation.ADJACENT, a.id, b.id))
            g.add(RelationalFact(Relation.ADJACENT, b.id, a.id))
            # 方向关系：仅当 a 为 avatar 时才记 IN_FRONT_OF
            if a.role == "avatar":
                direction = _direction_from(a, b)
                if direction:
                    g.add(RelationalFact(Relation.IN_FRONT_OF, a.id, b.id, direction))

    # CARRIES：由符号谓词 carries(avatar, X) 推导
    for pred in symbols.predicates:
        if pred.name == "carries" and pred.args and pred.args[0] == avatar.id:
            g.add(RelationalFact(Relation.CARRIES, avatar.id, pred.args[1]))
    return g


# ---------- 几何辅助（与 grid 解耦，单测友好） ----------

_DIR_VEC = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


def _are_adjacent(oa, ob) -> bool:
    dx = abs(oa.x - ob.x)
    dy = abs(oa.y - ob.y)
    return (dx + dy) == 1  # 四邻域


def _direction_from(oa, ob) -> Optional[str]:
    """oa -> ob 的方向（四向之一），否则 None。"""
    dx = ob.x - oa.x
    dy = ob.y - oa.y
    if dx == 1 and dy == 0:
        return "right"
    if dx == -1 and dy == 0:
        return "left"
    if dx == 0 and dy == 1:
        return "down"
    if dx == 0 and dy == -1:
        return "up"
    return None


def invert_direction(d: str) -> str:
    return {"up": "down", "down": "up", "left": "right", "right": "left"}[d]


def neighbors(pos: Tuple[int, int], grid_w: int, grid_h: int) -> List[Tuple[int, int]]:
    """泡壁邻域：返回 pos 的四邻域合法坐标。"""
    x, y = pos
    out = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_w and 0 <= ny < grid_h:
            out.append((nx, ny))
    return out
