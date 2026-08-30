"""
symbols.py — 符号层（Layer 0.5 桥接：感知 → 世界模型）

提供：
    - GameObject：带稳定 ID、角色、坐标、颜色的对象
    - SymbolTable：一帧的符号表（对象集合 + 谓词集合）
    - Predicate：一阶谓词（供关系图 build_relation_graph 使用）

设计原则：与 grid 解耦，单测友好；对象 ID 稳定（跨帧匹配）。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class GameObject:
    id: Any
    x: int
    y: int
    color: int = 0
    role: str = "unknown"  # avatar / box / wall / goal / target / unknown
    extra: Dict[str, Any] = field(default_factory=dict)

    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Predicate:
    """一阶谓词：name(args...)。例：carries(avatar, key)"""
    name: str
    args: List[Any] = field(default_factory=list)

    def __hash__(self):
        return hash((self.name, tuple(self.args)))

    def __eq__(self, o):
        return (self.name, tuple(self.args)) == (o.name, tuple(o.args))


class SymbolTable:
    """
    一帧的符号表示。
        objects:  {id: GameObject}
        predicates: 全局谓词集合（如 carries）
    """

    def __init__(self, grid_w: int = 64, grid_h: int = 64):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.objects: Dict[Any, GameObject] = {}
        self.predicates: List[Predicate] = []
        self.avatar_id: Optional[Any] = None
        self._seq = 0

    # ---------- 构造 ----------

    def add_object(
        self, x: int, y: int, color: int = 0, role: str = "unknown", obj_id: Any = None,
    ) -> Any:
        if obj_id is None:
            obj_id = f"obj_{self._seq}"
            self._seq += 1
        self.objects[obj_id] = GameObject(id=obj_id, x=x, y=y, color=color, role=role)
        return obj_id

    def add_predicate(self, name: str, args: List[Any]):
        self.predicates.append(Predicate(name=name, args=list(args)))

    # ---------- 查询 ----------

    def avatar(self) -> Optional[GameObject]:
        if self.avatar_id is None:
            # 兜底：按 role 找
            for o in self.objects.values():
                if o.role == "avatar":
                    self.avatar_id = o.id
                    return o
            return None
        return self.objects.get(self.avatar_id)

    def by_role(self, role: str) -> List[GameObject]:
        return [o for o in self.objects.values() if o.role == role]

    def at(self, x: int, y: int) -> Optional[GameObject]:
        for o in self.objects.values():
            if o.x == x and o.y == y:
                return o
        return None

    # ---------- 工具 ----------

    def copy(self) -> "SymbolTable":
        import copy
        return copy.deepcopy(self)

    def to_dict(self) -> Dict:
        return {
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "objects": {
                oid: {"x": o.x, "y": o.y, "color": o.color, "role": o.role, "extra": o.extra}
                for oid, o in self.objects.items()
            },
            "predicates": [{"name": p.name, "args": p.args} for p in self.predicates],
            "avatar_id": self.avatar_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SymbolTable":
        st = cls(grid_w=d.get("grid_w", 64), grid_h=d.get("grid_h", 64))
        for oid, o in d.get("objects", {}).items():
            st.objects[oid] = GameObject(
                id=oid, x=o["x"], y=o["y"], color=o.get("color", 0),
                role=o.get("role", "unknown"), extra=o.get("extra", {}),
            )
        for p in d.get("predicates", []):
            st.predicates.append(Predicate(name=p["name"], args=p.get("args", [])))
        st.avatar_id = d.get("avatar_id")
        return st
