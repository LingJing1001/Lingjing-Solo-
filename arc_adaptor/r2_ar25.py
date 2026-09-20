"""AR25 的 R2 观察层：把「引擎内部状态」归一成 §3.1 的公共 observation，并给出**正式**状态哈希。

为什么单独做这一层（设计文档 §8.6 给 ③ 留的那块）：
    ② 期间的 `input_state_hash` 是 `sha256(repr(arc_shadow._state_key(g)))` 前 16 位。
    那是**搜索去重键**，不是状态身份：
      * `_state_key` 只收 (各对象 x/y/pixels, 选中下标, 两个未命名引擎字段)，
        而 `arc_shadow._snapshot` 明确把「旋转距离 `ovoizfolxfq`／`ouurgkpbbjj`」和
        另外三个标志位算作必须保存的可变全局状态——搜索键漏掉的，证据里不能跟着漏；
      * `repr` 一个含 `bytes` 的元组会把内存表示当文本哈希进去，跨实现不稳定。
    所以本层的哈希建立在**逐字段命名、可 JSON 化、排序确定**的观测上；`_state_key`
    那份只留作对照（`engine_key_hash`），由 `tick_trail` 在真实跑测里双向审计，
    而不是在这里口头声明「新哈希一定更细」。

读法约定（`enrich` 为什么存在）：
    §3.2 的 plan 出口 `ar25_plan(perc, method, path, budget)` 手上**只有观测、没有引擎句柄**，
    所以本层把「需要引擎」的那三个字段（`state` / `rotation_distances` / `engine_flags`）
    先由 `enrich(g, perc)` 一次性抄进观测里，之后 `semantic_state(perc)` / `state_hash(perc)`
    / `observe(perc, ...)` 只吃观测。少了这一步就抛 `R2PerceptionError`：哈希**少收一类可变
    状态**比跑不起来严重得多，绝不能静默算出一个更粗的哈希冒充状态身份。

探针实测（2026-09-20，`environment_files/ar25/0c556536`，L1）：
    ACTION5 在只有一个拼块时位移为 0、只扣 1 步 → `sel_idx`/对象坐标/覆盖集全不变，
    `_state_key` 与 `pixels` 也不变，只有 `current_steps` 64→63。这跟 LS20 的教训
    （`r2_ls20.py:92` 只哈希玩法区，免得 HUD 把同一局面判成两个）同构：**步数条不进
    `state_hash`**，但作为 `game_specific.steps_left` 逐 tick 落盘，`moved=False` 的
    转移正是 R4 反循环要看的东西。

本模块只依赖 stdlib：不 import `arcengine`／`arc_shadow`／`numpy`，这样 Tier A 测试在
没有引擎包的解释器里也能跑（AR25 运行器自身的回归测试就没这待遇，见设计文档 §8.10 第 1 条）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

OBSERVATION_SCHEMA = "ar25-r2-v1"

#: AR25 只用 1..5；实测 `is_complex()` 全 False，所以 `requested_action.payload` 恒为 {}。
#: 运行器那边以边界块的 `AR25_ACTION_NAMES` 为准，传进 `observe(legal_actions=...)`；
#: 这里的默认值只给单测和不接引擎的调用方兜底。
AR25_LEGAL_ACTIONS: tuple[str, ...] = tuple(f"ACTION{n}" for n in (1, 2, 3, 4, 5))

#: 混淆名 → 语义。标了「未证实」的是本轮探针没读出语义、但因为 `arc_shadow._snapshot`
#: 把它当可变全局状态而收进哈希的字段——宁可不解释，也不让证据里的状态身份少一块。
AR25_STATE_FIELDS: dict[str, str] = {
    "jtkyjqznbnp": "轴精灵列表（tags 判横竖轴）",
    "ayyvxqrhnzw": "可切换对象列表（轴 + 拼块；index 即选择序）",
    "yvifanjrcyu": "当前选中对象（ACTION5 切换它）",
    "fswikrcrdmx": "目标格精灵列表（`grid[y, x] < 0` 即未覆盖）",
    "lelsvjlwneo": "步数条（current_steps 剩余 / ilqnjlrnkk 本关上限）——不进 state_hash",
    "naxbskjmlg()": "21×21 渲染网格（frame 来源，也用来判覆盖）",
    "vplrhaovhr()": "是否获胜（通关判定的权威来源，不用像素差冒充）",
    "_current_level_index": "当前关卡下标（0 基；引擎到最后一关不再 +1）",
    "_state": "引擎 GameState",
    "ovoizfolxfq": "旋转距离 sprite→distance（arc_shadow._snapshot 视作可变全局状态）",
    "ouurgkpbbjj": "参与旋转的对象列表（上面那个映射的键序）",
    "hsiusrsrdkswnt": "未证实：探针 L1 恒为 int 0，但 _state_key 与 _snapshot 都收着它",
    "qehjebksqcm": "未证实：探针 L1 恒为 bool False，同上",
    "hujpxmlafgh": "未证实：只有 _snapshot 收着（说明它会随动作变）",
    "xukxeewuexo": "未证实：同上",
    "xjwpeqpcxav": "未证实：同上",
}

#: 进 `semantic_state` 的未命名引擎标志位（顺序固定，跨版本对账时按名字读）。
ENGINE_FLAG_FIELDS: tuple[str, ...] = (
    "hsiusrsrdkswnt", "qehjebksqcm", "hujpxmlafgh", "xukxeewuexo", "xjwpeqpcxav",
)

#: `enrich()` 补进观测的键——`semantic_state` 见到缺一个就拒绝算哈希。
ENRICHED_KEYS: tuple[str, ...] = ("state", "rotation_distances", "engine_flags")


class R2PerceptionError(RuntimeError):
    """读不到 AR25 引擎字段、或观测没经 `enrich` 就要求哈希时抛出。

    两类都 fail-closed：引擎改版要显式失败，不能猜；观测缺字段也不能算出一个"看起来像
    状态身份"其实少收了一类可变状态的哈希。
    """


def dumps(value: Any) -> str:
    """规范序列化：键排序、元组降级成列表，跨进程/跨实现稳定。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=list)


def state_name(g: Any) -> str:
    s = getattr(g, "_state", None)
    return str(getattr(s, "name", s))


def rotation_distances(g: Any) -> list[list[int]]:
    """`ovoizfolxfq`（对象→旋转距离）按 `ouurgkpbbjj` 的下标序列化成 [[idx, dist], ...]。

    和 `arc_shadow._snapshot:157` 用同一个映射关系，只是那边要的是可还原对象、这边要的是
    可哈希标量；排序后写入，避免 dict 迭代序影响哈希。
    """
    mapping = getattr(g, "ovoizfolxfq", None) or {}
    sprites = getattr(g, "ouurgkpbbjj", None) or []
    index_of = {id(s): i for i, s in enumerate(sprites)}
    pairs = [(index_of.get(id(s), -1), int(d)) for s, d in mapping.items()]
    return [list(p) for p in sorted(pairs)]


def engine_flags(g: Any) -> dict[str, Any]:
    """未命名标志位的**值**（转成 JSON 标量）；缺字段说明引擎改版，按 None 记并留给审计。"""
    out: dict[str, Any] = {}
    for name in ENGINE_FLAG_FIELDS:
        value = getattr(g, name, None)
        out[name] = value if isinstance(value, (int, float, str, bool)) or value is None else repr(value)
    return out


def enrich(g: Any, perc: dict[str, Any]) -> dict[str, Any]:
    """把只有引擎句柄才读得到的三个字段抄进 R2 观测，之后就只需要观测了（就地改并返回）。

    幂等：已 enrich 过再调一次只是重读同一状态，不叠加。搜索层每次 `r2_perceive` 都调它会
    白读引擎，所以调用点只在「要出证据 / 要出 plan」的两处。
    """
    perc["state"] = state_name(g)
    perc["rotation_distances"] = rotation_distances(g)
    perc["engine_flags"] = engine_flags(g)
    return perc


def _hash_semantic(semantic: dict[str, Any]) -> str:
    """语义态 → 16 位十六进制。截断长度与 `r2_ls20.state_hash:92` 对齐，便于跨游戏对账。"""
    return hashlib.sha256(dumps(semantic).encode("utf-8")).hexdigest()[:16]


#: `semantic_state` 写入的字段（顺序即语义态的字段表，manifest 直接发布它）。
#: 少了哪个字段，`state_hash` 就不再是那件事的状态身份——所以这张表要跟 `semantic_state` 一起改。
SEMANTIC_STATE_FIELDS: tuple[str, ...] = (
    "schema", "level", "state", "axes", "pieces", "sel_idx", "total_targets", "uncovered",
    "rotation_distances", "engine_flags",
)


def semantic_state(perc: dict[str, Any]) -> dict[str, Any]:
    """状态身份 = 这些字段的规范序列化。逐条可解释，且不含步数条（见模块 docstring）。"""
    missing = [k for k in ENRICHED_KEYS if k not in perc]
    if missing:
        raise R2PerceptionError(
            f"观测未经 enrich（缺 {', '.join(missing)}）：先调 r2_ar25.enrich(g, perc)，"
            "否则哈希会静默漏掉引擎可变状态")
    axes = sorted([[a["x"], a["y"], a["type"]] for a in perc["axes"]])
    pieces = sorted([[s["x"], s["y"], sorted([list(c) for c in (s["cells"] or [])])]
                     for s in perc["movable"]])
    semantic = {
        "schema": OBSERVATION_SCHEMA,
        "level": int(perc["level"]),
        "state": str(perc["state"]),
        "axes": axes,
        "pieces": pieces,
        "sel_idx": int(perc["sel_idx"]),
        "total_targets": int(perc["total_targets"]),
        "uncovered": sorted([list(t) for t in perc["uncovered"]]),
        "rotation_distances": [list(p) for p in perc["rotation_distances"]],
        "engine_flags": dict(perc["engine_flags"]),
    }
    # 字段表与实算内容必须一字不差地对上：manifest 发布的是这张表，读的人据此判断哈希覆盖了
    # 什么。表改了代码没改（或反之）就是"声明的状态身份 ≠ 实际哈希的东西"，宁可抛错。
    if tuple(semantic) != SEMANTIC_STATE_FIELDS:
        raise R2PerceptionError(
            f"semantic_state 与 SEMANTIC_STATE_FIELDS 不一致: {tuple(semantic)} != "
            f"{SEMANTIC_STATE_FIELDS}")
    return semantic


def state_hash(perc: dict[str, Any]) -> str:
    """正式状态哈希（§3.1 observation.state_hash / §3.2 plan.input_state_hash 共用它）。"""
    return _hash_semantic(semantic_state(perc))


def engine_key_hash(perc: dict[str, Any]) -> str | None:
    """对照用的搜索键哈希（② 那份实现）。缺 `state_key` 时返回 None，不编一个假哈希。"""
    key = perc.get("state_key")
    if key is None:
        return None
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]


def observe(perc: dict[str, Any], *, tick: int, frame: Any,
            run_id: str = "", episode_id: str = "", game_id: str = "ar25",
            legal_actions: Sequence[str] = AR25_LEGAL_ACTIONS) -> dict[str, Any]:
    """§3.1 公共 observation。AR25 是完整可观测游戏，`visibility` 恒为 full。

    `frame` 必填且由调用方从引擎现读（`g.naxbskjmlg().tolist()`）：§5.2 的 tick 行必须带
    真帧，这里没有引擎句柄，造一个占位帧就等于在证据里撒谎。
    """
    if frame is None:
        raise R2PerceptionError("observe 需要真帧：frame 必须由运行器从引擎渲染网格读出，不给占位帧")
    semantic = semantic_state(perc)
    return {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": f"{run_id or 'local'}/{tick}",
        "game_id": game_id,
        "episode_id": episode_id,
        "tick": int(tick),
        "frame_ref": None,                      # 帧直接进 recording 的 frame 字段，不留外部引用
        "grid_shape": [len(frame), len(frame[0]) if frame else 0],
        "state_hash": _hash_semantic(semantic),
        "state": str(perc["state"]),
        "levels_completed": int(perc["level"]),
        "legal_actions": [str(a) for a in legal_actions],
        "objects_or_features": {
            "axes": semantic["axes"],
            "pieces": semantic["pieces"],
            "sel_idx": semantic["sel_idx"],
            "targets": sorted([list(t) for t in perc["targets"]]),
            "uncovered": semantic["uncovered"],
        },
        "visibility": "full",
        "game_specific": {
            "backend": "engine_state",
            "steps_left": int(perc["steps_left"]),
            "budget": int(perc["budget"]),
            "won": bool(perc["won"]),
            "n_axes": int(perc["n_axes"]),
            "rotation_distances": semantic["rotation_distances"],
            "engine_flags": semantic["engine_flags"],
            "semantic_state": semantic,
            "engine_key_hash": engine_key_hash(perc),
        },
    }


def hash_of(obs: dict[str, Any]) -> str:
    """从已建好的 observation 复算 state_hash（审计路径用，不重新读引擎）。"""
    return _hash_semantic(obs["game_specific"]["semantic_state"])


def transition(before: dict[str, Any], action: str, after: dict[str, Any]) -> dict[str, Any]:
    """§3.1 要求的 (state_before, action, state_after) 因果记录。

    `moved` 按**语义**判：只扣一步、几何与覆盖集没变的动作（单拼块时的 ACTION5）moved=False，
    这是 R4 反循环的信号而不是异常；步数变化另记在 `steps_delta`，别让读的人以为状态没动。
    """
    return {
        "before_hash": before["state_hash"],
        "action": {"name": action},
        "after_hash": after["state_hash"],
        "before_engine_key": before["game_specific"].get("engine_key_hash"),
        "after_engine_key": after["game_specific"].get("engine_key_hash"),
        "levels_delta": after["levels_completed"] - before["levels_completed"],
        "moved": before["state_hash"] != after["state_hash"],
        "steps_delta": (after["game_specific"]["steps_left"]
                        - before["game_specific"]["steps_left"]),
        "covered_delta": ((len(after["objects_or_features"]["targets"])
                           - len(after["objects_or_features"]["uncovered"]))
                          - (len(before["objects_or_features"]["targets"])
                             - len(before["objects_or_features"]["uncovered"]))),
    }
