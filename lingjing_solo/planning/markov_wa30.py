"""WA30 经验马尔可夫决策层

用"马尔可夫模型法"驱动 WA30 的自动播放, 三层结构:

1. 状态抽象 —— arc_shadow_wa30.get_state() 给出: 关卡号、玩家(位置+朝向)、
   箱子位置、抓取关系、目标格/墙格/障碍格、剩余步数、引擎状态。
   马尔可夫性实测过(_diag_20260921/wa30_markov_validity2.py):
   重放确定性 15/15 一致; 同一抽象状态由不同历史到达后施加同一动作,
   后继一致 60 项 0 分歧。去掉玩家朝向会出现分歧(抓取按朝向判定身前
   一格, 见 wa30.py 的 vwiozbtqgi), 所以朝向必须在状态里。

2. 转移模型 —— T[(状态哈希, 动作)] -> 后继状态哈希, 与状态表一起持久化到
   data/markov_wa30_transfers.json。某转移第一次出现才真实调用引擎(oracle),
   之后同一(状态,动作)直接查表, 不动引擎 —— 这就是从交互经验里学到的
   马尔可夫转移表, 也是同一关卡反复试探时越跑越快的原因。

3. 决策 —— 在模型上做加权 A*(启发式 = 各箱子到最近目标格的曼哈顿距离之和
   + 每箱一次抓取量)。搜索到"过关/关卡推进"即得整条计划; 计划跨回合复用,
   每回合执行前做闭环校验(实际状态哈希 == 上一步预测状态哈希), 一旦偏差
   (例如手动点过、或模型没覆盖到的转移)立即重新搜索。
   搜索超时则退回已找到的最优部分计划, 再不行退回 1 层贪心, 保证不卡住。
"""
import hashlib
import heapq
import json
import os
import time
from collections import deque as _pdq

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "markov_wa30_transfers.json")
MAX_STATES = 30000
MAX_TRANS = 60000
KEY_ACTIONS = (1, 2, 3, 4, 5)

# 每关内不变的字段。实测(data/markov_wa30_transfers.json 8218 状态 / 14.5MB 的
# 字段分解): walls 41.0% + targets 37.3% + spawn 8.2% = 86.5% 的文件体积都是
# 同一份墙格/目标格被逐状态重复存了 8218 遍 —— 落盘前按关抽成"关卡画像",
# 只在画像与本状态完全一致时才省略该键, 读回时再合并回去(见 Model.save/load)。
_STATIC_FIELDS = ("walls", "targets", "spawn", "locked", "blockers")

# 跨回合计划缓存: {"states": [s0, s1, ...], "actions": [a0, a1, ...], "i": 下一步下标}
_plan = None
_plan_hit = 0
_plan_miss = 0


# ---------------- 状态抽象 ----------------

def hash_state(sd, helper=None):
    """抽象状态哈希.

    steps_left **不参与哈希**: 它是"走了多少步"的记账, 不是局面的性质。带上它
    之后同一局面每消耗一步就换一个哈希, 经验转移表几乎永远命中不了 —— 实测
    (_diag_20260921/wa30_stall_probe.py) 110 个模型状态去掉 steps_left 只剩 26 个
    不同局面, 20 个局面被拆开, 一次决策 50 次引擎调用只换来 10 次表命中,
    0.7s 预算只能展开 10 个节点, A* 走不出 30 步。
    步数预算并没有被忽略: 耗尽时引擎把 _state 置为 GAME_OVER, 而 state 仍在哈希里,
    所以"没步数了"依然是模型中一个真实的终止转移。
    """
    return hashlib.sha1(json.dumps(_keyed(sd, helper=helper), sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _keyed(sd, helper=None):
    out = {k: v for k, v in sd.items() if k not in _HASH_SKIP}
    hs = helper if helper is not None else _helper_state(sd)
    if hs is not None:
        out["_helper"] = list(hs)
    return out


_HASH_SKIP = ("steps_left",)

# 影子状态里的助手字段 (关卡 2+ 存在自主搬运工 kdweefinfi, 它每玩家步动一次,
# 位置/抓取关系影响箱子后继 —— 不进哈希会导致同哈希不同后继, 破坏马尔可夫表)
_HELPER_TAG = "kdweefinfi"

def _helper_state(sd):
    """从影子读助手位置与抓取关系; 关卡无助手时返回 None.

    为避免改动 get_state 的持久化格式, 这里直接读引擎 (只读).
    """
    try:
        from . import arc_shadow_wa30 as sh
        g = sh._game
        helpers = g.current_level.get_sprites_by_tag(_HELPER_TAG)
        if not helpers:
            return None
        h = helpers[0]
        held = g.nsevyuople.get(h)
        return (int(h.x), int(h.y), int(held.x) if held is not None else -1,
                int(held.y) if held is not None else -1)
    except Exception:
        return None


def _cell_set(vals):
    return {(int(x) // 4, int(y) // 4) for x, y in vals}


def is_goal(sd, start_level=None):
    """过关: 关卡号推进, 或所有箱子都落在目标格上且没被抱着."""
    if start_level is not None and sd["level"] > start_level:
        return True
    if sd["held"]:
        return False
    ts = _cell_set(sd["targets"])
    return all((int(b[0]) // 4, int(b[1]) // 4) in ts for b in sd["boxes"])


def heuristic(sd, start_level=None):
    """剩余代价估计(单位: 步).

    - 每个未到位的箱子: 到最近目标格的曼哈顿距离(格) + 1 次抓取/释放;
      被抱着的箱子按玩家当前位置算, 因为它跟着玩家走。
    - 末尾再加"走到下一个待搬箱子"的距离: 少了这一项, 玩家朝箱子靠近时启发值
      完全不变, A*/贪心会卡在平地上(实测 8 次决策全选同一个动作, h 恒为 11)。
    """
    if is_goal(sd, start_level):
        return 0.0
    ts = _cell_set(sd["targets"])
    if not ts:
        return 1e9
    px = int(sd["player"][0]) // 4
    py = int(sd["player"][1]) // 4
    held_cells = {(int(h[2]) // 4, int(h[3]) // 4) for h in sd["held"]}
    total = 0.0
    pending = []
    for bx, by in sd["boxes"]:
        cx, cy = int(bx) // 4, int(by) // 4
        if (cx, cy) in held_cells:
            # 抱着的箱子跟着玩家走, 玩家离目标多远它就离目标多远
            total += min(abs(px - tx) + abs(py - ty) for tx, ty in ts) + 1.0
        elif (cx, cy) in ts:
            continue
        else:
            total += min(abs(cx - tx) + abs(cy - ty) for tx, ty in ts) + 1.0
            pending.append((cx, cy))
    if pending:
        total += min(abs(px - cx) + abs(py - cy) for cx, cy in pending) + 1.0
    return total


def terminal(sd):
    return sd["state"] in ("WIN", "GAME_OVER")


def meta(model, sha, sd):
    """(启发值, 箱子是否全落在目标格上) —— 按状态哈希缓存.

    这两个量只由抽象状态决定, 而 A* 会对同一状态反复求值: 实测一次决策里
    heuristic/_cell_set 占 27ms/69ms(热表搜索), 缓成一次查表。
    """
    m = model.meta.get(sha)
    if m is None:
        level = sd["level"]
        on_target = is_goal(sd, level)
        m = (0.0 if on_target else heuristic(sd, level), on_target)
        if len(model.meta) > 200000:
            model.meta.clear()
        model.meta[sha] = m
    return m


# ---------------- 马尔可夫转移表 ----------------

class Model:
    def __init__(self, path=MODEL_PATH):
        self.path = path
        self.states = {}   # sha -> state dict
        self.trans = {}    # "sha|action" -> sha
        self.meta = {}     # sha -> (启发值, 箱子是否全在目标格); 纯内存, 不落盘
        self.loaded = False
        self.hits = 0
        self.misses = 0
        self.dirty = False
        self.saved_at = 0.0

    def load(self):
        if self.loaded:
            return self
        self.loaded = True
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profiles = {str(k): v for k, v in (data.get("profiles") or {}).items()}
                slim = dict(data.get("states") or {})
                self.trans = dict(data.get("trans") or {})
                if profiles:
                    self.states = {}
                    for sha, sd in slim.items():
                        prof = profiles.get(str(sd.get("level")))
                        if prof:
                            merged = dict(prof)
                            merged.update(sd)
                            sd = merged
                        self.states[sha] = sd
                else:
                    self.states = slim
        except Exception:
            self.states = {}
            self.trans = {}
        return self

    def remember(self, sha, action, sd):
        """把 oracle 走出来的转移写进模型(经验积累)."""
        s2 = hash_state(sd)
        key = f"{sha}|{action}"
        if self.trans.get(key) is None:
            self.trans[key] = s2
            self.states.setdefault(s2, sd)
            self.dirty = True
        return s2

    def get(self, sha, action):
        """查表; 命中返回 (sha2, state2), 未命中返回 None."""
        s2 = self.trans.get(f"{sha}|{action}")
        if s2 is None:
            return None
        sd = self.states.get(s2)
        if sd is None:
            return None
        self.hits += 1
        return s2, sd

    def put_state(self, sha, sd):
        if sha not in self.states:
            self.states[sha] = sd
            self.dirty = True

    def save(self, min_interval=8.0):
        """落盘经验表(按关抽出静态字段).

        带最小间隔: 一次决策的搜索本身只有几十~几百 ms, 而表涨到几千条后
        json.dump 要 230ms(实测 profile) —— 每次决策都存等于把一半时间花在写文件上。
        自动播放是长跑, 8 秒一存足够, 进程被杀最多丢最近几秒的经验。

        为什么还要抽画像: 涨到 8218 状态后整份表 14.5MB, json.dump 实测 9.07s、
        读回 347ms —— 而这 9 秒是在 run_in_threadpool 里抱着 GIL 干的, 期间
        所有 HTTP 请求一起卡住。14.5MB 里 86.5% 是逐状态重复存的墙格/目标格/
        刷新点(见 _STATIC_FIELDS 注释), 抽成"每关一份"后同一份数据只落一次。
        画像只收录"该关每个状态都取值相同"的键, 所以读回时把画像合并进去
        与原始状态逐字节相等, 状态哈希不受影响(见 Model.load)。
        """
        if not self.dirty:
            return False
        now = time.perf_counter()
        if now - self.saved_at < min_interval:
            return False
        self.saved_at = now
        try:
            # 容量控制: 超限丢弃最早写入的条目(dict 保持插入顺序)
            if len(self.states) > MAX_STATES:
                for k in list(self.states.keys())[:len(self.states) - MAX_STATES]:
                    self.states.pop(k, None)
            if len(self.trans) > MAX_TRANS:
                for k in list(self.trans.keys())[:len(self.trans) - MAX_TRANS]:
                    self.trans.pop(k, None)
            profiles, slim = self._split()
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"states": slim, "trans": self.trans, "profiles": profiles}, f)
            os.replace(tmp, self.path)
            self.dirty = False
            return True
        except Exception:
            return False

    def _split(self):
        """把 self.states 拆成 (画像, 瘦身后的状态表); 画像键为 str(level)."""
        groups = {}
        for sd in self.states.values():
            groups.setdefault(str(sd.get("level")), []).append(sd)
        profiles = {}
        drop = {}
        for lv, sds in groups.items():
            base = sds[0]
            keep = {k: base[k] for k in _STATIC_FIELDS if k in base}
            for sd in sds[1:]:
                for k in list(keep):
                    if k not in sd or sd[k] != keep[k]:
                        del keep[k]
            if keep:
                profiles[lv] = keep
            drop[lv] = set(keep)
        slim = {}
        for sha, sd in self.states.items():
            keys = drop.get(str(sd.get("level")))
            if not keys:
                slim[sha] = sd
            else:
                slim[sha] = {k: v for k, v in sd.items() if k not in keys}
        return profiles, slim


_model = Model()


# ---------------- 引擎预言机 ----------------

class Oracle:
    """引擎预言机: 前缀一律**相对于锚点(实时状态)**, 能增量前进就不回锚点.

    实测成本: 单步 0.1~0.4ms; 整体重放 120 步 12.6ms, 代价随"已执行步数"线性
    增长(自动播放几百步后就不可接受了); 定向快照恒定 快照1.1ms/恢复1.5ms —— 所以
    锚点优先用快照, 快照失败才退回"重置+重放实时历史"。
    """

    def __init__(self, read, step, to_anchor, snapshot=None, restore_bag=None):
        self.read = read
        self.step = step
        self.to_anchor = to_anchor
        self.snapshot = snapshot
        self.restore_bag = restore_bag
        self.prefix = ()      # 当前相对锚点的动作前缀
        self.engine_steps = 0
        self.anchors = 0

    def go(self, rel):
        rel = tuple(rel)
        if rel == self.prefix:
            return
        cur = self.prefix
        if len(rel) > len(cur) and rel[:len(cur)] == cur:
            for a in rel[len(cur):]:
                if not self.step(a):
                    break
                self.engine_steps += 1
        else:
            self.to_anchor()
            self.anchors += 1
            self.prefix = ()
            for a in rel:
                if not self.step(a):
                    break
                self.engine_steps += 1
        self.prefix = rel

    def child(self, rel, action):
        """从相对前缀 rel 执行 action, 返回后继抽象状态(引擎真实执行)."""
        self.go(rel)
        if not self.step(action):
            return None
        self.prefix = tuple(rel) + (action,)
        self.engine_steps += 1
        return self.read()

    def expand(self, rel, actions):
        """批量扩展一个节点: 走到 rel 后只拍一次快照, 每个动作从该快照回滚.

        为什么不能直接用 child(): child 会把引擎留在 rel+a 上, 算下一个兄弟动作时
        又要 go(rel) —— 那是一次"回锚点 + 重放 |rel| 步"。实测每展开一个节点要
        重放 36 次引擎(一次决策 eng=1738 / 48 节点), 0.7s 预算只够展开几十个节点。
        批量版把每节点的引擎调用降到 |rel|(走过来) + |actions|(各一步 + 一次回滚)。

        快照只覆盖当前关, 所以某个动作把引擎带过关卡边界后这份快照就作废
        (restore 返回 None); 此时退回 child() 的重放路径, 正确性优先。
        """
        rel = tuple(rel)
        self.go(rel)
        bag = self.snapshot() if self.snapshot is not None else None
        for a in actions:
            if bag is not None:
                if self.restore_bag(bag) is None:
                    bag = None            # 引擎已跨过关卡边界, 快照失效
                else:
                    self.prefix = rel     # 回滚成功: 引擎确实又站在 rel 上
            if self.prefix != rel:
                self.go(rel)
            if not self.step(a):
                continue
            self.engine_steps += 1
            self.prefix = rel + (a,)
            sd = self.read()
            if sd is not None:
                yield a, sd
        if bag is not None:
            self.restore_bag(bag)
            self.prefix = rel

    def anchor(self):
        """把引擎放回锚点(实时状态). 搜索结束后必须调用一次."""
        self.to_anchor()
        self.anchors += 1
        self.prefix = ()


# ---------------- 搜索 ----------------

def search(oracle, model, start_sha, start_sd, actions, max_nodes, deadline, w=3.0):
    """加权 A*. 返回 (动作序列, 状态序列, 展开节点数, 是否找到完整解).

    节点前缀一律相对于 Oracle 的锚点(=实时状态)。早先版本在这里混用过绝对/相对
    前缀, 出过一个实质性 bug: 根节点前缀写成 (), 而 Oracle 把它当成"从初始状态
    起算的绝对前缀", 于是第一次扩展就把引擎倒回关卡起点, 记录下来的转移变成
    "实时状态哈希 -> 起点状态的后继", 整张马尔可夫转移表被污染。现在 Oracle 只认
    相对前缀, 锚点就是实时状态, 这一类错位不会再从参数上漏进来。

    未找到完整解时返回启发值最低的那条部分计划 —— 链上每一步都是马尔可夫模型里
    已有的转移(查表得来或刚让引擎真走过), 不是猜的。
    """
    level = start_sd["level"]
    model.put_state(start_sha, start_sd)
    h0 = meta(model, start_sha, start_sd)[0]
    # 堆元素: (f, g, sha, 相对动作前缀, 状态前缀)
    heap = [(h0 * w, 0, start_sha, (), (start_sha,))]
    best_g = {start_sha: 0}
    # 部分计划的评价: 启发值越低越好; 打平时取走得更深的那条 —— 绕过被挡住的箱子
    # 之前往往要侧移几步, 这几步 h 不变, 不按深度取舍就会返回空计划
    best_partial = (h0, 0, (), (start_sha,))
    nodes = 0

    while heap and nodes < max_nodes:
        if time.perf_counter() > deadline:
            break
        f, g, sha, prefix, chain = heapq.heappop(heap)
        if g < best_g.get(sha, 1e9) - 1:
            continue
        nodes += 1
        sd = model.states.get(sha)
        if sd is None:
            oracle.go(prefix)
            sd = oracle.read()
            if sd is None:
                continue
            model.put_state(sha, sd)
        h = meta(model, sha, sd)[0]
        key = (h, -len(prefix))
        if key < (best_partial[0], best_partial[1]):
            best_partial = (h, -len(prefix), prefix, chain)
        if terminal(sd):
            continue

        # 表里已有的转移直接用; 缺的让引擎一次批量走出来(每个节点只重放一次前缀)
        successors = {}
        missing = []
        for a in actions:
            cached = model.get(sha, a)
            if cached is not None:
                successors[a] = cached
            else:
                missing.append(a)
        if missing:
            for a, s2 in oracle.expand(prefix, missing):
                model.misses += 1
                successors[a] = (model.remember(sha, a, s2), s2)

        for a in actions:
            got = successors.get(a)
            if got is None:
                continue
            s2_sha, s2 = got
            ng = g + 1
            if s2["level"] < level or terminal(s2):
                continue
            if best_g.get(s2_sha, 1e9) <= ng:
                continue
            best_g[s2_sha] = ng
            child_prefix = prefix + (a,)
            child_chain = chain + (s2_sha,)
            h2, on_target = meta(model, s2_sha, s2)
            if on_target or s2["level"] > level:
                return list(child_prefix), list(child_chain), nodes, True
            heapq.heappush(heap, (ng + h2 * w, ng, s2_sha, child_prefix, child_chain))

    return list(best_partial[2]), list(best_partial[3]), nodes, False


def one_ply(oracle, model, sha, sd, actions):
    """1 层贪心: 挑启发式最小的后继动作. 搜索完全没有产出时的最后兜底."""
    level = sd["level"]
    best_key = None
    best_action = None
    for a in actions:
        s2 = oracle.child((), a)
        if s2 is None:
            continue
        model.misses += 1
        model.remember(sha, a, s2)
        if terminal(s2):
            continue
        key = (heuristic(s2, level), -len(s2["held"]))
        if best_key is None or key < best_key:
            best_key = key
            best_action = a
    if best_action is None and actions:
        best_action = actions[0]
    return best_action


def _make_oracle(sh, live_prefix):
    """建一个锚定在实时状态的 Oracle: 优先用定向快照, 失败则退回整体重放."""
    api = sh.search_engine()
    if api is None:
        return None
    _, read, step, restore, snapshot, restore_snapshot = api
    before = read()
    anchor_kind = "replay"
    replay_anchor = lambda: restore(live_prefix)  # noqa: E731
    to_anchor = replay_anchor
    snap = bag = None
    try:
        bag = snapshot()
        # 只有"存->取"能原样回到当前状态才敢把快照当锚点, 否则宁可退回重放
        if before is not None and restore_snapshot(bag) == before:
            snap, anchor_kind = snapshot, "snapshot"
            anchor_live = {"ok": True}

            def to_anchor():
                # 快照只覆盖当前关: 搜索中引擎一旦跨过关卡边界, 下一关的 sprite 也被
                # 走脏了, 这份 bag 从此不可信 —— 永久退回"重置 + 重放实时历史"。
                # (重置会把 9 个关卡全部还原, 所以之后逐节点快照仍然可用。)
                if anchor_live["ok"] and restore_snapshot(bag) is not None:
                    return None
                anchor_live["ok"] = False
                return replay_anchor()
    except Exception:
        snap = None
        to_anchor = replay_anchor
        anchor_kind = "replay"
    oracle = Oracle(read, step, to_anchor, snapshot=snap, restore_bag=restore_snapshot)
    oracle.anchor_kind = anchor_kind
    return oracle


def _check_sync(sh, expect_actions_taken):
    """影子记账是否和会话一致(只检查不修).

    对不上通常意味着有人手动点过/上次搜索中途异常; 会话历史才是权威, 由
    arc-smart-step 先 replay 修正。这里顺手作废跨回合计划, 避免拿旧预测继续走。
    """
    global _plan
    if expect_actions_taken is None:
        return True
    if sh.is_synced(expect_actions_taken):
        return True
    _plan = None
    return False


# ---------------- 对外入口 ----------------

def decide(t_limit=0.7, max_nodes=400, expect_actions_taken=None):
    """给 arc-smart-step 调用: 返回 {action, message, debug}, 只决策不改引擎.

    真正的执行由前端随后调用的 arc-step 完成, 所以这里结束时必须把引擎放回实时
    状态(oracle.anchor())。
    """
    global _plan, _plan_hit, _plan_miss
    from . import arc_shadow_wa30 as sh

    if not sh._ensure():
        return {"error": f"影子引擎未加载: {sh.status().get('error')}"}

    sh.search_acquire()
    try:
        synced = _check_sync(sh, expect_actions_taken)
        live_prefix = tuple(sh.get_history())
        oracle = _make_oracle(sh, live_prefix)
        if oracle is None:
            return {"error": "引擎不可用"}

        model = _model.load()
        sd = oracle.read()
        if sd is None:
            return {"error": "无法读取状态"}
        sha = hash_state(sd)
        model.put_state(sha, sd)
        actions = [a for a in sh.get_available_actions() if a in KEY_ACTIONS] or list(KEY_ACTIONS)
        t0 = time.perf_counter()
        base_debug = {"mode": "-", "level": sd["level"], "steps_left": sd["steps_left"],
                      "h": round(heuristic(sd, sd["level"]), 2), "boxes": len(sd["boxes"]),
                      "anchor": oracle.anchor_kind, "synced": synced,
                      "states_in_model": len(model.states),
                      "transitions_in_model": len(model.trans)}

        if terminal(sd):
            return {"error": f"本局已结束({sd['state']})", "action": None}

        # 1) 闭环校验 + 计划复用: 当前实际状态正是上一步预测的状态吗
        #    加严 (实测 wa30_l1_autopsy): 只对齐哈希不够 —— 计划是预算余量极小的
        #    情况下搜出的长链, 任何一步浪费 (如表中缺失的撞墙转身转移被当成无效)
        #    都会让余下计划变成致命拖沓。现在额外要求余量 ≥ 剩余步数 + 2 安全垫,
        #    余量不足时弃用重搜, 不再拿过期计划耗到 GAME_OVER。
        _reuse = (_plan and _plan["i"] < len(_plan["actions"])
                  and _plan["states"][_plan["i"]] == sha)
        if _reuse:
            _remain = len(_plan["actions"]) - _plan["i"]
            if _remain + 2 >= sd["steps_left"]:
                _reuse = False
                _plan = None
                _plan_miss += 1
        if _reuse:
            action = _plan["actions"][_plan["i"]]
            _plan["i"] += 1
            _plan_hit += 1
            model.save()
            dbg = dict(base_debug, mode="plan-reuse", plan_left=len(_plan["actions"]) - _plan["i"],
                       engine_steps=oracle.engine_steps, anchors=oracle.anchors,
                       elapsed_ms=int((time.perf_counter() - t0) * 1000))
            return {"action": action,
                    "message": f"马尔可夫计划复用 第{_plan['i']}/{len(_plan['actions'])}步",
                    "debug": dbg}

        # 1.5) 流水线优先 (关卡 2 型: 全高 blocker 墙 + 自主助手):
        #      A* 在这种关只能给出部分计划 (完整解搜不动), 部分计划会把预算磨光。
        #      单步闭环策略: 每次从实时状态推导"此刻该做什么", 天然免疫助手干扰。
        #      实测 (_diag_20260921/wa30_l2_pipeline_solve2.py) 同策略 80 步过关。
        if sd.get("blockers"):
            _pa = _wall_pipeline_action(sd)
            if _pa is not None:
                _plan = None
                dbg = dict(base_debug, mode="pipeline",
                           engine_steps=oracle.engine_steps, anchors=oracle.anchors,
                           elapsed_ms=int((time.perf_counter() - t0) * 1000))
                return {"action": _pa,
                        "message": "墙边交接流水线",
                        "debug": dbg}

        # 2) 在马尔可夫模型上重新 A* 规划
        #    末段加档 (实测 wa30_l1_autopsy): h 已很小时 (≤2) 若还按常规 6000 节点,
        #    搜不出 5~6 步的完整收尾链, 相邻两次决策的部分计划方向互相打架
        #    (抓起又放回、来回振荡), 在余量耗尽前差最后一步。小 h 时模型已热,
        #    大节点预算只是毫秒级 —— 直接拉满把尾部搜穿。
        _h_now = heuristic(sd, sd["level"])
        _nodes_now = max_nodes if _h_now > 2.0 else max(max_nodes, 40000)
        _tlim_now = max(t_limit, 3.0) if _h_now <= 2.0 else t_limit
        plan, chain, nodes, found = search(oracle, model, sha, sd, actions, _nodes_now,
                                           t0 + _tlim_now)
        oracle.anchor()                       # 引擎放回实时状态, 交给 arc-step 执行
        live_ok = tuple(sh.get_history()) == live_prefix
        elapsed = int((time.perf_counter() - t0) * 1000)
        model.save()

        if plan:
            _plan = {"states": chain, "actions": plan, "i": 1}
            dbg = dict(base_debug, mode="search", nodes=nodes, plan_len=len(plan),
                       complete=found, engine_steps=oracle.engine_steps, hist_ok=live_ok,
                       anchors=oracle.anchors, elapsed_ms=elapsed)
            return {"action": plan[0],
                    "message": f"马尔可夫+A* {'完整' if found else '部分'}规划{len(plan)}步",
                    "debug": dbg}

        # 2.5) A* 无产出且地形符合「全高 blocker 墙 + 自主助手」特征:
        #      回退墙边交接流水线 (关卡 2 型专用, 实测 80 动作过关); 流水线的
        #      states 占位用实时哈希, 计划复用的闭环校验会在引擎偏差时自动作废重算。
        pipeline = _wall_pipeline_plan(sd)
        if pipeline:
            _plan = {"states": [sha] * len(pipeline), "actions": pipeline, "i": 1}
            dbg = dict(base_debug, mode="pipeline", plan_len=len(pipeline),
                       engine_steps=oracle.engine_steps, hist_ok=live_ok,
                       anchors=oracle.anchors, elapsed_ms=int((time.perf_counter() - t0) * 1000))
            return {"action": pipeline[0],
                    "message": f"墙边交接流水线 {len(pipeline)}步",
                    "debug": dbg}

        # 3) 兜底: 1 层贪心
        action = one_ply(oracle, model, sha, sd, actions)
        oracle.anchor()
        model.save()
        if action is None:
            return {"error": "没有可用动作", "action": None}
        _plan = None
        _plan_miss += 1
        dbg = dict(base_debug, mode="greedy", nodes=nodes, engine_steps=oracle.engine_steps,
                   hist_ok=tuple(sh.get_history()) == live_prefix, anchors=oracle.anchors,
                   elapsed_ms=int((time.perf_counter() - t0) * 1000))
        return {"action": action, "message": "1层贪心兜底", "debug": dbg}
    finally:
        sh.search_release()


def invalidate_plan():
    """外部动作(手动点击/重试)后作废计划缓存, 下次决策重新搜索."""
    global _plan
    _plan = None


def _wall_pipeline_plan(sd):
    """关卡级专用求解器: 「墙边交接流水线」(实测 _diag_20260921/wa30_l2_pipeline_solve2.py
    用 80 动作/49 预算过关卡 2).

    适用条件 (关卡 2 地形特征): 全高 blocker 墙把地图切成两半, 玩家与目标区分居两侧,
    且存在自主搬运工 (kdweefinfi 助手): 每玩家步走一步、自动抓可及箱子、拖向目标区、
    到位自动松手, 被拖物落点不查 blockers (fuykgiiwit 的 tpos 校验)。

    策略: 玩家把左侧每只箱子拖到墙列 (x=WALL_X) 松手 (箱落在 blocker 格上),
    助手自动从右侧接走送进目标区。

    注意: 助手每玩家步走一步 —— 离线无法预知它的位置, 所以这里只生成「玩家侧脚本」;
    在线执行由 decide 每步闭环校验, 偏差即弃用重算。
    """
    from collections import deque as _dq
    blockers = {(int(x), int(y)) for x, y in sd.get("blockers", ())}
    if not blockers:
        return None
    xs = [x for x, _ in blockers]
    if max(xs) - min(xs) >= 4:      # blocker 不成单列, 不是本型
        return None
    wall_x = min(xs)
    targets = {(int(x) // 4, int(y) // 4) for x, y in sd.get("targets", ())}
    boxes = [(int(x), int(y)) for x, y in sd["boxes"]]
    left_boxes = [b for b in boxes if b[0] < wall_x]
    if not left_boxes or all((b[0] // 4, b[1] // 4) in targets for b in boxes):
        return None
    walls = {(int(x), int(y)) for x, y in sd.get("walls", ())
             if 0 <= int(x) < 64 and 0 <= int(y) < 64}

    def cell(p):
        return (p[0] // 4, p[1] // 4)

    def nav(start, goal, blocked):
        prev = {start: None}
        q = _dq([start])
        while q and goal not in prev:
            cur = q.popleft()
            for d, a in (((0, -1), 1), ((0, 1), 2), ((-1, 0), 3), ((1, 0), 4)):
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if 0 <= nxt[0] < 16 and 0 <= nxt[1] < 16 and nxt not in prev and nxt not in blocked:
                    prev[nxt] = (cur, a)
                    q.append(nxt)
        if goal not in prev:
            return None
        path = []
        cur = goal
        while prev[cur] is not None:
            path.append(prev[cur][1])
            cur = prev[cur][0]
        return list(reversed(path))

    plan = []
    boxes_dyn = list(boxes)
    p_cell = cell((int(sd["player"][0]), int(sd["player"][1])))
    for box in sorted(left_boxes, key=lambda b: (wall_x - b[0], b[1])):
        blocked = ({cell(b) for b in boxes_dyn if b != box}
                   | {cell(w) for w in walls}
                   | {cell(bl) for bl in blockers})
        seg = nav(p_cell, (box[0] // 4 - 1, box[1] // 4), blocked)
        if seg is None:
            return None
        plan += seg
        plan += [4, 5]                          # 朝右抓
        plan += [4] * ((wall_x - box[0]) // 4)  # 拖到墙列
        plan += [5]                             # 墙边松手 (箱子停在 blocker 格)
        boxes_dyn = [((wall_x, b[1]) if b == box else b) for b in boxes_dyn]
        p_cell = (wall_x // 4 - 1, box[1] // 4)
        plan += [3]                             # 转身 (兼待机步, 给助手时间)
    return plan if plan else None


def _wall_pipeline_action(sd):
    """关卡 2 型单步闭环策略 (每次只回答"此刻该做什么", 从实时状态推导).

    地形特征: 单列全高 blocker 墙把地图切成两半, 目标区与自主助手 (kdweefinfi,
    每玩家步走一步、自动抓拖送箱) 都在墙右侧; 玩家在左侧。
    玩家任务: 逐只把左箱拖到墙列松手 (被拖物落点不查 blockers, 箱停在墙上),
    助手会自动接走送进目标区 (实测 wa30_l2_pipeline_solve2.py 全程 80 步过关)。
    返回单个动作; 地形不适用返回 None。
    """
    blockers = {(int(x), int(y)) for x, y in sd.get("blockers", ())}
    if not blockers:
        return None
    xs = [x for x, _ in blockers]
    if max(xs) - min(xs) >= 4:          # blocker 不成单列, 不是本型
        return None
    wall_x = min(xs)
    px, py = int(sd["player"][0]), int(sd["player"][1])
    rot = int(sd["player"][2])
    boxes = [(int(x), int(y)) for x, y in sd["boxes"]]

    helper_held = set()
    player_held = None
    for hx, hy, tx, ty in sd.get("held", ()):
        if (int(hx), int(hy)) == (px, py) and (int(tx), int(ty)) in set(boxes):
            player_held = (int(tx), int(ty))
        else:
            helper_held.add((int(tx), int(ty)))

    # 玩家正拖着箱子:
    #   - 箱已到墙列 -> 松手 (箱停在 blocker 格上)
    #   - 抓反了方位 (箱在玩家左侧, 即玩家站在箱右侧) -> 立即松手重抓:
    #     这种方位玩家拖到墙前自己先被 blocker 挡住, 箱永远到不了墙列
    #     (实测卡死在 (7,11) 的根因)
    #   - 否则继续向右拖
    if player_held is not None:
        if player_held[0] >= wall_x or player_held[0] < px:
            return 5
        return 4

    # 选交接箱: 墙左侧、没被助手抓着
    cands = [b for b in boxes if b[0] < wall_x and b not in helper_held]
    if not cands:
        # 左侧全部交接完毕: 待机 (朝墙撞 = 纯转身不移动), 用玩家步驱动助手清扫
        return 4 if px < wall_x else 1

    # 已站在某交接箱正左侧: 面向右抓 (唯一能拖到墙的方位)
    for b in cands:
        if px == b[0] - 4 and py == b[1]:
            return 5 if rot == 90 else 4

    box_set = set(boxes)
    walls = {(int(x), int(y)) for x, y in sd.get("walls", ())
             if 0 <= int(x) < 64 and 0 <= int(y) < 64}

    def free(c):
        return c not in box_set and c not in walls and c not in blockers

    # BFS 到最近交接箱的**左侧格** (唯一正确方位), 不再接受任意贴身格
    goals = {(b[0] - 4, b[1]) for b in cands}
    start = (px, py)
    seen = {start}
    parents = {}
    goal = None
    q = _pdq([start])
    while q and goal is None:
        cur = q.popleft()
        if cur != start and cur in goals:
            goal = cur
            break
        for d, a in (((0, -4), 1), ((0, 4), 2), ((-4, 0), 3), ((4, 0), 4)):
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if nxt in seen or not free(nxt):
                continue
            seen.add(nxt)
            parents[nxt] = (cur, a)
            q.append(nxt)
    if goal is None:
        return 4 if px < wall_x else 1   # 左侧格都不可达: 待机 (助手可能先接走)

    cur = goal
    while cur in parents and parents[cur][0] != start:
        cur = parents[cur][0]
    return parents[cur][1]


def solve_current_level(t_limit=12.0, max_nodes=20000):
    """给 arc_shadow_wa30.solve_level 用: 一次规划出当前关的完整动作序列.

    先试 A* 完整解; 失败且地形符合「全高 blocker 墙 + 自主助手」特征时
    回退到墙边交接流水线 (_wall_pipeline_plan).
    """
    from . import arc_shadow_wa30 as sh
    if not sh._ensure():
        return None
    sh.search_acquire()
    try:
        oracle = _make_oracle(sh, tuple(sh.get_history()))
        if oracle is None:
            return None
        model = _model.load()
        sd = oracle.read()
        if sd is None:
            return None
        if is_goal(sd):
            oracle.anchor()
            return []
        actions = [a for a in sh.get_available_actions() if a in KEY_ACTIONS] or list(KEY_ACTIONS)
        plan, chain, nodes, found = search(
            oracle, model, hash_state(sd), sd, actions, max_nodes,
            time.perf_counter() + t_limit)
        model.save()
        if not found:
            pipeline = _wall_pipeline_plan(sd)
            if pipeline:
                oracle.anchor()
                return pipeline
        oracle.anchor()
        return plan if found else None
    finally:
        sh.search_release()


def stats():
    m = _model.load()
    return {"states": len(m.states), "transitions": len(m.trans),
            "oracle_calls": m.misses, "cache_hits": m.hits,
            "plan_hit": _plan_hit, "plan_miss": _plan_miss,
            "plan_remaining": (len(_plan["actions"]) - _plan["i"]) if _plan else 0}
