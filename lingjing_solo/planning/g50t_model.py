# -*- coding: utf-8 -*-
"""g50t 纯 python 关卡模型 —— 产品通道版 (2026-10-05, 模型 v4)。

来历: 探针 `_g50t_l2_model.py`(第 2 关) -> `_g50t_l37_model.py`(七关泛化, 模型 v1)
-> `_g50t_latch.py`(自锁活塞沿模拟, 模型 v2) -> `_g50t_v3.py`(**端点 + 传送门**, 模型 v3)
-> `_g50t_v4_*.py`(**移动者 `kgvnkyaimw` 状态 + A5 逐格回退**, 模型 v4)。
v2 实测战果: L1 17 手 / L2 31 手 / L3 64 手, 三关与引擎**逐手 0 差异**并同手过关。
v3 战果: L4 自搜 31 手, 逐手对拍 0 差异且第 31 手过关 —— 其中第 16 手与第 28 手各触发一次
传送门, 第 28 手把活体从端点 A 格 (0,5) 传送到端点 B 格 (0,7), 模型与引擎的成员表/位置**逐手一致**
(证据 `_g50t_v3_l4_evidence.txt`)。v3 同时保持 L1~L3 的回归口径(七关 20 手固定探针全部 0 分歧)。
v3 也解出 L5: 52 手, 331497 节点 / 15.1s, 逐手对拍 0 差异且第 52 手与引擎同手过关
(`_g50t_product.txt`)。**注意闸门口径**: 这条解只在产品默认节点上限 400000 下出现, 节点数压到
300000 时同一关报"A5 次数用尽"—— 报"某关无解"之前必须先说清用的是哪一档。

L5 之前一直被判成"无解", 真凶不在模型而在 `verify()` 的次序(10-05 修): 过关那一手引擎已经换关,
`g.vgwycxsxjz` 是下一关的控制器, 它的 `hamayflsib` 比本关短(L5 六个 gate / L6 五个), 所以
"先 `engine_view` 再判过关"会在第 52 手抛 IndexError, 又被 `solve()` 的 `except Exception` 咽成
"模型通道异常: IndexError" —— 一条看起来像诚实拒绝、实际是崩溃的 note。教训: **过关判定只读引擎
自己的计数器, 判完立刻 break**, 任何按本关 gate 下标取引擎对象的调用都必须在换关判定之后。

v2 -> v3 改了什么(全部对着 environment_files/g50t/5849a774/g50t.py 的源码顺序抄):
  A) **压制源集合进状态**: v2 每手开头用"当前位置"重建 mem, 隐含假设"没有残留成员"。引擎里
     `set_position` 类位移**不走** `ayhgaxoxce(obj, False)`(:2359 传送、:2760 阶段推进回 home),
     所以被传走/被搬走的 sprite 会**永远留在原 gate 的压制源集合里**, 直到同一个对象再压再放。
     v3 把每个 gate(导线+端点, 即 `hamayflsib` 下标序)的成员表成 token 位掩码放进状态, 并放进
     `key()` 去重键; token 编号: 0=活体, j+1=第 j 个幽灵(创建序, 与 `rloltuowth` 字典序一致)。
  B) **信号按 flag 贯穿**: `ayhgaxoxce` 把 `self.pohkooyzds` 当 flag 一路传到叶子(:2664-2668)。
     传送门 :2315-2326 的触发条件正是 flag/值的组合:
       flag=True  且 value=False(退带子中离开导线) => **当场** `oveynewjrw()`(:2320)
       flag=False 且 value=True (正常走步踩上导线)   => **排队**(:2321-2326), 4 倒计时 + 14 动画帧
                                                   后在同一个动作内的排空帧里 `oveynewjrw()`
     ⇒ v3 用 `pending` 表排队, 在"下一个静止界"(即下一次 press 之前)结算。
  C) **swap = 成员互换 + 传送**(:2344-2349 -> :2356-2359): A 的成员集合被 B 的旧集合**整表替换**并
     逐个 `set_position(A.x, A.y)`, 然后 B 同理得到 A 的旧集合。两个集合都有的 token 最终落在 **B 格**
     (第二次 `wbneuwcaoz` 覆盖了它的位置) —— 按这个先后顺序抄。
  D) **`rdgrjozeoh` 的次序**: 合法性用**位移前**的位置算(:2671), release 也在旧格, 但 `jagrwdyrhb`
     读的是**调用时**的 x/y(:2659) —— 所以 release 里那个"当场 swap"把对象传走后, 实际落点 =
     端点格 + 位移量。顺序照抄。
  E) **阶段推进时成员跟着对象改名**: :2760 活体 `set_position(home)` 不 release; :2769-2771 旧活体变
     幽灵、新活体是 clone(全新对象, 无成员资格) ⇒ bit0 的成员资格改名到"新幽灵"token; :2761-2765
     末阶段把幽灵整批 `yfcuzfdcnj()` **摘除画面但不 release** ⇒ 那些导线从此**常亮** ⇒ v3 用
     `dead_mask`(按 gate 位)表示"有已消亡成员压着", 不可再清除。
  F) `hold_cands` 的候选停点从"只有带活塞的导线"扩到"**传送门端点**"(把幽灵留在端点上才换得动传送)。

v3 -> v4 改了什么(移动者 `utmhrglxzq` / 容器 `kgvnkyaimw`, 同样对着源码抄):
  A) **走法只在玩家动成功之后**: `move()` :2678 玩家位移失败就整函数 return, 移动者一段(:2688-2696)
     根本走不到; 排在幽灵(:2682-2687)之后。每只最多试 4 次 `oymsmftmdy` :2451-2467 —— 候选顺序
     **(直行, 左, 右, 掉头)**, 判据是"候选**左上角**像素命中赛道 sprite"(`mvbhoupyur` :2469-2472,
     注意跟地板/活塞判据用的**中心**像素不是一个点, :1900-1906), 第一个命中的候选**立刻把朝向改成它**
     并返回位移 —— 哪怕随后 `rdgrjozeoh` 走不通也是(朝向已被改), 然后 `tbhfngaahn` :2448 再翻 180°。
     四次全败 => 既不走也**不往历史里追加**; 而 `oymsmftmdy` 找不到候选时返回 (0,0), 原地"走一格"
     是合法的 => 追加 (0,0) 并且**照样先 release 再 press**, 于是压在同一根导线上也会造一次下降沿+上升沿。
  B) **合法性 = 中心打在地板 mask 且不被"可见"活塞压住**(`rhvduhvfwn` :2644-2649 -> `vjpujwqrto`
     :2651-2655), 三处判据都用**动作前**的活塞位(同一帧内跑完)。它**不看玩家也不看幽灵**
     ⇒ 移动者会从玩家身上踩过去。
  C) **移动者是 gate 成员**: `step()` :2745-2749 的按压顺序是 玩家 -> **每只移动者** -> 幽灵, 所以
     token 编号 v4 改成 0=活体, 1..nm=移动者(创建序 == `kgvnkyaimw` 字典序 :2548-2554),
     nm+1..=幽灵。v3 的 `engine_view` 把"映射不到的成员"记进 `dead`, 正是这条把移动者咽掉了 ——
     也正因为如此, v3 在 L6/L7 报的"逐手 0 分歧"是**空的**(侦察实测 `移动者进 gate 成员表 0 人次`)。
  D) **A5 逐格跟着退**: `tqzseunaxh` :2701-2708 —— 玩家带子每弹一条, 就把"历史比玩家新长度长"的那只
     移动者弹**它自己的最后一条**(:2705 pop, 不是按玩家下标取)并 `rdgrjozeoh(-adx,-ady)`; 退格的合法性
     按 :2671 判, **但返回值被忽略**(:2706) => 退不动也照样把那条消费掉。弹之前先 `ehqdygkzrd()` :2704
     复活, **朝向全程不退**(源码里没有恢复 `wzgvpxcawd` 的那一支)。因为一次 A5 会在同一个动作里把带子
     退空(`_g50t_v4_a5.txt` 实测: 第 6 关第 10 手 ph 4->0, 第 7 关第 16 手 ph 5->0), 所以移动者一路退回
     **关卡起点格**; 它的历史长度与玩家带子同步(:2680 与 :2694 成对追加), 只有"死掉"或"四次全败"的手
     不追加 => 之后靠 :2702 的长度比较自然错位, v4 照这个长度规则建模, 不假设两者永远等长。
  E) **移动者会被夹死、也会夹死玩家**: 活塞回到 home 且静止时压住移动者中心 => `faggmbnvrz` :2248-2251
     (死 = `amjlzzsesf` :2065, `pddqxjztas` :1991); 死了的移动者**不再走也不再追加历史**, 但 :2746 的
     按压循环不看死活 => 它**继续把身上那根导线按住**, 直到某次 A5 弹到它那一条才 release(:2706 的
     `ayhgaxoxce(obj, False)` 在旧格上放)。反方向: 移动者动画收尾时与玩家碰撞且玩家像素压住它的中心
     => `dzxunlkwxt.faggmbnvrz()` :2485-2491 = 玩家死。
  F) **两处动作边界近似**(实测口径, 不假装是逐帧): 夹人/撞人的判据都按"静止界上的格重合"算;
     移动者与玩家同格 => 按 :2488 的"到达帧对方还差 1px"判成玩家死。去重键 `key()` 里放移动者的
     (格, 朝向, 死活) 与历史长度, **不放历史内容** —— 与 v3 对玩家带子 st[3] 的处理是同一个折中
     (同键的不同路径共用先到那条的带子/历史), A5 的结果因此代表其中一条路径。

模型建了什么 (直读源码):
  L0 几何: 玩家每手恰好走一格 `jarvstobjt`=6px(:2816-2823 -> move -> :2661), 于是世界可以按玩家起点
      铺成整数格; 地板/机关挡路格用像素级命中实测(:1962, 负值=透明)。端点/传送门两端实测都在格点上
      (七关侦察 `_g50t_l4_recon.txt`: 偏移全 0), 所以传送后的落点仍在格点坐标系内。
  L1 机关: 可按压对象 = `hamayflsib` = 导线+端点(:2623-2625); `ayhgaxoxce` 命中即 **break**(:2665-2668)
      ⇒ 一个对象一手最多压一条 gate(取列表序最靠前的, 导线优先于端点); 导线只在**有对象压着期间**为真
      (:2123-2139), 下游活塞位是"沿历史"的函数, 逐子步模拟; 自锁活塞(:2198-2206) off 沿忽略、
      on 沿翻转; 端点(:2361-2367)只进/出成员集合、**不传播信号**。
  L2 录像: A5 = `pmlawcgvcp` :2720-2726 —— 带子空 = 空操作; 否则当场退一步(`tqzseunaxh` :2698-2718,
      先弹带子再撤销, 撤销被挡也照弹), 之后每个静止帧"先 press 再退一步"; 带子退空那帧 press 完才
      推进阶段(:2758-2773); 阶段标记那一帧跳过 press(:2736-2744)。
  L3 计时: `ucorwtereb` 每手 +1(:2827), 预算 = 计时条耗尽手數(:2844-2849, 见 g50t_solver._budget)。

没建的(搜不到解时如实写进 note, 不当解收):
  - 移动者与玩家**同格**就判玩家死(:2485-2491 的到达帧只差 1px 没逐帧建), 以及"移动者四次候选全败
    就不追加历史"这条只在探针里按长度规则实现, 未拿到真实触发样本 —— 两条都归在上面的近似口径 F) 里;
  - 一台传送门两端不齐(:2316 直接 return, 按死物处理)、**同一动作内多台门排队**的先后顺序(:2730 逆序
    排空)未按多门关卡验证 —— L7 有两门;
  - 一个分电器被多条导线共同驱动时它的 `dijhfchobv == value` 去重(:2157)会吞掉后到的沿 —— v3 把沿
    直接抄给叶子, 遇到这种情况写进 gaps;
  - 非动作边界的像素级时序: 活塞回到 home 那一帧的夹人(:2237-2251)、动画中途被打断的漂移 ——
    判据都是"动作边界=静止态"。
  搜索分解仍是"停在某条导线/端点上再 A5"的阶段递归, 所以"借一次沿翻转但不在线上录像"那类型解表达不了。

零线上副作用: 只用调用方给的引擎对象, 不新建 Arcade、不发 HTTP; 收尾一律 `_restore_start` 回本关起点
(与 `g50t_solver.solve` 同一契约, 见 arc_shadow_g50t.solve_level 的 docstring)。
"""
import os
import time
from collections import Counter, deque

RANGE = 12
DIR = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
# 移动者自己的朝向 idx -> 位移 (ftmnrobwba :2474-2483); 注意这与玩家的**动作号**不是同一套编号
MOVER_DIR = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# 无移动者的关卡(L1-L5)每手都要拼 13 元组状态, 这四个空字段复用同一常量而不是每次重建:
# 同机交替实测 v4 的**每节点成本**是 v3 的 1.25~1.53x(第 5 关同为 331497 节点, 14.81s -> 22.70s),
# 正好把第 5 关顶出产品通道那道 20s 闸 —— 省掉这些空构造是行为不变的纯收益(见 _g50t_v4_cost.py)。
NO_MOVERS = ((), (), (), ())
MAX_DEPTH = 400000          # 单次阶段 BFS 弹出上限
DEFAULT_T_LIMIT = float(os.environ.get("G50T_MODEL_T", "20"))
DEFAULT_NODE_CAP = int(os.environ.get("G50T_MODEL_N", "400000"))

WIRE_CLS = "lqtxaumfed"        # 导线 :2111
TERMINAL_CLS = "ulhhdeoyok"    # 端点 :2352
PORTAL_CLS = "crfcpstubm"      # 传送门 :2263
SPLITTER_CLS = "alyzsfkumg"    # 分电器 :2142
PISTON_CLS = "yyzqramdhd"      # 活塞 :2179


def sprite_hit(sp, bx, by, px, py):
    """复刻 esidlbhbhw :1962 —— sprite 摆在 (bx,by) 时点 (px,py) 是否命中非透明像素(负值=透明)。"""
    dx, dy = px - bx, py - by
    if dx < 0 or dx >= sp.width or dy < 0 or dy >= sp.height:
        return False
    arr = sp.xmdjbwrpmv.pixels
    v = int(arr[dx, dy]) if int(sp.rotation or 0) in (90, 270) else int(arr[dy, dx])
    return v != -1


def follow(obj, max_depth=4):
    """沿 nexhtmlmxh(单输出)/ytztewxdin(多输出) 追信号下游: 返回 (下游对象, 途经类名)。"""
    objs, classes, seen = [], set(), {id(obj)}
    frontier = [obj]
    for _ in range(max_depth):
        nxt = []
        for o in frontier:
            for attr in ("nexhtmlmxh", "ytztewxdin"):
                t = getattr(o, attr, None)
                if t is None:
                    continue
                for c in (t if isinstance(t, (list, tuple, set)) else [t]):
                    if c is None or id(c) in seen:
                        continue
                    seen.add(id(c))
                    objs.append(c)
                    classes.add(type(c).__name__)
                    nxt.append(c)
        frontier = nxt
        if not frontier:
            break
    return objs, classes


class Facts(object):
    """L0 几何 + L1 机关拓扑(含端点/传送门): 一次性把关卡抽成整数集合, 之后模型不再调引擎。"""

    def __init__(self, g):
        import g50t_solver as gs
        C = g.vgwycxsxjz
        self.pitch = gs._pitch(g)
        self.px0, self.py0 = int(C.dzxunlkwxt.x), int(C.dzxunlkwxt.y)
        self.home = (0, 0)
        self.cx = C.dzxunlkwxt.width // 2
        self.cy = C.dzxunlkwxt.height // 2
        self.center = lambda a, b: (self.px0 + a * self.pitch + self.cx,
                                    self.py0 + b * self.pitch + self.cy)
        self.cell = lambda x, y: ((int(x) - self.px0) // self.pitch, (int(y) - self.py0) // self.pitch)
        self.cells = [(a, b) for b in range(-RANGE, RANGE + 1) for a in range(-RANGE, RANGE + 1)]
        fsp = C.afbbgvkpip
        fx, fy = int(fsp.x), int(fsp.y)
        self.floor = frozenset(c for c in self.cells if sprite_hit(fsp, fx, fy, *self.center(*c)))
        self.budget = gs._budget(g)
        self.stages = len(C.drofvwhbxb)
        self.movers = len(C.kgvnkyaimw)
        goal = C.whftgckbcu
        self.raw_goal_px = (int(goal.x) + 1, int(goal.y) + 1)
        self.goal_off = ((self.raw_goal_px[0] - self.px0) % self.pitch
                         or (self.raw_goal_px[1] - self.py0) % self.pitch)
        self.goal = self.cell(*self.raw_goal_px)
        self.haz_classes = Counter(type(h).__name__ for h in C.uwxkstolmf)
        self.gate_classes = Counter(type(w).__name__ for w in C.hamayflsib)
        self.pistons = []
        for haz in C.uwxkstolmf:
            hx, hy = int(haz.x), int(haz.y)
            movable = hasattr(haz, "hluvhlvimq")
            ddx, ddy = haz.hluvhlvimq() if movable else (0, 0)
            on_px = (hx + ddx * self.pitch, hy + ddy * self.pitch)
            self.pistons.append({
                "cls": type(haz).__name__, "moved": movable,
                "latch": bool(getattr(haz, "dpdubazedr", False)),
                "home": self.cell(hx, hy), "on": self.cell(*on_px), "dir": (ddx, ddy),
                "home_px": (hx, hy), "on_px": on_px, "vis": bool(haz.is_visible),
                "blk_home": self._blk(haz, hx, hy), "blk_on": self._blk(haz, *on_px),
            })
        self.wires = []
        self.start_on = []
        # `hamayflsib` 里只有 导线+端点(:2623-2625) —— 分电器/传送门/活塞都**不在**那张表里,
        # 只能沿信号下游按对象 id 数, 否则"本关有没有传送门"会永远报 0 (死代码)。
        self.down_class_ids = {}
        for w in C.hamayflsib:
            wx, wy = int(w.x), int(w.y)
            press = frozenset(c for c in self.cells if sprite_hit(w, wx, wy, *self.center(*c)))
            downs, classes = follow(w)
            for o in downs:
                self.down_class_ids.setdefault(type(o).__name__, set()).add(id(o))
            pis, only_on = [], []
            for o in downs:
                op = (int(o.x), int(o.y))
                for i, ps in enumerate(self.pistons):
                    if op == ps["home_px"]:
                        if i not in pis:
                            pis.append(i)
                    elif op == ps["on_px"]:
                        if i not in only_on:
                            only_on.append(i)
            self.wires.append({"cls": type(w).__name__, "press": press, "home": self.cell(wx, wy),
                               "pistons": tuple(pis), "via": tuple(sorted(classes)),
                               "on_init": bool(getattr(w, "dijhfchobv", False))})
            if only_on and not pis:
                self.start_on.append((len(self.wires) - 1, only_on))
        linked = {i for w in self.wires for i in w["pistons"]}
        self.unlinked_pistons = [i for i in range(len(self.pistons)) if i not in linked]
        self.latch_ids = [i for i, ps in enumerate(self.pistons) if ps["moved"] and ps["latch"]]
        self.multi_feed = [i for i in range(len(self.pistons))
                           if sum(1 for w in self.wires if i in w["pistons"]) > 1]
        # ---- v3: gate 表(= hamayflsib 序) + 传送门拓扑 ----
        idx_by_id = {id(o): i for i, o in enumerate(C.hamayflsib)}
        pist_idx = {id(o): i for i, o in enumerate(C.uwxkstolmf)}
        self.portals = []
        self.portal_by_id = {}
        self.unknown_leaves = []
        for w in C.hamayflsib:
            for o in self._leaf_objs(w):
                if type(o).__name__ == PORTAL_CLS and id(o) not in self.portal_by_id:
                    self.portal_by_id[id(o)] = len(self.portals)
                    a, b = o.lzcbbrzwdn, o.esbtgcjrfp
                    self.portals.append({
                        "a": None if a is None else idx_by_id.get(id(a)),
                        "b": None if b is None else idx_by_id.get(id(b)),
                        "a_cell": None if a is None else self.cell(int(a.x), int(a.y)),
                        "b_cell": None if b is None else self.cell(int(b.x), int(b.y)),
                        "anim": len(o.bqjuguzwil),
                    })
        splitter_up = {}
        self.gates = []
        for gi, w in enumerate(C.hamayflsib):
            cls = type(w).__name__
            kind = "wire" if cls == WIRE_CLS else ("terminal" if cls == TERMINAL_CLS else "other")
            leaves = []
            nxt = getattr(w, "nexhtmlmxh", None)
            if nxt is not None:
                splitter_up[id(nxt)] = splitter_up.get(id(nxt), 0) + 1
                for o in self._leaf_objs(w):
                    cn = type(o).__name__
                    if cn == PISTON_CLS:
                        leaves.append(("piston", pist_idx[id(o)]))
                    elif cn == PORTAL_CLS:
                        leaves.append(("portal", self.portal_by_id[id(o)]))
                    elif cn != SPLITTER_CLS:
                        self.unknown_leaves.append(cn)
            self.gates.append({"i": gi, "cls": cls, "kind": kind, "press": self.wires[gi]["press"],
                               "leaves": tuple(leaves)})
        self.multi_splitter = sum(1 for v in splitter_up.values() if v > 1)
        self.is_portal_end = set()
        for po in self.portals:
            for gi in (po["a"], po["b"]):
                if gi is not None:
                    self.is_portal_end.add(gi)
        # ---- v4: 移动者名册(kgvnkyaimw 创建序 :2548-2554)。赛道打在**左上角**像素(:2472),
        #      地板/活塞/gate 打在**中心**像素(:1900-1906 -> :2657/:2666), 两套判据不是一个点。
        self.nm = self.movers
        self.mv = []
        for mv in C.kgvnkyaimw:
            mk = mv.baygqyisjz                        # :2445 cssyikfflo 绑定的赛道 sprite
            mx, my = int(mv.x), int(mv.y)
            ccx, ccy = int(mv.width) // 2, int(mv.height) // 2
            ox, oy = (mx - self.px0) % self.pitch, (my - self.py0) % self.pitch
            cx0, cy0 = self.px0 + ox, self.py0 + oy
            tl = lambda a, b: (cx0 + a * self.pitch, cy0 + b * self.pitch)
            ct = lambda a, b: (cx0 + a * self.pitch + ccx, cy0 + b * self.pitch + ccy)
            lane = frozenset()
            if mk is not None:
                mkx, mky = int(mk.x), int(mk.y)
                lane = frozenset(c for c in self.cells if sprite_hit(mk, mkx, mky, *tl(*c)))
            press = {}
            for gi, gw in enumerate(C.hamayflsib):    # :2665-2668 命中即 break => 取列表序最靠前
                gx, gy = int(gw.x), int(gw.y)
                for c in self.cells:
                    if c not in press and sprite_hit(gw, gx, gy, *ct(*c)):
                        press[c] = gi
            same = (ox == 0 and oy == 0 and ccx == self.cx and ccy == self.cy)
            pblk = None
            if not same:
                pblk = []
                for haz in C.uwxkstolmf:
                    hx, hy = int(haz.x), int(haz.y)
                    dd = haz.hluvhlvimq() if hasattr(haz, "hluvhlvimq") else (0, 0)
                    pblk.append((
                        frozenset(c for c in self.cells if sprite_hit(haz, hx, hy, *ct(*c))),
                        frozenset(c for c in self.cells if sprite_hit(
                            haz, hx + dd[0] * self.pitch, hy + dd[1] * self.pitch, *ct(*c)))))
            self.mv.append({
                "cls": type(mv).__name__, "start": self.cell(mx, my), "d0": int(mv.wzgvpxcawd),
                "off": (ox, oy), "ctr": (ccx, ccy), "same": same, "lane": lane,
                "has_marker": mk is not None, "press": press,
                "allowed": self.floor if same else frozenset(
                    c for c in self.cells if sprite_hit(fsp, fx, fy, *ct(*c))),
                "pblk": pblk,
            })
        self.gaps = []
        broken = [i for i, po in enumerate(self.portals) if po["a"] is None or po["b"] is None]
        if broken:
            self.gaps.append("传送门端点不齐 x%s (:2316 直接 return, 按死物处理)" % broken)
        if len(self.portals) >= 2:
            self.gaps.append("多传送门同动作排队先后(:2730 逆序排空)未按多门关验证 x%d" % len(self.portals))
        if self.unknown_leaves:
            self.gaps.append("未识别叶子类型 %s" % sorted(set(self.unknown_leaves)))
        if any(g["kind"] == "other" for g in self.gates):
            self.gaps.append("hamayflsib 里有第三种 gate")
        if self.multi_splitter:
            self.gaps.append("分电器多输入 x%d (v3 把沿直接抄给叶子, 未模拟分电器去重)" % self.multi_splitter)
        if self.goal_off:
            self.gaps.append("胜利落点不在玩家格点上")
        if self.multi_feed:
            self.gaps.append("一活塞多线驱动 x%s (最后一条沿赢)" % self.multi_feed)

    def _blk(self, haz, bx, by):
        """摆 hazards 在 (bx,by) 时, 哪些格子的玩家中心像素会被它压住(= 挡路/夹人格)。"""
        return frozenset(c for c in self.cells if sprite_hit(haz, bx, by, *self.center(*c)))

    @staticmethod
    def _leaf_objs(w):
        """导线 -> 分电器 -> 叶子; 没有分电器时导线自己就是叶子的上游(:2120-2121, :2153-2154)。"""
        nxt = getattr(w, "nexhtmlmxh", None)
        if nxt is None:
            return []
        outs = getattr(nxt, "ytztewxdin", None)
        return list(outs) if outs else [nxt]

    def summary(self):
        return "地板%d格 目标%s 预算%d 阶段%d 活塞%d(自锁%s) 导线%d 端点%d 传送门%d 移动者%d" % (
            len(self.floor), self.goal, self.budget, self.stages, len(self.pistons),
            self.latch_ids or "-", sum(1 for w in self.wires if w["cls"] == WIRE_CLS),
            self.gate_classes.get(TERMINAL_CLS, 0), len(self.portals), self.movers)


class Sim(object):
    """一手(或一个退带子步)之内的信号沿模拟器: 位置 + 每个 gate 的成员位掩码 + 自锁位 + 排队传送。

    token 编号(v4): 0=活体, 1..nm=移动者(创建序), nm+1..=幽灵 —— 与 `step()` :2745-2749 的
    "玩家 -> 每只移动者 -> 每个幽灵"按压顺序一致。
    """

    def __init__(self, m, live, gcells, lbits, mem, dead, mcells=(), mdirs=(), mdead=(), mtapes=()):
        self.m = m
        self.nm = m.f.nm
        self.pos = {0: live}
        for i, c in enumerate(mcells):
            self.pos[i + 1] = c
        for j, c in enumerate(gcells):
            self.pos[self.nm + 1 + j] = c
        self.mem = list(mem)
        self.dead = dead
        self.lb = list(lbits)
        self.pending = []
        if self.nm:                      # 下面四样只被 mv_step/mv_rewind/mblk 用到, 而那三处
            self.md = list(mdirs)        # 都在 `range(nm)` 里跑 => 无移动者的关卡留空常量即可
            self.mdead = list(mdead)
            self.mhist = [tuple(t) for t in mtapes]
            self._mblk = {}
        else:
            self.md = self.mdead = self.mhist = ()
            self._mblk = None

    def toks(self):
        return range(len(self.pos))

    def gate_of(self, tok, cell):
        """活体/幽灵按玩家中心查 gate, 移动者按**它自己的中心**查(:2666 -> :1900-1906)。"""
        if 1 <= tok <= self.nm:
            return self.m.f.mv[tok - 1]["press"].get(cell)
        return self.m.gate_of_cell.get(cell)

    def mblk(self, mi, bits):
        """移动者摆在各格时会被哪些"可见"活塞压住(:2651-2655) —— 只在它不占玩家格点时才单独算。"""
        if mi not in self._mblk:
            mv = self.m.f.mv[mi]
            b = set()
            for i, (ph, po) in enumerate(mv["pblk"]):
                if not self.m.f.pistons[i]["vis"]:
                    continue
                b |= (po if bits[i] else ph)
            self._mblk[mi] = frozenset(b)
        return self._mblk[mi]

    def mv_legal(self, mi, c, blk, bits):
        """rhvduhvfwn :2644-2649: 中心打在地板 mask 上, 且不被可见活塞压住 —— **不看玩家也不看幽灵**。"""
        mv = self.m.f.mv[mi]
        if mv["same"]:
            return c in mv["allowed"] and c not in blk
        return c in mv["allowed"] and c not in self.mblk(mi, bits)

    def mv_choose(self, mi):
        """oymsmftmdy :2451-2467: 候选 (直行, 左, 右, 掉头); 第一个"左上角像素命中赛道"的就**改朝向**并返回。
        全不命中 => 返回 (0,0) 且**朝向不变**(:2467)。"""
        mv = self.m.f.mv[mi]
        d = self.md[mi]
        for cand in (d, (d - 1) % 4, (d + 1) % 4, (d + 2) % 4):
            dx, dy = MOVER_DIR[cand]
            if (self.pos[mi + 1][0] + dx, self.pos[mi + 1][1] + dy) in mv["lane"]:
                self.md[mi] = cand
                return (dx, dy)
        return (0, 0)

    def mv_step(self, mi, blk, bits):
        """move() :2688-2696: 最多 4 次, 走不通就 tbhfngaahn 掉头重试; 死的不走(:2689)且**不追加历史**。"""
        if self.mdead[mi]:
            return
        tok = mi + 1
        for _ in range(4):
            dx, dy = self.mv_choose(mi)
            c = (self.pos[tok][0] + dx, self.pos[tok][1] + dy)
            if self.mv_legal(mi, c, blk, bits):
                self.release(tok, False)                   # :2673 在**旧格**上放
                self.pos[tok] = c
                self.mhist[mi] = self.mhist[mi] + ((dx, dy),)
                return
            self.md[mi] = (self.md[mi] + 2) % 4            # :2696
        # 四次全败: 引擎既不走也不 append(:2694 在 break 分支里)

    def mv_rewind(self, mi, idx, blk, bits, flag):
        """tqzseunaxh :2701-2708: idx=玩家弹完后的新带子长度; 历史不比它长就**整段跳过**(连复活都不做)。"""
        hist = self.mhist[mi]
        if idx >= len(hist):
            return
        tok = mi + 1
        self.mdead[mi] = False                             # :2704 ehqdygkzrd 复活, 朝向不退
        adx, ady = hist[-1]                                # :2705 pop 它**自己的最后一条**
        self.mhist[mi] = hist[:-1]
        c = (self.pos[tok][0] - adx, self.pos[tok][1] - ady)
        if self.mv_legal(mi, c, blk, bits):                # :2706 返回值被忽略: 退不动也照样消费掉
            self.release(tok, flag)
            self.pos[tok] = c

    def on(self, gi):
        return bool(self.mem[gi]) or bool((self.dead >> gi) & 1)

    def bits(self):
        out = []
        for i, ps in enumerate(self.m.f.pistons):
            if not ps["moved"]:
                out.append(False)
            elif ps["latch"]:
                out.append(self.lb[self.m.lidx[i]])
            else:
                out.append(any(self.on(wi) for wi in self.m.feed[i]))
        return out

    def blk(self):
        b = set()
        bits = self.bits()
        for i, ps in enumerate(self.m.f.pistons):
            if not ps["vis"]:
                continue
            b |= (ps["blk_on"] if bits[i] else ps["blk_home"])
        return frozenset(b)

    def legal(self, blk, c):
        return c in self.m.f.floor and c not in blk

    def prop(self, gi, value, flag):
        """导线 on/off 变化 -> 沿分电器把 (value, flag) 抄给每个叶子(:2138-2139 -> :2168-2169)。"""
        for kind, k in self.m.gates[gi]["leaves"]:
            if kind == "piston":
                if value and self.m.f.pistons[k]["latch"]:
                    kk = self.m.lidx[k]
                    self.lb[kk] = not self.lb[kk]        # :2199-2206 只在上升沿翻转
            else:
                po = self.m.f.portals[k]
                if po["a"] is None or po["b"] is None:
                    continue                             # :2316 端点不齐 => 死物
                if flag:
                    if not value:
                        self.swap(k)                     # :2318-2320 当场换
                elif value and k not in self.pending:
                    self.pending.append(k)               # :2321-2326 排队, 排空帧落地

    def swap(self, k):
        """oveynewjrw :2344-2349 + wbneuwcaoz :2356-2359: 两端成员整表互换并传送到各自像素位。"""
        po = self.m.f.portals[k]
        a, b = po["a"], po["b"]
        a_mem, b_mem = self.mem[a], self.mem[b]
        self.mem[a] = b_mem
        self.mem[b] = a_mem
        for tok in range(len(self.pos)):
            if (b_mem >> tok) & 1:
                self.pos[tok] = po["a_cell"]
        for tok in range(len(self.pos)):
            if (a_mem >> tok) & 1:
                self.pos[tok] = po["b_cell"]
        self.m.swap_count += 1

    def flush_pending(self):
        while self.pending:
            self.swap(self.pending.pop(0))

    def press(self, tok, flag):
        gi = self.gate_of(tok, self.pos[tok])
        if gi is None or (self.mem[gi] >> tok) & 1:
            return
        was = self.on(gi)
        self.mem[gi] |= (1 << tok)
        if self.m.gates[gi]["kind"] == "wire" and not was:
            self.prop(gi, True, flag)                    # :2129-2132 空->非空才是上升沿

    def release(self, tok, flag):
        gi = self.gate_of(tok, self.pos[tok])
        if gi is None or not (self.mem[gi] >> tok) & 1:
            return
        self.mem[gi] &= ~(1 << tok)
        if self.m.gates[gi]["kind"] != "wire" or self.on(gi):
            return                                       # 还有成员(或已常亮) => 没有下降沿
        self.prop(gi, False, flag)                       # :2133-2136 非空->空

    def mask(self):
        return tuple(self.mem)


class Model(object):
    """状态 = (活体格, 幽灵格…, 幽灵脚本…, 本阶段带子, 阶段, 已花手数, 自锁位, 成员表, 常亮位,
    移动者格…, 移动者朝向…, 移动者死活…, 移动者各自的历史…)。"""

    def __init__(self, f):
        self.f = f
        self.gates = f.gates
        self.swap_count = 0
        self.latch = list(f.latch_ids)
        self.lidx = {i: k for k, i in enumerate(self.latch)}
        gc = {}
        for gi, g in enumerate(f.gates):            # hamayflsib 顺序 + break 语义 :2665-2668
            for c in g["press"]:
                gc.setdefault(c, gi)
        self.gate_of_cell = gc
        self.feed = [[] for _ in f.pistons]
        for gi, g in enumerate(f.gates):
            if g["kind"] != "wire":
                continue
            for kind, k in g["leaves"]:
                if kind == "piston":
                    self.feed[k].append(gi)

    def init(self):
        # 关卡起点: 引擎每个 gate 的压制源集合都是新空集(:2105), 活体在 home 上也没被 press 过
        return (self.f.home, (), (), (), 0, 0, tuple([False] * len(self.latch)),
                tuple([0] * len(self.f.gates)), 0,
                tuple(mv["start"] for mv in self.f.mv), tuple(mv["d0"] for mv in self.f.mv),
                tuple([False] * self.f.nm), tuple([()] * self.f.nm))

    def sim(self, st):
        return Sim(self, st[0], st[1], st[6], st[7], st[8], st[9], st[10], st[11], st[12])

    def bits(self, st):
        return self.sim(st).bits()

    def blocked(self, st):
        return self.sim(st).blk()

    def legal(self, blk, c):
        return c in self.f.floor and c not in blk

    def crushed(self, st):
        """活塞回到 home 并静止时压住活体中心 = 夹死 (:2237 -> :2520); 静态危险不会夹人。"""
        bits = self.bits(st)
        for i, on in enumerate(bits):
            ps = self.f.pistons[i]
            if on or not ps["vis"] or not ps["moved"]:
                continue
            if st[0] in ps["blk_home"]:
                return True
        return False

    def mover_deaths(self, st):
        """:2248-2251 活塞回到 home 静止时压住移动者中心 => 夹死(:2065)。死了就不再走、不再追加历史,
        但 :2746 的按压循环不看死活 => 它继续把身上那根导线按住, 直到某次 A5 弹到它那一条才 release。"""
        if not self.f.nm:
            return st
        bits = self.bits(st)
        md = list(st[11])
        for i, ps in enumerate(self.f.pistons):
            if not ps["moved"] or not ps["vis"] or bits[i]:
                continue
            for mi in range(self.f.nm):
                if not md[mi] and st[9][mi] in ps["blk_home"]:
                    md[mi] = True
        return st[:11] + (tuple(md),) + st[12:]

    def mover_on_live(self, st):
        """:2485-2491 移动者动画收尾时与玩家碰撞且玩家像素压住它的中心 = **玩家**死。动作边界近似见口径 F)。"""
        for mi in range(self.f.nm):
            if not st[11][mi] and st[9][mi] == st[0]:
                return True
        return False

    def keep(self, st, live, gcells, tape, stage, acts, sim):
        return (live, gcells, st[2], tape, stage, acts, tuple(sim.lb), sim.mask(), sim.dead,
                st[9], st[10], st[11], st[12])

    def step(self, st, a):
        live, gcells, gscripts, tape, stage, acts = st[0], st[1], st[2], st[3], st[4], st[5]
        acts += 1
        sim = self.sim(st)
        ng = len(gcells)
        nm = self.f.nm
        if a == 5:
            if not tape:
                return self.finish(self.keep(st, live, gcells, tape, stage, acts, sim))
            snapshot = tape
            cur = self._pop(sim, snapshot, gscripts, False)      # :2724 当场退第一步, flag 还是 False
            while cur:
                sim.flush_pending()                              # 排空帧里传送落地, 在这之后才 press
                for tok in sim.toks():                           # :2745-2749 活体 -> 每只移动者 -> 幽灵
                    sim.press(tok, True)
                cur = self._pop(sim, cur, gscripts, True)
            sim.flush_pending()
            for tok in sim.toks():                               # 带子退空那帧的 press: flag 已 False
                sim.press(tok, False)
            if sim.pending:                                      # press 上升沿排队 -> 落地后再 press 一次
                sim.flush_pending()
                for tok in sim.toks():
                    sim.press(tok, False)
            home = self.f.home
            old_cells = tuple(sim.pos[nm + 1 + j] for j in range(ng))
            mcells2 = tuple(sim.pos[i + 1] for i in range(nm))
            mdirs2, mdead2, mtapes2 = tuple(sim.md), tuple(sim.mdead), tuple(sim.mhist)
            new_mem = list(sim.mem)
            dead = sim.dead
            nstage = stage + 1
            keep_mask = (1 << (nm + 1)) - 1        # bit0(活体) + 移动者位; 幽灵位从 token nm+1 起
            if nstage == self.f.stages:
                # :2761-2765 幽灵整批摘除画面但**不 release** => 按住的导线从此常亮; 活体/移动者成员保留
                for gi in range(len(new_mem)):
                    if any((new_mem[gi] >> t) & 1 for t in range(nm + 1, nm + 1 + ng)):
                        dead |= (1 << gi)
                    new_mem[gi] &= keep_mask
                gcells2, scripts2, nstage = (), (), 0
            else:
                # :2769-2771 旧活体(已 set_position 到 home)变**最后一个**幽灵; 新活体是 home 上的
                #        clone = 全新对象 => 不带任何成员资格(bit0 改名到 token nm+ng+1)
                for gi in range(len(new_mem)):
                    if new_mem[gi] & 1:
                        new_mem[gi] = (new_mem[gi] & ~1) | (1 << (nm + ng + 1))
                gcells2 = tuple(old_cells) + (home,)
                scripts2 = tuple(list(gscripts) + [snapshot])
            sim3 = Sim(self, home, gcells2, tuple(sim.lb), new_mem, dead, mcells2, mdirs2,
                       mdead2, mtapes2)
            for tok in sim3.toks():   # :2736-2744 阶段标记那帧不 press, 下一帧才 press
                sim3.press(tok, False)
            return self.finish((home, gcells2, scripts2, (), nstage, acts,
                                tuple(sim3.lb), sim3.mask(), sim3.dead) + (
                    (tuple(sim3.pos[i + 1] for i in range(nm)), tuple(sim3.md),
                     tuple(sim3.mdead), tuple(sim3.mhist)) if nm else NO_MOVERS))
        d = DIR[a]
        blk = sim.blk()                                      # :2671 用动作前的活塞位判合法性
        if not sim.legal(blk, (live[0] + d[0], live[1] + d[1])):
            return self.finish(self.keep(st, live, gcells, tape, stage, acts, sim))
        bits0 = sim.bits() if nm else ()                     # 只为 mv_step 的"动作前活塞位"服务;
                                                             # 无移动者的关卡不必多跑一遍 bits()
        idx = len(tape)
        sim.release(0, False)                                # :2664 在**旧格**上放
        sim.pos[0] = (live[0] + d[0], live[1] + d[1])
        for j, sc in enumerate(gscripts):                    # :2682-2687 幽灵(引擎排在移动者之前)
            if idx >= len(sc):
                continue
            gdx, gdy = sc[idx]
            cand = (gcells[j][0] + gdx, gcells[j][1] + gdy)
            if sim.legal(blk, cand):
                sim.release(nm + 1 + j, False)
                sim.pos[nm + 1 + j] = cand
        for mi in range(nm):                                 # :2688-2696 移动者(只在玩家动成功之后)
            sim.mv_step(mi, blk, bits0)
        sim.flush_pending()                                  # 放线触发的排队传送在排空帧落地
        for tok in sim.toks():                               # :2745-2749 活体 -> 每只移动者 -> 幽灵
            sim.press(tok, False)
        if sim.pending:                                      # press 上升沿排队 -> 落地后再 press 一次
            sim.flush_pending()
            for tok in sim.toks():
                sim.press(tok, False)
        return self.finish((sim.pos[0], tuple(sim.pos[nm + 1 + j] for j in range(ng)), gscripts,
                            tape + (d,), stage, acts, tuple(sim.lb), sim.mask(), sim.dead) + (
            (tuple(sim.pos[i + 1] for i in range(nm)), tuple(sim.md),
             tuple(sim.mdead), tuple(sim.mhist)) if nm else NO_MOVERS))

    @staticmethod
    def _pop(sim, cur, gscripts, flag):
        """tqzseunaxh :2698-2718: 先弹带子, 再 移动者 -> 幽灵 -> 活体 各退一步; 合法性用退之前的活塞位。"""
        dx, dy = cur[-1]
        idx = len(cur) - 1                                   # = :2700 弹完玩家之后的新带子长度
        blk = sim.blk()
        nm = sim.nm
        bits = sim.bits() if nm else ()                      # 只喂 mv_rewind 的判据
        for mi in range(nm):                                 # :2701-2708 先移动者
            sim.mv_rewind(mi, idx, blk, bits, flag)
        for j, sc in enumerate(gscripts):                    # :2709-2715 幽灵
            if idx >= len(sc):
                continue
            gdx, gdy = sc[idx]
            tok = nm + 1 + j
            c = (sim.pos[tok][0] - gdx, sim.pos[tok][1] - gdy)
            if sim.legal(blk, c):
                sim.release(tok, flag)                       # flag=True 时这里可能**当场**传送
                sim.pos[tok] = c
        c = (sim.pos[0][0] - dx, sim.pos[0][1] - dy)
        if sim.legal(blk, c):
            sim.release(0, flag)
            sim.pos[0] = c
        return cur[:-1]

    def finish(self, st):
        nm = self.f.nm
        if nm:
            st = self.mover_deaths(st)
        won = (st[0] == self.f.goal)
        dead = (self.crushed(st) or (nm and self.mover_on_live(st))
                or st[5] >= self.f.budget)
        return st, won, dead

    def render(self, st):
        sim = self.sim(st)
        return dict(live=st[0], ghosts=tuple(sorted(st[1])), stage=st[4], acts=st[5],
                    pistons=tuple(sorted(self.f.pistons[i]["on"] if b else self.f.pistons[i]["home"]
                                         for i, b in enumerate(sim.bits()))),
                    tape=st[3], lbits=st[6], mem=st[7], dead=st[8],
                    movers=tuple(zip(st[9], st[10], st[11])),
                    mhlen=tuple(len(t) for t in st[12]))


def key(st):
    """去重键: 带子**内容**不进键(进了就变路径空间, v1 挂过 420s+), 只留长度;
    v3 多带两项 —— 成员表和常亮位, 因为传送/阶段推进留下的残留成员会改变未来;
    v4 再加移动者的(格, 朝向, 死活) 与各自历史**长度**(内容同上, 见口径 F) 的折中)。"""
    return (st[0], st[1], st[2], len(st[3]), st[4], st[6], st[7], st[8],
            st[9], st[10], st[11], tuple(len(t) for t in st[12]) if st[12] else ())


def stage_paths(m, st0, pred, max_len, max_cands, max_nodes, deadline):
    """阶段内 BFS: 幽灵(格,脚本)已定, 从 home 搜若干条到 pred / 直奔目标的最短活体路径。"""
    seen = {key(st0)}
    q = deque([(st0, ())])
    popped = 0
    cands, cand_tapes = [], set()
    while q:
        if popped >= max_nodes or time.time() > deadline:
            break
        st, path = q.popleft()
        popped += 1
        if len(path) >= max_len:
            continue
        for a in (1, 2, 3, 4):
            ns, won, dead = m.step(st, a)
            if dead:
                continue
            npath = path + (a,)
            if won or pred(ns):
                if ns[3] in cand_tapes:
                    continue
                cand_tapes.add(ns[3])
                cands.append((npath, ns, won))
                if len(cands) >= max_cands:
                    return cands, popped
                continue
            k = key(ns)
            if k in seen:
                continue
            seen.add(k)
            q.append((ns, npath))
    return cands, popped


def search(f, m, deadline, node_cap):
    """阶段递归: 每层先问"直奔目标行不行", 不行就把"停在某条导线/端点上再 A5"的候选当分叉。"""
    stats = {"nodes": 0, "calls": 0, "bfs": 0, "why": "", "cands": 0}
    seen_rec = set()

    def ghost_final(ghosts):
        cells = []
        for c0, sc in ghosts:
            cx, cy = c0
            for dx, dy in sc:
                cx += dx
                cy += dy
            cells.append((cx, cy))
        return cells

    def worth_holding(gi):
        """值得把幽灵留在上面的 gate: 驱动活塞/传送门的导线, 以及传送门的两个端点。"""
        g = f.gates[gi]
        if g["kind"] == "terminal":
            return gi in f.is_portal_end
        return any(kind in ("piston", "portal") for kind, _ in g["leaves"])

    def hold_cands(st):
        """**逐条 gate 分别枚举**候选: 合成一条"压任意线"的 BFS 会让最短的几条候选全是"幽灵已经
        按住的那条线"(v1 实测: 真候选 12 手被 max_cands 截掉 => 已知解丢失)。"""
        ghosts = list(zip(st[1], st[2]))
        held = set()
        for c in ghost_final(ghosts):
            gi = m.gate_of_cell.get(c)
            if gi is not None:
                held.add(gi)
        out = []
        for gi, g in enumerate(f.gates):
            if not worth_holding(gi) or gi in held:
                continue
            c, n = stage_paths(m, st, (lambda ns, pr=g["press"]: ns[0] in pr),
                               max_len=min(f.budget, 40), max_cands=4,
                               max_nodes=MAX_DEPTH, deadline=deadline)
            stats["nodes"] += n
            stats["bfs"] += 1
            for path, ns, won in c:
                out.append((len(path), gi, path, ns, won))
        # 排序键(现役=旧键): `out.sort()` = (手数, gate), 等于"最便宜的压线先钻"。
        # 10-07 应用户明确指令改回旧键。v5 那行留档, 不要删:
        #   hm = f.home
        #   out.sort(key=lambda e: (-abs(e[3][0][0] - hm[0]) - abs(e[3][0][1] - hm[1]), e[0], e[1]))
        # v5 键当时立项的依据(记录留档, 出处 _g50t_v4_l7order_L7.txt、_g50t_v4_orderall_dhome.txt、
        # _g50t_replay_trace*.txt): 同一批候选里 L7 真骨架要的那两条在旧键下排第 10/20 与 11/20,
        # 前面全是家附近两三手就能压到的线, 每条错兄弟又各自钻一整支 => 深度优先把 20s/40 万节点整个
        # 花在错分支上, 真分支轮不到。两档实测: 旧键 = 六关 244 手且 L7 未解; v5 = 七关 287 手逐手 0 分歧
        # (前六关手数两档完全一致, 且 v5 多数关更快: L3 2.18s→0.49s、L5 17.24s→11.21s、L6 4.17s→2.19s)。
        out.sort()
        out = out[:12]
        stats["cands"] += len(out)
        return out

    def rec(st, prefix, a5n):
        if time.time() > deadline:
            stats["why"] = stats["why"] or "超时"
            return None
        if stats["nodes"] > node_cap:
            stats["why"] = stats["why"] or "节点上限"
            return None
        sig = (st[1], st[2], st[4], st[6], st[7], st[8], st[9], st[10], st[11],
               tuple(len(t) for t in st[12]) if st[12] else (), a5n)
        if sig in seen_rec:
            return None
        seen_rec.add(sig)
        stats["calls"] += 1
        c, n = stage_paths(m, st, lambda ns: False, max_len=f.budget, max_cands=1,
                           max_nodes=MAX_DEPTH, deadline=deadline)
        stats["nodes"] += n
        stats["bfs"] += 1
        if c:
            return list(prefix) + list(c[0][0])
        if a5n >= f.stages:
            stats["why"] = stats["why"] or "A5 次数用尽"
            return None
        for _ln, _gi, path, ns, _w in hold_cands(st):
            a5, _w2, _d2 = m.step(ns, 5)
            r = rec(a5, list(prefix) + list(path) + [5], a5n + 1)
            if r:
                return r
        if stats["nodes"] <= node_cap and time.time() <= deadline:
            stats["why"] = stats["why"] or "本层无可用候选(穷尽)"
        return None

    plan = rec(m.init(), [], 0)
    return plan, stats


def engine_view(f, g):
    """引擎侧同一份摘要: 活体格 / 幽灵格 / **移动者(格, 朝向, 死活)与其历史长度** / 活塞格 /
    自锁位 / **每 gate 成员表** / **常亮位** / 本关步数。

    token 归属按对象身份映射(v4): 0 = `dzxunlkwxt`(活体), 1..nm = `kgvnkyaimw` 的键(创建序),
    nm+1+j = `rloltuowth` 第 j 个键(创建序)。v3 把"映射不到的成员"一律记进 `dead`, 正是这一支
    把移动者咽掉了 —— 那会让 L6/L7 的"逐手 0 分歧"变成**空的**。映射不到的(被 `yfcuzfdcnj` 摘掉的
    幽灵、被替换的旧活体)仍按 gate 位记进 `dead`。
    """
    C = g.vgwycxsxjz
    live_obj = C.dzxunlkwxt
    mv_objs = list(C.kgvnkyaimw.keys())
    ghost_objs = list(C.rloltuowth.keys())
    tok_by_id = {id(live_obj): 0}
    for j, o in enumerate(mv_objs):
        tok_by_id[id(o)] = j + 1
    for j, o in enumerate(ghost_objs):
        tok_by_id[id(o)] = len(mv_objs) + j + 1
    mem = [0] * len(f.gates)
    dead = 0
    for gi, gate in enumerate(f.gates):
        for src in C.hamayflsib[gi].rqxibulywa:
            t = tok_by_id.get(id(src))
            if t is None:
                dead |= (1 << gi)
            else:
                mem[gi] |= (1 << t)
    return dict(live=f.cell(live_obj.x, live_obj.y),
                ghosts=tuple(sorted(f.cell(o.x, o.y) for o in ghost_objs)),
                movers=tuple((f.cell(o.x, o.y), int(o.wzgvpxcawd), bool(o.pddqxjztas))
                             for o in mv_objs),
                mhlen=tuple(len(v) for v in C.kgvnkyaimw.values()),
                pistons=tuple(sorted(f.cell(h.x, h.y) for h in C.uwxkstolmf)),
                lbits=tuple(bool(h.dijhfchobv) for h in C.uwxkstolmf
                            if hasattr(h, "hluvhlvimq") and getattr(h, "dpdubazedr", False)),
                mem=tuple(mem), dead=dead,
                acts=int(g.ucorwtereb), lvl=int(g._current_level_index), state=str(g._state))


CMP_FIELDS = ("live", "ghosts", "movers", "mhlen", "pistons", "lbits", "mem", "dead")


def verify(f, m, g, plan, idx0):
    """把模型方案在这台引擎上**逐手重放对拍**一遍; 返回 (分歧数, 首分歧手, 引擎过关手)。

    过关那一手引擎已换关、`ucorwtereb` 归 0, 格坐标天然不可比 —— 单列成"过关边界", 不算分歧。
    比较字段见 `CMP_FIELDS`(v4 起含移动者的格/朝向/死活与历史长度 —— 少了这两项, 对拍看不见移动者,
    "0 分歧"就是空的)。
    """
    import g50t_solver as gs
    from arcengine import ActionInput, GameAction
    aid = {a: getattr(GameAction, "ACTION%d" % a) for a in (1, 2, 3, 4, 5)}
    gs._restore_start(g, idx0)
    st = m.init()
    e0 = engine_view(f, g)
    mm = m.render(st)
    bad = 0
    first_bad = None
    if any(e0[k] != mm[k] for k in CMP_FIELDS):
        bad, first_bad = 1, 0
    won_at = None
    for i, a in enumerate(plan, 1):
        st, _won, dead = m.step(st, a)
        g.perform_action(ActionInput(id=aid[a], data={}, reasoning=None), raw=True)
        # **顺序是承重的**: 过关那一手引擎已经换关, `g.vgwycxsxjz` 变成下一关的控制器, 它的
        # `hamayflsib` 比本关短(L5 有 6 个 gate, L6 只有 5 个) —— 先碰 `engine_view` 就会
        # IndexError(10-05 实测: 第 5 关 52 手模型解重放到过关手时炸在这里, 被 solve 的 except
        # 咽成"模型通道异常")。所以过关判定只用引擎自己的计数器, 判完立刻 break。
        lvl = int(g._current_level_index)
        state = str(g._state)
        if lvl > idx0 or state.endswith("WIN"):
            won_at = i
            break
        e = engine_view(f, g)
        mm = m.render(st)
        if any(e[k] != mm[k] for k in CMP_FIELDS + ("acts",)):
            bad += 1
            if first_bad is None:
                first_bad = i
        if dead or state.endswith("GAME_OVER"):
            first_bad = first_bad or i
            bad += 1
            break
    return bad, first_bad, won_at


def solve(g, t_limit=None, node_cap=None):
    """在**关卡起点**的引擎对象上搜本关解。

    返回 `(plan, note)`:
      plan = list[int 1-5] —— 已过 `verify` 逐手对拍(0 分歧 + 同手过关)的模型方案;
      plan = None          —— 模型没解/对拍不过, `note` 里如实写为什么(缺口点名或分歧手数)。
    收尾总把引擎恢复到本关起点(与 g50t_solver.solve 同契约)。
    """
    import g50t_solver as gs
    t_limit = DEFAULT_T_LIMIT if t_limit is None else float(t_limit)
    node_cap = DEFAULT_NODE_CAP if node_cap is None else int(node_cap)
    idx0 = int(g._current_level_index)
    if int(getattr(g, "ucorwtereb", 0) or 0) != 0:
        return None, "本关已走步(ucorwtereb!=0), 模型只能在关卡起点上搜"
    try:
        f = Facts(g)
        m = Model(f)
        t0 = time.time()
        plan, stats = search(f, m, time.time() + t_limit, node_cap)
        el = time.time() - t0
        if not plan:
            return None, "模型未搜到解: %s (%.2fs / %d 节点 / 阶段层 %d; 本关仍缺: %s)" % (
                stats["why"] or "未知", el, stats["nodes"], stats["calls"],
                " | ".join(f.gaps) or "无(机制已建模, 是搜索/表达式的墙)")
        bad, first_bad, won_at = verify(f, m, g, plan, idx0)
        if bad or won_at != len(plan):
            return None, "模型解 %d 手未通过逐手对拍(分歧 %d 处, 首分歧手=%s, 引擎过关手=%s); 缺口: %s" % (
                len(plan), bad, first_bad if first_bad is not None else "-",
                won_at if won_at else "-", " | ".join(f.gaps) or "无")
        note = ("模型自搜 %d 手, 逐手对拍 0 差异且同手(第 %d 手)过关 | %.2fs / %d 节点 | %s"
                % (len(plan), won_at, el, stats["nodes"], f.summary()))
        if f.gaps:
            note += " | 未建模: " + "; ".join(f.gaps)
        return plan, note
    except Exception as e:
        return None, "模型通道异常: %s: %s" % (type(e).__name__, e)
    finally:
        try:
            gs._restore_start(g, idx0)
        except Exception:
            pass
