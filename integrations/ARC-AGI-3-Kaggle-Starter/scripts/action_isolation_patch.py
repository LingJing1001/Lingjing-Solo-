"""GameAction 线程隔离补丁 —— **标准件**（供任何 agent 复用）。

为什么需要它
------------
`arcengine.GameAction` 是**进程级枚举单例**：

```python
class GameAction(Enum):
    ACTION6 = (6, ComplexAction)
    def __init__(self, action_id, action_type):
        self.action_data = action_type()            # 每个成员只有一份
    def set_data(self, data):
        self.action_data = self.action_type(**data) # 直接覆写这个全进程唯一的成员
```

框架 `agents/agent.py::do_action_request` 的发包顺序是：

```python
action = self.choose_action(...)          # 里面 action.set_data({"x":..,"y":..})  ← 写共享成员
data = action.action_data.model_dump()    # ← 下一行才读回来
raw  = self.arc_env.step(action, data=data, ...)
```

写入和读取之间隔着一次函数返回。竞赛 rerun 里 25~110 局 = 25~110 个线程
**共用一个 Python 进程**，于是：

```
线程 A（cd82）: set_data({x:46,y:4})       ← 写进全局 ACTION6
线程 B（别的局）: set_data({x:11,y:29})     ← 覆盖！A 的坐标没了
线程 A: data = action.action_data.model_dump()  ← 读到的是 B 的坐标
⇒ A 局的点击落到 B 局的格子上 → 计划型动作全线跑歪
```

实测（2026-10-22，本地 25 局并发复刻）：
| | 修复前 | 修复后 |
|---|---|---|
| 总分 | 408.20 | **1449.82** |
| 完成关数 | 38 | **103** |
| 满分局 | 3 | **13**（与顺序基准一致） |
| 单元级坐标串号率 | 10.3%~13.9% | **0%** |

怎么用
------
**首选：把 `install()` 放在你自己的 agent 模块顶层**（import 时执行，早于任何一局开跑）：

```python
from action_isolation_patch import install
install()
```

**注意约束**：竞赛 rerun 的框架树来自 Kaggle 官方 input 包
（`/kaggle/input/.../ARC-AGI-3-Agents`），**框架代码改不了**；
notebook 只会把 `my_agent.py` 拷进 `agents/templates/`。
所以**提交用的 agent 必须是自包含的**——
`integrations/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py` 里是内联的一份拷贝，
`scripts/check_action_isolation.py` 会校验它与本文件的实现**逐字一致**（防漂移）。

跑测试：
    python scripts/check_action_isolation.py
"""
from __future__ import annotations

import threading
from typing import Any

__all__ = ["install", "is_installed"]


def install() -> bool:
    """把 GameAction 的可变载荷改成线程本地。返回是否成功安装。

    成功时：同一线程内 `set_data()` → `action_data` 读写一致（框架照常工作）；
    跨线程互不可见（竞态消失）。失败时打印一行 FAILED 并退回原行为，
    绝不拖垮开局。
    """
    try:
        from arcengine.enums import GameAction as ga
    except Exception as exc:  # noqa: BLE001
        print(
            f"[action-isolation] FAILED to import GameAction: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    if getattr(ga, "_lingjing_tls_actions", False):
        return True
    try:
        # 先快照原始实例属性（此时读到的还是 instance __dict__ 里的值）
        base = {m.name: m.action_data for m in ga}
        tls = threading.local()

        def _get_action_data(self: Any) -> Any:
            store = getattr(tls, "data", None)
            if store is not None:
                val = store.get(self.name)
                if val is not None:
                    return val
            return base.get(self.name)

        def _set_action_data(self: Any, value: Any) -> None:
            base[self.name] = value
            store = getattr(tls, "data", None)
            if store is None:
                store = {}
                tls.data = store
            store[self.name] = value

        def _get_reasoning(self: Any) -> Any:
            store = getattr(tls, "reasoning", None)
            if store is not None and self.name in store:
                return store[self.name]
            return self.__dict__.get("_lingjing_reasoning_base")

        def _set_reasoning(self: Any, value: Any) -> None:
            self.__dict__["_lingjing_reasoning_base"] = value
            store = getattr(tls, "reasoning", None)
            if store is None:
                store = {}
                tls.reasoning = store
            store[self.name] = value

        # property 是 data descriptor，优先级高于实例 __dict__，
        # 因此旧的 instance attribute 会被自动遮蔽，不需要改写成员本身。
        ga.action_data = property(_get_action_data, _set_action_data)  # type: ignore[attr-defined]
        ga.reasoning = property(_get_reasoning, _set_reasoning)  # type: ignore[attr-defined]
        ga._lingjing_tls_actions = True  # type: ignore[attr-defined]
        ga._lingjing_tls_base = base  # type: ignore[attr-defined]
        ga._lingjing_tls_store = tls  # type: ignore[attr-defined]
        return True
    except Exception as exc:  # noqa: BLE001 — 装不上就退回原行为
        print(
            f"[action-isolation] FAILED to install thread-local action state: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return False


def is_installed() -> bool:
    """补丁是否已生效。"""
    try:
        from arcengine.enums import GameAction as ga
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(ga, "_lingjing_tls_actions", False))


if __name__ == "__main__":
    ok = install()
    print(f"[action-isolation] installed={ok} "
          f"(thread-local GameAction state: {'ON' if ok else 'OFF'})")
    raise SystemExit(0 if ok else 1)
