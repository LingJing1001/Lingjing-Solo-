# lingjing_explore — 用 C++ 加强 ARC-AGI-3「未知环境」冲关

## 为什么要上 C++？

灵境战队当前主路径是 **Python Agent + 离线脚本挖计划**。  
冲关慢点不在「算式」，而在 **搜索前沿**：

| 瓶颈（Python） | C++ 做什么 |
|----------------|------------|
| MD5 / 大 tuple 当 `seen` 键 | `lj_hash_*` → u64 |
| `set`/`dict` 海量插入查找 | `Visited` 开放哈希 |
| beam 截断排序 | `lj_beam_top_k` |
| 路径 `list` 复制膨胀 | `CompactBFS` 父指针存路径 |

**环境转移**（`perform_action` / `deepcopy`）仍留在 Python——这是正确边界：  
C++ 先加速「数字世界探索的数据结构」，再考虑把单游戏抽象转移模型下沉。

## 快速开始

```bat
cd cpp_explore
build_msvc.bat
python scripts\bench_hash.py
```

离线 wa30 L3（C++ visited + beam）：

```bat
cd ..
python cpp_explore\scripts\wa30_l3_engine_search_cpp.py
```

## 接到正式 Agent

在 `agent/my_agent.py` 的 `_bfs_search` / `_iddfs_search` 里：

1. `from lingjing_explore import CompactBFS, state_key_u64, hash_bytes`
2. 用 `CompactBFS` 替 Python `deque` + `seen`
3. 每次 `perform_action` 后对 frame bytes 做 `state_key_u64(...)`，`try_child`

（镜像同步 `vendor/.../templates/my_agent.py`。）

## 与「未知环境探索」的关系

ARC-AGI-3 的未知关 = 没有先验计划，只能：

1. **感知**格子/精灵 → bytes  
2. **行动** 1–6  
3. **去重探索** 避免同一状态死循环  
4. **启发式 / beam** 把算力压到有希望的分支  

C++ 模块专门加速 3+4，让你们在同样 60–300 秒预算里多扩几个数量级的节点——这就是「用 C++ 能力去冲未知环境」的落地方式。

## 版本

`lingjing_explore/1.0.0-arcagi3`
