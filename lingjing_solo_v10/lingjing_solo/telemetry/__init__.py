"""
telemetry/__init__.py — v1.0 实验观测层

职责：记录 Agent 每一步的完整决策链路，输出为 JSONL，供跑分后分析。

为什么需要：用户要去真实环境跑分，实验数据再喂回来分析。
没有结构化观测 = 拿到分数也无法诊断"哪一级决策出了问题"。

记录字段（每步一条 JSON）：
    step          : 真实步数
    action        : 本步选择的动作
    source        : 决策来源（transition / wmp / object / score / macro / win）
    sim_calls     : 本步模拟器调用次数（= BFS 展开节点数）
    real_actions  : 累计真实步数
    drift         : 当前 WMP 漂移度
    wmp_compiled  : WMP 是否已编译
    rules_count   : 当前已晋升规则数
    cache_hits    : 后继缓存命中数
    cache_misses  : 后继缓存未命中数
    decision_ms   : choose_action 耗时（毫秒）
    win           : 是否通关

设计原则：零开销可关闭（采集本身不拖慢决策）。
"""
import json
import time
from collections import defaultdict
from typing import Optional, Dict, Any, List


class Telemetry:
    """
    轻量级观测器：append 到内存 + 可选落盘 JSONL。

    用法：
        tel = Telemetry(log_path="run.jsonl")
        tel.start_step()
        ... 决策 ...
        tel.record(source="wmp", sim_calls=12)
        tel.end_step(action="right")
        tel.flush()   # 落盘
    """

    def __init__(self, log_path: Optional[str] = None, enabled: bool = True):
        self.log_path = log_path
        self.enabled = enabled
        self.records: List[Dict[str, Any]] = []
        self._current: Dict[str, Any] = {}
        self._step_start: Optional[float] = None
        # 聚合统计（供 summary 使用）
        self.source_counts: Dict[str, int] = defaultdict(int)
        self.total_sim_calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    # ---------- 生命周期 ----------

    def start_step(self):
        """标记一步决策的开始（计时）。"""
        if not self.enabled:
            return
        self._current = {}
        self._step_start = time.time()

    def record(self, **fields):
        """记录当前步的任意字段。"""
        if not self.enabled:
            return
        self._current.update(fields)

    def end_step(self, action: Optional[str] = None, win: bool = False):
        """一步决策结束：打包成记录。"""
        if not self.enabled:
            return
        rec = dict(self._current)
        rec.setdefault("action", action)
        rec.setdefault("win", win)
        if self._step_start is not None:
            rec["decision_ms"] = round((time.time() - self._step_start) * 1000, 3)
        self.records.append(rec)
        # 聚合
        src = rec.get("source", "?")
        self.source_counts[src] += 1
        self.total_sim_calls += rec.get("sim_calls", 0) or 0
        self.cache_hits += rec.get("cache_hits", 0) or 0
        self.cache_misses += rec.get("cache_misses", 0) or 0
        self._current = {}
        self._step_start = None

    # ---------- 持久化 ----------

    def flush(self, path: Optional[str] = None):
        """把内存记录追加写入 JSONL 文件。"""
        if not self.enabled:
            return
        target = path or self.log_path
        if not target:
            return
        with open(target, "a", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.records.clear()

    def save(self, path: Optional[str] = None):
        """全量保存（覆盖写）所有记录。"""
        if not self.enabled:
            return
        target = path or self.log_path
        if not target:
            return
        with open(target, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- 摘要（跑分后喂回分析的入口）----------

    def summary(self) -> Dict[str, Any]:
        """聚合统计：供用户跑分后快速了解决策分布。"""
        total = len(self.records)
        return {
            "total_steps": total,
            "total_sim_calls": self.total_sim_calls,
            "avg_sim_calls_per_step": (self.total_sim_calls / total) if total else 0.0,
            "source_distribution": dict(self.source_counts),
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            "wmp_used": self.source_counts.get("wmp", 0) > 0,
        }

    def reset(self):
        """新一关开始时清空（保留文件）。"""
        self.records.clear()
        self._current = {}
        self._step_start = None
