# 灵境引擎 V14.4 变更记录 —— 「度规张量 + 迷雾泊松展开」

> 版本主题：**从标量曲率代理升级为真正的度规张量 g_μν，并实现协议 9 的迷雾区拉普拉斯展开**。
> 新增两个模块：`core/geometry.py`（度规生成）、`core/fog.py`（迷雾展开）。

---

## 一、动机

V14.3 补全了场方程（耗散 + 自相互作用），但几何层仍是标量 `curvature_proxy = grad_energy/8`。对照 v13.2 §3，真正的几何生成应该输出度规张量 `g_μν`，而不是一个标量代理。

同时协议 9（迷雾展开）在 V14.2 里只是"新 ROI 并入清晰区统计"，没有真正的重构机制——新区域内部的值是黑盒。V14.4 用拉普拉斯方程 ∇²φ=0 做 Dirichlet 边界延拓。

---

## 二、核心改动

### 2.1 `core/geometry.py`（新建）—— 度规张量

```python
@dataclass
class MetricConfig:
    h: float = 1.0
    I_Pl: float = 1.0      # Planck 尺度
    kappa: float = 1.0     # 曲率耦合
    phi0: float = 0.0      # 真空基线

def compute_spatial_metric(phi, cfg) -> np.ndarray:
    """g_ij = (1 + κ(Φ²−Φ₀²))·δ_ij + (1/I_Pl)·∂_iΦ·∂_jΦ
    返回 (Nx,Ny,Nz,3,3)"""

def curvature_proxy(phi, cfg) -> np.ndarray:
    """弱场 Ricci scalar 近似：R ≈ −∇²[κ(Φ²−Φ₀²) + (1/2I_Pl)‖∇Φ‖²]
    返回 (Nx,Ny,Nz)"""
```

**设计决策**：
- 只算空间度规 g_ij (3×3)，时间分量 g_00 留待 V15.0（多 Agent 共识层需要时间导数语义）
- 梯度用中心差分（内部）+ 单侧差分（边界，与 Neumann 一致）
- 度规迹 `tr(g) = g_ii` 作为"整体弯曲度"的标量摘要

### 2.2 `core/fog.py`（新建）—— 迷雾拉普拉斯展开

```python
def expand_fog_laplacian(phi, unknown_mask, h) -> (phi_new, cons_err):
    """对 unknown_mask 内节点解 ∇²φ = 0（Dirichlet 边界）。
    边界节点值固定，内部节点由拉普拉斯方程唯一确定。"""

def identify_fog_shell(clear_mask, new_clear_mask) -> shell_mask:
    """识别新进入清晰区的薄壳层（面积律 O(R²)）。"""
```

**面积律保证**：展开只作用于新进入清晰区的薄壳层节点，不碰整个 N³ 体积。

---

## 三、新增契约测试（4 项）

文件：`tests/v14/test_metric_fog.py`

| 契约 | 内容 | 验证 |
|---|---|---|
| **C14** | 均匀场 Φ=0 → 度规 g_ij = δ_ij（完全平坦）；曲率 R=0 | ✅ |
| **C15** | 高斯点源 → 源中心度规偏离单位阵，远处回归平坦；曲率代理响应 | ✅ |
| **C16** | 拉普拉斯展开精确恢复调和函数（线性场 φ=x/15），rel L2 < 1e-6 | ✅ |
| **C17** | 展开结果满足拉普拉斯方程（unknown 区域 max\|∇²φ\| < 1e-6） | ✅ |

---

## 四、验证结果

```
V14.4 新测试:   4 passed (C14-C17)
全量回归:       452 passed (448 + 4 新增) / 19 skipped / 0 failed
```

---

## 五、诚实边界

1. **只算空间度规 g_ij**：时间分量 g_00 和混合项 g_0i 未实现。v13.2 公式里 ∂_0Φ（时间导数）在当前单步演化里没有直接量，留待 V15.0 多 Agent 共识层接入 TickRecord 后补。
2. **弱场近似**：`curvature_proxy` 是 Ricci scalar 的弱场线性化，不是精确 Ricci 张量收缩。强场（Φ 大、梯度大）下精度下降。
3. **拉普拉斯展开只恢复调和部分**：迷雾展开是 Dirichlet 边界延拓，只能恢复满足 ∇²φ=0 的分量。非调和分量（如正弦波的振荡）会被平滑掉——这是协议 9 的固有特性，不是 bug。
4. **C17 不是"Σφ 守恒"**：拉普拉斯展开是边界值的调和延拓，内部值由边界唯一确定，不创造信息量但也不保持 Σφ 不变（边界固定时内部解唯一，Σφ 是解的结果而非约束）。
5. **N≥128 规模**：当前 CSR 矩阵逐节点构造，N=16 时快；N=128（200 万节点）时构造和求解都会慢，届时需要向量化构造或 CUDA。

---

## 六、V14.4 → V15.0 候选

1. **多 Agent 共识层**：统一提交队列 + R1-R5 规则（协议 6）
2. **C 值只读接入**：`ReflectionSignal.c_value` 作为场的只读监视器接入 tick 循环
3. **第二个场景**：`CollaborationScenario`（2-3 Agent 推箱子）验证共识层

---

*灵境引擎 · IFC 信息场范式工程 v1.0 · V14.4「度规张量 + 迷雾泊松展开」*
