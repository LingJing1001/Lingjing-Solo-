# Duck + SmartRouter LLM 接入（2026-10-12）

## 布局

| 路径 | 用途 |
|------|------|
| `integrations/duck-harness/` | 开源 [Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness) vendor（gitignore，本地 clone） |
| `integrations/duck-kaggle/` | 灵境推送/补丁（入库）：`patch_cp313.py`、kernel-metadata、脚本 |
| `integrations/duck-kaggle/scripts/smoke_vendor.py` | 无 GPU 结构校验（对 vendor 目录） |
| `integrations/duck-kaggle/scripts/submit_duck_kaggle.sh` | 平行 Duck kernel 提交（WSL/Linux） |
| `integrations/ARC-AGI-3-Kaggle-Starter/agent/llm_policy.py` | SmartRouter 本地 JSON 策略 |
| `integrations/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py` | `+llm-policy` 路由（默认 fail-closed） |

## Kaggle datasets（禁网挂载）

| Ref | 内容 |
|-----|------|
| `bilgisarmal/vllm031-cp313-wheelhouse` | vLLM cp313 wheelhouse（Kaggle 镜像现为 Python 3.13） |
| `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` | Qwen 3.6 27B FP8 权重 |
| `jeroencottaar/taaf-kaggle-source-share` | Duck/TAAF 源码包（平行 kernel） |
| Served name | `vrfai/Qwen3.6-27B-FP8` |

> 坑：Duck 默认 `arc3-vllm-h100-wheelhouse-v3` 是 **cp312**；Kaggle 镜像现为 Python 3.13，会 `No matching distribution for aiohttp`。  
> 第二坑：cp313 house 的 wheel 在 `wheels/` 子目录，pip `--find-links` 必须指到该目录，否则 `agent-detector` 找不到。  
> `kaggle_push/patch_cp313.py` 每次从官方 share notebook 重建并打上这两处补丁。

## Phase 1 — 平行 Duck kernel（不覆盖 SmartRouter）

**已推送（2026-10-12）：**  
https://www.kaggle.com/code/liaojiaxuan/lingjing-duck-harness-v1  

Windows 可复现推送（基于公开 share notebook + 官方 Qwen/vLLM datasets）：

```powershell
cd integrations/duck-harness
python scripts/smoke_vendor.py
# 产物在 kaggle_push/ ；用 Starter 的 kaggle CLI：
$env:KAGGLE_API_TOKEN = (Get-Content ..\ARC-AGI-3-Kaggle-Starter\.kaggle\access_token -Raw).Trim()
..\ARC-AGI-3-Kaggle-Starter\.venv\Scripts\kaggle.exe kernels push -p kaggle_push
```

挂载 datasets：

- `jeroencottaar/taaf-kaggle-source-share`（公开 TAAF 源码包）
- `bilgisarmal/vllm031-cp313-wheelhouse`
- `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`

推送前先跑：`python kaggle_push/patch_cp313.py`

Linux 完整 `make kaggle-duck`（自打包源码 dataset）仍可用：

```bash
bash scripts/submit_duck_kaggle.sh
```

验收：kernel COMPLETE 且 `publicScore > 0`（Phase B 竞赛重跑后看 Submissions）。

## Phase 2 — SmartRouter hybrid

Env：

| Var | 默认 | 含义 |
|-----|------|------|
| `LINGJING_LLM` | GPU 构建 `on` / CPU `off` | 总开关 |
| `LINGJING_LLM_URL` | `http://127.0.0.1:1234/v1` | vLLM |
| `LINGJING_LLM_MODEL` | `vrfai/Qwen3.6-27B-FP8` | served name |
| `LINGJING_LLM_STALL` | `80` | stall 步数触发终端 override |
| `LINGJING_KAGGLE_ACCEL` | `rtx6000` | `cpu` 则关掉 GPU/数据集 |

触发点（plan/INLINE/engine 地板不覆盖）：

1. CEAX giveup 前先问 LLM  
2. `r3-none` 后 `_llm_armed`  
3. `choose_action` 末端 stall ≥ `LINGJING_LLM_STALL`

构建与提交：

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter
# GPU + LLM
LINGJING_KAGGLE_ACCEL=rtx6000 python scripts/build_notebook.py
make submit   # 或 kaggle kernels push -p notebooks/

# CPU 地板回归
LINGJING_KAGGLE_ACCEL=cpu python scripts/build_notebook.py
```

## 失败指纹

| 日志 | 含义 |
|------|------|
| `Expected 1 CUDA GPU(s), found ['Tesla T4', 'Tesla T4']` | 未设 `machine_shape: NvidiaRtxPro6000`（仅 `enable_gpu` 会落到 T4×2） |
| `No matching distribution found for aiohttp==…` | 用了 cp312 H100 wheelhouse，但镜像是 Python 3.13 → 换 cp313 house |
| `No matching distribution found for agent-detector` | `--find-links` 指到 house 根目录，wheel 实际在 `wheels/` |
| `IndentationError` in setup cell | 补丁字符串含真实换行；改用 `chr(10)` 拼接 |
| `Maximum batch GPU session count of 2 reached` | 账号并行 GPU session 满，等上一跑结束再 push |
| `VLLM_SKIP` / `VLLM_START_FAIL` | 权重或 wheel 未挂上 / 起服失败 → `LINGJING_LLM=off` |
| `llm_policy request failed` | vLLM 未就绪或超时 |
| `llm_policy import fail` | notebook 未拷贝 `llm_policy.py` |
| Duck OOM / 9h kill | 降 `CONCURRENT_JOBS` / `MAX_RUNTIME_MINUTES` |

## kernel-metadata 要点

```json
{
  "enable_gpu": true,
  "machine_shape": "NvidiaRtxPro6000",
  "enable_internet": false,
  "dataset_sources": [
    "jeroencottaar/taaf-kaggle-source-share",
    "bilgisarmal/vllm031-cp313-wheelhouse",
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
  ],
  "competition_sources": ["arc-prize-2026-arc-agi-3"]
}
```

`build_notebook.py` 在 `LINGJING_KAGGLE_ACCEL=rtx6000` 时会自动写入 `machine_shape`。
