# Lingjing-Solo

Lingjing-Solo 是一个面向交互式推理基准的单 Agent 世界模型框架。它把感知、世界模型、探索、轻量规划和反思组合成一个可复用的 Python package。

## 安装

要求 Python 3.11 或更新版本。

### 用户安装

```bash
python -m pip install lingjing-solo
```

如果项目尚未发布到 PyPI，可以直接从本仓库安装：

```bash
python -m pip install \
  "lingjing-solo @ git+ssh://git@github.com/LingJing1001/Lingjing-Solo-.git"
```

也可以使用 HTTPS：

```bash
python -m pip install \
  "lingjing-solo @ git+https://github.com/LingJing1001/Lingjing-Solo-.git"
```

### 本地开发安装

在本仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

也可以使用 `uv`：

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

安装后验证公开入口：

```bash
python -c "from lingjing_solo import LingjingSoloAgent; print(LingjingSoloAgent.__name__)"
```

## 安装使用最短路径

如果目标是运行 ARC-AGI-3 Starter，而不是只使用核心 Python package，直接执行：

```bash
git clone git@github.com:LingJing1001/Lingjing-Solo-.git
cd Lingjing-Solo-/integrations/ARC-AGI-3-Kaggle-Starter
make setup
make verify-local
make play-local STEPS=400
```

这条路径会使用仓库内的 `vendor/requirements.lock`、固定的
`vendor/ARC-AGI-3-Agents/` 和 `environment_files/`。本地 Kaggle token 只在
执行 `make submit` 时需要，放在 Starter 目录的 `.kaggle/access_token`，不应
提交到 Git。若只开发核心 package，则从仓库根目录执行上一节的 `.venv` 安装
和 `pytest -q` 即可。

## 目录结构与职责

```text
Lingjing-Solo-/
├── lingjing_solo/                         # 唯一 canonical 核心 package
├── tests/                                 # 核心 package 测试
├── integrations/
│   ├── ARC-AGI-3-Kaggle-Starter/         # ARC-AGI-3 本地/Kaggle 运行入口
│   │   ├── agent/my_agent.py              # 通常只修改这个 Agent
│   │   ├── environment_files/             # 已固定的 25 个本地游戏环境
│   │   ├── vendor/requirements.lock       # ARC 运行依赖锁
│   │   ├── vendor/ARC-AGI-3-Agents/       # 固定 framework 源码
│   │   ├── scripts/                       # 本地运行、benchmark、notebook 构建
│   │   └── notebooks/                     # Kaggle metadata/submission
│   └── ARC-AGI-3/                         # ARC 相关源码/研究资料快照
├── scripts/                               # 项目级同步和 notebook 工具
└── pyproject.toml                         # package、测试和开发依赖配置
```

`lingjing_solo/` 是唯一 canonical source。ARC integration 不应再创建或依赖
`integrations/*/lingjing_solo` 的副本；运行入口通过项目根路径加载 canonical package。

## ARC-AGI-3 Starter 安装与使用

以下命令在仓库根目录执行。ARC Starter 要求 Python 3.12；核心 package
本身支持 Python 3.11 及以上。

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter
make setup
```

`make setup` 会创建 `integrations/ARC-AGI-3-Kaggle-Starter/.venv/`，按
`vendor/requirements.lock` 安装 `arc-agi`、`arcengine`、`python-dotenv`、
`pandas`、`pyarrow`、pytest 等依赖，并使用仓库中固定提交的
`ARC-AGI-3-Agents` framework。安装生成的本地 site-packages 不进 Git，fresh
checkout 会通过这个命令重建。

本地快速验证：

```bash
make verify-local                         # ls20 + vc33，50 步 smoke test
make play-local GAME=ls20 STEPS=400       # 单局 400 步
make play-local STEPS=400                 # 25 个游戏完整本地跑测
python scripts/benchmark_all_games.py 400 \
  --out ui/static/games_benchmark_phase_a_400.json \
  --label phase-a
```

Kaggle 提交前，在 Starter 目录创建本地 token 文件；不要把 token 写进
README、代码或 `.env`：

```bash
mkdir -p .kaggle
printf '%s\n' '<your-kaggle-token>' > .kaggle/access_token
chmod 600 .kaggle/access_token
# 首次提交前修改 notebooks/kernel-metadata.json 中的 Kaggle username
make submit
make status
```

`make submit` 只负责构建并上传 Phase A notebook。Kaggle kernel 完成后，
还需要在 Kaggle 页面点击 **Submit to Competition** 才会进入 Phase B 隐藏集
评分。`notebooks/submission.ipynb` 是已生成的参考产物；修改
`agent/my_agent.py` 后应运行 `make notebook` 重新生成它。

## 测试、skip 与 ARC 运行时边界

核心测试和 Starter 测试分开运行：

```bash
# 根目录
pytest -q

# ARC Starter 目录
cd integrations/ARC-AGI-3-Kaggle-Starter
pytest -q
```

当前测试结果中的 skip 有明确边界：

- 根目录 18 个 skip：`tests/test_ar25_plan_contract.py` 的 ARC engine
  contract tests。当前执行解释器没有安装 `arc_agi`，所以这些测试没有执行，
  不代表 AR25 ARC engine 契约已经通过。
- Starter 1 个 skip：`tests/test_lincore.py` 的真实 `arcengine` 运行测试；
  在没有 ARC Kaggle runtime 的环境中主动跳过。
- Starter 的 3 个 `Mean of empty slice` 是现有 memory 测试 warning，不是
  测试失败；它表示该 fixture 某些列没有观测值，当前测试仍通过。

要覆盖这些 ARC 运行时测试，应在 Starter 的锁定环境中执行：

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter
make setup
.venv/bin/pytest -q -rs
```

如果仍显示 ARC runtime skip，先检查：

```bash
.venv/bin/python -c "import arc_agi, arcengine; print('ARC SDK OK')"
```

普通核心开发不需要安装 ARC SDK；不要为了让 skip 数变成 0 而把 ARC 专用
依赖加入核心 `pyproject.toml`。

## 基本用法

```python
import numpy as np
from lingjing_solo import LingjingSoloAgent

agent = LingjingSoloAgent()
agent.reset()

grid = np.zeros((8, 8), dtype=np.int8)
action = agent.choose_action(
    frames=[],
    latest_frame=grid,
    valid_actions=["ACTION1", "ACTION2"],
)
print(action)
```

`choose_action` 返回 ARC/Kaggle 适配层使用的动作字符串。实际 ARC 接入请使用单独的 adaptor，不要让核心 package 依赖 ARC SDK。

## 可选 CNN 依赖

基础框架只需要 NumPy。需要 Torch CNN 编码时安装可选依赖：

```bash
uv pip install -e ".[cnn]"
```

没有 Torch 时，编码器会使用轻量降级路径；ARC/Kaggle 评测环境不得依赖外部网络服务。

## 测试与 lint

```bash
uv run pytest -q
uv run ruff check .
```

项目中的 `test_solo.py` 和 `notebook_template.py` 也可用于本地闭环自检：

```bash
python test_solo.py
python notebook_template.py
```

## 可选 R5 反思顾问

R5 不是“每一步都问 LLM”。只有检测到循环、规则冲突或步数告急时，
`ReflectionTrigger` 才会打包摘要。`lingjing_solo/reflection/skill.py` 中的
`R5_SKILL` 是固定工作规则，`prompt.py` 中的 `build_r5_prompt()` 会把规则、
最近转移、目标假设、触发原因和合法动作拼成一次模型输入。

默认模型标识为 `minimax-m3`（`R5_DEFAULT_MODEL`）。仓库不会自动联网或读取
密钥；调用方可以注入 `llm_fn`，但模型输出必须是合法动作字符串。R5 只提供
建议，不绕过动作合法性、预算和 ARC/Kaggle 适配层的安全边界。

```python
from lingjing_solo import LingjingSoloAgent
from lingjing_solo.reflection import R5_DEFAULT_MODEL

agent = LingjingSoloAgent()

def local_minimax(prompt):
    # 由调用方提供本地模型适配器；不要把 API key 提交到 Git。
    return adapter.generate(model=R5_DEFAULT_MODEL, prompt=prompt)

agent.llm.inject_prompt_llm(local_minimax)
```

## API key 与安全

- API key 只放在本地 `.env` 或环境变量中。
- 不要把 `.env`、API key、私有 endpoint、机器路径提交到 Git。
- ARC API key 示例：

```bash
export ARC_API_KEY="<your-key>"
```

- 发布 package 前检查 Git 状态和 secret scan。

## 许可证

本项目使用 MIT-0（MIT No Attribution）许可证，详见 [LICENSE](LICENSE)。提交第三方代码时，必须确认其许可证允许公开分发，并保留必要的版权声明。

## 当前限制

- `WorldModelField.detect_win` 仍需要按具体环境完善。
- 轻量规划器目前是框架骨架，不能保证解决所有 ARC 游戏。
- LLM 顾问只适用于本地实验；正式 ARC/Kaggle 评测不能依赖网络 API。
