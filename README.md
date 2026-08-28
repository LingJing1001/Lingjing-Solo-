# Lingjing-Solo

Lingjing-Solo 是一个面向交互式推理基准的单 Agent 世界模型框架。它把感知、世界模型、探索、轻量规划和反思组合成一个可复用的 Python package。

## 安装

要求 Python 3.11 或更新版本。

### 用户安装

```bash
python -m pip install lingjing-solo
```

如果项目尚未发布到 PyPI，可以直接从公开 Git 仓库安装：

```bash
python -m pip install \
  "lingjing-solo @ git+https://github.com/<organization>/Lingjing-Solo-.git"
```

### 本地开发安装

在本仓库根目录执行：

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

安装后验证公开入口：

```bash
python -c "from lingjing_solo import LingjingSoloAgent; print(LingjingSoloAgent.__name__)"
```

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

## ARC-AGI-3 adaptor

ARC adaptor 应直接依赖本 package，而不是依赖本机 sibling directory：

```python
from lingjing_solo import LingjingSoloAgent
```

ARC 项目本地开发时可以安装本仓库：

```bash
uv pip install -e /srv/agent-platform/projects/Lingjing-Solo-
```

团队协作时，应把 `<organization>/Lingjing-Solo-` 替换为公开仓库地址，并在 ARC 项目的依赖文件中固定一个版本、tag 或 commit，保证不同开发者使用相同代码。

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
