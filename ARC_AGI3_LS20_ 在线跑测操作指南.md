# ARC-AGI-3 LS20 L1-L7 复现指南

本文说明团队成员如何从干净环境运行 Lingjing-Solo Agent，并在 ARC 在线环境中生成 LS20 Scorecard，目标是复现 **Level 1–7 pass（7/7）**。L1-L7 罐头解（共 309 步）已通过本地引擎离线重放与官方线上运行双重验证。

> 本文档只记录已经验证过的运行链路。`exit 0` 或生成 Scorecard 只证明 runner 链路可用；只有 `levels_completed >= 7` 和正分才证明 Level 1–7 实际通过。

## 1. 重要前提：需要两个仓库

真实 ARC runner 不是只运行 `Lingjing-Solo-`。需要同时准备：

| 仓库 | 作用 |
|---|---|
| `Lingjing-Solo-` | 核心 Agent、感知、规划和 LS20 solver |
| `ARC-AGI-3-Agents` | 官方 ARC `Agent` adaptor、runner 和 Scorecard 提交入口 |

仓库地址：

```text
https://github.com/LingJing1001/Lingjing-Solo-
https://github.com/arcprize/ARC-AGI-3-Agents
```

### 当前协作限制

`ARC-AGI-3-Agents` 中的 Lingjing-Solo adaptor 已同步到本仓库 `arc_adaptor/`（分支 `fix/ls20-plan-reseed`，内含 L1-L7 罐头解）。其他成员应独立 checkout 官方 ARC 仓库到自己的本地目录，再从该分支把两个 adaptor 文件复制到本地 checkout 中。

```text
Lingjing-Solo-/arc_adaptor/agents/templates/lingjing_solo_agent.py
Lingjing-Solo-/arc_adaptor/agents/__init__.py
```

这两个文件保持 ARC 仓库中的相对目录结构。其他成员 checkout 两个仓库后，直接复制到 ARC checkout 的对应位置即可。

## 1.1 独立 checkout ARC 仓库并复制 adaptor

每个人独立 checkout 官方 ARC 仓库到自己的本地目录，然后把本项目携带的两个 adaptor 文件复制进去。这里不依赖任何 shared adaptor branch/commit，也不要求在 ARC 仓库上使用其他人的分支：

```bash
cd ~/projects

git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents

mkdir -p agents/templates
cp ../Lingjing-Solo-/arc_adaptor/agents/templates/lingjing_solo_agent.py \
   agents/templates/lingjing_solo_agent.py
cp ../Lingjing-Solo-/arc_adaptor/agents/__init__.py \
   agents/__init__.py
```

此时可以确认两个 adaptor 文件已经进入本地 ARC checkout：

```bash
git status --short

test -f agents/templates/lingjing_solo_agent.py
grep -n "lingjing_solo_agent\|lingjingsolo" agents/__init__.py
```

预期 `git status` 显示两个文件被修改/新增，并且能看到 adaptor 文件和 `lingjingsolo` 注册导入。

> `agents/__init__.py` 是 ARC 仓库的完整文件，不是只追加一行。复制时应使用本项目 `arc_adaptor/agents/__init__.py` 的完整版本，避免遗漏现有 imports。

> 如果 ARC checkout 中已有本地修改，先处理或保存这些修改，再执行复制；复制会覆盖 ARC checkout 中的同名文件。不要把本机绝对路径、`.env` 或 API key 提交到 Git。

## 2. Checkout Lingjing-Solo

如果已按第 1.1 节完成两个仓库的 clone，则直接进入已有目录；不要重复执行 `git clone`：

```bash
cd ~/projects/Lingjing-Solo-
```

如果只准备了 ARC 仓库，单独 clone Lingjing-Solo：

```bash
git clone https://github.com/LingJing1001/Lingjing-Solo-.git
cd Lingjing-Solo-
```

使用 `main` 创建自己的复现分支，并在该分支上运行：

```bash
git switch main
git pull --ff-only origin main
git switch -c reproduce/lingjing-solo-ls20
```

如果该个人分支已经存在，则直接切换到它：

```bash
git switch reproduce/lingjing-solo-ls20
```

确认当前分支和工作树：

```bash
git branch --show-current
git status --short --branch
```

## 3. 安装依赖

在 ARC adaptor 仓库目录执行：

```bash
cd ../ARC-AGI-3-Agents
uv sync
uv pip install -e ../Lingjing-Solo-
```

`uv pip install -e` 很重要：它让 ARC adaptor 使用当前 checkout 的 Lingjing-Solo 代码，而不是其他位置的旧副本。

验证核心 package 可以导入：

```bash
uv run python -c \
  "from lingjing_solo import LingjingSoloAgent; print(LingjingSoloAgent.__name__)"
```

预期输出：

```text
LingjingSoloAgent
```

## 4. 配置 ARC 在线 API

在 `ARC-AGI-3-Agents` 目录：

```bash
cp .env.example .env
chmod 600 .env
```

编辑 `.env`，至少设置：

```dotenv
SCHEME=https
HOST=arcprize.org
PORT=443
OPERATION_MODE=online
ARC_API_KEY=<自己的 ARC API key>
```

每个成员应使用自己的 API key。不要把 key 发到群聊、写进 README、提交到 Git 或打印到日志。

只检查 key 是否配置，不打印 key 内容：

```bash
uv run python -c \
  "from pathlib import Path; p=Path('.env'); s=p.read_text(); print('ARC_API_KEY configured:', any(x.startswith('ARC_API_KEY=') and x.split('=',1)[1].strip() for x in s.splitlines()))"
```

预期：

```text
ARC_API_KEY configured: True
```

## 5. 运行前清除实验覆盖变量

默认 adaptor 计划覆盖 Level 1 → Level 7（罐头解，共 309 步）。运行前清除可能覆盖默认路线的变量：

```bash
unset LINGJING_LS20_PLAN
unset LINGJING_EXPERIMENT_ACTIONS
```

不要在第一次复现时猜测并设置 `ACTION1,ACTION2,...`。这些变量只用于单动作或自定义路线实验，错误设置会覆盖默认 L1-L7 计划。

## 6. 正式运行 LS20

```bash
cd ARC-AGI-3-Agents

env -u LINGJING_LS20_PLAN \
    -u LINGJING_EXPERIMENT_ACTIONS \
    uv run main.py \
      --agent=lingjingsolo \
      --game=ls20 \
      --tags=lingjing-solo,explore-plan
```

这条命令会：

1. 连接 ARC 在线环境；
2. 创建 LS20 游戏；
3. 调用 `lingjingsolo` adaptor；
4. 按默认罐头计划依次执行 Level 1–7（每关独立播种、按 `levels_completed` 自动续跑，直接弹动作执行，不做运动学失效检查）；
5. 结束游戏并关闭 Scorecard；
6. 输出新的 Scorecard ID。

每次运行都会生成新的 Scorecard ID。Scorecard URL 格式为：

```text
https://arcprize.org/scorecards/<scorecard-id>
```

## 7. 结果验收

从 runner 输出中检查：

```text
scorecard: <新的 UUID>
levels_completed: 7
score: <大于 0>
state: WIN
```

最新已验证的真实运行结果（2026-09-02）：

```text
scorecard: 5fcf8efa-6932-4243-951d-d72521311b40
levels_completed: 7
total_actions: 309
score: 100.0
state: WIN
```

Scorecard 地址：https://arcprize.org/scorecards/5fcf8efa-6932-4243-951d-d72521311b40

已实测：官方 `main.py --agent=lingjingsolo --game=ls20` 一次跑到 **L7 pass（7/7）**，score=100.0，state=WIN。全部 7 关均有确定性罐头解。

运行记录通常位于：

```text
ARC-AGI-3-Agents/recordings/
```

## 8. 运行本地测试

### Lingjing-Solo 核心测试

```bash
cd Lingjing-Solo-
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
```

已验证基线：

```text
21 passed, 1 warning
```

### ARC adaptor 定向测试

```bash
cd ../ARC-AGI-3-Agents
uv run pytest -q tests/unit/test_lingjing_solo_agent.py
```

### L1-L7 罐头解本地引擎验证（离线，不消耗 Scorecard）

罐头解已内嵌在 adaptor 中，可离线重放验证，无需线上运行。前提：ARC checkout 里有本地 ls20 环境文件 `environment_files/ls20/9607627b`，并把 `arc_agi.Arcade` 切到 `OperationMode.OFFLINE`。本地脚本会直接弹动作执行 L1-L7 计划，预期输出逐关完成、309 步全通。

如果要运行 ARC 仓库完整测试：

```bash
uv run pytest -q
```

完整测试集可能包含与 Lingjing adaptor 无关的仓库既有问题；真实 Level 1/2 验收仍以定向测试加在线 runner 的 `levels_completed`、`score` 和 Scorecard 为准。

## 9. 常见问题

### 只 checkout `Lingjing-Solo-` 能不能运行？

不能完成官方 Scorecard 流程。Lingjing-Solo 是核心 package，官方 ARC `main.py` 和 `Agent` 接口在 `ARC-AGI-3-Agents`。

### 只 checkout 官方 ARC `main` 能不能运行 Lingjing Agent？

不一定。必须确认官方 checkout 中存在 `lingjing_solo_agent.py` 并注册了 `lingjingsolo`。如果没有，需要 checkout 团队 adaptor 分支/commit。

### 生成 Scorecard 但 `levels_completed=0` 怎么办？

先确认：

```bash
grep -R "lingjingsolo" agents main.py
env | grep '^LINGJING_'
git -C ../Lingjing-Solo- branch --show-current
```

然后确保没有设置 `LINGJING_LS20_PLAN` 或 `LINGJING_EXPERIMENT_ACTIONS`，重新执行第 6 节的命令。

### 能否把 API key 写在命令行里？

不建议。命令行可能进入 shell history 或进程列表。使用 ARC 仓库内权限为 `600` 的 `.env`。

## 10. 复现完成清单

```text
[ ] 两个仓库都已 checkout
[ ] ARC 仓库包含并注册 Lingjing-Solo adaptor
[ ] uv sync 成功
[ ] uv pip install -e ../Lingjing-Solo- 成功
[ ] ARC_API_KEY 已配置且未泄露
[ ] OPERATION_MODE=online
[ ] LINGJING_LS20_PLAN 未设置
[ ] LINGJING_EXPERIMENT_ACTIONS 未设置
[ ] runner exit code = 0
[ ] 输出新的 Scorecard ID
[ ] levels_completed >= 7
[ ] score > 0
[ ] 保存 Scorecard URL 和 recording 路径
```

## 11. 当前限制

- 本文档的真实 L1-L7 复现依赖 ARC adaptor 已被共享；当前 adaptor 位于 `arc_adaptor/`（分支 `fix/ls20-plan-reseed`），需手动复制进各成员的官方 ARC checkout，尚未自动进入。
- 全部 7 关均有确定性罐头解（L1-L7 共 309 步），已在线上验证 score=100.0, state=WIN。
- `exit 0`、recording 或 Scorecard 生成本身不能替代 Level pass 证据。
- 不同成员必须使用匹配的 adaptor commit 和 Lingjing-Solo commit；只比较仓库名称不足以保证结果一致。
