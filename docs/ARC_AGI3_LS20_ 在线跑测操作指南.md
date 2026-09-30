# ARC-AGI-3 LS20 L1-L7 复现指南

本文说明团队成员如何从干净环境运行 Lingjing-Solo Agent，并在 ARC 在线环境中生成 LS20 Scorecard，目标是复现 **Level 1–7 pass（7/7）**。L1-L7 罐头解（共 309 步）已通过本地引擎离线重放与官方线上运行双重验证。

> 本文档只记录已经验证过的运行链路。`exit 0` 或生成 Scorecard 只证明 runner 链路可用；只有 `levels_completed >= 7` 和正分才证明 Level 1–7 实际通过。

## 1. 重要前提：需要两个仓库

真实 ARC runner 不是只运行 `Lingjing-Solo-`。需要同时准备：

| 仓库 | 作用 |
|---|---|
| `Lingjing-Solo-` | 核心 Agent、感知、规划和可复用策略核心 |
| `ARC-AGI-3-Agents` | 官方 ARC `Agent` adaptor、runner 和 Scorecard 提交入口 |

仓库地址：

```text
https://github.com/LingJing1001/Lingjing-Solo-
https://github.com/arcprize/ARC-AGI-3-Agents
```

### 当前协作限制

`ARC-AGI-3-Agents` 中的 Lingjing-Solo reproduction bundle 已同步到本仓库 `arc_adaptor/`。当前新架构基准分支为 `feature/arc-strategy-registry`，包含 L1-L7 罐头解、strategy registry、测试、调试工具和可选 patch。其他成员应独立 checkout 官方 ARC 仓库到自己的本地目录，再使用 bundle 的同步脚本复制到本地 checkout 中。

```text
Lingjing-Solo-/arc_adaptor/agents/templates/lingjing_solo_agent.py
Lingjing-Solo-/arc_adaptor/agents/strategies/
Lingjing-Solo-/arc_adaptor/tests/
Lingjing-Solo-/arc_adaptor/tools/
Lingjing-Solo-/arc_adaptor/patches/
```

这些文件保持 ARC 仓库中的相对目录结构。其他成员 checkout 两个仓库后，运行 `sync_to_arc.sh` 即可同步 adaptor、测试和调试工具。完整清单及版本基准见 `arc_adaptor/MANIFEST.md`。

## 1.1 独立 checkout ARC 仓库并复制 adaptor

每个人独立 checkout 官方 ARC 仓库到自己的本地目录，然后从本项目的 reproduction bundle 同步文件。这里不依赖 ARC 仓库上的 shared adaptor branch：

```bash
cd ~/projects

git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents

bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh
```

此时可以确认 adaptor、测试和调试工具已经进入本地 ARC checkout：

```bash
git status --short

test -f agents/templates/lingjing_solo_agent.py
grep -n "lingjing_solo_agent\|lingjingsolo" agents/__init__.py
test -f agents/strategies/registry.py
test -f agents/strategies/ls20.py
test -f tests/unit/test_lingjing_solo_agent.py
test -f tests/unit/test_action_recording.py
test -f tools/ls20_single_action_probe.py
```

预期 `git status` 显示 adaptor、测试和工具文件被修改/新增，并且能看到 adaptor 文件和 `lingjingsolo` 注册导入。

> `agents/__init__.py` 保留 ARC 仓库原生版本。同步脚本不会覆盖它；同步后请确认该文件注册了 `LingjingSolo`，并且包含 `lingjingsolo` 名称。

> 如果 ARC checkout 中已有本地修改，先处理或保存这些修改，再执行同步；同步会覆盖 ARC checkout 中的同名 adaptor、strategy、测试和工具文件，但不会覆盖原生 `agents/__init__.py`。脚本不会复制 `.env`、API key、虚拟环境或缓存。

### 可选 recording patch

`arc_adaptor/patches/arc-agent-recording.patch` 会让 ARC recording 额外记录 requested action。它不是 LS20 正式运行的必要条件，只在需要审计 action recording 时应用：

```bash
cd ../ARC-AGI-3-Agents
bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh . --with-recording-patch
```

如果不需要 recording 增强，不要应用该 patch。测试 `test_action_recording.py` 只有在应用 patch 后才运行。

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

使用 `feature/arc-strategy-registry` 作为新架构的团队统一复现基线：

```bash
git fetch origin
git switch --track -c feature/arc-strategy-registry origin/feature/arc-strategy-registry
```

如果该本地分支已经存在，则直接切换并快进更新：

```bash
git switch feature/arc-strategy-registry
git pull --ff-only origin feature/arc-strategy-registry
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
      --game=ls20-9607627b \
      --tags=restructure,ls20-reverify
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
scorecard: 904b4c76-f8bf-44fa-a218-b4cd665060f6
levels_completed: 7
total_actions: 309
resets: 0
score: 100.0
state: WIN
```

Scorecard 地址：https://arcprize.org/scorecards/904b4c76-f8bf-44fa-a218-b4cd665060f6

已实测：官方 `main.py --agent=lingjingsolo --game=ls20-9607627b` 一次跑到 **L7 pass（7/7）**，score=100.0，state=WIN。全部 7 关均有确定性罐头解。

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

### ARC adaptor 和 bundle 定向测试

```bash
cd ../ARC-AGI-3-Agents
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/unit/test_lingjing_solo_agent.py
```

如果已应用可选 recording patch，再运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/unit/test_lingjing_solo_agent.py \
  tests/unit/test_action_recording.py
```

### 单动作调试工具（可选，在线）

该工具每次创建新游戏并执行一个动作，会产生远程环境访问；只在需要诊断动作行为时使用：

```bash
cd ../ARC-AGI-3-Agents
uv run python tools/ls20_single_action_probe.py ACTION1
```

### L1-L7 罐头解本地引擎验证（离线，不消耗 Scorecard）

罐头解已内嵌在 adaptor 中，可离线重放验证，无需线上运行。前提：ARC checkout 里有本地 LS20 环境文件 `environment_files/ls20/9607627b`，并把 `arc_agi.Arcade` 切到 `OperationMode.OFFLINE`。当前 bundle 不包含该环境文件；它必须通过 ARC 官方允许的方式单独取得，不能从仓库或他人机器复制 API/受限数据。

如果要运行 ARC 仓库完整测试：

```bash
uv run pytest -q
```

完整测试集可能包含与 Lingjing adaptor 无关的仓库既有问题；真实 L1-L7 验收仍以定向测试加在线 runner 的 `levels_completed`、`score` 和 Scorecard 为准。

## 9. 常见问题

### 只 checkout `Lingjing-Solo-` 能不能运行？

不能完成官方 Scorecard 流程。Lingjing-Solo 是核心 package，官方 ARC `main.py` 和 `Agent` 接口在 `ARC-AGI-3-Agents`。

### 只 checkout 官方 ARC `main` 能不能运行 Lingjing Agent？

不一定。必须运行 `arc_adaptor/sync_to_arc.sh`，确认官方 checkout 中存在 `lingjing_solo_agent.py` 并注册了 `lingjingsolo`。方案 A 不要求在 ARC 仓库使用团队分支。

### 生成 Scorecard 但 `levels_completed=0` 怎么办？

先确认：

```bash
grep -R "lingjingsolo" agents main.py
env | grep '^LINGJING_'
git -C ../Lingjing-Solo- branch --show-current
```

然后确保没有设置 `LINGJING_LS20_PLAN` 或 `LINGJING_EXPERIMENT_ACTIONS`，重新执行第 6 节的命令。若要检查 adaptor 测试，先确认是否应用了 optional recording patch。

### 能否把 API key 写在命令行里？

不建议。命令行可能进入 shell history 或进程列表。使用 ARC 仓库内权限为 `600` 的 `.env`。

## 10. 复现完成清单

```text
[ ] Lingjing-Solo checkout `feature/arc-strategy-registry`
[ ] 两个仓库 commit 基准已记录在 `arc_adaptor/MANIFEST.md`
[ ] ARC 仓库包含并注册 Lingjing-Solo adaptor
[ ] `arc_adaptor/sync_to_arc.sh` 执行成功
[ ] ARC adaptor 测试通过
[ ] 可选 recording patch（如需要）通过 `git apply --check`
[ ] 可选调试工具已同步且无 `__pycache__`/`.pyc`
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

- 本文档的真实 L1-L7 复现依赖 ARC reproduction bundle 已被共享；当前 bundle 位于 `arc_adaptor/`（分支 `feature/arc-strategy-registry`，重构 commit `2276016`，文档移动 commit `fc7f4b0`），各成员需在自己的官方 ARC checkout 运行同步脚本。
- `test_action_recording.py` 依赖可选的 `arc-agent-recording.patch`；不应用 patch 时不要运行该测试。
- `ls20_single_action_probe.py` 是可选在线诊断工具，每次调用会创建新的远程游戏；正式跑测前不需要运行。
- 全部 7 关均有确定性罐头解（L1-L7 共 309 步），已在线上验证 score=100.0, state=WIN。
- `exit 0`、recording 或 Scorecard 生成本身不能替代 Level pass 证据。
- 不同成员必须使用匹配的 adaptor commit 和 Lingjing-Solo commit；只比较仓库名称不足以保证结果一致。
