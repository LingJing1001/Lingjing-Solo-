# Lingjing-Solo- GitHub 协作操作指南

本文档面向 Linux 和 Windows 开发者，说明如何获取 Lingjing-Solo- 项目、创建开发分支、同步代码、提交并推送，以及通过 Pull Request 合并到 `main`。

项目地址：<https://github.com/LingJing1001/Lingjing-Solo->

> **与 Starter / 提交对齐**：分数与 ScriptBank 权威说明见仓库根目录 [`CURRENT_STATUS.md`](../CURRENT_STATUS.md)、[`docs/SYNC_AND_SUBMIT.md`](../docs/SYNC_AND_SUBMIT.md)。拉取 `main` 后务必重新同步 `planning/data/*_scripts.json` 与桥接改动。

## 1. 协作规则

- `main` 是稳定分支，不直接提交代码。
- 每项功能、修复或文档修改都从最新的 `main` 创建独立分支。
- 开发者将分支推送到 GitHub 后创建 Pull Request（PR）。
- PR 必须经过至少一名成员审批，并通过要求的自动检查，才能合并到 `main`。
- 不要把密码、Token、私钥、API Key 或本地配置文件提交到仓库。
- 提交信息应说明实际变更，建议使用中文，例如：`完善项目安装与运行文档`。

## 2. 首次准备

### 2.1 安装 Git

Linux（Debian/Ubuntu）：

```bash
sudo apt update
sudo apt install -y git
```

Windows：

- 安装 [Git for Windows](https://git-scm.com/download/win)。
- 可使用 **Git Bash** 或 **PowerShell**。

检查安装：

```bash
git --version
```

### 2.2 配置提交身份

在每台开发机器上配置一次：

```bash
git config --global user.name "你的姓名"
git config --global user.email "你的 GitHub 邮箱"
```

检查配置：

```bash
git config --global --list
```

提交身份不必与 GitHub 昵称完全相同，但邮箱应尽量使用已经添加到 GitHub 账户的邮箱，这样提交可以关联到个人账户。

## 3. 首次 checkout 项目

### 3.1 Linux

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/LingJing1001/Lingjing-Solo-.git
cd Lingjing-Solo-
git switch main
git pull --ff-only origin main
```

### 3.2 Windows PowerShell

```powershell
New-Item -ItemType Directory -Force "$HOME\projects" | Out-Null
Set-Location "$HOME\projects"
git clone https://github.com/LingJing1001/Lingjing-Solo-.git
Set-Location ".\Lingjing-Solo-"
git switch main
git pull --ff-only origin main
```

如果 GitHub 要求登录：

- 推荐使用 GitHub CLI：`gh auth login`。
- HTTPS 不再接受 GitHub 账户密码作为 Git 密码，应使用 Personal Access Token 或 Git Credential Manager。
- 也可以配置 SSH，然后使用 SSH 地址克隆：

```bash
git clone git@github.com:LingJing1001/Lingjing-Solo-.git
```

## 4. 创建开发分支

开始新工作前，先同步 `main`：

```bash
git switch main
git pull --ff-only origin main
```

创建并切换到新分支：

```bash
git switch -c feature/简短名称
```

推荐分支命名：

```text
feature/xxx       新功能
fix/xxx           Bug 修复
docs/xxx          文档修改
refactor/xxx      重构
experiment/xxx    实验性工作
```

示例：

```bash
git switch -c docs/github-workflow
```

确认当前分支：

```bash
git branch --show-current
git status
```

## 5. 开发前和开发中的常用操作

查看状态：

```bash
git status
```

查看本地分支：

```bash
git branch
```

查看远程分支：

```bash
git branch -r
```

从远程获取最新信息，但不修改当前文件：

```bash
git fetch origin
```

查看当前分支与远程分支的差异：

```bash
git diff origin/main...HEAD
```

## 6. 提交代码

先查看修改内容：

```bash
git status
git diff
```

运行项目测试。当前项目可使用：

```bash
python test_solo.py
python notebook_template.py
```

只添加明确需要提交的文件：

```bash
git add path/to/file.py
```

确认暂存区内容：

```bash
git diff --cached
```

提交，使用中文提交信息：

```bash
git commit -m "完善 GitHub 协作操作文档"
```

查看最近提交：

```bash
git log --oneline -5
```

### 不小心添加了文件怎么办？

从暂存区移除，但保留本地文件：

```bash
git restore --staged path/to/file
```

撤销尚未提交的文件修改（谨慎使用）：

```bash
git restore path/to/file
```

## 7. Push 到 GitHub

第一次推送当前分支：

```bash
git push -u origin feature/你的分支名
```

例如：

```bash
git push -u origin docs/github-workflow
```

以后在同一分支继续推送：

```bash
git push
```

检查远程跟踪关系：

```bash
git branch -vv
```

常见错误：

### `src refspec ... does not match any`

通常是分支名称拼写错误，或者本地还没有该分支：

```bash
git branch --show-current
git branch
```

如果分支还不存在：

```bash
git switch -c feature/正确的分支名
git push -u origin feature/正确的分支名
```

### `rejected` 或远程分支领先

先同步远程分支，再重放本地提交：

```bash
git fetch origin
git rebase origin/你的分支名
git push
```

如果发生冲突，参见第 10 节。不要随意对共享分支使用 `git push --force`。

## 8. 创建 Pull Request

推送成功后，在 GitHub 仓库页面：

1. 打开 **Pull requests**。
2. 点击 **New pull request**。
3. Base repository 选择 `Lingjing-Solo-`。
4. Base branch 选择 `main`。
5. Compare branch 选择你的开发分支。
6. 检查变更文件和差异。
7. 填写清楚的 PR 标题和说明。
8. 指定 Reviewer，点击 **Create pull request**。

PR 描述建议包括：

```markdown
## 变更内容
- 做了什么

## 测试结果
- `python test_solo.py`：通过
- `python notebook_template.py`：通过

## 注意事项
- 是否需要配置环境变量
- 是否有已知限制
```

## 9. 审批和合并流程

`main` 分支建议配置 Branch protection 或 Ruleset：

- 必须通过 Pull Request。
- 至少需要 1 名成员批准。
- 新提交后旧的审批可以自动失效。
- 必须解决 Review 评论。
- 必须通过 CI 测试（如果仓库配置了 Actions）。
- 禁止直接 push 到 `main`。

PR 创建后：

1. 作者自查代码、测试和敏感信息。
2. Reviewer 检查设计、正确性、测试和兼容性。
3. Reviewer 选择 **Approve**、**Request changes** 或 **Comment**。
4. 作者根据意见修改代码并再次 push 到同一个分支。
5. 检查通过且审批满足要求后，由有权限的成员点击 **Merge pull request**。
6. 合并后可删除已经完成的远程分支，但不要删除仍在使用的分支。

开发者通常不需要重新创建 PR；对同一个分支再次 `git push`，PR 会自动更新。

## 10. 处理合并冲突

如果 PR 提示与 `main` 冲突：

```bash
git switch 你的开发分支
git fetch origin
git rebase origin/main
```

打开冲突文件，保留正确内容并删除冲突标记：

```text
[当前分支内容]
[分隔线]
[origin/main 内容]
```

然后：

```bash
git add 冲突文件
git rebase --continue
```

重复处理直到 rebase 完成，运行测试后推送：

```bash
git push --force-with-lease
```

`--force-with-lease` 只用于你自己的开发分支，并且比 `--force` 安全。不要对 `main` 或其他成员共享的分支执行强制推送。

如果想取消正在进行的 rebase：

```bash
git rebase --abort
```

## 11. 合并后开始下一项工作

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/下一项工作
```

删除已经合并的本地分支：

```bash
git branch -d 已合并的分支名
```

删除远程分支（确认不再需要后）：

```bash
git push origin --delete 已合并的分支名
```

## 12. Linux / Windows 注意事项

- Linux 路径使用 `/home/user/projects/...`；PowerShell 路径通常使用 `C:\Users\用户名\projects\...`。
- Windows 项目建议配置统一换行符，避免整个文件出现无意义的换行变化：

```bash
git config --global core.autocrlf true
```

- Linux 开发者可以使用：

```bash
git config --global core.autocrlf input
```

- 不要提交 `.venv/`、`__pycache__/`、编辑器配置、日志和本地密钥。
- 提交前检查：

```bash
git status
git diff --cached
```

## 13. 推荐的完整日常流程

```bash
# 进入项目
git switch main
git pull --ff-only origin main

# 创建工作分支
git switch -c feature/my-change

# 修改代码后检查
git status
git diff
python test_solo.py
python notebook_template.py

# 提交并推送
git add path/to/changed-file
git commit -m "实现某项功能"
git push -u origin feature/my-change

# 在 GitHub 创建 PR，等待审批和检查
# PR 合并后清理本地分支
git switch main
git pull --ff-only origin main
git branch -d feature/my-change
```
