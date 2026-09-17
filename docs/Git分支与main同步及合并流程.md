# Git 分支与 `main` 同步及合并流程

本文说明当功能分支和 `main` 分支都存在新提交时，如何安全地同步和合并。

> 文中的 `feature/r4-hotspot-detection` 只是示例分支名，不代表固定流程必须使用这个名称。实际操作时，应替换为当前正在开发的功能分支名称，例如 `feature/<功能名>`、`fix/<问题名>` 或项目约定的其他分支名。

## 一、两个方向的含义

需要区分两个不同动作：

```text
最新 main  →  功能分支
功能分支   →  main
```


| 动作            | 目的                           | 一般发生在什么时候     |
| ------------- | ---------------------------- | ------------- |
| `main` → 功能分支 | 让功能分支获得 `main` 的最新改动，并提前解决冲突 | 开发中、准备提 PR 前  |
| 功能分支 → `main` | 将已完成并验证的功能发布到主分支             | PR 审查和 CI 通过后 |


当两个分支都有更新时，通常不是二选一，而是按顺序执行这两个动作：

1. 先把最新的 `main` 同步到功能分支。
2. 在功能分支上解决冲突并运行测试。
3. 测试通过后，再通过 Pull Request 将功能分支合并到 `main`。

## 二、推荐流程

以下命令中的 `feature/r4-hotspot-detection` 仅为示例。请替换成实际功能分支名。

### 1. 获取远程最新状态

```bash
git fetch origin
```

查看远程 `main` 和当前分支的状态：

```bash
git log --oneline --decorate --graph --all -20
git branch -vv
```



### 2. 切换到功能分支并确认工作树干净

```bash
git switch feature/r4-hotspot-detection
git status
```

如果有未提交改动，应先根据需要提交、暂存或保存：

```bash
git add <文件>
git commit -m "<说明>"
```

或者临时保存：

```bash
git stash push -m "temporary work before syncing main"
```

不要在未确认工作树内容的情况下直接合并或 rebase。

### 3. 将最新 `main` 合并到功能分支

```bash
git merge origin/main
```

如果出现冲突：

```bash
git status
git diff --name-only --diff-filter=U
```

逐个解决冲突文件，删除冲突标记：

```text
[current-branch content]
[conflicting separator]
[origin/main content]
```

然后验证并完成合并：

```bash
git add <已解决的文件>
git diff --check
git commit
```

如果确认不应继续此次合并，可以撤销尚未完成的 merge：

```bash
git merge --abort
```



### 4. 在功能分支上验证

使用项目实际规定的测试和静态检查命令。例如：

```bash
pytest
ruff check .
git diff --check
```

如果项目有专门的集成测试、构建命令或运行检查，也应在此阶段执行。

必须区分：

- 合并前已有的基线失败；
- `main` 同步后新出现的失败；
- 冲突解决导致的新问题。



### 5. 推送更新后的功能分支

```bash
git push origin feature/r4-hotspot-detection
```

实际使用时，把示例分支名替换成当前功能分支名。

### 6. 创建 Pull Request 合并到 `main`

在 GitHub 上创建：

```text
实际功能分支 → main
```

例如：

```text
feature/<实际功能分支> → main
```

Pull Request 应至少确认：

- 功能分支已包含最新 `main`；
- 冲突已解决；
- 测试和 CI 通过；
- 代码审查完成；
- 没有把临时文件、凭据或不相关修改带入合并。

合并完成后，本地 `main` 可以同步：

```bash
git switch main
git pull --ff-only origin main
```



## 三、`merge` 和 `rebase` 的选择



### 使用 `merge`

```bash
git switch <实际功能分支>
git fetch origin
git merge origin/main
```

优点：

- 不改写已有提交；
- 保留真实的分支历史；
- 适合多人共享或已经推送的功能分支。



### 使用 `rebase`

```bash
git switch <实际功能分支>
git fetch origin
git rebase origin/main
```

优点：

- 提交历史更线性；
- 可以减少额外的 merge commit。

注意：rebase 会改写功能分支提交。如果该分支已经被其他人使用，或已经有其他工作基于它开发，不应未经沟通直接 rebase。rebase 后通常需要：

```bash
git push --force-with-lease origin <实际功能分支>
```

`--force-with-lease` 比 `--force` 更安全，但仍属于改写远程历史的操作。

## 四、常见错误做法



### 直接把旧功能分支合并到 `main`

如果功能分支没有先同步最新 `main`，可能会：

- 把冲突推迟到 `main` 或 Pull Request 阶段；
- 在过时基线上完成测试；
- 遗漏 `main` 的兼容性变化；
- 让审查者难以判断问题来自功能代码还是分支落后。



### 在有未提交改动时直接切换或合并

这可能造成：

- 本地改动与合并结果混在一起；
- 冲突范围扩大；
- 难以准确回滚。



### 把“合并成功”当成“功能完成”

Git merge 成功只说明版本历史已经连接，不代表代码正确。仍需运行测试、lint、构建和必要的集成检查。

## 五、最简决策规则

```text
开发过程中同步上游：      main → 功能分支
功能完成并验证后发布：    功能分支 → main
两边都有更新时：          先 main → 功能分支，再功能分支 → main
```

推荐原则：

> 先在功能分支上吸收最新 `main` 并完成验证，再把经过验证的功能分支合并回 `main`。