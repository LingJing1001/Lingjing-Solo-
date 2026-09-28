# 协作与分支规范（四人）

仓库：https://github.com/liaomile/ARC-AGI-3  
默认分支：`main`（**只接受 Pull Request 合并**，禁止直接推送业务改动）

---

## 1. 长期分支（认人）

| 分支 | 角色 | 职责范围 |
|------|------|----------|
| `main` | 全员（只读+PR） | 可交榜的稳定主线 |
| `team/lead` | 负责人 | 整合发版、路由表审批、Kaggle 提交 |
| `team/a-floors` | 伙伴 A | ls20/ar25/ft09 地板与回归 |
| `team/b-delivery` | 伙伴 B | Phase B / notebook / 提交工程 |
| `team/c-unknown` | 伙伴 C | 未知局挖分与对照实验 |

每人先：

```bash
git clone https://github.com/liaomile/ARC-AGI-3.git
cd ARC-AGI-3
git checkout team/<你的分支>
git pull
```

---

## 2. 日常开发（谁改的一目了然）

**不要**在 `team/*` 上堆很多无关提交。请从 `main` 拉短生命周期功能分支：

```text
feat/<角色缩写>-<简述>
```

示例：

- `feat/a-ft09-regression`
- `feat/b-phaseb-checklist`
- `feat/c-sc25-probe`
- `feat/lead-router-v1.1`

规则：

1. 分支名必须以 `feat/a-` / `feat/b-` / `feat/c-` / `feat/lead-` 开头（审核时靠前缀认人）。  
2. Commit message 建议带角色前缀，例如：`[A] restore ls20 L7 53 steps`。  
3. 一个 PR 只做一件事；过闸门再合并（见分工文档 G1–G4）。

```bash
git fetch origin
git checkout main
git pull origin main
git checkout -b feat/a-xxx
# ... 改代码 ...
git add -A
git commit -m "[A] short why"
git push -u origin feat/a-xxx
```

然后在 GitHub 开 **Pull Request → base: `main`**。

---

## 3. 审核与合并（必须）

1. **Assignee**：作者自己  
2. **Reviewer**：至少 1 人  
   - A 的 PR → 负责人或 B 审  
   - B 的 PR → 负责人审  
   - C 的 PR → 负责人审（涉及 `ROUTE_INLINE` 必须负责人批）  
   - 负责人的 PR → A 或 B 审  
3. **合并方式**：GitHub 上 **Squash and merge**（或 Merge commit），保留 PR 标题里的角色信息  
4. **合并后**：删除已合并的 `feat/*` 远程分支；`team/*` 长期保留，定期 `git merge main` 同步  

**禁止**

- 直接 `git push origin main` 业务代码（负责人紧急热修除外，事后补 PR 说明）  
- 未本地复现 WIN 就改 INLINE 序列  
- 把 SSA/neural 未经对照直接换主路径  

---

## 4. GitHub 网页设置（负责人做一次）

仓库 → **Settings → Collaborators** ：把三位伙伴加成 Write。  

仓库 → **Settings → Branches → Add rule**（保护 `main`）：

- Require a pull request before merging  
- Require approvals: **1**  
- Do not allow bypassing the above settings（可选）  
- Restrict who can push to matching branches：清空直接推送，或仅管理员  

（若暂无权限改 Settings，先靠约定：所有人只推 `feat/*` / `team/*`。）

---

## 5. 与分工文档的对应

详见 `docs/冲榜技术说明与四人分工.md`。

| 角色 | 分支前缀 | 合并进 main 前最低验收 |
|------|----------|------------------------|
| A | `feat/a-` | 三地板 WIN 回归绿 |
| B | `feat/b-` | Phase B 检查清单 / 指纹说明 |
| C | `feat/c-` | 对照表；WIN 才动路由 |
| 负责人 | `feat/lead-` | 发版说明 + BUILD_TAG |

---

## 6. 伙伴 GitHub 账号

| 角色 | 邮箱 | GitHub 用户名 | 长期分支 |
|------|------|---------------|----------|
| 负责人 | （待填） | @liaomile | `team/lead` |
| 伙伴 A · 地板 | **aiff@outlook.com** | （待对方提供） | `team/a-floors` |
| 伙伴 B · 交付 | （待填） | （待填） | `team/b-delivery` |
| 伙伴 C · 未知 | （待填） | （待填） | `team/c-unknown` |

完整花名册见 [`TEAM.md`](TEAM.md)。伙伴 A 提供 GitHub 用户名后，补进 `.github/CODEOWNERS`。
