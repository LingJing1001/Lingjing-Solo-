# 团队花名册

仓库：https://github.com/liaomile/ARC-AGI-3

| 角色 | 联系邮箱 | GitHub | 长期分支 | 职责 |
|------|----------|--------|----------|------|
| 负责人 | （待填） | @liaomile（仓库所有者） | `team/lead` | 整合发版、路由审批、Kaggle 提交 |
| 伙伴 A · 地板 | **aiff@outlook.com** | （待对方提供用户名） | `team/a-floors` | ls20 / ar25 / ft09 锁分与回归 |
| 伙伴 B · 交付 | （待填） | （待填） | `team/b-delivery` | Phase B / notebook / 提交工程 |
| 伙伴 C · 未知 | （待填） | （待填） | `team/c-unknown` | 未知局挖分与对照 |

## 伙伴 A 入职清单

1. 用绑定 `aiff@outlook.com` 的 GitHub 账号接受仓库邀请（Write）。  
2. 把 **GitHub 用户名** 发回负责人，以便写入 `CODEOWNERS` 自动审 PR。  
3. 本地：

```bash
git clone https://github.com/liaomile/ARC-AGI-3.git
cd ARC-AGI-3
git checkout team/a-floors
git pull
```

4. 日常开分支：`feat/a-<简述>` → PR 合入 `main`，Reviewer 至少 1 人（负责人或 B）。  
5. 合并前必须：ls20 / ar25 / ft09 本地 WIN（见分工文档 G1）。
