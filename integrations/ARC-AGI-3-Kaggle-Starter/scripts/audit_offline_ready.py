"""离线就绪审计 CLI：正式比赛无网络，发货前跑一遍。

    .venv/Scripts/python.exe scripts/audit_offline_ready.py

退出码：0 = 就绪；1 = 有阻断项（会在比赛里静默降级或直接崩）。

它做四件事：
  1. 打一份离线环境变量清单（哪些已钉死、哪些还缺）
  2. 列出网络放行名单（比赛用的是容器内 `gateway:8001` 边车，不是全禁）
  3. 审计提交包 vs `agent/my_agent.py` 的真实导入闭包
  4. 把阻断项一句话一条打出来，方便直接抄进结论

只读 —— 不写文件、不改环境、不联网。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTER = HERE.parent
REPO = STARTER.parents[1]

sys.path.insert(0, str(REPO))

from lingjing_solo.harness import offline as O  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="离线（无网络）比赛就绪审计")
    ap.add_argument("--entry", default=str(STARTER / "agent" / "my_agent.py"))
    ap.add_argument("--notebook", default=str(STARTER / "notebooks" / "submission.ipynb"))
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是人读报告")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    report = O.readiness_report(repo, Path(args.entry), Path(args.notebook))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1

    audit = O.audit_packaging(Path(args.entry), Path(args.notebook), repo)
    print("=" * 72)
    print("离线就绪审计（enable_internet=false）")
    print("=" * 72)
    print()
    print(audit.render())
    print()

    print("-- 网络放行名单 --")
    print("   比赛走容器内边车，所以放行：" + ", ".join(O.DEFAULT_ALLOW_HOSTS))
    print("   （'没网络' ≠ '没 HTTP'；全禁会连游戏环境都拿不到）")
    print()

    changed = {k: v for k, v in O.OFFLINE_ENV.items()}
    print("-- 离线环境变量（应全部按此值钉死）--")
    for key in sorted(changed):
        print(f"   {key}={changed[key]}")
    print(f"   应用方式：lingjing_solo.harness.offline.apply_offline_env()")
    print()

    print("-- 数据资产 --")
    for asset in audit.assets:
        in_pkg = "在包内" if asset.payload_rel in audit.shipped_assets else "不在包内"
        exists = "存在" if asset.present else "缺失"
        size = f"{asset.size:,} B" if asset.present else "-"
        print(f"   [{exists}/{in_pkg}] {asset.key}  {asset.path.name}  {size}")
        if asset.why:
            print(f"          {asset.why}")
    print()

    print("-- 第三方硬依赖（必须由赛方 wheel / 镜像提供）--")
    for name in audit.hard_third_party:
        print(f"   {name}")
    print()

    print("=" * 72)
    if audit.ok:
        print("结论：OK —— 提交包与 agent 的真实依赖一致。")
        return 0
    print(f"结论：{len(audit.blockers())} 条阻断项")
    for item in audit.blockers():
        print("   * " + item)
    print()
    print("提示：缺模块多半是打包器收录集合没跟上代码；外部包补进")
    print("      build_notebook.EXTRA_ROOTS，大资产走 dataset_sources。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
