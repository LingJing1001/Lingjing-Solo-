"""AR25 求解器命令行入口。

用法:
    python -m ar25_solver                      # 求解所有关卡
    python -m ar25_solver --level 3            # 求解L3
    python -m ar25_solver --strategy astar     # 指定A*搜索
    python -m ar25_solver --strategy macro     # 指定宏搜索
    python -m ar25_solver --save out.json      # 保存解法
    python -m ar25_solver --quiet               # 静默模式
    python -m ar25_solver --data custom.json   # 自定义关卡数据
"""
import argparse
import json
import sys

from . import (
    load_levels, solve, solve_all, verify,
    AR25Simulator, Advisor, __version__,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ar25_solver",
        description="AR25 通用求解器 — 纯Python逆向实现，不依赖官方引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m ar25_solver                      求解所有关卡
  python -m ar25_solver --level 3            求解L3
  python -m ar25_solver --strategy macro     使用宏搜索
  python -m ar25_solver --save result.json   保存解法到文件
  python -m ar25_solver --quiet              静默模式(仅输出结果)

API 调用:
  from ar25_solver import load_levels, solve, verify
  levels = load_levels()
  actions = solve(levels[0], strategy="auto")
  print(verify(levels[0], actions))
""",
    )
    parser.add_argument("--level", type=int, default=None,
                        help="指定关卡编号(1-based)，不指定则求解所有")
    parser.add_argument("--strategy", choices=["auto", "astar", "macro"],
                        default="auto", help="求解策略 (默认: auto)")
    parser.add_argument("--t-limit", type=int, default=120,
                        help="每关时间限制秒数 (默认: 120)")
    parser.add_argument("--data", default=None,
                        help="关卡数据JSON路径 (默认: 内置数据)")
    parser.add_argument("--save", default=None,
                        help="保存解法到JSON文件")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="静默模式，仅输出最终结果")
    parser.add_argument("--advisor", action="store_true",
                        help="启用LLM反思顾问 (步数阈值触发自我审视)")
    parser.add_argument("--version", "-v", action="version",
                        version=f"ar25_solver {__version__}")

    args = parser.parse_args(argv)
    verbose = not args.quiet

    levels = load_levels(args.data)
    advisor = True if args.advisor else None

    if args.level is not None:
        idx = args.level - 1
        if idx < 0 or idx >= len(levels):
            print(f"关卡 {args.level} 不存在 (共 {len(levels)} 关)")
            return 1

        actions = solve(levels[idx], strategy=args.strategy,
                        t_limit=args.t_limit, verbose=verbose,
                        advisor=advisor)

        if actions:
            result = verify(levels[idx], actions)
            status = "通关" if result["won"] else "未通关"
            print(f"\n结果: {status} {result['steps']}步 "
                  f"({result['covered']}/{result['total']}覆盖)")
            if verbose:
                print(f"动作序列: {actions}")
            if args.save:
                with open(args.save, "w") as f:
                    json.dump({"level": args.level, "actions": actions,
                               "result": result}, f, indent=2)
                print(f"已保存到 {args.save}")
            return 0 if result["won"] else 1
        else:
            print("\n结果: ❌ 未找到解")
            return 1
    else:
        results = solve_all(levels, strategy=args.strategy,
                            t_limit=args.t_limit, verbose=verbose,
                            advisor=advisor)

        if args.save:
            with open(args.save, "w") as f:
                json.dump({str(k): v for k, v in results.items()},
                           f, indent=2)
            print(f"\n解法已保存到 {args.save}")

        won_count = sum(1 for v in results.values() if v)
        print(f"\n通关: {won_count}/{len(levels)}")
        return 0 if won_count == len(levels) else 1


if __name__ == "__main__":
    sys.exit(main())
