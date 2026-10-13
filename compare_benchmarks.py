#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比两个 benchmark 结果"""
import json

def compare_benchmarks(baseline_file, optimized_file):
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(optimized_file) as f:
        optimized = json.load(f)

    print('=' * 80)
    print('📊 BENCHMARK 对比分析')
    print('=' * 80)
    print(f'基线: {baseline["label"]}')
    print(f'优化: {optimized["label"]}')
    print()

    baseline_results = {r['game_id'].split('-')[0]: r for r in baseline['results']}
    optimized_results = {r['game_id'].split('-')[0]: r for r in optimized['results']}
    all_games = sorted(set(list(baseline_results.keys()) + list(optimized_results.keys())))

    total_baseline = 0
    total_optimized = 0
    degraded_games = ['cn04', 'g50t', 'ka59', 'lf52', 'm0r0']
    improved = []
    worsened = []

    print(f'{"Game":<8} | {"Baseline":>8} | {"Optimized":>9} | {"Delta":>6} | {"Status":>12}')
    print('-' * 80)

    for gid in all_games:
        b_res = baseline_results.get(gid, {'final_levels': 0, 'elapsed_s': 0})
        o_res = optimized_results.get(gid, {'final_levels': 0, 'elapsed_s': 0})

        b_levels = b_res['final_levels']
        o_levels = o_res['final_levels']
        delta = o_levels - b_levels

        total_baseline += b_levels
        total_optimized += o_levels

        if delta > 0:
            status = '✅ IMPROVED'
            improved.append(gid)
        elif delta < 0:
            status = '❌ DEGRADED'
            worsened.append(gid)
        elif gid in degraded_games and b_levels == 0:
            status = '⚠️ STILL 0'
        else:
            status = '➡️ SAME'

        print(f'{gid:<8} | {b_levels:>8} | {o_levels:>9} | {delta:>+6} | {status:>12}')

    print('-' * 80)
    print(f'{"TOTAL":<8} | {total_baseline:>8} | {total_optimized:>9} | {total_optimized-total_baseline:>+6}')
    print()

    print('=' * 80)
    print('📈 关键发现:')
    print('=' * 80)
    print(f'总关卡数: 基线 {total_baseline} → 优化 {total_optimized} (变化: {total_optimized-total_baseline:+d})')
    print()
    print(f'✅ 改进的游戏 ({len(improved)}个): {improved if improved else "无"}')
    print(f'❌ 退化的游戏 ({len(worsened)}个): {worsened if worsened else "无"}')
    print()
    print('⚠️  退化游戏详情 (cn04, g50t, ka59, lf52, m0r0):')
    for gid in degraded_games:
        b = baseline_results.get(gid, {})
        o = optimized_results.get(gid, {})
        print(f'   {gid}: 基线={b.get("final_levels", 0)}关 ({b.get("elapsed_s", 0):.1f}s) → '
              f'优化={o.get("final_levels", 0)}关 ({o.get("elapsed_s", 0):.1f}s)')

if __name__ == '__main__':
    import sys
    baseline_file = sys.argv[1] if len(sys.argv) > 1 else 'ui/static/games_benchmark_post_main_merge_offline_400.json'
    optimized_file = sys.argv[2] if len(sys.argv) > 2 else 'ui/static/games_benchmark_optimized.json'
    compare_benchmarks(baseline_file, optimized_file)