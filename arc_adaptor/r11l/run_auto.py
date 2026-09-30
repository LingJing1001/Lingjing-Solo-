#!/usr/bin/env python
"""R11L AI 自动闯关 - 健壮版（带重试、错误恢复）
用法: python run_r11l_auto.py
"""
import requests, json, time, sys, os

BASE = 'http://localhost:8089'
TASK_ID = 'r11l'
MAX_STEPS = 500
STEP_DELAY = 0.3  # 步间延迟
REQUEST_TIMEOUT = 240  # 单次请求超时（秒）


def start_game():
    """开启新游戏"""
    r = requests.post(f'{BASE}/api/proxy/arc-start',
        json={'task_id': TASK_ID},
        headers={'Content-Type': 'application/json'},
        timeout=60)
    if r.status_code != 200:
        return None, f"arc-start失败: {r.status_code} {r.text[:200]}"
    data = r.json()
    return data, None


def auto_step():
    """单步 AI 执行，返回 (data, error)"""
    try:
        r = requests.post(f'{BASE}/api/proxy/arc-auto',
            json={'task_id': TASK_ID, 'mode': 'astar'},
            headers={'Content-Type': 'application/json'},
            timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return None, "请求超时"
    except requests.exceptions.ConnectionError as e:
        return None, f"连接错误: {e}"

    if r.status_code != 200:
        try:
            err_data = r.json()
            err_msg = err_data.get('error', r.text[:200])
        except Exception:
            err_msg = r.text[:200]
        return None, f"HTTP {r.status_code}: {err_msg}"

    try:
        data = r.json()
    except Exception as e:
        return None, f"JSON解析失败: {e}"

    if data.get('error'):
        return None, f"AI错误: {data['error']}"

    return data, None


def get_session():
    """读取当前 session 状态"""
    path = f'data/arc_session_{TASK_ID}.json'
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    print('=' * 60)
    print('R11L AI 自动闯关 - 健壮版')
    print('=' * 60)

    # 检查后端
    try:
        r = requests.get(f'{BASE}/', timeout=5)
        print(f'[OK] 后端运行正常 ({r.status_code})')
    except Exception as e:
        print(f'[ERROR] 后端无响应: {e}')
        sys.exit(1)

    # 检查外部网络
    try:
        r = requests.get('https://arcprize.org/api/games', timeout=10)
        print(f'[OK] arcprize.org 连通 ({r.status_code})')
    except Exception as e:
        print(f'[WARN] arcprize.org 连通异常: {e}')

    # 开始新游戏
    print('\n>>> 开启新游戏...')
    data, err = start_game()
    if err:
        print(f'[ERROR] {err}')
        sys.exit(1)
    print(f'[OK] 游戏开始: 关卡 {data.get("levels_completed")}, 可用动作 {data.get("available_actions")}')

    print('\n>>> 开始 AI 自动闯关循环...\n')
    consecutive_errors = 0
    max_consecutive = 5

    for step in range(1, MAX_STEPS + 1):
        data, err = auto_step()

        if err:
            consecutive_errors += 1
            print(f'  Step {step}: ❌ {err} (连续错误 {consecutive_errors}/{max_consecutive})')
            if consecutive_errors >= max_consecutive:
                print('\n[ABORT] 连续错误过多，停止')
                break
            # 等待后重试
            wait = min(5 * consecutive_errors, 30)
            print(f'    等待 {wait}s 后重试...')
            time.sleep(wait)
            continue
        else:
            consecutive_errors = 0

        levels = data.get('levels_completed', 0)
        status = data.get('status', '')
        game_over = data.get('game_over', False)
        win = data.get('win', False)
        msg = data.get('message', '')

        # 进度报告（每10步或关键节点）
        if step % 10 == 0 or levels > 0 or game_over:
            print(f'  Step {step}: 🏆 {levels}关 | {status} | {msg}')

        if game_over:
            print()
            if win:
                print(f'🎉 通关! 共完成 {levels} 关！')
            else:
                print(f'💀 游戏结束（完成 {levels} 关）')
            break

        time.sleep(STEP_DELAY)
    else:
        print(f'\n达到最大步数 ({MAX_STEPS})')

    # 最终状态
    session = get_session()
    if session:
        print('\n=== 最终状态 ===')
        print(f'关卡: {session.get("levels_completed")}')
        print(f'步数: {session.get("actions_taken")}')
        print(f'状态: {session.get("status")}')


if __name__ == '__main__':
    main()
