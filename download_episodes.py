# -*- coding: utf-8 -*-
"""通过 API key 下载 episodes 数据

使用 ARC-AGI-3 API 下载游戏记录到 episodes 目录
"""
import sys
import os
import time
import json
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def download_episodes(api_key: str = None, game_ids: list = None):
    """通过 API key 下载 episodes 数据"""
    
    # 获取 API key
    if api_key is None:
        api_key = os.environ.get('ARC_API_KEY', '')
    
    if not api_key:
        print("未找到 API key，使用 anonymous key")
        # 使用 arc_agi 获取 anonymous key
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"))
        os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
        
        import arc_agi
        from arc_agi import OperationMode
        
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        # arc 会自动获取 anonymous key
        api_key = arc._api_key if hasattr(arc, '_api_key') else ''
        
        if not api_key:
            print("无法获取 API key")
            return
    
    print(f"使用 API key: {api_key[:8]}...")
    
    # API 端点
    base_url = "https://three.arcprize.org"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # 1. 获取所有 scorecards
    print("\n1. 获取 scorecards...")
    try:
        response = requests.get(
            f"{base_url}/api/scorecards",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            scorecards = response.json()
            print(f"   找到 {len(scorecards) if isinstance(scorecards, list) else '未知'} 个 scorecards")
        else:
            print(f"   获取失败: {response.status_code}")
            print(f"   响应: {response.text[:200]}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 2. 创建新 scorecard 并运行游戏
    if game_ids is None:
        game_ids = ["cn04", "sk48", "ar25", "ls20", "vc33"]
    
    print(f"\n2. 为 {len(game_ids)} 个游戏创建 scorecards...")
    
    for game_id in game_ids:
        print(f"\n   --- {game_id} ---")
        
        # 创建 scorecard
        try:
            response = requests.post(
                f"{base_url}/api/scorecard/open",
                headers=headers,
                json={},
                timeout=30
            )
            
            if response.status_code == 200:
                card_id = response.json().get('card_id')
                print(f"   Scorecard: {card_id}")
                
                # 获取 scorecard 数据
                time.sleep(1)
                response = requests.get(
                    f"{base_url}/api/scorecard/{card_id}",
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   数据: {json.dumps(data, indent=2)[:500]}")
                    
                    # 保存到 episodes 目录
                    episodes_dir = ROOT / "data" / "aop" / "episodes"
                    episode_dir = episodes_dir / f"{game_id}_api_rep0"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(episode_dir / "scorecard.json", 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"   已保存到: {episode_dir}")
                else:
                    print(f"   获取数据失败: {response.status_code}")
            else:
                print(f"   创建 scorecard 失败: {response.status_code}")
        except Exception as e:
            print(f"   错误: {e}")


def download_replay(replay_url: str, output_path: str = None):
    """下载 replay 数据"""
    print(f"\n下载 replay: {replay_url}")
    
    # replay_url 格式: https://arcprize.org/replay/{card_id}
    card_id = replay_url.split('/')[-1]
    
    # 尝试获取 replay 数据
    try:
        response = requests.get(
            f"https://three.arcprize.org/api/scorecard/{card_id}",
            headers={"X-API-Key": os.environ.get('ARC_API_KEY', '')},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"已保存到: {output_path}")
            
            return data
        else:
            print(f"下载失败: {response.status_code}")
    except Exception as e:
        print(f"错误: {e}")
    
    return None


def main():
    """主函数"""
    print("=" * 60)
    print("通过 API key 下载 episodes 数据")
    print("=" * 60)
    
    # 检查 API key
    api_key = os.environ.get('ARC_API_KEY', '')
    
    if api_key:
        print(f"找到 API key: {api_key[:8]}...")
    else:
        print("未找到 ARC_API_KEY 环境变量")
        print("请设置: export ARC_API_KEY=your-key")
        print("或从 https://arcprize.org/platform 获取")
        print("\n尝试使用 anonymous key...")
    
    # 下载 episodes
    download_episodes(api_key)
    
    # 下载特定 replay
    print("\n" + "=" * 60)
    print("下载特定 replay")
    print("=" * 60)
    
    # 用户提供的 replay 链接
    replays = [
        "https://arcprize.org/replay/b62df293-1d00-4689-83ef-75fa091b59d3",  # sk48
    ]
    
    for replay_url in replays:
        download_replay(replay_url)


if __name__ == "__main__":
    main()
