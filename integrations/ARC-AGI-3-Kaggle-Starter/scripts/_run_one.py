"""临时脚本：单跑一个游戏，捕获 shadow_solver 日志。"""
import sys, os, time, importlib.util, logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from arc_agi import Arcade, OperationMode

ENV_DIR = os.path.join(ROOT, "environment_files")

def load_agent(path):
    spec = importlib.util.spec_from_file_location("bench_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MyAgent

def gid_to_full(gid):
    arc = Arcade(environments_dir=ENV_DIR, operation_mode=OperationMode.OFFLINE)
    for e in arc.get_environments():
        if e.game_id.split("-")[0] == gid:
            return e.game_id
    return None

gid = sys.argv[1] if len(sys.argv) > 1 else "cn04"
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400

full_id = gid_to_full(gid)
if not full_id:
    print(f"[ERROR] 找不到 {gid}", flush=True)
    sys.exit(1)

print(f"[run-one] gid={gid} full={full_id} n_steps={n_steps}", flush=True)

agent_cls = load_agent(os.path.join(ROOT, "agent", "my_agent.py"))
arc = Arcade(environments_dir=ENV_DIR, logger=logging.getLogger("bench"),
             operation_mode=OperationMode.OFFLINE)

env = arc.make(full_id, seed=0, save_recording=False)
agent = agent_cls("bench-card", full_id, "smart-router-bench", "local", False, env)

frame = env.reset()
agent.frames[0] = agent.frames[0]  # trigger init
steps = 0
max_levels = 0
t0 = time.time()

while not agent.is_done(agent.frames, agent.frames[-1]) and steps < n_steps:
    action = agent.choose_action(agent.frames, agent.frames[-1])
    steps += 1
    agent.action_counter += 1
    raw = env.step(action)
    if hasattr(agent, 'append_frame'):
        from arc_agi import GameState
        fd = type('FD', (), {
            'state': raw.state,
            'levels_completed': raw.levels_completed,
            'frame': raw.frame,
            'available_actions': raw.available_actions,
        })()
        agent.frames.append(fd)
    max_levels = max(max_levels, int(raw.levels_completed))
    if raw.state is getattr(raw, 'state', None) and hasattr(raw.state, 'name') and raw.state.name == 'WIN':
        break

elapsed = time.time() - t0
print(f"\n[RESULT] gid={gid} max_levels={max_levels} steps={steps} elapsed={elapsed:.1f}s", flush=True)