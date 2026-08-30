"""debug4.py — 手动复现 step 逻辑并逐行打印"""
src = (
    "def step(state, action):\n"
    "    s = dict(state)\n"
    "    s['objects'] = dict(state['objects'])\n"
    "    s['extras'] = dict(state['extras'])\n"
    "    objs = s['objects']\n"
    "    aid = s['avatar_id']\n"
    "    if aid is None:\n"
    "        return s\n"
    "    if action not in ('up', 'down', 'left', 'right'):\n"
    "        return s\n"
    "    _dirs = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}\n"
    "    dx, dy = _dirs[action]\n"
    "    av = objs[aid]\n"
    "    tx = av['x'] + dx\n"
    "    ty = av['y'] + dy\n"
    "    for oid in list(objs):\n"
    "        if oid == aid:\n"
    "            continue\n"
    "        o = objs[oid]\n"
    "        role = o['role']\n"
    "        if role == 'box' or role == 'pushable':\n"
    "            if o['x'] == tx and o['y'] == ty:\n"
    "                o['x'] = o['x'] + dx\n"
    "                o['y'] = o['y'] + dy\n"
    "                break\n"
    "    av['x'] = tx\n"
    "    av['y'] = ty\n"
    "    return s\n"
)

state = {
    "objects": {
        "a": {"x": 1, "y": 1, "role": "avatar"},
        "b": {"x": 2, "y": 1, "role": "box"},
    },
    "avatar_id": "a",
    "extras": {},
}

ns = {"__builtins__": {}, "UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"}
exec(compile(src, "<t>", "exec"), ns)
step = ns["step"]

print("调用前 state:", state)
res = step(dict(state), "right")
print("调用后 state:", state)
print("返回值:", res['objects'])
print("是否为同一 dict:", state['objects']['a'] is res['objects']['a'])
print("av 修改前 tx 计算: tx=1+1=2")
print("若 av 与返回 objs 不同 dict，则修改丢失")
