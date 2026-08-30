"""debug6.py — 完全手工复现 step 逻辑，逐行打印"""
state = {
    "objects": {
        "a": {"x": 1, "y": 1, "role": "avatar"},
        "b": {"x": 2, "y": 1, "role": "box"},
    },
    "avatar_id": "a",
    "extras": {},
}

s = dict(state)
s['objects'] = dict(state['objects'])
s['extras'] = dict(state['extras'])
objs = s['objects']
aid = s['avatar_id']
action = "right"

_dirs = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
dx, dy = _dirs = _dirs[action]
av = objs[aid]
tx = av['x'] + dx
ty = av['y'] + dy
print(f"av={av}, tx={tx}, ty={ty}, objs keys={list(objs)}")
for oid in list(objs):
    print(f"  循环 oid={oid}, aid={aid}, continue?={oid==aid}")
    if oid == aid:
        continue
    o = objs[oid]
    role = o['role']
    print(f"    o={o}, role={role}, box?={role in ('box','pushable')}")
    print(f"    o.x==tx? {o['x']}=={tx}, o.y==ty? {o['y']}=={ty}")
    if role == 'box' or role == 'pushable':
        if o['x'] == tx and o['y'] == ty:
            print("    → 推箱!")
            o['x'] = o['x'] + dx
            o['y'] = o['y'] + dy
            break
print(f"before av assign: av={av}")
av['x'] = tx
av['y'] = ty
print(f"final: objects={s['objects']}")
