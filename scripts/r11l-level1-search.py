#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

GAME_ID='r11l-495a7899'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('state/r11l-level1-search.jsonl')
OUT.parent.mkdir(parents=True, exist_ok=True)

def emit(obj):
    with OUT.open('a') as f: f.write(json.dumps(obj,sort_keys=True)+'\n')

def act(env,x,y):
    a=GameAction.ACTION6; a.set_data({'x':x,'y':y,'game_id':GAME_ID})
    return env.step(a,data=a.action_data.model_dump())

arcade=Arcade(); emit({'type':'start','utc':time.time(),'game_id':GAME_ID})
coords=[(x,y) for y in range(0,64,4) for x in range(0,64,4)]
trials=0
for selector in [(4,4),(28,60),(24,56),(32,56),(28,56)]:
  for dest in coords:
    trials += 1
    try:
      env=arcade.make(GAME_ID,save_recording=True); reset=env.reset()
      if reset is None: raise RuntimeError('reset-none')
      obs=act(env,*selector)
      seq=[('select',selector)]
      seq.append(('move',dest)); obs=act(env,*dest)
      for tick in range(8):
        obs=act(env,*dest)
        if obs is None: break
        seq.append(('tick',dest))
        lv=int(getattr(obs,'levels_completed',0) or 0)
        state=str(getattr(obs,'state',None))
        if lv>=1:
          emit({'type':'success','trial':trials,'selector':selector,'dest':dest,'sequence':seq,'levels_completed':lv,'state':state,'action_input':str(getattr(obs,'action_input',None))})
          print(json.dumps({'success':True,'trial':trials,'selector':selector,'dest':dest,'levels_completed':lv,'state':state}))
          raise SystemExit(0)
      if trials % 20 == 0:
        emit({'type':'heartbeat','trial':trials,'selector':selector,'last_dest':dest})
    except Exception as e:
      emit({'type':'trial_error','trial':trials,'selector':selector,'dest':dest,'error':repr(e)})
emit({'type':'complete','success':False,'trials':trials})
print(json.dumps({'success':False,'trials':trials}))
raise SystemExit(1)
