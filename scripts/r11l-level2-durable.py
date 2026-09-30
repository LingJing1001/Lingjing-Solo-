import itertools,json,time,os
from arc_agi import Arcade
from arcengine import GameAction
G='r11l-495a7899';OUT='/srv/agent-platform/projects/Lingjing-Solo-/state/r11l-level2-durable.jsonl';os.makedirs(os.path.dirname(OUT),exist_ok=True)
def log(x):
 with open(OUT,'a') as f:f.write(json.dumps(x)+'\n')
def run(order):
 e=Arcade().make(G,save_recording=True);z=e.reset();n=0
 def go(x,y):
  nonlocal z,n;n+=1;a=GameAction.ACTION6;a.set_data({'x':x,'y':y,'game_id':G});z=e.step(a,data=a.action_data.model_dump());return z
 last = None
 def wait(k=1):
  # The environment's ACTION6 (0,0) advances animation and is the
  # empirically verified no-op/heartbeat for R11L.
  for _ in range(k):
   z=go(0,0)
   if z is None or str(z.state).endswith('GAME_OVER'):return False
  return True
 def move(x,y,settle=1):
  nonlocal last
  z=go(x,y); last=(x,y)
  if z is None or str(z.state).endswith('GAME_OVER'): return False
  return wait(settle)
 go(7,36);go(38,20);last=(38,20);wait(1);go(27,59);go(42,21);last=(42,21);wait(1)
 routes={
 (15,4):[(55,4),(55,26),(55,50)],(6,19):[(55,19),(55,26),(55,50)],(47,7):[(55,7),(55,26),(55,50)]}
 for i,sel in enumerate(order):
  go(sel[0]+2,sel[1]+2)
  for p in routes[sel]:move(p[0]+2,p[1]+2)
  dest={(15,4):(35,50),(6,19):(39,50),(47,7):(37,52)}[sel]
  move(*dest,20 if i==len(order)-1 else 1)
 if z is None or z.levels_completed<2:return z.levels_completed if z else -1
 # pumlzd, try direct safe route with 6 tick segments
 for i,(sel,way) in enumerate([((43,33),[(47,33),(47,27),(48,27),(48,26),(49,26),(49,25),(50,25),(50,24),(54,24)]),((52,46),[(47,46),(47,27),(48,27),(48,26),(49,26),(49,25),(50,25),(50,24),(54,24)])]):
  move(sel[0]+2,sel[1]+2,0)
  for p in way:move(p[0]+2,p[1]+2)
  move(56,17,20 if i==1 else 1)
 return z.levels_completed if z else -1
for i,o in enumerate(itertools.permutations([(15,4),(6,19),(47,7)])):
 try:
  lev=run(o);log({'trial':i,'order':o,'levels_completed':lev,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print({'trial':i,'order':o,'levels_completed':lev},flush=True)
  if lev>=2:print('SUCCESS',flush=True);break
 except Exception as e:log({'trial':i,'error':repr(e)});print('ERROR',repr(e),flush=True)
