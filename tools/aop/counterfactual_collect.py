#!/usr/bin/env python3
"""Collect same-state counterfactual action outcomes from ARC-AGI-3.

For each scout prefix, replay the exact prefix in a fresh scorecard for every
candidate action. A group is retained only when the pre-action state hash is
identical across all branches. This makes candidate rows comparable without
claiming that unexecuted actions have an outcome.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from pathlib import Path
sys.path.insert(0, '/srv/agent-platform/projects/Lingjing-Solo-/integrations/ARC-AGI-3-Kaggle-Starter/vendor')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

GAMES = ['ar25','vc33','ls20','dc22','re86']

def dump(frame):
    return frame.model_dump(mode='json') if frame is not None else None

def stable_hash(value):
    if isinstance(value, dict):
        value = {k: stable_hash(v) for k, v in value.items() if k not in {'guid', 'action_input', 'frame_id'}}
    elif isinstance(value, list):
        value = [stable_hash(v) for v in value]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()[:16]

def obs(frame, tick):
    d = dump(frame) or {}
    legal = list(d.get('available_actions') or [])
    return {'state': d.get('state'), 'levels_completed': d.get('levels_completed', 0),
            'available_actions': legal, 'legal_actions': legal, 'tick': tick,
            'frame_id': d.get('guid'), 'frame': d}


def recording_transition(out, card, candidate):
    files = sorted((out / 'remote_recordings' / str(card)).glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(files) != 1:
        return None
    records = [json.loads(line)['data'] for line in files[0].read_text().splitlines() if line.strip()]
    for index, data in enumerate(records):
        if (data.get('action_input') or {}).get('id') == f'ACTION{candidate}' and index > 0:
            return records[index - 1], data
    return None

def obs_from_record(data, tick):
    legal = list(data.get('available_actions') or [])
    return {'state': data.get('state'), 'levels_completed': data.get('levels_completed', 0),
            'available_actions': legal, 'legal_actions': legal, 'tick': tick,
            'frame_id': data.get('guid'), 'frame': data}

def choose_prefix(arc, game, seed, length):
    card = arc.open_scorecard(tags=['aop','counterfactual-scout',game,f'prefix{length}'])
    try:
        env = arc.make(game, scorecard_id=card, save_recording=False)
        if env is None:
            return None
        frame = env.reset(); actions=[]
        rng=random.Random(seed)
        for tick in range(length):
            if frame is None: return None
            legal=list(frame.available_actions or [])
            candidates=[x for x in legal if x not in (0,6)] or legal
            if not candidates: return None
            aid=rng.choice(candidates); actions.append(aid)
            frame=env.step(GameAction.from_id(aid), reasoning={'probe':'counterfactual scout','tick':tick})
        if frame is None: return None
        return {'actions':actions,'before':obs(frame,length),'hash':stable_hash(dump(frame))}
    finally:
        try: arc.close_scorecard(card)
        except Exception: pass

def branch(arc, game, prefix, candidate, group_id, branch_no, out):
    card=arc.open_scorecard(tags=['aop','counterfactual',game,group_id,f'candidate{candidate}'])
    try:
        env=arc.make(game, scorecard_id=card, save_recording=True)
        if env is None:
            return None
        frame=env.reset()
        for tick,aid in enumerate(prefix['actions']):
            if frame is None or aid not in list(frame.available_actions or []): return None
            frame=env.step(GameAction.from_id(aid), reasoning={'probe':'counterfactual replay','group':group_id,'tick':tick})
        if frame is None: return None
        before=obs(frame,len(prefix['actions']))
        replay_hash=stable_hash(dump(frame))
        if replay_hash != prefix['hash']: return None
        legal=list(frame.available_actions or [])
        if candidate not in legal: return None
        after_frame=env.step(GameAction.from_id(candidate), reasoning={'probe':'counterfactual candidate','group':group_id,'candidate':candidate})
        after=obs(after_frame,len(prefix['actions'])+1)
        transition=recording_transition(out, card, candidate)
        if transition is not None:
            before=obs_from_record(transition[0],len(prefix['actions']))
            after=obs_from_record(transition[1],len(prefix['actions'])+1)
        before_hash=stable_hash(before); after_hash=stable_hash(after)
        row={
          'episode_id':f'{game}_{group_id}_candidate{candidate}_b{branch_no}',
          'step_id':len(prefix['actions']),
          'requested_action':{'name':f'ACTION{candidate}','id':candidate,'payload':{}},
          'observation_before':before,'observation_after':after,
          'state_delta':{
            'before_hash':before_hash,'after_hash':after_hash,
            'state_changed':before_hash!=after_hash,
            'visual_changed':before.get('frame')!=after.get('frame'),
            'visual_delta':sum(a != b for row_a, row_b in zip((before.get('frame') or {}).get('frame') or [], (after.get('frame') or {}).get('frame') or []) for a, b in zip(row_a, row_b)) if isinstance((before.get('frame') or {}).get('frame'), list) and isinstance((after.get('frame') or {}).get('frame'), list) else None,
            'action_effective':before_hash!=after_hash or before.get('state')!=after.get('state'),
            'progressed':after.get('levels_completed',0)>before.get('levels_completed',0) or after.get('state')=='WIN',
            'state_before':before.get('state'),'state_after':after.get('state'),
            'levels_delta':after.get('levels_completed',0)-before.get('levels_completed',0),
            'score_delta':0.0,
          },
          'reward':1.0 if after.get('levels_completed',0)>before.get('levels_completed',0) or after.get('state')=='WIN' else 0.0,
          'level_completed':after.get('levels_completed',0),
          'legal_actions':before['legal_actions'],
          'temporal':{'tick_before':len(prefix['actions']),'tick_after':len(prefix['actions'])+1,'dt':1},
          'counterfactual_group_id':group_id,'candidate_action_id':candidate,
          'prefix_action_ids':prefix['actions'],'replay_state_hash':replay_hash,
          'replay_verified':True,'source':'remote_counterfactual_v1',
        }
        return row
    finally:
        try: arc.close_scorecard(card)
        except Exception: pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--groups-per-game',type=int,default=3)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    arc=arc_agi.Arcade(arc_api_key=os.environ['ARC_API_KEY'],operation_mode=OperationMode.ONLINE, recordings_dir=str(args.out/'remote_recordings'))
    rows=[]; groups=[]; prefixes=[50,75,100,125]
    for gi,game in enumerate(GAMES):
      for n in range(args.groups_per_game):
        length=prefixes[n % len(prefixes)]; gid=f'{game}_g{n}_p{length}'
        try:
          prefix=choose_prefix(arc,game,20261010+gi*100+n,length)
          if not prefix: continue
          candidates=[x for x in prefix['before']['legal_actions'] if x not in (0,)]
          branch_rows=[]
          for b,candidate in enumerate(candidates):
            try:
              row=branch(arc,game,prefix,candidate,gid,b,args.out)
            except Exception as exc:
              print(json.dumps({'group':gid,'candidate':candidate,'error':type(exc).__name__}), file=sys.stderr)
              row=None
            if row is not None: branch_rows.append(row)
          if len(branch_rows)>=2:
            rows.extend(branch_rows); groups.append({'group_id':gid,'game':game,'prefix':prefix['actions'],'candidates':[r['candidate_action_id'] for r in branch_rows],'state_hash':prefix['hash']})
            with (args.out/'counterfactual_labels.jsonl').open('a') as f:
              for row in branch_rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\\n')
        except Exception as exc:
          print(json.dumps({'group':gid,'error':type(exc).__name__,'message':str(exc)[:160]}), file=sys.stderr)

    with (args.out/'counterfactual_labels.jsonl').open('w') as f:
      for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    (args.out/'groups.json').write_text(json.dumps(groups,indent=2))
    print(json.dumps({'rows':len(rows),'groups':len(groups),'games':sorted({g['game'] for g in groups}),'output':str(args.out/'counterfactual_labels.jsonl')},ensure_ascii=False))
    return 0 if rows else 1
if __name__=='__main__': raise SystemExit(main())
