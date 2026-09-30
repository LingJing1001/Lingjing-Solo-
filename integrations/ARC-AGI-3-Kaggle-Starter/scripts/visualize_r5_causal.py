"""Generate + serve R5 UpdateCausalModel visualization.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\visualize_r5_causal.py
  .\\.venv\\Scripts\\python.exe scripts\\visualize_r5_causal.py --port 8791 --open
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "ui" / "static"
DEFAULT_JSON = ROOT / "R5更新反思报告_LLM_causal.json"


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>R5 更新因果模型可视化</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet" />
<style>
:root {
  --bg:#0e141b; --panel:#161e27; --line:#2a3542; --text:#e8eef6; --muted:#8b9bb0;
  --ok:#3ecf8e; --warn:#e0a106; --bad:#ef476f; --accent:#4cc9c0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:"IBM Plex Sans",sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:end}
h1{margin:0;font-size:1.35rem}
.sub{color:var(--muted);font-size:.9rem}
.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px;padding:16px 24px}
@media(max-width:980px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h2{margin:0 0 10px;font-size:1rem}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.stat{background:#111820;border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:120px}
.stat b{display:block;font-size:1.15rem}
.stat span{color:var(--muted);font-size:.78rem}
svg{width:100%;height:auto;background:#111820;border-radius:8px;border:1px solid var(--line)}
.node rect{stroke-width:1.5}
.node text{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--text)}
.edge{fill:none;stroke-width:1.6;opacity:.9}
.edge.bad{stroke:var(--bad);stroke-dasharray:5 4}
.edge.ok{stroke:var(--ok)}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.pill{font-size:.75rem;padding:3px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.pill.obs{border-color:var(--warn);color:var(--warn)}
.pill.bad{border-color:var(--bad);color:var(--bad)}
.pill.ok{border-color:var(--ok);color:var(--ok)}
table{width:100%;border-collapse:collapse;font-size:.86rem}
th,td{border-bottom:1px solid var(--line);padding:7px 6px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.bar{display:flex;align-items:center;gap:8px;margin:4px 0}
.bar .name{width:150px;font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--muted)}
.bar .track{flex:1;height:8px;background:#0b1015;border-radius:4px;overflow:hidden}
.bar .fill{height:100%;background:var(--accent)}
.bar .val{width:42px;text-align:right;font-family:"IBM Plex Mono",monospace;font-size:.72rem}
ul.breaks{margin:0;padding-left:18px;color:var(--muted)}
ul.breaks li{margin:4px 0}
.gh{margin-top:8px;padding:10px;border:1px solid var(--line);border-radius:8px}
.gh code{font-family:"IBM Plex Mono",monospace;color:var(--accent)}
</style>
</head>
<body>
<header>
  <div>
    <h1>R5 更新因果模型</h1>
    <div class="sub">UpdateCausalModel · Layer 3b/4 · 与对局 WorldModelField 同构</div>
  </div>
  <div class="sub" id="meta"></div>
</header>
<div class="grid">
  <div class="card">
    <h2>因果图</h2>
    <div class="stats" id="stats"></div>
    <svg id="dag" viewBox="0 0 920 420"></svg>
    <div class="legend">
      <span class="pill obs">黄框 = 已观测</span>
      <span class="pill bad">红 = 断点 / 0.15</span>
      <span class="pill ok">绿 = 目标 SCORE_ALIGNED</span>
      <span class="pill">虚线 = 偏离/误报边</span>
    </div>
  </div>
  <div>
    <div class="card" style="margin-bottom:16px">
      <h2>假设库 C1–C6</h2>
      <table id="hypo"><thead><tr><th>ID</th><th>conf</th><th>status</th><th>陈述</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card" style="margin-bottom:16px">
      <h2>effect_ema</h2>
      <div id="ema"></div>
    </div>
    <div class="card">
      <h2>路径断点</h2>
      <ul class="breaks" id="breaks"></ul>
      <div class="gh" id="gh"></div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const BAD_ACTIONS = new Set(["competitions_submit","run_partial_script","misreport_success","assume_same_code","select_old_version"]);
const LAYOUT = {
  LOCAL_SCRIPT_OK:[40,180],
  KERNEL_PUSHED:[220,120],
  SUBMIT_API_OK:[400,60],
  PHASE_B_BOUND:[580,100],
  HARDCODED_LOADED:[740,60],
  MULTI_LEVEL_SCORE:[860,40],
  SCORE_ALIGNED:[860,120],
  QUOTA_BLOCK:[400,200],
  MISREPORT_SUBMITTED:[580,240],
  SCORE_015:[740,220],
};
const W=128,H=36;

function draw(){
  const observed = new Set(DATA.observed_states||[]);
  const graph = DATA.predict_graph||[];
  const svg = document.getElementById('dag');
  let edges='', nodes='';
  for(const e of graph){
    if(e.state===e.next && e.action==='verify_local_win') continue;
    const a=LAYOUT[e.state], b=LAYOUT[e.next];
    if(!a||!b) continue;
    const bad = BAD_ACTIONS.has(e.action) || (e.state==='KERNEL_PUSHED' && e.next==='QUOTA_BLOCK');
    const x1=a[0]+W, y1=a[1]+H/2, x2=b[0], y2=b[1]+H/2;
    edges += `<path class="edge ${bad?'bad':'ok'}" d="M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}"><title>${e.state} --${e.action}--> ${e.next}</title></path>`;
  }
  for(const [id,pos] of Object.entries(LAYOUT)){
    const obs=observed.has(id);
    const isGoal=id==='SCORE_ALIGNED';
    const isBad=id==='QUOTA_BLOCK'||id==='SCORE_015'||id==='MISREPORT_SUBMITTED';
    let fill='#161e27', stroke='#2a3542';
    if(isBad){fill='#2a151b';stroke='#ef476f';}
    else if(isGoal){fill='#13271d';stroke='#3ecf8e';}
    else if(obs){fill='#2a2312';stroke='#e0a106';}
    nodes += `<g class="node" transform="translate(${pos[0]},${pos[1]})">
      <rect width="${W}" height="${H}" rx="6" fill="${fill}" stroke="${stroke}"/>
      <text x="${W/2}" y="22" text-anchor="middle">${id}</text>
    </g>`;
  }
  svg.innerHTML = edges+nodes;
  document.getElementById('meta').textContent = `version ${DATA.version} · nodes ${DATA.nodes.length} · edges ${graph.length}`;
  document.getElementById('stats').innerHTML = [
    ['已观测', observed.size, 'warn'],
    ['断点', 'QUOTA_BLOCK', 'bad'],
    ['目标', 'SCORE_ALIGNED', 'ok'],
    ['假设', (DATA.hypotheses||[]).length, ''],
  ].map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

  const tb = document.querySelector('#hypo tbody');
  tb.innerHTML = (DATA.hypotheses||[]).map(h=>`<tr><td>${h.id}</td><td>${h.confidence}</td><td>${h.status}</td><td>${h.statement}</td></tr>`).join('');

  const ema = Object.entries(DATA.effect_ema||{}).sort((a,b)=>b[1]-a[1]);
  document.getElementById('ema').innerHTML = ema.map(([k,v])=>`<div class="bar"><div class="name">${k}</div><div class="track"><div class="fill" style="width:${Math.round(v*100)}%"></div></div><div class="val">${v}</div></div>`).join('');

  document.getElementById('breaks').innerHTML = (DATA.path_breaks||[]).map(b=>`<li>${b}</li>`).join('');
  document.getElementById('gh').innerHTML = `<div><b>GitHub 更新</b></div>
    <div><code>a02287b</code> AR25 canned L0–L7 · offline 8/8 WIN / 276 actions</div>
    <div><code>5bbc2f9</code> AR25 encoder/field + arcsage reference</div>`;
}
draw();
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=str(DEFAULT_JSON))
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--open", action="store_true")
    p.add_argument("--no-serve", action="store_true")
    args = p.parse_args()

    data_path = Path(args.json)
    if not data_path.exists():
        raise SystemExit(f"missing causal json: {data_path}")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    out = STATIC / "r5_causal.html"
    STATIC.mkdir(parents=True, exist_ok=True)
    out.write_text(
        HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"[viz] wrote {out}")

    if args.no_serve:
        return

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(STATIC), **kw)

        def log_message(self, fmt, *args):  # noqa: A003
            print("[http]", fmt % args)

    url = f"http://127.0.0.1:{args.port}/r5_causal.html"
    print(f"[viz] serving {url}")
    if args.open:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
