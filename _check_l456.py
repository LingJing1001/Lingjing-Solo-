import json, os, glob

for name in ["wa30_l5_plan.json", "wa30_l56_ext_plans.json", "wa30_l5_ext_plan.json", "wa30_l6_ext_plan.json"]:
    matches = glob.glob(f"C:/newtask-pi/_diag_*/{name}", recursive=True) + glob.glob(f"C:/newtask-pi/_diag_*/**/{name}", recursive=True)
    for m in matches:
        try:
            data = json.load(open(m, encoding="utf-8"))
            print(f"=== {name} ({os.path.getsize(m)} bytes) ===")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        print(f"  {k}: {len(v)} items, first 20: {v[:20]}")
                    else:
                        print(f"  {k}: {v}")
            elif isinstance(data, list):
                print(f"  list of {len(data)} items, first 30: {data[:30]}")
        except Exception as e:
            print(f"  ERROR: {e}")

# Also check L456 chain verify output
for name in ["wa30_l456_chain_verify_out.txt"]:
    matches = glob.glob(f"C:/newtask-pi/_diag_*/{name}", recursive=True)
    for m in matches:
        print(f"\n=== {name} ===")
        with open(m, encoding="utf-8", errors="replace") as f:
            print(f.read()[:2000])
