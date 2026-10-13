"""提取 L4/L5/L6 plan 并生成完整代码片段。"""
import json, glob

# L3 plan (from prefix[150:])
l3 = [1, 1, 4, 5, 2, 4, 4, 5, 1, 5, 2, 5, 2, 5, 2, 5, 3, 3, 5, 3, 1, 5, 4, 5, 4, 1, 4, 5, 3, 1, 1, 3, 5, 3, 2, 5, 4, 4, 4, 5, 4, 2, 5, 3, 3, 4, 4, 5, 2, 2, 3, 5, 1, 4, 2, 5, 2, 5, 1, 1, 3, 4, 1, 1, 2, 1, 3, 1]

# L4 plan (from wa30_l5_plan.json "plan")
for m in glob.glob("C:/newtask-pi/_diag_*/wa30_l5_plan.json", recursive=True):
    data = json.load(open(m, encoding="utf-8"))
    l4 = data["plan"]
    break

# L5 plan (from wa30_l5_ext_plan.json)
for m in glob.glob("C:/newtask-pi/_diag_*/wa30_l5_ext_plan.json", recursive=True):
    data = json.load(open(m, encoding="utf-8"))
    l5 = data["plan"]
    break

# L6 plan (from wa30_l6_ext_plan.json)
for m in glob.glob("C:/newtask-pi/_diag_*/wa30_l6_ext_plan.json", recursive=True):
    data = json.load(open(m, encoding="utf-8"))
    l6 = data["plan"]
    break

print(f"L3: {len(l3)} steps")
print(f"L4: {len(l4)} steps")
print(f"L5: {len(l5)} steps")
print(f"L6: {len(l6)} steps")
print(f"Total L0-L6: {26+48+76+len(l3)+len(l4)+len(l5)+len(l6)} steps")

# Generate code
print("\n=== CODE ===")
for lv, plan in [(3, l3), (4, l4), (5, l5), (6, l6)]:
    print(f"    {lv}: {plan},")
