import sys

# Read the file
with open('integrations/ARC-AGI-3-Kaggle-Starter/agent/shadow_adapter.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with 'if isinstance(head, str):' after the 'c'/'click' check
# and insert new dialects before it
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    # Check if this is the line after the 'c'/'click' block
    if i > 0 and 'return (6, x, y) if x is not None and y is not None else None' in lines[i-1]:
        # Next line should be the 'if isinstance(head, str):' check
        if i + 1 < len(lines) and 'if isinstance(head, str):' in lines[i+1]:
            # Insert new dialects before this line
            new_lines.append('        # 方言 C：("i", 动作号) —— 键盘动作（arc_shadow_cn04 格式）\n')
            new_lines.append('        if isinstance(head, str) and head.strip().lower() == "i" and len(item) == 2:\n')
            new_lines.append('            val = _as_int(item[1])\n')
            new_lines.append('            return val if val is not None and 1 <= val <= 7 else None\n')
            new_lines.append('        # 方言 D：("r",) —— 旋转（arc_shadow_cn04 格式，旋转次数由调用方决定）\n')
            new_lines.append('        if isinstance(head, str) and head.strip().lower() == "r":\n')
            new_lines.append('            return 5  # ACTION5 = rotate\n')
    i += 1

# Write back
with open('integrations/ARC-AGI-3-Kaggle-Starter/agent/shadow_adapter.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('OK')
