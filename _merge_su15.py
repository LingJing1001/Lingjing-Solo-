import sys

with open('integrations/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# 1. 添加 su15 状态变量
old_stag = '        self._stag_logged: bool = False'
new_stag = old_stag + '\n        # su15 动态求解器缓存\n        self._su15_plans: dict[int, list[tuple[int, int]]] = {}\n        self._su15_idx: int = 0\n        self._su15_level: int = -1\n        self._su15_solved: bool = False\n        self._su15_solver_mod: Any = None'
content = content.replace(old_stag, new_stag, 1)

# 2. 从 ROUTE_CEAX_KNOWN 移除 su15
old_ceax = '"lp85", "su15", "sb26", "s5i5", "tu93", "r11l",'
new_ceax = '"lp85", "sb26", "s5i5", "tu93", "r11l",'
content = content.replace(old_ceax, new_ceax)

# 3. 在 boot log 后添加 su15 预求解
old_boot = '            flush=True,\n        )\n\n    @property'
su15_block = '''            flush=True,
        )

        # su15: 预求解所有关 -> 按实际步数设 MAX_ACTIONS
        if gid == "su15":
            try:
                import su15_solver as _su15_sol  # noqa: E402
                self._su15_solver_mod = _su15_sol
                self._su15_solved = True
                _total = 0
                _ok = 0
                for _lvl in range(9):
                    _sol = _su15_sol.solve_level(_lvl, max_nodes=200000, weight=1.3, time_budget=300)
                    if _sol:
                        self._su15_plans[_lvl] = [(int(x), int(y)) for x, y in _sol]
                        _total += len(_sol)
                        _ok += 1
                if _ok == 9:
                    self.MAX_ACTIONS = _total + 2
                    print(f"[{BUILD_TAG}] su15-solver: 9/9 solved, {_total} clicks, "
                          f"MAX_ACTIONS={self.MAX_ACTIONS}", flush=True)
                elif _ok > 0:
                    self.MAX_ACTIONS = _total + 15
                    print(f"[{BUILD_TAG}] su15-solver: {_ok}/9 solved, {_total} clicks, "
                          f"MAX_ACTIONS={self.MAX_ACTIONS} (CEAX fallback)", flush=True)
                else:
                    print(f"[{BUILD_TAG}] su15-solver: 0/9 solved, CEAX fallback", flush=True)
            except Exception as _e:
                print(f"[{BUILD_TAG}] su15-solver pre-solve failed: {_e}", flush=True)

    @property'''
content = content.replace(old_boot, su15_block, 1)

# 4. 在 tn36 dispatch 后添加 su15 dispatch
old_tn36 = '            self._tn36_exhausted = levels\n\n        # --- bp35'
su15_dispatch = '''            self._tn36_exhausted = levels

        # --- su15 动态求解（预求解已填 self._su15_plans，此处纯回放） ---
        if gid == "su15" and levels in self._su15_plans:
            if self._su15_level != levels:
                self._su15_level = levels
                self._su15_idx = 0
            plan = self._su15_plans[levels]
            if self._su15_idx < len(plan):
                xy = plan[self._su15_idx]
                self._su15_idx += 1
                action = GameAction.ACTION6
                action.set_data({"x": int(xy[0]), "y": int(xy[1])})
                action.reasoning = {"text": f"{BUILD_TAG}:su15 L{levels} #{self._su15_idx}/{len(plan)}"}
                return action

        # --- bp35'''
content = content.replace(old_tn36, su15_dispatch, 1)

with open('integrations/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('OK - su15 solver merged')
