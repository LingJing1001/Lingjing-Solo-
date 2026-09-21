#!/usr/bin/env bash
set -euo pipefail

# Sync the versioned ARC reproduction bundle into an existing ARC checkout.
#
# Usage: bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh [ARC_DIR]
#                                   [--with-recording-patch] [--check] [--allow-overwrite]
#
#   --check / --check-only   只跑预检、一个字都不写；有"不可恢复的覆盖"就 exit 1（可当 CI 闸门）
#                            两个名字等价：`--check` 是本仓库先起的，`--check-only` 是 ARC 侧
#                            那版（origin/main 33b91aa）先起的，谁的手都别改， hence both.
#   --allow-overwrite        允许覆盖不可恢复的差异文件，但覆盖前先把 ARC 侧那份存档（见下）
#
# 为什么要有预检这一步（设计文档 §8.10 的真实代价）：原来这里是
# `cp -R "$SCRIPT_DIR/agents/strategies/." agents/strategies/`。ARC 侧那个目录**整目录
# untracked**，所以 `cp` 盖掉同名文件时 `git status`/`git diff` 完全看不见被盖掉的内容。
# 上一次真跑同步就把 ARC 侧一份 `run_ar25_r234.py`（35622 字节 / 9-19 22:47）静默盖成了
# ② 版——那份副本在 git 里对不上任何版本，`__pycache__` 里唯一能反推的 .pyc 又同时被重新
# 生成，**不可恢复**。规则 ② 要求"动手前先看 ARC 的 diff"，在那个目录里此前是一句空话；
# 本脚本现在把这句话变成机器判定。
#
# 这份文件是两条独立改动的**语义合并**，不是选边：
#   · 清单化复制（`SYNC_PAIRS`）+ 四类预检 + fail-closed + 覆盖前存档  ← #10
#   · `--check-only` 这个开关名、复制后逐文件 sha256 复核、
#     `official_agents_init_sha256` 打印                            ← origin/main `33b91aa`
# 两边其实是同一个动机的两份实现（都为了"别再静默盖掉 ARC 侧内容"），所以合而不是挑一份。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINGJING_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARC_DIR=""
ARC_GIVEN=0
WITH_RECORDING_PATCH=0
CHECK_ONLY=0
ALLOW_OVERWRITE=0

for arg in "$@"; do
  case "$arg" in
    --with-recording-patch) WITH_RECORDING_PATCH=1 ;;
    --check|--check-only) CHECK_ONLY=1 ;;
    --allow-overwrite) ALLOW_OVERWRITE=1 ;;
    *) if (( ! ARC_GIVEN )); then ARC_DIR="$arg"; ARC_GIVEN=1; fi ;;
  esac
done
if [[ -z "$ARC_DIR" ]]; then ARC_DIR="$PWD"; fi

ARC_DIR="$(cd -- "$ARC_DIR" && pwd)"
cd "$ARC_DIR"

git rev-parse --show-toplevel >/dev/null
test -f main.py

# ── 声明清单：复制范围只由 SYNC_PAIRS 决定（规则 ①）──────────────────────
# 每一对是 "源绝对路径<TAB>目标相对 ARC 根的路径"。目录形态的旧 `cp -R` 换成展开成逐文件，
# 并显式排除 `__pycache__/` 与 `*.pyc`——前者此前确实被搬过去过（两侧都在，§8.11 第 7 条已核实），
# 后者是 ARC 侧那版在复核循环里单独挡的一道，合并后一并收进清单展开这一步。
SYNC_PAIRS=()
add_pair() { SYNC_PAIRS+=("$1$(printf '\t')$2"); }
add_tree() {                       # $1=源目录 $2=ARC 下的目标目录
  local root="$1" dest="$2" f rel
  while IFS= read -r f; do
    rel="${f#"$root"/}"
    add_pair "$f" "$dest/$rel"
  done < <(find "$root" -type f -not -path '*/__pycache__/*' -not -name '*.pyc' | LC_ALL=C sort)
}

add_pair "$SCRIPT_DIR/agents/templates/lingjing_solo_agent.py" agents/templates/lingjing_solo_agent.py
# agents/strategies/ 整目录（含离线 runner 与探针）。**这一条是 §7.3 登记的规则 ① 越界**：
# 线上真正需要的只有 __init__.py + base/generic/ar25/ls20/r11l/registry 六份，其余是诊断件。
# 缩小到那七份要改的是"生产包该含什么"，那是你的决定，本脚本只保证复制范围可见、可比对。
add_tree "$SCRIPT_DIR/agents/strategies" agents/strategies
add_pair "$SCRIPT_DIR/tests/test_lingjing_solo_agent.py" tests/unit/test_lingjing_solo_agent.py
add_pair "$SCRIPT_DIR/tests/test_action_recording.py" tests/unit/test_action_recording.py
add_pair "$SCRIPT_DIR/tests/test_r11l_probe.py" tests/unit/test_r11l_probe.py
# 统一 R3 plan 契约（§3.2 九字段）：只依赖 stdlib + editable 装的 lingjing_solo，
# 在 ARC 里跑它才算"契约在线上包内可用"，在本仓库跑只证明它自己能 import。
add_pair "$LINGJING_ROOT/tests/test_plan_contract.py" tests/unit/test_plan_contract.py
# AR25 运行器的 §3.2 出口回归（§8.8）。**在 ARC 里它是 skip 不是 assert**：被测的
# run_ar25_r234.py 虽然在清单里，但它 import 依赖 arc_adaptor/paths.py，ARC 没有那个目录。
# 带过去仍值得：它证明这份测试进生产 checkout 不会 error 掉别人的 tests/unit。
add_pair "$LINGJING_ROOT/tests/test_ar25_plan_contract.py" tests/unit/test_ar25_plan_contract.py
# AR25 逐 tick 证据层（③：r2_ar25 哈希 + tick_trail 写入器）。同样是**模块级 skip**：
# 被测的 r2_ar25.py / tick_trail.py 在 arc_adaptor/ 根，不在清单里（引擎后端不得上线，规则 ①），
# 而 skip 判断写在那两个 import 之前，所以带过去只会 skip、不会崩。
add_pair "$LINGJING_ROOT/tests/test_ar25_recording.py" tests/unit/test_ar25_recording.py
add_pair "$SCRIPT_DIR/tools/ls20_single_action_probe.py" tools/ls20_single_action_probe.py
add_pair "$SCRIPT_DIR/tools/r11l_single_action_probe.py" tools/r11l_single_action_probe.py

# ── 清单自检：每条源文件必须真的在盘上 ────────────────────────────────
# ARC 侧那版靠"找不到就去 arc_adaptor/tests/ 再不去仓库根 tests/"来回避路径分叉，结果把
# 仓库根那份回退删掉了（test_plan_contract.py 等三条只在 <root>/tests/，而 arc_adaptor/tests/
# 在 git 里是空树）。这里不做回退猜测：清单里写的就是绝对路径，对不上就直接停——
# 路径漂移要么在这一行暴露，要么根本不该由复制脚本兜着。
missing_sources=()
for pair in "${SYNC_PAIRS[@]}"; do
  src="${pair%%$'\t'*}"
  [[ -f "$src" ]] || missing_sources+=("${src#"$LINGJING_ROOT"/}")
done
if (( ${#missing_sources[@]} )); then
  printf 'BLOCKED — 清单里有 %s 条源文件不在盘上（路径漂移或被删）:\n' "${#missing_sources[@]}"
  printf '  %s\n' "${missing_sources[@]}"
  exit 1
fi

# ── 预检：把每个目标分成 identical / new / 可恢复差异 / **不可恢复差异** 四类 ────────
# "可恢复"的定义是可机械判定的：ARC 侧那份内容（去 CRLF 后，两侧都要归一，单边归一会得出
# 假的"有差异"）与本仓库该路径**任一历史版本**逐字节相同 ⇒ 盖掉不丢东西。
# 对不上任一版本 = 只有 ARC 侧有这份内容 = 上次丢代码的那种，默认不许盖。
sha16() { sha256sum "$1" | cut -c1-16; }
normalized_sha16_of_file() { tr -d '\r' < "$1" | sha256sum | cut -c1-16; }
normalized_sha16_of_stdin() { tr -d '\r' | sha256sum | cut -c1-16; }

# in_history <ARC 侧文件> <本仓库相对路径> ：该内容的归一形态是否在我们历史的任一版本里
# 只看该路径最近 50 个提交；更早的版本对不上时**保守判成不可恢复**（宁可停，别盖）。
# 先 `cat-file -e` 确认那个提交里**真有**这条路径：`git show` 对不存在的路径只会报错并吐出
# 空串，而空串的哈希就是"空文件"的哈希 —— 少了这一步，一份被删过路径的旧提交会让任何空
# 的 ARC 侧文件都被误判成"历史里对得上、盖掉不丢内容"（实测过：72b9069 那批删除即触发）。
in_history() {
  local file="$1" rel="$2" h c
  h="$(normalized_sha16_of_file "$file")"
  while IFS= read -r c; do
    [[ -n "$c" ]] || continue
    git -C "$LINGJING_ROOT" cat-file -e "$c:$rel" 2>/dev/null || continue
    if [[ "$(git -C "$LINGJING_ROOT" show "$c:$rel" | normalized_sha16_of_stdin)" == "$h" ]]; then
      return 0
    fi
  done < <(git -C "$LINGJING_ROOT" log --all --format=%H -- "$rel" | head -50)
  return 1
}

identical=0
new_files=0
recoverable_diffs=()
unrecoverable_diffs=()
uncommitted_mod=()
uncommitted_new=()
# 一次快照，别在每个清单条目上重复调 git（§8.10 第二条：本仓库里**未提交**的内容照样会跟着
# 同步进生产 checkout，而且没人看见）。这里只**摊开**、不阻塞——哪些散改算"该进包的"是
# 提交分组的事，由你定；脚本至少让"包里有多少未提交内容"变成一个每次都会打印的数。
STATUS_RAW="$(git -C "$LINGJING_ROOT" status --porcelain -uall -- \
  arc_adaptor/agents tests tools 2>/dev/null || true)"
for pair in "${SYNC_PAIRS[@]}"; do
  src="${pair%%$'\t'*}"
  dst="${pair#*$'\t'}"
  rel="${src#"$LINGJING_ROOT"/}"
  if grep -qxF "?? $rel" <<<"$STATUS_RAW"; then
    uncommitted_new+=("$rel")
  elif grep -qE "^[ MADRC!]{1,2} \"?${rel//./\\.}\"?$" <<<"$STATUS_RAW"; then
    uncommitted_mod+=("$rel")
  fi
  if [[ ! -f "$dst" ]]; then
    new_files=$((new_files + 1)); continue
  fi
  if cmp -s "$src" "$dst"; then identical=$((identical + 1)); continue; fi
  if in_history "$dst" "$rel"; then
    recoverable_diffs+=("$dst|$(sha16 "$src")|$(sha16 "$dst")")
  else
    unrecoverable_diffs+=("$dst|$(sha16 "$src")|$(sha16 "$dst")")
  fi
done

printf 'arc_root=%s\n' "$ARC_DIR"
printf 'arc_commit=%s\n' "$(git rev-parse HEAD)"
printf 'preflight: declared=%s identical=%s new=%s overwritable=%s unrecoverable=%s\n' \
  "${#SYNC_PAIRS[@]}" "$identical" "$new_files" \
  "${#recoverable_diffs[@]}" "${#unrecoverable_diffs[@]}"
printf 'sha 取 sha256 前 16 位；比对口径为两侧都去掉 CR 后的字节（CRLF 归一）\n'

printf 'uncommitted_sources: modified=%s untracked=%s (这些条目本仓库里还没提交，同步等于把工作树内容送进生产 checkout)\n' \
  "${#uncommitted_mod[@]}" "${#uncommitted_new[@]}"
if (( ${#uncommitted_mod[@]} )); then
  printf '  M %s\n' "${uncommitted_mod[@]:0:15}"
  if (( ${#uncommitted_mod[@]} > 15 )); then printf '  ... 共 %s 项\n' "${#uncommitted_mod[@]}"; fi
fi
if (( ${#uncommitted_new[@]} )); then
  printf '  ? %s\n' "${uncommitted_new[@]:0:15}"
  if (( ${#uncommitted_new[@]} > 15 )); then printf '  ... 共 %s 项\n' "${#uncommitted_new[@]}"; fi
fi

if (( ${#recoverable_diffs[@]} )); then
  printf 'would_overwrite (ARC 侧那份在本仓库历史里对得上版本，盖掉不丢内容):\n'
  printf '  %s\n' "${recoverable_diffs[@]}"
fi
if (( ${#unrecoverable_diffs[@]} )); then
  printf 'BLOCKED — ARC 侧内容在本仓库任何历史版本里都对不上，覆盖即不可恢复:\n'
  printf '  %s\n' "${unrecoverable_diffs[@]}"
  printf '  处置：先把上面每行第一段 `diff -- <ARC 相对路径>` 看一遍，把该留的内容收进本仓库并提交，\n'
  printf '        再重跑本脚本。确实要用本仓库版本盖掉时加 --allow-overwrite（会先存档到\n'
  printf '        %s/.lingjing-sync-overwritten/<时间戳>/，那份存档属 ARC 侧，按规则 ③ 不得提回 Lingjing）。\n' "$ARC_DIR"
  if (( CHECK_ONLY )); then exit 1; fi
  if (( ! ALLOW_OVERWRITE )); then
    printf 'sync=ABORTED (未写任何文件)\n'
    exit 1
  fi
fi

if (( CHECK_ONLY )); then
  printf 'sync=CHECK_ONLY (未写任何文件)\n'
  exit 0
fi

# ── 覆盖前存档（只在 --allow-overwrite 且确有不可恢复差异时才有内容）────────────
if (( ${#unrecoverable_diffs[@]} )); then
  BACKUP_DIR="$ARC_DIR/.lingjing-sync-overwritten/$(date +%Y%m%d_%H%M%S)"
  for entry in "${unrecoverable_diffs[@]}"; do
    dst="${entry%%|*}"
    mkdir -p "$BACKUP_DIR/$(dirname "$dst")"
    cp "$dst" "$BACKUP_DIR/$dst"
  done
  printf 'backed_up_to=%s (ARC 侧，勿提回 Lingjing)\n' "$BACKUP_DIR"
fi

# ── 复制：只复制清单里声明过的目标 ──────────────────────────────────
mkdir -p agents/templates agents/strategies tests/unit tools
for pair in "${SYNC_PAIRS[@]}"; do
  src="${pair%%$'\t'*}"
  dst="${pair#*$'\t'}"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
done

# ── 复制后逐文件复核（ARC 侧那版带来的闸门）──────────────────────────
# 每个目标必须与它的源**逐字节**相同，否则 `manifest_mismatch` + exit 1。
# 打印格式沿用那版的 `<sha256>  <路径>`，别动：那是下游比对的口径。
# 与预检的分工：预检管"动手前 ARC 侧有什么会被盖"，这一环管"动手后落没落对"。
# 路径来自清单，所以不存在那版"仓库根 tests/ 找不到"的问题。
for pair in "${SYNC_PAIRS[@]}"; do
  src="${pair%%$'\t'*}"
  dst="${pair#*$'\t'}"
  source_hash="$(sha256sum "$src" | cut -d' ' -f1)"
  target_hash="$(sha256sum "$dst" | cut -d' ' -f1)"
  [[ "$source_hash" == "$target_hash" ]] || { echo "manifest_mismatch=$dst" >&2; exit 1; }
  printf '%s  %s\n' "$target_hash" "$dst"
done

if (( WITH_RECORDING_PATCH )); then
  git apply --check --unidiff-zero "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  git apply --unidiff-zero "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  echo "recording_patch=APPLIED"
else
  echo "recording_patch=NOT_APPLIED (optional; use --with-recording-patch)"
fi

# synced_files 现在是从清单**推出来的**，不再是手写的第二份名单——
# 之前"`cp -R` 实际搬了什么、`synced_files` 声明了什么"两者无法机械核对，就是这个原因。
printf 'synced_files (%s):\n' "${#SYNC_PAIRS[@]}"
for pair in "${SYNC_PAIRS[@]}"; do printf '%s\n' "${pair#*$'\t'}"; done

# agents/__init__.py **不在**上面：规则 2 要求不覆盖 ARC 原生 exports，
# 策略注册是 ARC checkout 里的手工步骤（改了要单独 git diff 复核，别当成同步产物）。
# 打印它的哈希只为留下一个可比对点——它变了就说明有人手工动过注册，不是同步造成的。
official_init_hash="$(sha256sum agents/__init__.py | cut -d' ' -f1)"
printf 'official_agents_init_sha256=%s\n' "$official_init_hash"
printf 'agents/__init__.py=NOT_SYNCED (register strategies manually, see docs/ARC-AGI3-adapter-architecture.md)\n'
printf '__pycache__=NOT_SYNCED (excluded from the manifest; the old `cp -R` did copy it)\n'

git diff --check
