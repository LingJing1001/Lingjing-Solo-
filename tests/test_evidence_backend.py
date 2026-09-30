"""#9 的验收闸门：`evidence_compat` 到底接哪一份实现，以及它**不许静默降级**。

为什么单开一个文件：待办 #9 的验收条件写在团队规范里——`report.json` 的 `protocol_backend`
要从 `"arc_adaptor.evidence_compat(fallback)"` 变成 `"lingjing_solo.evidence.protocol"`。
那是**唯一一个会撒谎的字段**：`evidence_compat` 原本是裸 `except ImportError`，
`lingjing_solo.evidence` 已经在了、但它自己 import 时炸了（缺三方依赖、符号改名），
也会被当成"分支还没合"而退回本地实现，字段于是继续报 fallback——
看起来"没合所以走兜底"，实际是"合了但坏了"。本文件钉住这件事的正反面。

盯住的五件事：
  1. 后端取值只有两种可能，且与"`lingjing_solo.evidence` 能不能 import"严格一致；
  2. 真的还没合并时，本地实现照常可用，且 build/validate/replay 三件套都在；
  3. **盘上有 `evidence/__init__.py`（= 分支已合）却 import 失败 → 抛出去**，不再静默退回；
  4. **父包自己缺依赖（真·合并前）→ 仍走兜底**——这条是本轮 `py -3.13` 逼出来的回归测试：
     上一版闸门按异常 `exc.name` 分家，而 `import lingjing_solo.evidence.protocol` 要先执行父包
     `lingjing_solo/__init__.py:5 → core/__init__.py:2 → core/types.py:11`（那里要 numpy），
     于是没装 numpy 的解释器在**合并前**也被认成"合了但坏了"，整条兜底被堵死；
  5. 无论哪个后端，§3.2 的 plan 闸门都生效——合并本身不许把九字段校验洗掉。

第 2/3/4 件事都得造"合并的另一侧"，而**在测试进程里造不真**：`lingjing_solo` 是 editable 装的，
在不在盘上由仓库自己决定，改 `sys.modules` 或挡 `sys.meta_path` 都改不了
`_evidence_pkg_on_disk()` 看的那份文件（合并后挡链会让闸门判成"已合但坏了"，测试就成了假绿）。
所以这三条一律**合成一个仓库桩**（`<tmp>/repo_stub/{arc_adaptor/evidence_compat.py, lingjing_solo/…}`）
再用 `python -I -S` 起子进程去 import 它：`-S` 让 `.pth` 根本不处理，桩就是唯一的那份 `lingjing_solo`。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

_ADAPTOR = pathlib.Path(__file__).resolve().parents[1] / "arc_adaptor"
if not _ADAPTOR.is_dir():                    # ARC checkout 里没有 arc_adaptor/（规则 ① 的已知越界）
    pytest.skip(f"被测模块不在当前 checkout 内: {_ADAPTOR}", allow_module_level=True)
sys.path.insert(0, str(_ADAPTOR))

COMPAT = _ADAPTOR / "evidence_compat.py"
FALLBACK = "arc_adaptor.evidence_compat(fallback)"
PROTOCOL = "lingjing_solo.evidence.protocol"

# 最小可用 protocol：够 compat 的 try 分支把那 10 个名字 import 进来即可。
_STUB_PROTOCOL = '''
"""桩：只为证明"盘上有且能 import"时后端会翻到 protocol 那一侧。"""
SCHEMA_VERSION = "lingjing-evidence-v1"


class EvidenceValidationError(ValueError):
    pass


class ReplayResult:
    pass


def _passthrough(**kwargs):
    return dict(kwargs)


def build_tick(**kwargs):
    return _passthrough(schema_version=SCHEMA_VERSION, **kwargs)


build_manifest = _passthrough
build_verification_report = _passthrough
validate_manifest = _passthrough
validate_tick = _passthrough
validate_verification_report = _passthrough
replay_recording = _passthrough
'''

# 桩协议故意坏掉时用的正文。
_BROKEN_PROTOCOL = "import definitely_not_a_real_module_xyz\n"

_PROBE = '''
"""子进程里 import 桩仓库的 evidence_compat，把三件事如实报回来（只 print ASCII JSON）。"""
import json
import sys

root = sys.argv[1]
sys.path.insert(0, root + "/arc_adaptor")
sys.path.insert(0, root)
out = {}
try:
    import evidence_compat as ev
    out["backend"] = ev.BACKEND
except ImportError as exc:
    print(json.dumps({"backend": "IMPORT_ERROR", "error": str(exc)}))
    raise SystemExit(0)

out["tick_schema"] = ev.build_tick(
    run_id="r", episode_id="e", tick=0, frame=[[0]], state="NOT_FINISHED",
    levels_completed=0, legal_actions=["ACTION1"], state_hash="h")["schema_version"]
for name in ("build_manifest", "build_tick", "build_verification_report", "validate_manifest",
             "validate_tick", "validate_verification_report", "replay_recording",
             "EvidenceValidationError", "SCHEMA_VERSION"):
    if not hasattr(ev, name):
        out.setdefault("missing", []).append(name)
try:                                            # §3.2 九字段闸门：缺字段必须现形
    ev.build_tick(run_id="r", episode_id="e", tick=0, frame=[[0]], state="NOT_FINISHED",
                  levels_completed=0, legal_actions=["ACTION1"], state_hash="h",
                  plan={"schema_version": "lingjing-r3-plan-v1"})
    out["plan_gate"] = "MISSING"
except Exception as exc:
    out["plan_gate"] = type(exc).__name__
print(json.dumps(out))
'''


def _make_stub(tmp_path: pathlib.Path, *, merged: bool, parent_body: str = "",
               protocol_body: str = _STUB_PROTOCOL) -> pathlib.Path:
    """造一个"仓库桩"：compat 文件照抄，`lingjing_solo` 换成正文由本函数决定的空壳包。

    桩父包默认不 import 任何东西（真父包要 numpy，会把这组测试拖成 numpy 依赖测试），
    只留 `evidence/` 这个目录在不在，作为"分支合没合"的唯一差异。
    """
    root = tmp_path / "repo_stub"
    (root / "arc_adaptor").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(COMPAT, root / "arc_adaptor" / "evidence_compat.py")
    pkg = root / "lingjing_solo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(parent_body, encoding="utf-8")
    if merged:
        evidence = pkg / "evidence"
        evidence.mkdir()
        (evidence / "__init__.py").write_text("", encoding="utf-8")
        (evidence / "protocol.py").write_text(protocol_body, encoding="utf-8")
    (root / "probe.py").write_text(_PROBE, encoding="utf-8")
    return root


def _run_stub(root: pathlib.Path) -> dict:
    """`-I -S` 起子进程：`-S` 让 site 的 `.pth`（本仓 editable 装的那份）根本不处理。"""
    done = subprocess.run([sys.executable, "-I", "-S", str(root / "probe.py"), str(root)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=120)
    assert done.returncode == 0, f"桩进程非零退出：{done.stderr[-400:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def compat():
    import evidence_compat as ev                    # noqa: PLC0415
    return ev


def test_backend_matches_whether_the_branch_package_is_importable(compat):
    ok = (COMPAT.parents[1] / "lingjing_solo" / "evidence" / "__init__.py").is_file()
    assert compat.BACKEND in {FALLBACK, PROTOCOL}, f"未知后端 {compat.BACKEND}"
    assert (compat.BACKEND == PROTOCOL) is ok, (
        f"后端与包在不在盘上不一致: BACKEND={compat.BACKEND} on_disk={ok}")


def test_merged_and_importable_selects_the_protocol_backend(tmp_path):
    """桩里放了能 import 的 evidence/ → 必须翻到 protocol（#9 验收的那一侧）。"""
    result = _run_stub(_make_stub(tmp_path, merged=True))
    assert result["backend"] == PROTOCOL, result
    assert "missing" not in result, f"protocol 侧少件：{result['missing']}"
    assert result.get("plan_gate") == "EvidenceValidationError", result


def test_not_merged_still_uses_the_local_fallback(tmp_path):
    """桩里没有 evidence/ → 兜底实现照常可用（这是兜底存在的全部理由）。"""
    result = _run_stub(_make_stub(tmp_path, merged=False))
    assert result["backend"] == FALLBACK, result
    assert result["tick_schema"] == "lingjing-evidence-v1", result
    assert "missing" not in result, f"兜底实现少了件：{result['missing']}"
    assert result.get("plan_gate") == "EvidenceValidationError", result


def test_merged_but_broken_package_is_not_silently_downgraded(tmp_path):
    """盘上有 `evidence/__init__.py` 却 import 炸 → 抛出去，不许谎报 fallback。"""
    root = _make_stub(tmp_path, merged=True, protocol_body=_BROKEN_PROTOCOL)
    result = _run_stub(root)
    assert result["backend"] == "IMPORT_ERROR", f"静默退回了：{result}"
    assert "拒绝静默退回" in result["error"], result["error"]
    assert "definitely_not_a_real_module_xyz" in result["error"], result["error"]


def test_parent_import_error_is_not_mistaken_for_merged(tmp_path):
    """父包缺依赖（`py -3.13` 的真形状）且**没合并** → 必须仍是兜底，不是硬崩。

    上一版按 `exc.name` 分家就是在这条上翻车的：抛出来的名字是三方件（numpy），不在
    "lingjing_solo.evidence" 那条命名链里，于是被当成"合了但坏了"。这条测试不需要真找一个
    没装 numpy 的解释器——桩父包 import 一个不存在的模块，形状完全一样。
    """
    root = _make_stub(tmp_path, merged=False, parent_body=_BROKEN_PROTOCOL)
    result = _run_stub(root)
    assert result["backend"] == FALLBACK, f"被误判成已合并：{result}"
    assert result["tick_schema"] == "lingjing-evidence-v1", result


def test_plan_gate_survives_whichever_backend_wins(compat):
    """九字段校验长在 compat 的包装层里，不许因为换后端而消失（§3.2 落盘即校验）。"""
    with pytest.raises(compat.EvidenceValidationError):
        compat.build_tick(run_id="r", episode_id="e", tick=0, frame=[[0]],
                          state="NOT_FINISHED", levels_completed=0,
                          legal_actions=["ACTION1"], state_hash="h",
                          plan={"schema_version": "lingjing-r3-plan-v1"})      # 缺其余字段
    # 不带 plan（旧 recording 的形状）必须仍放过，否则 replay 既有证据会红
    compat.build_tick(run_id="r", episode_id="e", tick=0, frame=[[0]],
                      state="NOT_FINISHED", levels_completed=0,
                      legal_actions=["ACTION1"], state_hash="h")
