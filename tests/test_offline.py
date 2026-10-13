"""`lingjing_solo.harness.offline` 的测试：无网络比赛环境的就绪检查与护栏。

覆盖四块：
  1. 环境变量固化（幂等、尊重实验开关）
  2. 网络护栏（放行边车/回环、拦公网、退出后还原 socket）
  3. 打包审计（缺模块 / 外部包 / 资产 / 保护链传播）
  4. 载荷自述解析（`_LJ_FILES` / `_EXT_FILES` / `_LJ_ASSETS`）

不 import numpy / torch / arcengine —— 本模块的设计前提就是"没有比赛依赖也能跑"。
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from lingjing_solo.harness import offline as O


# --------------------------------------------------------------------------- #
# 夹具：一棵最小的假仓库 + 一个假 notebook
# --------------------------------------------------------------------------- #


def _make_repo(tmp_path: Path) -> Path:
    """造一个最小仓库：受保护导入 + 硬导入 + 外部包。"""
    root = tmp_path / "repo"
    pkg = root / "lingjing_solo"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("", encoding="utf-8")
    # a.py：受保护导入 present（已发货）与 missing（未发货）
    (pkg / "a.py").write_text(
        "try:\n"
        "    from lingjing_solo.present import thing\n"
        "except Exception:\n"
        "    thing = None\n"
        "try:\n"
        "    from lingjing_solo.absent import other\n"
        "except Exception:\n"
        "    other = None\n"
        "from lingjing_solo.sub.leaf import hard  # 无保护\n",
        encoding="utf-8",
    )
    (pkg / "present.py").write_text("thing = 1\n", encoding="utf-8")
    (pkg / "absent.py").write_text("other = 2\n", encoding="utf-8")
    (pkg / "sub" / "leaf.py").write_text("hard = 3\n", encoding="utf-8")
    # 外部包：在 lingjing_solo 之外，结构上不属于 payload
    ext = root / "arc_ext"
    ext.mkdir()
    (ext / "__init__.py").write_text("", encoding="utf-8")
    (ext / "core.py").write_text("value = 4\n", encoding="utf-8")
    (pkg / "a.py").write_text(
        (pkg / "a.py").read_text(encoding="utf-8")
        + "from arc_ext.core import value\n",
        encoding="utf-8",
    )
    # 资产
    (root / "state").mkdir()
    (root / "state" / "w.npz").write_bytes(b"\x00\x01")
    (root / "my_agent.py").write_text(
        "from lingjing_solo.a import thing\n", encoding="utf-8"
    )
    return root


def _make_notebook(
    path: Path,
    lj_files: dict[str, str] | None = None,
    ext_files: dict[str, str] | None = None,
    assets: dict[str, str] | None = None,
) -> Path:
    cells = [
        {
            "cell_type": "code",
            "source": [
                "_LJ_FILES = {\n",
                *[f"    {k!r}: {v!r},\n" for k, v in (lj_files or {}).items()],
                "}\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "_EXT_FILES = {\n",
                *[f"    {k!r}: {v!r},\n" for k, v in (ext_files or {}).items()],
                "}\n",
            ],
        },
        {
            "cell_type": "code",
            "source": [
                "_LJ_ASSETS = {\n",
                *[f"    {k!r}: {v!r},\n" for k, v in (assets or {}).items()],
                "}\n",
            ],
        },
    ]
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 1. 环境变量
# --------------------------------------------------------------------------- #


def test_apply_offline_env_sets_all_and_is_idempotent():
    env: dict[str, str] = {}
    first = O.apply_offline_env(env)
    assert set(first) == set(O.OFFLINE_ENV)
    assert env["HF_HUB_OFFLINE"] == "1"
    # 第二次什么都不该改
    assert O.apply_offline_env(env) == {}
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_apply_offline_env_respects_explicit_ablation_switch():
    """已显式设置的消融开关不能被静默改掉，否则本地实验结论不可信。"""
    env = {"CLICK_HEATMAP_WEIGHTS": "/my/ablation.npz"}
    changed = O.apply_offline_env(env)
    assert env["CLICK_HEATMAP_WEIGHTS"] == "/my/ablation.npz"
    assert "CLICK_HEATMAP_WEIGHTS" not in changed
    # 其它变量照旧写入
    assert env["HF_HUB_OFFLINE"] == "1"


def test_apply_offline_env_fills_blank_heatmap_var():
    """空值不算"显式设置"，应当被填上。"""
    env = {"CLICK_HEATMAP_WEIGHTS": ""}
    changed = O.apply_offline_env(env)
    assert env["CLICK_HEATMAP_WEIGHTS"] == O.OFFLINE_ENV["CLICK_HEATMAP_WEIGHTS"]
    assert "CLICK_HEATMAP_WEIGHTS" in changed


# --------------------------------------------------------------------------- #
# 2. 网络护栏
# --------------------------------------------------------------------------- #


def test_network_guard_blocks_public_host():
    with pytest.raises(O.OfflineViolation):
        with O.network_guard():
            socket.create_connection(("example.com", 80), timeout=0.01)


def test_network_guard_allows_gateway_the_competition_sidecar():
    """`gateway` 是比赛边车，必须放行 —— 否则连游戏环境都拿不到。"""
    assert O._host_allowed("gateway", O.DEFAULT_ALLOW_HOSTS)
    with O.network_guard():
        with pytest.raises(OSError) as excinfo:
            # 放行了才会走到真正的连接尝试（并因无人监听而失败）
            socket.create_connection(("127.0.0.1", 9), timeout=0.05)
    assert not isinstance(excinfo.value, O.OfflineViolation)


def test_network_guard_record_mode_collects_without_raising():
    with O.network_guard(on_violation="record") as violations:
        for _ in range(2):
            try:
                socket.create_connection(("93.184.216.34", 80), timeout=0.01)
            except Exception:
                pass
    assert violations and violations[0][0] == "93.184.216.34"


def test_network_guard_restores_socket_on_exit():
    before = socket.socket.connect
    with O.network_guard():
        assert socket.socket.connect is not before
    assert socket.socket.connect is before


def test_host_allowed_treats_unix_socket_and_loopback_as_local():
    assert O._host_allowed(None, O.DEFAULT_ALLOW_HOSTS)
    assert O._host_allowed("::1", O.DEFAULT_ALLOW_HOSTS)
    assert not O._host_allowed("8.8.8.8", O.DEFAULT_ALLOW_HOSTS)


# --------------------------------------------------------------------------- #
# 3. 导入闭包与保护链
# --------------------------------------------------------------------------- #


def test_import_closure_finds_unguarded_and_guarded(tmp_path):
    root = _make_repo(tmp_path)
    local, external, third = O.import_closure(root / "my_agent.py", root)
    assert "lingjing_solo.a" in local
    assert "lingjing_solo.present" in local
    # present/absent 都在 try 里 → 受保护
    assert local["lingjing_solo.present"].guarded
    assert local["lingjing_solo.absent"].guarded
    # leaf 由 a.py 的无保护导入到达 → 不受保护
    assert "lingjing_solo.sub.leaf" in local
    assert not local["lingjing_solo.sub.leaf"].guarded
    assert "arc_ext" in external


def test_guard_propagates_along_the_chain(tmp_path):
    """保护要沿链传播：受保护的导入，它自己再硬导入别的模块也算受保护。

    这正是 `engine_solver`（被 my_agent 的 try 包着）在模块级硬导入
    `harness.engine_action_adapter` 的情况 —— 缺文件只是降级，不会崩。
    """
    root = tmp_path / "repo"
    pkg = root / "lingjing_solo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "outer.py").write_text("from lingjing_solo.inner import x\n", encoding="utf-8")
    (pkg / "inner.py").write_text("x = 1\n", encoding="utf-8")
    (root / "my_agent.py").write_text(
        "try:\n    from lingjing_solo.outer import x\nexcept Exception:\n    x = None\n",
        encoding="utf-8",
    )
    local, _, _ = O.import_closure(root / "my_agent.py", root)
    assert local["lingjing_solo.outer"].guarded
    assert local["lingjing_solo.inner"].guarded, "保护没有沿链传播"


def test_third_party_split_hard_vs_soft(tmp_path):
    root = tmp_path / "repo"
    pkg = root / "lingjing_solo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "m.py").write_text(
        "import numpy\n"
        "try:\n"
        "    import torch\n"
        "except Exception:\n"
        "    torch = None\n",
        encoding="utf-8",
    )
    (root / "my_agent.py").write_text("from lingjing_solo.m import _\n", encoding="utf-8")
    _, _, third = O.import_closure(root / "my_agent.py", root)
    assert third["numpy"]["hard"]
    assert third["torch"]["soft"]
    assert not third["torch"]["hard"]


# --------------------------------------------------------------------------- #
# 4. 打包审计
# --------------------------------------------------------------------------- #


def test_audit_flags_missing_module_and_asset(tmp_path):
    root = _make_repo(tmp_path)
    nb = _make_notebook(
        tmp_path / "nb.ipynb",
        lj_files={"__init__.py": "", "a.py": "", "present.py": "", "sub/__init__.py": ""},
    )
    audit = O.audit_packaging(
        root / "my_agent.py",
        nb,
        root,
        asset_spec={"w": "state/w.npz"},
        asset_payload_map={"w": "assets/w.npz"},
    )
    # absent.py 没发货但受保护 → 降级；sub/leaf.py 没发货且硬引用 → 会崩
    assert "lingjing_solo.absent" in audit.will_degrade
    assert "lingjing_solo.sub.leaf" in audit.will_crash
    assert not audit.ok
    assert any("会崩" in b for b in audit.blockers())


def test_audit_passes_when_payload_is_complete(tmp_path):
    root = _make_repo(tmp_path)
    nb = _make_notebook(
        tmp_path / "nb.ipynb",
        lj_files={
            "__init__.py": "",
            "a.py": "",
            "present.py": "",
            "absent.py": "",
            "sub/__init__.py": "",
            "sub/leaf.py": "",
        },
        ext_files={"arc_ext/core.py": "", "arc_ext/__init__.py": ""},
        assets={"assets/w.npz": "AAAA"},
    )
    audit = O.audit_packaging(
        root / "my_agent.py",
        nb,
        root,
        asset_spec={"w": "state/w.npz"},
        asset_payload_map={"w": "assets/w.npz"},
    )
    assert audit.will_crash == []
    assert audit.will_degrade == []
    assert audit.unshipped_external == {}
    assert audit.missing_assets == []
    assert audit.ok, audit.blockers()


def test_external_package_is_a_blocker_until_shipped(tmp_path):
    root = _make_repo(tmp_path)
    nb = _make_notebook(tmp_path / "nb.ipynb", lj_files={"__init__.py": ""})
    audit = O.audit_packaging(root / "my_agent.py", nb, root)
    assert "arc_ext" in audit.unshipped_external
    assert any("arc_ext" in b for b in audit.blockers())


def test_asset_present_locally_but_not_in_payload_is_a_blocker(tmp_path):
    """仓库里有 ≠ 提交包里有 —— 这正是热图权重曾经的坑。"""
    root = _make_repo(tmp_path)
    nb = _make_notebook(
        tmp_path / "nb.ipynb",
        lj_files={
            "__init__.py": "",
            "a.py": "",
            "present.py": "",
            "absent.py": "",
            "sub/__init__.py": "",
            "sub/leaf.py": "",
        },
        ext_files={"arc_ext/core.py": ""},
    )
    audit = O.audit_packaging(
        root / "my_agent.py",
        nb,
        root,
        asset_spec={"w": "state/w.npz"},
        asset_payload_map={"w": "assets/w.npz"},
    )
    assert (root / "state" / "w.npz").is_file()
    assert [a.key for a in audit.missing_assets] == ["w"]


def test_notebook_extra_payload_reads_self_description(tmp_path):
    nb = _make_notebook(
        tmp_path / "nb.ipynb",
        lj_files={"__init__.py": ""},
        ext_files={"arc_adaptor/click_heatmap.py": "x=1", "arc_adaptor/__init__.py": ""},
        assets={"assets/click_heatmap_v3.npz": "AAAA"},
    )
    extra, assets = O.notebook_extra_payload(nb)
    assert extra == {"arc_adaptor"}
    assert assets == {"assets/click_heatmap_v3.npz"}


def test_notebook_payload_files_missing_var_raises(tmp_path):
    """格式变了要炸，不能静默返回空（否则审计误报"全都缺"）。"""
    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["x = 1\n"]}]}))
    with pytest.raises(KeyError):
        O.notebook_payload_files(nb)


def test_readiness_report_shape(tmp_path):
    root = _make_repo(tmp_path)
    nb = _make_notebook(tmp_path / "nb.ipynb", lj_files={"__init__.py": ""})
    report = O.readiness_report(root, root / "my_agent.py", nb)
    assert set(report) >= {"ok", "env", "allow_hosts", "packaging", "blockers"}
    assert report["ok"] is False
    assert "gateway" in report["allow_hosts"]
    assert report["packaging"]["will_crash"]


def test_deferred_gap_is_accepted_not_blocker(tmp_path, monkeypatch):
    """有意不发货的东西要单列成"已接受"，不能算缺口、也不能被静默吞掉。"""
    root = _make_repo(tmp_path)
    nb = _make_notebook(
        tmp_path / "nb.ipynb",
        lj_files={
            "__init__.py": "",
            "a.py": "",
            "present.py": "",
            "absent.py": "",
            "sub/__init__.py": "",
            "sub/leaf.py": "",
        },
        # 刻意**不**声明 arc_ext 已随包：这样它才是"未发货但被 defer"。
    )
    monkeypatch.setattr(O, "DEFERRED", {"arc_ext": "有意不发：理由"})
    audit = O.audit_packaging(root / "my_agent.py", nb, root, asset_spec={})
    assert audit.unshipped_external == {}
    assert [n for n, _ in audit.accepted_gaps] == ["arc_ext"]
    assert not any("arc_ext" in b for b in audit.blockers())
    assert "已接受" in audit.render()


def test_dangling_local_import_is_not_treated_as_packaging_gap(tmp_path):
    """引用了仓库里**不存在**的模块 → 不是打包缺口（补包也修不了），
    应当落到第三方软依赖里，而不是让审计报一条查不动的阻断项。"""
    root = tmp_path / "repo"
    pkg = root / "lingjing_solo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "m.py").write_text(
        "try:\n    import ghost_adapter\n"
        "except Exception:\n    ghost_adapter = None\n",
        encoding="utf-8",
    )
    (root / "my_agent.py").write_text("from lingjing_solo.m import _\n", encoding="utf-8")
    nb = _make_notebook(tmp_path / "nb.ipynb", lj_files={"__init__.py": "", "m.py": ""})
    audit = O.audit_packaging(root / "my_agent.py", nb, root, asset_spec={})
    assert audit.external_packages == {}
    assert audit.third_party["ghost_adapter"]["soft"]
    assert audit.ok, audit.blockers()


def test_module_file_resolves_package_init(tmp_path):
    root = _make_repo(tmp_path)
    assert O._module_file("lingjing_solo", root) == root / "lingjing_solo" / "__init__.py"
    assert (
        O._module_file("lingjing_solo.sub", root)
        == root / "lingjing_solo" / "sub" / "__init__.py"
    )
