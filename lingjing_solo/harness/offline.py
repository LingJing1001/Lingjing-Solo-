"""正式比赛无网络环境下的**就绪检查**与**硬护栏**。

比赛执行的拓扑（从 `notebooks/submission.ipynb` 实证，不是猜）：

    ┌─────────────────────── Kaggle 容器 ───────────────────────┐
    │  enable_internet = false  →  容器出不了公网                 │
    │  arc_agi_3_wheels/       →  赛方挂载的 wheel，供离线 pip     │
    │  ARC-AGI-3-Agents/       →  赛方挂载的框架（含 agents 包）   │
    │  environment_files/      →  25 个游戏的源码（本地，无需下载） │
    │  gateway:8001            →  **容器内的边车**，喂环境 + 记分   │
    └────────────────────────────────────────────────────────────┘

三个容易搞错的推论：

1. **"没网络"不等于"没 HTTP"**。官方 `main.py` 用
   ``OPERATION_MODE=online`` + ``ARC_BASE_URL=http://gateway:8001/``，
   走的是容器内的 gateway 边车。所以网络护栏的**放行名单**必须是
   ``localhost / 127.0.0.1 / ::1 / gateway``，而不是"全禁"。
2. **构建期能上网，运行期不能**。本地 `OperationMode.NORMAL` 会真的去
   拉 25 个环境（实测能拉到）。所以在本地验证"离线能不能跑"，
   必须**显式** ``mode="offline"``，否则测的是联网路径，白测。
3. **任何运行期下载都是致命的**。``from_pretrained`` / HF hub / pip
   一律不可用，权重只能来自 ``model_sources`` 挂载的只读输入目录。

本模块只做三件事，都是**只读/校验**性质，不改环境语义：

- :func:`apply_offline_env` —— 把离线环境变量钉死（幂等，返回改了哪些）
- :func:`network_guard` —— 出网即抛 :class:`OfflineViolation`（放行边车）
- :func:`audit_packaging` / :func:`readiness_report` —— 审计提交包缺了什么

纯 stdlib，不 import arcengine / numpy / torch。理由同 `harness/arc_env.py`：
这个模块要在**没有任何比赛依赖**的解释器里也能跑，否则它自己就成了依赖。
"""

from __future__ import annotations

import ast
import ipaddress
import json
import os
import socket
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

__all__ = [
    "OfflineViolation",
    "OFFLINE_ENV",
    "DEFAULT_ALLOW_HOSTS",
    "apply_offline_env",
    "network_guard",
    "LocalAsset",
    "DEFAULT_ASSETS",
    "check_assets",
    "ImportRef",
    "import_closure",
    "notebook_payload_files",
    "notebook_extra_payload",
    "PackagingAudit",
    "audit_packaging",
    "readiness_report",
]


# --------------------------------------------------------------------------- #
# 1. 环境变量固化
# --------------------------------------------------------------------------- #

#: 离线环境应当钉死的变量。全部是"**关闭私网外呼**"语义，不是"加速"。
#: 之所以要在启动时显式设置：这些库的默认值都是"联网优先、失败回退"，
#: 在无网环境里那意味着每次加载都先等一轮 DNS 超时 —— 纯浪费 9 小时预算。
OFFLINE_ENV: Dict[str, str] = {
    # HuggingFace 栈：禁止任何 hub 往返（含隐式 token 探测）
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_ETAG_TIMEOUT": "1",
    # vLLM：关统计上报（离线环境下会重试到超时）
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
    # 通用隐私 / 遥测开关
    "DO_NOT_TRACK": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # 缺省权重路径：见 §3，必须指到"会被打包进提交"的位置
    "CLICK_HEATMAP_WEIGHTS": "/tmp/lingjing_solo/assets/click_heatmap_v3.npz",
}

#: 这些变量属于实验开关，已显式设置时**不覆盖**（否则本地消融会被静默改掉）。
_RESPECT_EXISTING: Tuple[str, ...] = tuple(
    k for k in OFFLINE_ENV if k.startswith("CLICK_HEATMAP_")
)


def apply_offline_env(environ: Optional[Dict[str, str]] = None) -> Dict[str, Tuple[str, str]]:
    """把 :data:`OFFLINE_ENV` 写进环境变量，**幂等**。

    返回 ``{变量名: (旧值, 新值)}``，只含**真正发生变化**的项。空 dict
    表示环境已经就绪 —— 可以直接拿来断言"启动路径没有偷偷联网"。
    """
    target = os.environ if environ is None else environ
    changed: Dict[str, Tuple[str, str]] = {}
    for key, value in OFFLINE_ENV.items():
        old = target.get(key, "")
        if old == value:
            continue
        if old and key in _RESPECT_EXISTING:
            continue
        target[key] = value
        changed[key] = (old, value)
    return changed


# --------------------------------------------------------------------------- #
# 2. 网络护栏
# --------------------------------------------------------------------------- #

#: 放行名单 = 容器内边车 + 回环。**不要**在这里加公网域名。
DEFAULT_ALLOW_HOSTS: Tuple[str, ...] = (
    "gateway",                 # 比赛的边车服务名（http://gateway:8001）
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "host.docker.internal",
)


class OfflineViolation(RuntimeError):
    """在无网络环境里尝试出网。"""


def _host_allowed(host: Optional[str], allow_hosts: Iterable[str]) -> bool:
    if host is None:                       # AF_UNIX 等：不涉及网络
        return True
    host = str(host).strip().lower()
    if not host:
        return True
    if host in {h.lower() for h in allow_hosts}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@contextmanager
def network_guard(
    allow_hosts: Iterable[str] = DEFAULT_ALLOW_HOSTS,
    on_violation: str = "raise",
) -> Iterator[List[Tuple[str, int]]]:
    """拦截对非放行主机的 TCP 连接。

    ``on_violation="raise"``（默认）出网即抛 :class:`OfflineViolation`；
    ``"record"`` 只记录不抛 —— 给"我想知道到底谁在联网"的探测场景用，
    退出后从 yield 出来的列表读。

    只包 ``socket.socket.connect`` / ``socket.create_connection``，覆盖
    requests / urllib / httpx / aiohttp 的同步路径（asyncio 也落到底层同一个
    ``connect``，所以一并拦住）。**刻意不拦 UDP**：本项目没有 UDP 依赖，
    拦了只增加误报面。
    """
    violations: List[Tuple[str, int]] = []
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def _check(host: Any, port: Any) -> None:
        if _host_allowed(host, allow_hosts):
            return
        violations.append((str(host), int(port) if port is not None else -1))
        if on_violation == "raise":
            raise OfflineViolation(
                f"出网被拦截: {host}:{port}（放行名单 {tuple(allow_hosts)}）"
                " —— 正式比赛 enable_internet=false，运行期下载必然失败"
            )

    def connect(self, address):  # type: ignore[no-untyped-def]
        if isinstance(address, tuple) and len(address) >= 2:
            _check(address[0], address[1])
        return real_connect(self, address)

    def create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(address, tuple) and len(address) >= 2:
            _check(address[0], address[1])
        return real_create(address, *args, **kwargs)

    socket.socket.connect = connect                # type: ignore[assignment]
    socket.create_connection = create_connection   # type: ignore[assignment]
    try:
        yield violations
    finally:
        socket.socket.connect = real_connect               # type: ignore[assignment]
        socket.create_connection = real_create             # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 3. 本地资产校验
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalAsset:
    """一个需要**随提交包一起到达**的非 .py 文件。"""

    key: str
    path: Path
    why: str = ""
    optional: bool = False
    payload_rel: str = ""      # 在提交包里的落点（相对 /tmp/lingjing_solo）

    @property
    def present(self) -> bool:
        return self.path.is_file()

    @property
    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


#: 提交包里必须有、否则对应能力**静默降级**的数据资产。
DEFAULT_ASSETS: Dict[str, str] = {
    "heatmap_weights": "state/click_heatmap_v3.npz",
    "aop_checkpoint": "data/aop/collected_10eps.pt",
}

#: 资产在提交包里的落点约定（与 `build_notebook.ASSET_FILES` 对齐）。
#: 注意：**默认**没有 aop_checkpoint —— 288 KB 的 .pt base64 内联会直接顶穿
#: Kaggle 的 1 MiB 源码上限，所以它只走 dataset_sources，不走内联通道。
DEFAULT_ASSET_PAYLOAD: Dict[str, str] = {
    "heatmap_weights": "assets/click_heatmap_v3.npz",
}

_ASSET_WHY: Dict[str, str] = {
    "heatmap_weights": "arc_adaptor.click_heatmap 的权重，缺则点击热图提议全灭",
    "aop_checkpoint": (
        "neural.aop_controller 的检查点，缺则 AOP 控制器不加载"
        "（288 KB，只能走 kernel-metadata.json 的 dataset_sources，"
        "base64 内联会顶穿 Kaggle 的 1 MiB 源码上限）"
    ),
}


def check_assets(
    repo_root: Path,
    spec: Optional[Dict[str, str]] = None,
    optional_keys: Iterable[str] = (),
    payload_map: Optional[Dict[str, str]] = None,
) -> List[LocalAsset]:
    """检查数据资产在**仓库里**是否存在（存在 ≠ 已进提交包，见 §4）。"""
    root = Path(repo_root)
    spec = DEFAULT_ASSETS if spec is None else spec
    payload = DEFAULT_ASSET_PAYLOAD if payload_map is None else payload_map
    optional = set(optional_keys)
    return [
        LocalAsset(
            key=key,
            path=root / rel,
            why=_ASSET_WHY.get(key, ""),
            optional=key in optional,
            payload_rel=payload.get(key, ""),
        )
        for key, rel in spec.items()
    ]


# --------------------------------------------------------------------------- #
# 4. 打包缺口审计：my_agent 要的 vs 提交包里有的
# --------------------------------------------------------------------------- #

#: 打包器会把**这些包内部**的模块写进 `/tmp/lingjing_solo`，因此可以递归跟。
FOLLOW_PACKAGES: Tuple[str, ...] = ("lingjing_solo",)

#: 显式声明的"外部包"兜底名单。实际判定是**动态**的：某个顶层名在仓库根下
#: 存在同名目录或 `.py` 文件就当外部包（见 :func:`_repo_local_top`）。留这个
#: 常量是为了给"目录还没落盘但已被引用"的情况留个显式出口；当前为空。
EXTERNAL_PACKAGES: Tuple[str, ...] = ()


def _repo_local_top(name: str, root: Path) -> bool:
    """``name`` 是不是"仓库根下的顶层包/模块"（= 本项目的代码，但不是子包）。"""
    if not name:
        return False
    return (root / name).is_dir() or (root / f"{name}.py").is_file()


#: **有意不随提交包发货**的东西，及理由。列入这里不算"缺口"，但会在报告里
#: 单独列成"已接受"—— 目的是让"我们知道自己放弃了什么"变成一条记录，
#: 而不是一个被审计静默吞掉的疏漏。
DEFERRED: Dict[str, str] = {
    "su15_solver": (
        "仓库根的单文件求解器，只在内联 _SU15_PLANS 不全时才走；"
        "它还要 environment_files/su15/<hash>/su15.py 这类游戏源码上 sys.path，"
        "打包成本高、收益仅覆盖兜底路径。主路径（内联计划）已随包。"
    ),
    "aop_checkpoint": (
        "288 KB 的 .pt，base64 内联会顶穿 Kaggle 的 1 MiB 源码上限；"
        "要启用须走 kernel-metadata.json 的 dataset_sources。"
    ),
}


@dataclass(frozen=True)
class ImportRef:
    """一次 import 引用。"""

    name: str
    referenced_by: Path
    guarded: bool          # 在 try/except 或 if 里 → 失败只降级不崩
    lazy: bool             # 在函数体内 → 惰性导入，但仍需文件在场


def _iter_imports(path: Path) -> List[ImportRef]:
    """列出一个文件里所有 import（含惰性 / 受保护的），带上下文标记。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out: List[ImportRef] = []

    def walk(node: ast.AST, guarded: bool, lazy: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                names: List[str] = []
                if isinstance(child, ast.Import):
                    names = [a.name for a in child.names]
                elif child.level == 0 and child.module:
                    names = [child.module]
                out.extend(ImportRef(n, path, guarded, lazy) for n in names)
            elif isinstance(child, (ast.Try, ast.If)):
                walk(child, True, lazy)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, guarded, True)
            else:
                walk(child, guarded, lazy)

    walk(tree, False, False)
    return out


def _build_index(repo_root: Path) -> Dict[str, Path]:
    """点号模块名 → 仓库内文件路径。"""
    root = Path(repo_root).resolve()
    index: Dict[str, Path] = {}
    for pkg_name in FOLLOW_PACKAGES:
        pkg = root / pkg_name
        if not pkg.is_dir():
            continue
        index[pkg_name] = pkg / "__init__.py"
        for f in pkg.rglob("*.py"):
            parts = list(f.relative_to(pkg).with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            if parts:
                index[f"{pkg_name}." + ".".join(parts)] = f
    return index


def import_closure(
    entry: Path,
    repo_root: Path,
) -> Tuple[Dict[str, ImportRef], Dict[str, List[str]], Dict[str, Dict[str, bool]]]:
    """从 ``entry`` 出发遍历**真实导入闭包**。

    与 `build_notebook.py` 的差别是刻意造的：跟进**惰性导入**（函数体内），
    因为打包器的 roots 是手写清单，会随代码增长而失准。区分 ``guarded``：
    受保护的引用缺了只是**静默降级**，不受保护的缺了**直接崩** —— 都要报，
    但严重度不同。

    **保护沿链传播**：``engine_solver`` 自己在模块级硬导入
    ``harness.engine_action_adapter``，但它是被 `my_agent` 的 ``try`` 包着导入的，
    所以整条链上缺文件都只是降级、不会崩。判定方式是对每个文件记录
    "**是否存在一条完全不受保护的到达路径**"，而不是看单条边的标记。

    返回 ``(本地模块名 -> ImportRef, 外部包名 -> 引用它的文件, 第三方依赖)``。
    """
    root = Path(repo_root).resolve()
    index = _build_index(root)
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    local_names: Dict[str, Path] = {}
    local_ref: Dict[str, ImportRef] = {}
    external: Dict[str, List[str]] = {}
    third_party: Dict[str, Dict[str, bool]] = {}

    # file -> 是否存在"全程无保护"的到达路径
    naked: Dict[Path, bool] = {}
    stack: List[Tuple[Path, bool]] = [(Path(entry).resolve(), False)]
    while stack:
        current, chain_guarded = stack.pop()
        if not current.is_file():
            continue
        this_naked = not chain_guarded
        if current in naked and (naked[current] or not this_naked):
            continue          # 已按"至少一样激进"的方式访问过
        naked[current] = naked.get(current, False) or this_naked

        for ref in _iter_imports(current):
            top = ref.name.split(".")[0]
            if top in stdlib or top == "__future__":
                continue
            child_guarded = chain_guarded or ref.guarded
            if top in FOLLOW_PACKAGES:
                parts = [p for p in ref.name.split(".") if p]
                resolved = next(
                    (
                        ".".join(parts[:i])
                        for i in range(len(parts), 0, -1)
                        if ".".join(parts[:i]) in index
                    ),
                    top,
                )
                local_names.setdefault(resolved, index.get(resolved, Path()))
                local_ref.setdefault(resolved, ref)
                target = index.get(resolved)
                if target is not None:
                    stack.append((target, child_guarded))
                continue
            if top in EXTERNAL_PACKAGES or _repo_local_top(top, root):
                # 仓库里存在、但不是 lingjing_solo 子包：打包器**结构上**
                # 放不进 /tmp/lingjing_solo，必须走独立通道。
                external.setdefault(top, []).append(
                    str(ref.referenced_by.relative_to(root))
                )
                continue
            flags = third_party.setdefault(top, {"hard": False, "soft": False})
            flags["soft" if child_guarded else "hard"] = True

    local = {
        name: ImportRef(
            name=local_ref[name].name,
            referenced_by=local_ref[name].referenced_by,
            guarded=not naked.get(path, False),
            lazy=local_ref[name].lazy,
        )
        for name, path in local_names.items()
    }
    for name in external:
        external[name] = sorted(set(external[name]))
    return local, external, third_party


def _notebook_dict(notebook_path: Path, var: str) -> Dict[str, str]:
    """从提交 notebook 里读出某个顶层字符串字典（``_LJ_FILES`` / ``_EXT_FILES`` / ``_LJ_ASSETS``）。

    读不出来就抛 :class:`KeyError`，**不静默返回空** —— 否则审计会把
    "打包格式变了"误报成"所有东西都缺"，反而误导。
    """
    nb = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if var not in src:
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            if not any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
                continue
            return {
                k.value: (v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "")
                for k, v in zip(node.value.keys, node.value.values)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    raise KeyError(f"{notebook_path} 里找不到 {var}（打包格式变了？）")


def notebook_payload_files(notebook_path: Path) -> Dict[str, str]:
    """提交 notebook 的 ``_LJ_FILES``（键 = 相对 ``/tmp/lingjing_solo`` 的路径）。"""
    return _notebook_dict(notebook_path, "_LJ_FILES")


def notebook_extra_payload(notebook_path: Path) -> Tuple[Set[str], Set[str]]:
    """提交 notebook 自描述的**外部包**与**资产**载荷。

    返回 ``(外部包顶层名集合, 资产落点集合)``。外部包来自 ``_EXT_FILES`` 的键
    （``<pkg>/<rel>``），资产来自 ``_LJ_ASSETS`` 的键。这样审计就不需要另配
    一份"哪些包被带了"的清单 —— 直接读包里的自述，不会漂移。
    """
    extra: Set[str] = set()
    try:
        for key in _notebook_dict(notebook_path, "_EXT_FILES"):
            head = key.split("/", 1)[0].strip()
            if head:
                extra.add(head)
    except KeyError:
        pass
    assets: Set[str] = set()
    try:
        assets = set(_notebook_dict(notebook_path, "_LJ_ASSETS"))
    except KeyError:
        pass
    return extra, assets



def _payload_key(path: Path, repo_root: Path) -> Optional[str]:
    """仓库内文件 → 提交包 ``_LJ_FILES`` 的键（相对 ``lingjing_solo/``）。"""
    try:
        return (
            Path(path)
            .resolve()
            .relative_to((Path(repo_root) / "lingjing_solo").resolve())
            .as_posix()
        )
    except ValueError:
        return None


@dataclass
class PackagingAudit:
    """提交包 vs 真实依赖的差集报告。"""

    entry: Path
    repo_root: Path
    local_modules: Dict[str, Path] = field(default_factory=dict)
    unguarded: Set[str] = field(default_factory=set)
    shipped_keys: Set[str] = field(default_factory=set)
    shipped_external: Set[str] = field(default_factory=set)
    shipped_assets: Set[str] = field(default_factory=set)
    external_packages: Dict[str, List[str]] = field(default_factory=dict)
    third_party: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    assets: List[LocalAsset] = field(default_factory=list)
    deferred: Dict[str, str] = field(default_factory=dict)

    # ---- 判定 ------------------------------------------------------------- #

    @property
    def missing_local(self) -> List[str]:
        """在闭包里、但提交包没带的本地模块（硬引用优先排前）。"""
        missing = [
            name
            for name, path in self.local_modules.items()
            if (key := _payload_key(path, self.repo_root)) is None
            or key not in self.shipped_keys
        ]
        return sorted(missing, key=lambda n: (n not in self.unguarded, n))

    @property
    def will_crash(self) -> List[str]:
        """缺失且**没有 try 保护** —— 这些会让提交直接 ImportError。"""
        return [n for n in self.missing_local if n in self.unguarded]

    @property
    def will_degrade(self) -> List[str]:
        """缺失但有 try 保护 —— 提交能起来，但能力悄悄没了。"""
        return [n for n in self.missing_local if n not in self.unguarded]

    @property
    def unshipped_external(self) -> Dict[str, List[str]]:
        """被引用、但提交包**确实没带**、且**没有被列入 defer** 的外部包。

        判定依据是 notebook 里 ``_EXT_FILES`` 的自述（见
        :func:`notebook_extra_payload`），而不是另配清单 —— 免得两处漂移。
        """
        return {
            pkg: where
            for pkg, where in self.external_packages.items()
            if pkg not in self.shipped_external and pkg not in self.deferred
        }

    @property
    def missing_assets(self) -> List[LocalAsset]:
        """仓库里没有、或没进提交包载荷的资产（defer 的不算）。"""
        return [
            a
            for a in self.assets
            if not a.optional
            and a.key not in self.deferred
            and (not a.present or a.payload_rel not in self.shipped_assets)
        ]

    @property
    def accepted_gaps(self) -> List[Tuple[str, str]]:
        """**有意不发货**的东西（在报告里单列，不算缺口）。

        这一项的用意是把"我们知道自己放弃了什么"变成一条记录 ——
        审计如果不记它，下次没人分得清"漏了"和"放弃的"。
        """
        out: List[Tuple[str, str]] = []
        for pkg in sorted(self.external_packages):
            if pkg not in self.shipped_external and pkg in self.deferred:
                out.append((pkg, self.deferred[pkg]))
        for asset in self.assets:
            if asset.key in self.deferred and asset.payload_rel not in self.shipped_assets:
                out.append((asset.key, self.deferred[asset.key]))
        return out

    @property
    def hard_third_party(self) -> List[str]:
        return sorted(n for n, f in self.third_party.items() if f.get("hard"))

    @property
    def ok(self) -> bool:
        return not self.missing_local and not self.unshipped_external and not self.missing_assets

    def blockers(self) -> List[str]:
        """一句话一条的阻断项清单，可直接抄进结论。"""
        out = [f"提交包缺模块 {n}（硬引用 → 会崩）" for n in self.will_crash]
        out += [f"提交包缺模块 {n}（受保护 → 静默降级）" for n in self.will_degrade]
        for pkg, where in sorted(self.unshipped_external.items()):
            out.append(
                f"提交包结构上带不上外部包 {pkg}"
                f"（被 {len(where)} 个文件引用，如 {', '.join(where[:2])}）"
            )
        out += [f"提交包缺资产 {a.path.name}（{a.why}）" for a in self.missing_assets]
        return out

    def render(self) -> str:
        lines = [
            f"提交包审计  entry={self.entry.name}  repo={self.repo_root.name}",
            f"  闭包内本地模块 {len(self.local_modules)} 个 / "
            f"提交包内嵌 {len(self.shipped_keys)} 个文件",
            f"  缺模块 {len(self.missing_local)}"
            f"（会崩 {len(self.will_crash)} / 静默降级 {len(self.will_degrade)}）"
            f" / 缺资产 {len(self.missing_assets)}",
        ]
        if self.will_crash:
            lines.append("  -- 会崩 --")
            lines += [f"     {n}" for n in self.will_crash]
        if self.will_degrade:
            lines.append("  -- 静默降级 --")
            lines += [f"     {n}" for n in self.will_degrade]
        if self.external_packages:
            tag = "已随包" if not self.unshipped_external else "有缺口"
            lines.append(f"  -- 外部包（{tag}）--")
            for pkg in sorted(self.external_packages):
                mark = "OK " if pkg in self.shipped_external else "缺 "
                lines.append(f"    {mark} {pkg}")
        if self.assets:
            lines.append("  -- 数据资产 --")
            for a in self.assets:
                mark = "OK " if a.present and a.payload_rel in self.shipped_assets else "缺 "
                lines.append(f"    {mark} {a.key}  ({a.path.name})")
        if self.accepted_gaps:
            lines.append("  -- 已接受（有意不发货）--")
            for name, why in self.accepted_gaps:
                lines.append(f"    ~  {name}")
                lines.append(f"       {why}")
        if self.hard_third_party:
            lines.append(f"  -- 第三方硬依赖：{', '.join(self.hard_third_party)}")
        return "\n".join(lines)



def audit_packaging(
    entry: Path,
    notebook: Path,
    repo_root: Path,
    asset_spec: Optional[Dict[str, str]] = None,
    asset_payload_map: Optional[Dict[str, str]] = None,
) -> PackagingAudit:
    """把"agent 真正要什么"与"提交包实际带了什么"做差集。"""
    root = Path(repo_root).resolve()
    notebook = Path(notebook)
    local, external, third_party = import_closure(Path(entry), root)
    audit = PackagingAudit(entry=Path(entry), repo_root=root)
    audit.shipped_keys = set(notebook_payload_files(notebook))
    audit.shipped_external, audit.shipped_assets = notebook_extra_payload(notebook)
    audit.third_party = third_party
    audit.external_packages = external
    audit.deferred = dict(DEFERRED)
    audit.assets = check_assets(root, asset_spec, payload_map=asset_payload_map)
    audit.unguarded = {n for n, r in local.items() if not r.guarded}
    audit.local_modules = {
        name: (_module_file(name, root) or Path()) for name in sorted(local)
    }
    return audit


def _module_file(name: str, repo_root: Path) -> Optional[Path]:
    cand = Path(repo_root).joinpath(*name.split("."))
    for f in (cand.with_suffix(".py"), cand / "__init__.py"):
        if f.is_file():
            return f
    return None


# --------------------------------------------------------------------------- #
# 5. 总报告
# --------------------------------------------------------------------------- #


def readiness_report(
    repo_root: Path,
    entry: Path,
    notebook: Path,
    asset_spec: Optional[Dict[str, str]] = None,
    asset_payload_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """离线就绪总报告。``ok=False`` 时看 ``blockers``。"""
    audit = audit_packaging(entry, notebook, repo_root, asset_spec, asset_payload_map)
    return {
        "ok": audit.ok,
        "env": {k: os.environ.get(k, "") for k in OFFLINE_ENV},
        "allow_hosts": list(DEFAULT_ALLOW_HOSTS),
        "packaging": {
            "local_modules": len(audit.local_modules),
            "shipped_files": len(audit.shipped_keys),
            "will_crash": audit.will_crash,
            "will_degrade": audit.will_degrade,
            "external_shipped": sorted(audit.shipped_external),
            "external_missing": sorted(audit.unshipped_external),
            "assets_shipped": sorted(audit.shipped_assets),
            "missing_assets": [str(a.path) for a in audit.missing_assets],
            "hard_third_party": audit.hard_third_party,
            "accepted_gaps": [n for n, _ in audit.accepted_gaps],
        },
        "blockers": audit.blockers(),
    }
