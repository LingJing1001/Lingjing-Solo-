# patches/

针对外部仓库 `ARC-AGI-3-Kaggle-Starter`（origin = `arcprize/ARC-AGI-3-Kaggle-Starter`，本地没有推送权限，重新 clone 就会丢改动）的上游补丁存放处。文件本身不带入本仓库代码，只负责在搭环境时打回 Starter。

## kaggle-starter-bundle-lingjing-solo.patch

基线 `eeb1535`，只改 `scripts/build_notebook.py`（+132 / −20），blob `9926459`。做两件事：

1. 把 `lingjing_solo/` 打包进 notebook。原脚本只将 `agent/my_agent.py` 用 `%%writefile` 写进 `/tmp`，而 SmartRouter 在模块顶层 `from lingjing_solo.core import ...`；Kaggle 加速会话断网，装不了也拷不进这个包，Phase B 一 import 就炸，落到榜上就是 0 分。补丁新增 unpack cell：内嵌固定时间戳/权限的确定性 zip 的 base64（payload `sha256=3bacccea1604`，100 个文件；
2026-09-23 S1 懒加载改动后变为 `86af244fe3a2`，文件数不变，见
`docs/SSA白皮书关联核实与整合方案.md`），解到 `/tmp/lingjing_solo`，run cell 以 `MPLBACKEND=agg PYTHONPATH=/tmp` 启动。走 `/tmp` 而非 `/kaggle/working` 是为了不往 notebook 输出目录丢上百个文件（官方 README 警告过输出文件会干扰 "Submit to Competition" 的候选选择）。cell 只在 agent 源码确实引用 `lingjing_solo` 时生成，随机 starter 的老链路行为不变。
2. 修一处上游编码 bug：`AGENT_SRC.read_text()` 不带 encoding，中文 Windows 默认 cp936 读到 agent 里的破折号即 `UnicodeDecodeError: 'gbk' codec can't decode byte 0x94`（随机 starter 全 ASCII 所以从未暴露）；四处文本 IO 现全部显式 `encoding="utf-8"`。

## 怎么用

```powershell
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
cd ARC-AGI-3-Kaggle-Starter
git apply ..\ARC-AGI-3\patches\kaggle-starter-bundle-lingjing-solo.patch

# 等价于 make setup（本机没有 make；venv 布局是 Scripts\ 而非 bin\）
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe "arc-agi>=0.9.6" "kaggle>=2.2" python-dotenv pandas pyarrow
git clone --depth 1 https://github.com/arcprize/ARC-AGI-3-Agents.git vendor\ARC-AGI-3-Agents
.venv\Scripts\python.exe scripts\slim_framework.py     # 注意下方「未覆盖」的 GBK 坑

powershell -NoProfile -File ..\ARC-AGI-3\scripts\sync_into_starter.ps1   # 灌 SmartRouter agent + lingjing_solo
.venv\Scripts\python.exe scripts\build_notebook.py
```

补丁是 LF 行尾，且仓库根目录 `.gitattributes` 用 `*.patch -text` 钉死了字节（否则 `core.autocrlf=true` 的机器 checkout 出来会变 CRLF，`git apply` 的行尾匹配就会漂）。在 `core.autocrlf=true` 的机器上已实测：全新 clone（干净 `eeb1535` 工作树）里 `git apply --check` 与实际 apply 均通过，结果文件与打补丁前的工作树 `git hash-object` 一致。验证记录（2026-09-22）：pristine clone 打完补丁后 `py_compile` 通过，原始随机 agent 走 build 得 4 个 code cell、无 bundle cell；同步 SmartRouter 后得 5 个 code cell（含首行 markdown 共 6 cells），Phase B 模拟（只读 notebook）三地板 `ls20 L7/309`、`ar25 L8/276`、`ft09 L6/81` 全 WIN。

## 未覆盖

`scripts/slim_framework.py` 同族问题仍在：它 `INIT.write_text(SLIM)` 不带 encoding，每次 `make setup` 后会把 `vendor/ARC-AGI-3-Agents/agents/__init__.py` 写成 GBK，下次 import 报 `SyntaxError: 'utf-8' codec can't decode byte 0xa1`。临时办法是把该文件按 UTF-8 重写（vendor 已被 gitignore）。要根治就把 `encoding="utf-8"` 一起加进补丁。

## 重新生成

```powershell
cd ARC-AGI-3-Kaggle-Starter
git diff -- scripts/build_notebook.py > ..\ARC-AGI-3\patches\kaggle-starter-bundle-lingjing-solo.patch
```
