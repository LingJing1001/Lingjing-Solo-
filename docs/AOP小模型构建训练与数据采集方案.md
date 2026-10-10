# AOP 小模型构建、训练与数据采集方案

> 项目根目录：`/srv/agent-platform/projects/Lingjing-Solo-`
> 目标：为 ARC/CEAX 游戏的点击、动作和关卡推进预测建立可审计的数据闭环。

## 1. 标签原则

每个训练样本表示一个真实动作转移：

```text
(observation_before, requested_action, observation_after)
```

必须分开记录：

- `visual_changed`：画面或网格是否变化；
- `action_effective`：语义状态是否变化或权威状态是否变化；
- `progressed`：关卡、目标、得分或权威 `WIN` 是否推进。

**视觉变化不等于动作成功，动作生效不等于任务推进。** `RESET` 只建立 episode 基线，不生成动作标签。

## 2. 阶段状态

- [x] **阶段 1：标签契约**
  - [x] 定义 `episode_id`、`step_id`、`requested_action`、before/after observation；
  - [x] 定义 state hash、visual delta、action effectiveness、progress delta；
  - [x] 对缺失观测、缺失 episode、RESET、坏 tick fail-closed。
- [x] **阶段 2：recording → labels 转换器**
  - [x] `models/aop/labels.py`：stdlib-only、确定性 JSON 标签转换；
  - [x] `tools/aop/build_labels.py`：JSONL CLI；
  - [x] 支持 ARC/CEAX 公共字段和 `game_specific` score/level 字段。
  - [x] `models/aop/recording_gate.py` / `tools/aop/validate_recording.py`：真实采集前的 settled-after、合法动作、episode/tick 单调性和权威状态字段闸门。
- [ ] **阶段 3：真实 ARC/CEAX recording 采集**
  - [ ] 从官方/授权环境生成包含 settled after-state 的 recording；
  - [ ] 每个 game profile 校验 legal actions、状态字段和 episode 边界；
  - [ ] 采集 click/action/no-op/失败/推进各类平衡样本。
- [ ] **阶段 4：训练与评估**
  - [x] `lingjing_solo/neural/aop_encoder.py`：16 维有限值特征编码，拒绝 NaN/Inf；
  - [x] `lingjing_solo/neural/aop_model.py`：AOP v1 多任务模型、严格 checkpoint round-trip；
  - [x] `tools/aop/dataset.py`：按 episode 切分 train/validation，禁止泄漏；
  - [x] `tools/aop/train.py`：三头 loss + checkpoint 写入；synthetic smoke 已通过；
  - [ ] 使用真实 recording 训练并独立评估 action effectiveness 与 progress prediction；
  - [ ] 低置信度回退确定性 planner，不直接执行模型建议。
- [ ] **阶段 5：线上接入**
  - [ ] 仅接入排序/候选筛选，不替代证据层和权威 WIN 判定；
  - [ ] 记录 prediction/actual mismatch，支持回放与回滚。

## 3. 当前实现

```text
models/aop/labels.py       # 标签契约和转换逻辑
models/aop/recording_gate.py # 真实 recording 采集前闸门
lingjing_solo/neural/aop_encoder.py # 16 维有限值编码器
lingjing_solo/neural/aop_model.py   # AOP v1 多任务模型和 checkpoint
tools/aop/build_labels.py  # recording JSONL → labels JSONL
tools/aop/validate_recording.py # normalized recording schema/settled-after 校验
tools/aop/normalize_arc_recording.py # ARC Recorder → AOP transition JSONL
tools/aop/dataset.py       # episode-disjoint train/validation 切分
tools/aop/train.py         # 离线训练 CLI
tests/test_aop_labels.py   # 标签单元、边界、确定性测试
tests/test_recording_gate.py # recording 闸门测试
tests/test_aop_training.py # encoder/model/split/checkpoint 测试
```

运行：

```bash
# 如果输入已经是 normalized AOP recording：
python3 tools/aop/validate_recording.py recording.jsonl
python3 tools/aop/build_labels.py recording.jsonl labels.jsonl

# 如果输入来自 ARC-AGI-3 Recorder：
python3 tools/aop/normalize_arc_recording.py \
  raw.recording.jsonl normalized.recording.jsonl
python3 tools/aop/validate_recording.py normalized.recording.jsonl --min-episodes 2
python3 tools/aop/build_labels.py normalized.recording.jsonl labels.jsonl

pytest -q tests/test_aop_labels.py tests/test_recording_gate.py \
  tests/test_aop_recording_normalize.py tests/test_aop_training.py
python3 tools/aop/train.py labels.jsonl \
  --output models/aop/checkpoints/aop-v1.pt --epochs 10
```

训练命令要求 labels 至少包含两个不同 episode；按 episode 切分防止泄漏。首次 smoke 可使用 `--synthetic`，但生成的 checkpoint 只能作为流水线证据，不得作为 ARC 能力或线上模型。

## 4. 下一步：真实数据采集闸门

在进入阶段 3 前，每种 ARC/CEAX 游戏必须提供至少一份 recording，并通过：

1. 每个非 RESET 行存在 before、after、合法 requested action；
2. after 是动作 settle 后状态，不是动作请求时的重复快照；
3. level/state/score 等权威字段可回放；
4. `visual_changed`、`action_effective`、`progressed` 三者可独立重算；
5. recording 重复转换产生字节一致的 labels。

未经这些闸门，不训练、不把视觉变化当成成功、不将模型输出直接送入执行器。
