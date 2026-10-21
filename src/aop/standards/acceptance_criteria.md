# AOP Sandbox 数据闭环验收标准

本文件定义 model thread、sandbox thread 以及未来 gameplay verification 共享的数据进入标准。标准是 fail-closed：任一硬门槛失败，数据只能进入 quarantine，不能进入训练或正式验证 packet。

## 1. Canonical label schema

每条 canonical label 必须满足：

- `schema_version == "aop-canonical-v1"`；
- `episode_id` 非空字符串；
- `requested_action.id` 与 `requested_action.name` 非空，且 `requested_action.id` 出现在 `legal_actions`；
- `observation_before`、`observation_after`、`state_delta` 为对象；
- counterfactual 数据必须有非空 `counterfactual_group_id`、`replay_state_hash`、`replay_verified == true`；
- `provenance.kind`、`provenance.shard_id`、`provenance.sha256` 非空；
- 不接受无法追溯到 source artifact 的手工拼接 labels。

## 2. Shard manifest

每个可导入 shard 必须有同目录 `manifest.json`，并满足：

- `schema_version == "aop-counterfactual-shard-v1"`；
- `collector_exit == 0`；
- `provenance_complete == true`；
- `training_performed == false`；
- `source_sha256` 与输入 labels 文件 SHA-256 完全一致；
- `rows == labels.jsonl 实际非空行数`；
- `labels_sha256` 与 labels 文件 SHA-256 完全一致；
- `shard_id`、`thread`、`source`、`created_at` 存在；
- counterfactual shard 每个 group 至少有 2 个 candidate，且 group 内不能有重复 semantic key。

## 3. Fail-closed ingestion

- 缺 manifest、manifest JSON 非法、provenance 不完整、SHA 不匹配、collector 非零退出、row schema 非法时必须拒绝；
- 拒绝时不产生可被 freeze 发现的成功 artifact，状态必须记录为 rejected/quarantine；
- 不得因为已有 output 存在而跳过 source integrity 校验；
- 重复导入同一 shard 必须幂等；不同 group 的相同 action/state 不能被误去重；
- ingest 不得执行训练，manifest 必须保留 `training_performed=false`。

## 4. QA and split

- 所有 counterfactual rows 必须 replay verified；
- train/holdout 按 episode 或 counterfactual group 切分，不能按 row 随机切分；
- train/holdout split keys 必须不相交；
- candidate packet 只有在 freeze gate 通过后才能训练；
- sandbox 正式 packet 必须保留输入 artifact path、SHA、rows、groups 和 split manifest。

## 5. Gameplay verification boundary

- canonical data、freeze packet 和 evaluator 可以被 gameplay verification 复用；
- sandbox candidate 默认 shadow-only；
- 未通过独立 evaluator、runtime legal-action 检查和人工批准前，不得成为 runtime authority；
- 采集失败、凭据缺失或 provenance 缺失时不得产生可训练数据。

## 6. Acceptance evidence

实现验收必须提供：

1. schema/provenance/hash 拒绝测试；
2. group-aware dedupe 和幂等测试；
3. counterfactual group QA 拒绝测试；
4. episode/group split 无泄漏测试；
5. 至少一个真实 shard 的 canonical ingest 结果；
6. 全仓测试、lint、编译检查及 changed-file scope。
