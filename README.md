# Driving Risk Multi-stage Training

面向公交与营运车辆视频的多阶段风险判断训练流水线。项目将视频、平台事件、学生模型回答和专家标签组织成八个连续阶段，支持多模态 SFT、GRPO、模型导出、推理及质量评估。

## 流水线

1. 最终风险判断 SFT
2. 学生答案排序 SFT
3. 学生答案排序 GRPO
4. 学生答案批评 SFT
5. 学生答案批评 GRPO
6. 最终标签推理 SFT
7. 最终标签推理 GRPO
8. 排序理由 SFT

## 目录

```text
src/pipeline/       数据构建、阶段定义、奖励函数、脚本生成
src/evaluation/     分类指标、启发式评分和可选 LLM Judge
tests/              不依赖模型权重的快速测试
docs/               设计和数据格式说明
```

模型权重、原始视频、业务标签、训练 checkpoint 和推理结果均不进入仓库。

## 快速开始

```bash
python -m venv .venv
pip install -r requirements.txt
python src/pipeline/build_datasets.py \
  --input-dir /path/to/source-a \
  --input-dir /path/to/source-b \
  --output-dir ./artifacts
python src/pipeline/generate_stage_scripts.py \
  --manifest ./artifacts/manifest.json
```

运行基础检查：

```bash
python -m unittest discover -s tests -v
```

## 数据与隐私

输入目录需包含可配对的 `labels/*_labels.json` 与 `rankings/*_ranking.json`。请勿提交视频、车牌、司机身份信息或内部绝对路径。正式实验建议按车辆或司机分组切分数据，避免实体跨越训练集和测试集。

## 当前限制

- 底层训练依赖本地 `ms-swift` 与兼容 Qwen3.5 的 Transformers 环境。
- 默认切分逻辑仍按样本执行；用于正式报告前应升级为实体分组切分。
- 评估脚本会报告跳过样本数；正式指标应额外将预测解析失败计作错误。

