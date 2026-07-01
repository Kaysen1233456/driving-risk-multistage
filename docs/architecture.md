# 架构说明

原始标签与模型排序结果首先由 `build_datasets.py` 配对、规范化并生成统一记录。随后按阶段生成 SFT 或 GRPO 数据集，`generate_stage_scripts.py` 根据 manifest 生成训练、导出和推理命令。

三个 GRPO 奖励分别衡量模型排序一致性、异常标签集合相似度和最终风险标签准确性。推理结果可进入分类指标、启发式文本质量评分及可选的多模态 LLM Judge。

```text
labels + rankings
        |
        v
record normalization -> train/val/test manifest
        |
        v
SFT warm-up -> GRPO optimization -> export/inference
        |
        v
classification + heuristic quality + optional LLM judge
```

生产化前应完成车辆级分组切分、配置路径环境化、失败预测惩罚、训练数据版本记录和端到端冒烟测试。

