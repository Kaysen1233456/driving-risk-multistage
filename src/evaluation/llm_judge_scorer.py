"""
LLM Judge 多模态评分模块（L1-L5）

使用 DashScope 多模态 API，视频 + 预测文本 → 5 维评分 JSON。

各维度权重：
  L1 视觉接地性              25%
  L2 推理步骤连贯性          20%
  L3 特例排除论证完整性      20%
  L4 干预话术推理一致性      20%
  L5 解释忠实性（负向）      15%
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------

LLM_WEIGHTS: dict[str, float] = {
    "L1": 0.25,
    "L2": 0.20,
    "L3": 0.20,
    "L4": 0.20,
    "L5": 0.15,
}

# ---------------------------------------------------------------------------
# Judge Prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
你是一名专业的AI输出质量评估专家，正在对驾驶员疲劳检测大模型的解释性输出进行多维度评分。
你将收到：
1. 一段驾驶员监控视频
2. 大模型对该视频的疲劳分析解释文本（包含疲劳等级判断、风险解读、干预话术）

请严格按照评分标准输出一个合法的JSON对象，不要在JSON之外输出任何内容。\
"""

JUDGE_USER_TEMPLATE = """\
## 待评估的模型输出文本

{pred_text}

---

## 评分维度与标准

### L1. 视觉接地性（Visual Grounding）— 权重25%
核心问题：该解释在多大程度上将每个推断直接锚定到视频中可观测的具体视觉事件？
- 9-10分：每个关键判断均有具体时间戳+动作描述，且证据与视频高度吻合
- 6-8分：大部分判断有证据，但部分细节缺乏具体时间定位
- 3-5分：证据泛化（如"司机多次闭眼"而无具体量化），或仅描述现象而无判断联系
- 1-2分：解释几乎不依赖视频内容，或出现与视频不符的描述

### L2. 推理步骤连贯性（Reasoning Coherence）— 权重20%
核心问题：从"观察证据"到"疲劳等级判断"再到"干预方案"的推导链，每一步是否由上一步合理推出？
- 9-10分：原始观察→特征归因→等级判断→干预推导，每步逻辑清晰，因果链完整无跳跃
- 6-8分：大体逻辑合理，但个别步骤有跳跃或前后表述轻微矛盾
- 3-5分：推理链存在明显断层，结论与中间分析脱节
- 1-2分：解释缺乏逻辑结构，各部分独立堆砌，无因果联系

### L3. 特例排除论证完整性（Counter-Evidence Reasoning）— 权重20%
核心问题：模型是否主动考虑并排除了可能导致误判的替代解释（强光眯眼、分心下瞟、唱歌闭眼、生理性眨眼等）？
- 9-10分：主动识别并排除当前场景中潜在混淆因素，给出明确排除理由
- 6-8分：部分提及特例说明，但排除论证不够完整或缺乏针对性
- 3-5分：仅描述疲劳症状，未考虑备选解释
- 1-2分：完全忽略特例排除，或将非疲劳行为错误归类为疲劳依据

### L4. 干预话术与疲劳等级的推理一致性（Intervention Alignment）— 权重20%
核心问题：干预话术的设计是否能从疲劳等级判断和风险解读中合理推导出来？
注意：不评估"话术措辞是否恰当"，只评估话术设计过程是否源于等级判断的逻辑推导。
- 9-10分：话术语气强度、紧迫程度与疲劳等级完全对应，且文本中明确说明推导逻辑
- 6-8分：话术与等级大体匹配，但推导逻辑未显式说明
- 3-5分：话术与等级存在程度不匹配
- 1-2分：话术逻辑与等级判断完全脱节，或相互矛盾

### L5. 解释忠实性（Faithfulness）— 权重15%【负向维度，基础分10分】
核心问题：解释文本中是否存在幻觉内容——即描述了视频中不存在的事实？
基础分10分，按规则扣分：
- 描述了视频中不存在的动作：每处 -2分
- 引用了视频时间范围以外的时间戳：每处 -1分
- 使用推测性语言（"可能""或许"等）但呈现为确定性描述：每处 -0.5分

---

## 输出格式（严格JSON，不要输出其他内容）

{{
  "L1_visual_grounding": {{
    "score": <1-10的整数>,
    "weight": 0.25,
    "evidence": "<具体说明，引用解释文本原句>",
    "deduction_points": []
  }},
  "L2_reasoning_coherence": {{
    "score": <1-10的整数>,
    "weight": 0.20,
    "evidence": "<推理链分析>",
    "deduction_points": []
  }},
  "L3_counter_evidence": {{
    "score": <1-10的整数>,
    "weight": 0.20,
    "evidence": "<混淆因素处理分析>",
    "deduction_points": []
  }},
  "L4_intervention_alignment": {{
    "score": <1-10的整数>,
    "weight": 0.20,
    "evidence": "<话术与等级对应分析>",
    "deduction_points": []
  }},
  "L5_faithfulness": {{
    "base_score": 10,
    "deductions": [
      {{"type": "幻觉动作描述", "content": "<原文引用>", "score_deducted": 2}},
      {{"type": "越界时间戳",   "content": "<原文引用>", "score_deducted": 1}},
      {{"type": "推测当确定",   "content": "<原文引用>", "score_deducted": 0.5}}
    ],
    "final_score": <计算后分值，最低0>,
    "weight": 0.15,
    "evidence": "<逐句核验说明>"
  }},
  "weighted_total": <加权总分，保留两位小数>,
  "overall_assessment": "<50字以内综合评价>",
  "critical_issues": ["<最需改进的1-3个问题>"]
}}
"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LLMJudgeScores:
    L1_visual_grounding:       float = 0.0
    L2_reasoning_coherence:    float = 0.0
    L3_counter_evidence:       float = 0.0
    L4_intervention_alignment: float = 0.0
    L5_faithfulness:           float = 0.0
    weighted_total:            float = 0.0
    raw_response:              dict  = field(default_factory=dict)
    error:                     Optional[str] = None


# ---------------------------------------------------------------------------
# JSON 解析（多策略容错）
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[dict]:
    # 策略1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2：提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 策略3：提取首个完整 { ... }
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _recompute_weighted_total(parsed: dict) -> float:
    """重新计算加权总分，防止模型计算出错。"""
    total = 0.0
    for key, weight in [
        ("L1_visual_grounding",       LLM_WEIGHTS["L1"]),
        ("L2_reasoning_coherence",    LLM_WEIGHTS["L2"]),
        ("L3_counter_evidence",       LLM_WEIGHTS["L3"]),
        ("L4_intervention_alignment", LLM_WEIGHTS["L4"]),
    ]:
        if key in parsed:
            total += float(parsed[key].get("score", 0)) * weight

    if "L5_faithfulness" in parsed:
        l5     = parsed["L5_faithfulness"]
        base   = float(l5.get("base_score", 10))
        deduct = sum(float(d.get("score_deducted", 0)) for d in l5.get("deductions", []))
        total += max(0.0, base - deduct) * LLM_WEIGHTS["L5"]

    return round(total, 2)


def _extract_l5_final(parsed: dict) -> float:
    """从解析结果中提取 L5 最终分（含扣分计算）。"""
    l5 = parsed.get("L5_faithfulness", {})
    if "final_score" in l5:
        return float(l5["final_score"])
    base   = float(l5.get("base_score", 10))
    deduct = sum(float(d.get("score_deducted", 0)) for d in l5.get("deductions", []))
    return max(0.0, base - deduct)


# ---------------------------------------------------------------------------
# 单条 Judge 调用（DashScope 多模态）
# ---------------------------------------------------------------------------

def call_judge_single(
    pred_text:   str,
    video_path:  str,
    api_key:     str,
    model:       str  = "qwen-vl-max",
    base_url:    str  = "https://dashscope.aliyuncs.com/api/v1",
    video_fps:   int  = 3,
    max_retries: int  = 3,
    retry_delay: float = 2.0,
) -> LLMJudgeScores:
    """调用 DashScope 多模态 Judge 对单条样本评分。

    video_path 支持：
      - 本地绝对路径（自动加 file:// 前缀）
      - http(s):// URL
      - file:// URL
    """
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except ImportError:
        return LLMJudgeScores(error="dashscope 未安装，请执行: pip install dashscope")

    dashscope.api_key = api_key
    dashscope.base_http_api_url = base_url

    # 构建视频 URL
    if video_path.startswith(("http://", "https://", "file://")):
        video_url = video_path
    else:
        video_url = f"file://{video_path}"

    user_content = [
        {"video": video_url, "fps": video_fps},
        {"text": JUDGE_USER_TEMPLATE.format(pred_text=pred_text)},
    ]
    messages = [
        {"role": "system", "content": [{"text": JUDGE_SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]

    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            response = MultiModalConversation.call(
                model=model,
                messages=messages,
                result_format="message",
            )
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.message}"
                logger.warning(f"[Judge] 第{attempt+1}次请求失败: {last_error}")
                time.sleep(retry_delay)
                continue

            raw_text = response.output.choices[0].message.content[0].get("text", "")
            parsed   = _parse_json_response(raw_text)

            if parsed is None:
                last_error = f"JSON解析失败，原始: {raw_text[:300]}"
                logger.warning(f"[Judge] 第{attempt+1}次JSON解析失败")
                time.sleep(retry_delay)
                continue

            weighted_total = _recompute_weighted_total(parsed)

            return LLMJudgeScores(
                L1_visual_grounding=       float(parsed.get("L1_visual_grounding",       {}).get("score", 0)),
                L2_reasoning_coherence=    float(parsed.get("L2_reasoning_coherence",    {}).get("score", 0)),
                L3_counter_evidence=       float(parsed.get("L3_counter_evidence",       {}).get("score", 0)),
                L4_intervention_alignment= float(parsed.get("L4_intervention_alignment", {}).get("score", 0)),
                L5_faithfulness=           _extract_l5_final(parsed),
                weighted_total=            weighted_total,
                raw_response=              parsed,
            )

        except Exception as exc:
            last_error = str(exc)
            logger.warning(f"[Judge] 第{attempt+1}次异常: {exc}")
            time.sleep(retry_delay)

    return LLMJudgeScores(error=f"重试{max_retries}次后仍失败: {last_error}")
