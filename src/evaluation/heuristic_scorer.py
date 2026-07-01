"""
启发式规则评分模块（H1-H5）

各维度权重：
  H1 格式规范性        20%
  H2 关键证据词覆盖率   25%
  H3 时间定位精确性    15%
  H4 等级-话术一致性   25%
  H5 语义相似度        15%

所有子维度满分均为 10 分。
加权总分 = Σ Hi × weight_i，范围 [0, 10]。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------

HEURISTIC_WEIGHTS: dict[str, float] = {
    "H1": 0.20,
    "H2": 0.25,
    "H3": 0.15,
    "H4": 0.25,
    "H5": 0.15,
}


# H1 格式字段与等级表达式
REQUIRED_FIELDS = ["疲劳等级判断", "风险解读", "干预话术"]
LEVEL_PATTERNS  = [
    r"[0-3]级",
    r"无疲劳", r"轻微疲劳", r"严重疲劳", r"极度危险疲劳",
]

# ---------------------------------------------------------------------------
# H2 证据词典（按维度分组）
# ---------------------------------------------------------------------------

EVIDENCE_VOCAB: dict[str, list[str]] = {
    "眼部":   ["闭眼", "眯眼", "眼睑", "眼皮", "频繁闭眼", "长时间闭眼"],
    "头部":   ["点头", "低头", "头部晃动", "后仰", "头部下垂"],
    "肢体":   ["搓脸", "抿嘴", "甩头", "托腮", "揉眼", "拍脸"],
    "抗拒":   ["自我抗拒", "对抗疲劳", "短暂清醒", "恢复清醒"],
    "特例排除": ["分心", "强光", "生理眨眼", "下瞟", "唱歌"],
}


# H3 时间戳正则（HH:MM:SS 或 H:MM:SS）
TIMESTAMP_RE = re.compile(r"\b(\d{1,2}:\d{2}:\d{2})\b")



# ---------------------------------------------------------------------------
# H4 等级-话术规则
# ---------------------------------------------------------------------------

# forbidden_if_nonempty: 0级话术非空且含实质内容时扣分
INTERVENTION_RULES: dict[int, dict] = {
    0: {
        "forbidden_if_nonempty": True,
        "forbidden_words": [], # 禁止词
        "required_words":  [], # 必须词
        "missing_penalty":    0,
        "violation_penalty":  5,
    },
    1: {
        "forbidden_if_nonempty": False,
        "forbidden_words": ["立即停车", "强制", "紧急", "危险"],
        "required_words":  ["建议", "注意", "提醒", "请"],
        "missing_penalty":    3,
        "violation_penalty":  3,
    },
    2: {
        "forbidden_if_nonempty": False,
        "forbidden_words": [],
        "required_words":  ["停车", "休息", "换班", "靠边"],
        "missing_penalty":    3,
        "violation_penalty":  0,
    },
    3: {
        "forbidden_if_nonempty": False,
        "forbidden_words": [],
        "required_words":  ["立即", "强制", "危险", "调度", "紧急"],
        "missing_penalty":    5,
        "violation_penalty":  0,
    },
}






# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class HeuristicScores:
    H1_format:        float = 0.0
    H2_evidence:      float = 0.0
    H3_temporal:      float = 0.0
    H4_intervention:  float = 0.0
    H5_semantic:      float = 0.0
    weighted_total:   float = 0.0
    h5_available:     bool  = False   # 标记 H5 是否实际计算
    details:          dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# H1：格式规范性（满分 10 分）
# ---------------------------------------------------------------------------

def score_H1(pred_text: str) -> tuple[float, dict]:
    score  = 0.0
    detail: dict = {}

    # 各字段存在性（每项 2 分，共 6 分）
    for fname in REQUIRED_FIELDS:
        present = fname in pred_text
        detail[f"has_{fname}"] = present
        if present:
            score += 2.0

    # 三字段顺序正确（1 分）
    positions = [pred_text.find(f) for f in REQUIRED_FIELDS]
    order_ok  = (
        all(p != -1 for p in positions)
        and positions[0] < positions[1] < positions[2]
    )
    detail["field_order_correct"] = order_ok
    if order_ok:
        score += 1.0

    # 等级表述规范（3 分）
    level_ok = any(re.search(pat, pred_text) for pat in LEVEL_PATTERNS)
    detail["level_expression_valid"] = level_ok
    if level_ok:
        score += 3.0

    return min(score, 10.0), detail


# ---------------------------------------------------------------------------
# H2：关键证据词覆盖率（满分 10 分）
# ---------------------------------------------------------------------------

def score_H2(pred_text: str, gt_level: int) -> tuple[float, dict]:
    hit_words: list[str] = []
    hit_dims:  set[str]  = set()

    for dim, words in EVIDENCE_VOCAB.items():
        for w in words:
            if w in pred_text:
                hit_words.append(w)
                hit_dims.add(dim)

    if gt_level == 0:
        # 期望 ≥2 个特例排除词
        exclusion_hits = sum(1 for w in EVIDENCE_VOCAB["特例排除"] if w in pred_text)
        expected, achieved = 2, exclusion_hits

    elif gt_level == 1:
        # 期望 ≥3 个眼部词 + ≥1 个其他维度词
        eye_hits   = sum(1 for w in EVIDENCE_VOCAB["眼部"] if w in pred_text)
        other_hits = sum(
            1 for dim in ("头部", "肢体", "抗拒")
            for w in EVIDENCE_VOCAB[dim] if w in pred_text
        )
        expected  = 4
        achieved  = min(eye_hits, 3) + min(other_hits, 1)

    else:  # 2 级、3 级
        # 期望 ≥5 个跨维度词
        cross_words = sum(
            len([w for w in words if w in pred_text])
            for words in EVIDENCE_VOCAB.values()
        )
        expected, achieved = 5, cross_words

    ratio    = min(achieved / expected, 1.0) if expected > 0 else 1.0
    h2_score = ratio * 10.0

    detail = {
        "hit_words":      hit_words,
        "hit_dimensions": sorted(hit_dims),
        "achieved":       achieved,
        "expected":       expected,
        "ratio":          round(ratio, 4),
    }
    return h2_score, detail


# ---------------------------------------------------------------------------
# H3：时间定位精确性（满分 10 分）
# ---------------------------------------------------------------------------

def _ts_to_sec(ts: str) -> float:
    parts = ts.split(":")
    return sum(float(p) * (60 ** (len(parts) - 1 - i)) for i, p in enumerate(parts))


def score_H3(
    pred_text: str,
    video_duration_sec: Optional[float] = None,
) -> tuple[float, dict]:
    score  = 0.0
    detail: dict = {}

    # 只在风险解读段落中查找时间戳
    m = re.search(r"风险解读[：:](.*?)(?=干预话术[：:]|$)", pred_text, re.S)
    risk_text  = m.group(1) if m else pred_text
    timestamps = TIMESTAMP_RE.findall(risk_text)
    detail["timestamps_found"] = timestamps

    if timestamps:
        score += 4.0   # 出现时间戳
        score += 2.0   # 格式合法（正则已保证）

        if len(timestamps) >= 2:
            score += 2.0
        detail["timestamp_count"] = len(timestamps)

        if video_duration_sec is not None:
            in_range = all(_ts_to_sec(ts) <= video_duration_sec for ts in timestamps)
            detail["all_timestamps_in_range"] = in_range
            if in_range:
                score += 2.0
        else:
            score += 2.0   # 无元数据，满分补偿
            detail["all_timestamps_in_range"] = "skipped_no_metadata"
    else:
        detail["timestamp_count"] = 0

    return min(score, 10.0), detail


# ---------------------------------------------------------------------------
# H4：等级-话术一致性（满分 10 分，扣分制）
# ---------------------------------------------------------------------------

def score_H4(pred_text: str, pred_level: int) -> tuple[float, dict]:
    m = re.search(r"干预话术[：:](.*?)$", pred_text, re.S)
    intervention = m.group(1).strip() if m else ""

    rule      = INTERVENTION_RULES.get(pred_level, INTERVENTION_RULES[1])
    deduction = 0.0
    detail: dict = {"intervention_text_length": len(intervention)}

    if pred_level == 0:
        is_empty_or_none = (
            len(intervention) == 0
            or "无需干预" in intervention
            or "不需要干预" in intervention
            or len(intervention) < 15
        )
        if not is_empty_or_none:
            deduction += rule["violation_penalty"]
            detail["violation"] = (
                f"0级话术非空且含实质内容（{len(intervention)}字）"
                f" → -{rule['violation_penalty']}分"
            )
    else:
        # 检查禁止词（命中任意一个扣一次）
        violated_fw = next(
            (fw for fw in rule["forbidden_words"] if fw in intervention), None
        )
        if violated_fw:
            deduction += rule["violation_penalty"]
            detail["forbidden_word"] = (
                f"出现禁止词 '{violated_fw}' → -{rule['violation_penalty']}分"
            )

        # 检查必须词
        has_required = any(rw in intervention for rw in rule["required_words"])
        if rule["required_words"] and not has_required:
            if intervention:
                deduction += rule["missing_penalty"]
                detail["missing_required"] = (
                    f"缺少必要词 {rule['required_words']}"
                    f" → -{rule['missing_penalty']}分"
                )
            else:
                deduction += rule["missing_penalty"]
                detail["empty_intervention"] = (
                    f"{pred_level}级话术为空 → -{rule['missing_penalty']}分"
                )

    detail["deduction_total"] = deduction
    return max(0.0, 10.0 - deduction), detail


# ---------------------------------------------------------------------------
# H5：BERTScore 语义相似度（满分 10 分，批量计算）
# ---------------------------------------------------------------------------

def score_H5_batch(
    pred_texts: list[str],
    ref_texts:  list[str],
    lang: str = "zh",
) -> list[tuple[float, dict]]:
    """批量计算 BERTScore F1。返回每条 (score×10, detail) 列表。"""
    if not BERT_SCORE_AVAILABLE:
        return [
            (5.0, {"error": "bert-score not installed, fallback=5.0"})
            for _ in pred_texts
        ]
    if not pred_texts:
        return []

    _, _, F1 = bert_score_fn(pred_texts, ref_texts, lang=lang, verbose=False)
    return [
        (min(float(f) * 10.0, 10.0), {"bertscore_f1": round(float(f), 4)})
        for f in F1.tolist()
    ]


# ---------------------------------------------------------------------------
# 单条综合启发式评分（H5 由外部批量传入）
# ---------------------------------------------------------------------------

def score_heuristic_single(
    pred_text:          str,
    gt_level:           int,
    pred_level:         int,
    video_duration_sec: Optional[float] = None,
    h5_score:           Optional[float] = None,
) -> HeuristicScores:
    """计算单条样本 H1-H5 及加权总分。

    h5_score: 由 score_H5_batch() 批量得到后传入；为 None 时排除 H5 权重重新归一化。
    """
    h1, d1 = score_H1(pred_text)
    h2, d2 = score_H2(pred_text, gt_level)
    h3, d3 = score_H3(pred_text, video_duration_sec)
    h4, d4 = score_H4(pred_text, pred_level)

    h5_available = h5_score is not None
    h5 = h5_score if h5_available else 5.0

    if h5_available:
        total = (
            h1 * HEURISTIC_WEIGHTS["H1"]
            + h2 * HEURISTIC_WEIGHTS["H2"]
            + h3 * HEURISTIC_WEIGHTS["H3"]
            + h4 * HEURISTIC_WEIGHTS["H4"]
            + h5 * HEURISTIC_WEIGHTS["H5"]
        )
    else:
        # H5 未计算时，将其权重等比分配给其余四项
        # 各维度分已在 0-10 范围，乘以归一化权重后结果仍在 0-10 范围，无需再乘 10
        w_no_h5 = {k: v for k, v in HEURISTIC_WEIGHTS.items() if k != "H5"}
        w_sum   = sum(w_no_h5.values())
        total   = (
            h1 * w_no_h5["H1"] / w_sum
            + h2 * w_no_h5["H2"] / w_sum
            + h3 * w_no_h5["H3"] / w_sum
            + h4 * w_no_h5["H4"] / w_sum
        )

    return HeuristicScores(
        H1_format=       round(h1,    4),
        H2_evidence=     round(h2,    4),
        H3_temporal=     round(h3,    4),
        H4_intervention= round(h4,    4),
        H5_semantic=     round(h5,    4),
        weighted_total=  round(total, 4),
        h5_available=    h5_available,
        details={"H1": d1, "H2": d2, "H3": d3, "H4": d4},
    )
