"""
Compute classification metrics (Accuracy, Precision, Macro-F1, Weighted-F1, Macro-Recall)
for fatigue level predictions stored in driving_video_test_pred.json.

Each sample has:
  - videos[0]                      -> ground-truth label (extracted from video path, same as
                                      build_driving_video_dataset_split.py)
  - messages[role="qwen3_5_0.8B"] -> model prediction (extracted from response text)

Fatigue level mapping (aligned with build_driving_video_dataset_split.py):
  0 -> 无疲劳（含分心行为）
  1 -> 轻微疲劳（TTS 干预）
  2 -> 中度疲劳
  3 -> 严重/极度危险疲劳（人工干预唤醒/停车）

Safety-oriented metrics:
  高危漏报率      - GT∈{2,3} 但预测为 0或1 的样本占所有高危样本的比例
  高危召回率      - GT∈{2,3} 且预测也∈{2,3} 的比例（= 1 - 高危漏报率）
  风险降级误判率  - GT∈{2,3} 但预测等级严格小于真实等级的比例（占所有高危样本）
  误报率          - GT=0 但被预测为 1/2/3 的比例（占所有 GT=0 样本）
  安全加权综合分  - Σ F1(lbl) × weight(lbl)，权重按安全敏感度：{0:0.05, 1:0.15, 2:0.30, 3:0.50}
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_DEFAULT_PRED_ROLE = "qwen3_5_0.8B"


# ---------------------------------------------------------------------------
# GT level extraction from video path (same logic as build_driving_video_dataset_split.py)
# ---------------------------------------------------------------------------

def extract_fatigue_level_from_path(video_path: str) -> int | None:
    """从视频路径中提取疲劳等级，逻辑与 build_driving_video_dataset_split.py 保持一致。

    支持：
      疲劳0级 / 疲劳1级 / 疲劳2级         -> 对应数字
      疲劳3_人工干预唤醒 / 疲劳3_... 等    -> 3
      疲劳_人工干预停车                     -> 3
      分心_无需干预 / 分心_...              -> 0
    """
    if not video_path:
        return None
    m = re.search(r"疲劳(\d+)级", video_path)
    if m:
        return int(m.group(1))
    m2 = re.search(r"疲劳(\d+)_", video_path)
    if m2:
        return int(m2.group(1))
    if re.search(r"疲劳_", video_path):
        return 3
    if re.search(r"分心_", video_path):
        return 0
    return None

LABEL_NAMES = {
    0: "0级-无疲劳",
    1: "1级-轻微疲劳",
    2: "2级-中度疲劳",
    3: "3级-严重疲劳",
}

# 安全加权综合分权重：等级越高、安全风险越大，权重越高
SAFETY_WEIGHTS: dict[int, float] = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.50}


# ---------------------------------------------------------------------------
# Level extraction
# ---------------------------------------------------------------------------

def extract_level_from_text(text: str) -> int | None:
    """从模型输出文本中提取疲劳等级（0-3）。

    等级定义与 build_driving_video_dataset_split.py 保持一致：
      0 -> 无疲劳（含分心）
      1 -> 轻微疲劳（TTS 干预）
      2 -> 中度疲劳
      3 -> 严重/极度危险疲劳（人工干预唤醒/停车）

    提取优先级：
      1. 数字等级：匹配"疲劳等级判断：X级"或括号形式"（X级）"
      2. 关键词等级：在输出首行中匹配中文疲劳程度描述（含常见变体）
      3. 模糊写法回退：如"疲劳等级判断：疲劳"（无具体级别），按中度疲劳（2级）回退
    """
    if not text:
        return None
    # API 错误响应，跳过
    if text.lstrip().startswith("{") and "status_code" in text:
        return None

    # 预处理：去掉 markdown 加粗等符号，避免关键词被 ** ** 包裹导致漏匹配
    # 例如："**疲劳等级判断：** 严重疲劳（高风险）"
    norm = text.replace("**", "")

    # --- 优先：数字等级匹配（容忍不同标题与空格变体，扫描全文） ---
    for pattern in [
        r"(疲劳等级判断|疲劳等级|疲劳程度)[：:]\s*(\d+)\s*级",       # 疲劳等级判断：0级 / 疲劳等级：0 级
        r"(疲劳等级判断|疲劳等级|疲劳程度)[：:][^\n（(]*[（(]\s*(\d+)",  # 疲劳等级判断：无疲劳（0级）
        r"(疲劳等级判断|疲劳等级|疲劳程度)[：:][^\n]*?(\d+)\s*级",   # 疲劳等级判断：...0级...
    ]:
        m = re.search(pattern, norm)
        if m:
            # 数字始终在最后一个捕获组
            level = int(m.groups()[-1])
            # 超出已知范围时归入最高级
            return min(level, 3)

    # --- 其次：在“疲劳等级(判断/程度)”这一行内做关键词映射（最安全） ---
    lines = norm.split("\n")
    level_line = next((ln for ln in lines if ("疲劳等级判断" in ln or "疲劳等级" in ln or "疲劳程度" in ln)), "")
    context = (level_line or lines[0])[:400]
    # 若找到的标题行仅有“疲劳等级判断：”且后面为空，则不要被它“绑死”，扩大到前段全文做匹配
    if level_line and re.fullmatch(r".*(疲劳等级判断|疲劳等级|疲劳程度)[：:]\s*", level_line.strip()):
        context = norm[:600]

    # 按严重程度从高到低匹配，避免"轻微"包含在"中度"描述中误判
    HIGH_KWS = (
        "极度危险疲劳",
        "严重疲劳",
        "重度疲劳",
        "中重度疲劳",
        "较重疲劳",
        "人工干预唤醒",
        "人工干预停车",
    )
    if any(kw in context for kw in HIGH_KWS):
        return 3
    if "中度疲劳" in context:
        return 2
    if "轻微疲劳" in context or "轻度疲劳" in context or "TTS干预" in context:
        return 1
    # 无疲劳：明确出现且不包含更高等级词汇
    if "无疲劳" in context and not any(
        kw in context
        for kw in (
            "轻微疲劳",
            "轻度疲劳",
            "中度疲劳",
            "严重疲劳",
            "极度危险疲劳",
            "重度疲劳",
            "中重度疲劳",
            "较重疲劳",
        )
    ):
        return 0
    # "无需干预"也视为无疲劳（build_driving_video_dataset_split 中分心类归 0 级）
    if "无需干预" in context and not any(
        kw in context
        for kw in (
            "轻微疲劳",
            "轻度疲劳",
            "中度疲劳",
            "严重疲劳",
            "极度危险疲劳",
            "重度疲劳",
            "中重度疲劳",
            "较重疲劳",
        )
    ):
        return 0

    # --- 补充：常见“风险”措辞（不一定出现在等级标题行里） ---
    # 例如："无疲劳风险"、"严重疲劳风险"、"高风险"、"低风险"
    # 说明：这里优先匹配“无疲劳/严重疲劳”等明确等级词；"高风险/低风险" 仅作弱提示
    m_risk = re.search(r"(无疲劳|轻微疲劳|轻度疲劳|中度疲劳|重度疲劳|严重疲劳|极度危险疲劳|中重度疲劳|较重疲劳)\s*风险", context)
    if m_risk:
        kw = m_risk.group(1)
        if kw in ("无疲劳",):
            return 0
        if kw in ("轻微疲劳", "轻度疲劳"):
            return 1
        if kw in ("中度疲劳",):
            return 2
        return 3

    # --- 补充：英文/中文序号表达（一级/二级/三级/四级） ---
    # 数据集中主要用 0-3，但有的模型会写“一级疲劳/二级疲劳/三级疲劳”
    m_cn = re.search(r"(一|二|三|四)级\s*疲劳", context)
    if m_cn:
        cn2num = {"一": 1, "二": 2, "三": 3, "四": 3}
        return cn2num.get(m_cn.group(1), None)

    # --- 回退1：若明确出现"疲劳等级判断：疲劳/疑似疲劳/存在疲劳迹象/疲劳风险"等模糊表达 ---
    m3 = re.search(
        r"(疲劳等级判断|疲劳等级|疲劳程度)[：:]\s*"
        r"(无疲劳|轻微疲劳|轻度疲劳|中度疲劳|重度疲劳|严重疲劳|极度危险疲劳|中重度疲劳|较重疲劳|疑似疲劳|存在疲劳迹象|疲劳风险|疲劳)\b",
        context,
    )
    if m3:
        kw = m3.groups()[-1]
        kw2level = {
            "无疲劳": 0,
            "轻微疲劳": 1,
            "轻度疲劳": 1,
            "中度疲劳": 2,
            "较重疲劳": 3,
            "重度疲劳": 3,
            "严重疲劳": 3,
            "极度危险疲劳": 3,
            "中重度疲劳": 3,
            "疑似疲劳": 1,
            "存在疲劳迹象": 1,
            "疲劳风险": 1,
            "疲劳": 2,  # 模糊：默认中度
        }
        return kw2level.get(kw, None)

    # --- 回退2：若未找到标题行，则在全文早段（前600字符）做关键词兜底 ---
    early = norm[:600]
    if any(kw in early for kw in HIGH_KWS):
        return 3
    if "中度疲劳" in early:
        return 2
    if any(kw in early for kw in ("轻微疲劳", "轻度疲劳", "疑似疲劳", "存在疲劳迹象", "疲劳风险", "TTS干预")):
        return 1
    if "无疲劳" in early and not any(kw in early for kw in ("轻微疲劳", "轻度疲劳", "中度疲劳", "严重疲劳", "极度危险疲劳", "重度疲劳", "中重度疲劳", "较重疲劳")):
        return 0

    # --- 回退3：极端情况下只有“疲劳/存在疲劳”描述但没写级别 ---
    # 注意：这是最弱的兜底，仅在明确出现“疲劳”但没有“无疲劳”时触发，默认按中度疲劳(2级)
    if "疲劳" in early and "无疲劳" not in early:
        return 2

    return None


# ---------------------------------------------------------------------------
# sklearn-based metrics
# ---------------------------------------------------------------------------

def compute_safety_metrics(y_true: list[int], y_pred: list[int], per_class: dict) -> dict:
    """计算安全驾驶场景下的五项专项指标。

    高危漏报率      = GT∈{2,3} 且 Pred∈{0,1} 的数量 / GT∈{2,3} 的总数
    高危召回率      = 1 - 高危漏报率
    风险降级误判率  = GT∈{2,3} 且 Pred < GT 的数量 / GT∈{2,3} 的总数
    误报率          = GT=0 且 Pred∈{1,2,3} 的数量 / GT=0 的总数
    安全加权综合分  = Σ F1(lbl) × SAFETY_WEIGHTS[lbl]，对所有出现的等级求和
    """
    HIGH_RISK = {2, 3}
    SAFE_LABEL = 0

    high_risk_total   = sum(1 for t in y_true if t in HIGH_RISK)
    high_risk_missed  = sum(1 for t, p in zip(y_true, y_pred) if t in HIGH_RISK and p not in HIGH_RISK)
    high_risk_demoted = sum(1 for t, p in zip(y_true, y_pred) if t in HIGH_RISK and p < t)
    safe_total        = sum(1 for t in y_true if t == SAFE_LABEL)
    false_alarm_cnt   = sum(1 for t, p in zip(y_true, y_pred) if t == SAFE_LABEL and p != SAFE_LABEL)

    # 当无高危样本时，设置为 None，以便报告中显示为 N/A
    if high_risk_total:
        miss_rate   = high_risk_missed  / high_risk_total
        recall_high = 1.0 - miss_rate
        demote_rate = high_risk_demoted / high_risk_total
    else:
        miss_rate   = None
        recall_high = None
        demote_rate = None
    false_alarm   = false_alarm_cnt   / safe_total      if safe_total      else 0.0

    safety_score = sum(
        per_class[lbl]["f1"] * SAFETY_WEIGHTS.get(lbl, 0.0)
        for lbl in per_class
    )

    return {
        "high_risk_miss_rate":   miss_rate,
        "high_risk_recall":      recall_high,
        "risk_downgrade_rate":   demote_rate,
        "false_alarm_rate":      false_alarm,
        "safety_weighted_score": safety_score,
        # 辅助计数，方便核查
        "_high_risk_total":      high_risk_total,
        "_high_risk_missed":     high_risk_missed,
        "_high_risk_demoted":    high_risk_demoted,
        "_safe_total":           safe_total,
        "_false_alarm_cnt":      false_alarm_cnt,
    }


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """使用 sklearn 计算所有汇总指标及每类指标，并附加安全专项指标。"""
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)

    accuracy        = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    macro_recall    = recall_score(   y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    macro_f1        = f1_score(       y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    weighted_f1     = f1_score(       y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # 每类指标
    per_cls_p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_cls_r = recall_score(   y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_cls_f = f1_score(       y_true, y_pred, labels=labels, average=None, zero_division=0)
    count_true = Counter(y_true)

    per_class = {
        lbl: {
            "precision": float(per_cls_p[i]),
            "recall":    float(per_cls_r[i]),
            "f1":        float(per_cls_f[i]),
        }
        for i, lbl in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    safety = compute_safety_metrics(y_true, y_pred, per_class)

    return {
        "accuracy":        accuracy,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "macro_f1":        macro_f1,
        "weighted_f1":     weighted_f1,
        "labels":          labels,
        "per_class":       per_class,
        "count_true":      count_true,
        "confusion_matrix": cm,
        "n":               n,
        "safety":          safety,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path, pred_role: str) -> tuple[list[int], list[int], int]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    y_true, y_pred = [], []
    skipped = 0

    for item in data:
        # GT 等级从视频路径提取，与 build_driving_video_dataset_split.py 的分类标准一致
        video_path = item.get("videos", [""])[0]
        gt_lv = extract_fatigue_level_from_path(video_path)

        pred_text = next(
            (m["content"] for m in item["messages"] if m["role"] == pred_role), ""
        )
        pred_lv = extract_level_from_text(pred_text)

        if gt_lv is None or pred_lv is None:
            skipped += 1
            continue

        y_true.append(gt_lv)
        y_pred.append(pred_lv)

    return y_true, y_pred, skipped


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    labels      = metrics["labels"]
    per_cls     = metrics["per_class"]
    count_true  = metrics["count_true"]

    # ---- 汇总指标 ----
    print("=" * 58)
    print(f"  总样本数: {metrics['n']}")
    print("=" * 58)
    print(f"  {'指标':<22} {'值':>10}")
    print("-" * 58)
    print(f"  {'Accuracy':<22} {metrics['accuracy']:>10.4f}")
    print(f"  {'Macro-Precision':<22} {metrics['macro_precision']:>10.4f}")
    print(f"  {'Macro-Recall':<22} {metrics['macro_recall']:>10.4f}")
    print(f"  {'Macro-F1':<22} {metrics['macro_f1']:>10.4f}")
    print(f"  {'Weighted-F1':<22} {metrics['weighted_f1']:>10.4f}")
    print("=" * 58)

    # ---- 各类别明细 ----
    print()
    header = f"  {'类别':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'样本数':>8}"
    print(header)
    print("-" * 58)
    for lbl in labels:
        c = per_cls[lbl]
        name = LABEL_NAMES.get(lbl, f"{lbl}级")
        n_support = count_true.get(lbl, 0)
        print(
            f"  {name:<16} {c['precision']:>10.4f} {c['recall']:>10.4f}"
            f" {c['f1']:>10.4f} {n_support:>8}"
        )
    print("=" * 58)

    # ---- 混淆矩阵 ----
    cm = metrics["confusion_matrix"]
    col_w = 10
    print()
    print("  混淆矩阵（行=真实，列=预测）")
    print()
    header_parts = [f"{'':16}"] + [f"{LABEL_NAMES.get(lbl, str(lbl)):>{col_w}}" for lbl in labels]
    print("  " + "".join(header_parts))
    print("  " + "-" * (16 + col_w * len(labels)))
    for i, lbl in enumerate(labels):
        row_name = LABEL_NAMES.get(lbl, f"{lbl}级")
        row_parts = [f"{row_name:<16}"] + [f"{cm[i, j]:>{col_w}}" for j in range(len(labels))]
        print("  " + "".join(row_parts))
    print("=" * 58)

    # ---- 安全专项指标 ----
    s = metrics["safety"]
    print()
    print("  安全驾驶专项指标")
    print("=" * 58)
    print(f"  {'指标':<22} {'值':>10}  说明")
    print("-" * 58)
    def _fmt_optional(value: float | None) -> str:
        return f"{value:>10.4f}" if value is not None else f"{'N/A':>10}"
    print(
        f"  {'高危漏报率':<22} {_fmt_optional(s['high_risk_miss_rate'])}"
        f"  高危(2/3级)被判低危的比例"
        f"  [{s['_high_risk_missed']}/{s['_high_risk_total']}]"
    )
    print(
        f"  {'高危召回率':<22} {_fmt_optional(s['high_risk_recall'])}"
        f"  高危(2/3级)被正确捕捉的比例"
    )
    print(
        f"  {'风险降级误判率':<22} {_fmt_optional(s['risk_downgrade_rate'])}"
        f"  高危被预测等级严格低于真实等级的比例"
        f"  [{s['_high_risk_demoted']}/{s['_high_risk_total']}]"
    )
    print(
        f"  {'误报率':<22} {s['false_alarm_rate']:>10.4f}"
        f"  正常驾驶(0级)被误判为疲劳的比例"
        f"  [{s['_false_alarm_cnt']}/{s['_safe_total']}]"
    )
    print(
        f"  {'安全加权综合分':<22} {s['safety_weighted_score']:>10.4f}"
        f"  Σ F1(lbl)×w(lbl)，w={SAFETY_WEIGHTS}"
    )
    print("=" * 58)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="计算疲劳驾驶预测的分类指标")
    parser.add_argument(
        "--pred-file",
        type=Path,
        required=True,
        help="预测结果 JSON 文件路径",
    )
    parser.add_argument(
        "--pred-role",
        default=_DEFAULT_PRED_ROLE,
        help=f"预测角色名，即 messages 中的 role 字段（默认: {_DEFAULT_PRED_ROLE}）",
    )
    args = parser.parse_args()

    y_true, y_pred, skipped = load_data(args.pred_file, args.pred_role)

    if skipped:
        print(f"[Warning] 跳过 {skipped} 条样本（等级无法解析）\n")

    if not y_true:
        print("未找到有效样本，退出。")
        sys.exit(1)

    print(f"有效样本数: {len(y_true)}")
    print(f"GT   分布: {dict(sorted(Counter(y_true).items()))}")
    print(f"Pred 分布: {dict(sorted(Counter(y_pred).items()))}")
    print()

    metrics = compute_metrics(y_true, y_pred)
    print_report(metrics)


if __name__ == "__main__":
    main()
