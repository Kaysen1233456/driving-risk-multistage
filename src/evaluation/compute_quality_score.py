"""
驾驶员疲劳分析输出质量综合评分主入口

总体指标 = 大模型评估（LLM Judge）× 0.6 + 启发式规则评估 × 0.4

每条样本保留：
  - LLM Judge 各维度分（L1-L5）及加权总分
  - 启发式各维度分（H1-H5）及加权总分
  - 综合加权总分

最终输出：
  1. 终端报告：样本级均值 + 各维度均值 + 按 GT 等级分组均值
  2. JSON 文件（--output-detail）：{"samples": [...], "summary": {...}}

用法示例：
  # 仅启发式评分（离线，不调用 API）
  python eval/compute_quality_score.py --heuristic-only

  # 完整评分（LLM Judge + 启发式）
  python eval/compute_quality_score.py --config config.yaml

  # LLM Judge 仅对前 N 条（其余自动降级为启发式）
  python eval/compute_quality_score.py --config config.yaml --llm-eval-sample 50

  # 保存每条详细评分
  python eval/compute_quality_score.py --config config.yaml --output-detail eval/quality_detail.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml

# ── 同目录模块 ──────────────────────────────────────────────
_EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(_EVAL_DIR))

from heuristic_scorer import (
    BERT_SCORE_AVAILABLE,
    HEURISTIC_WEIGHTS,
    HeuristicScores,
    score_H5_batch,
    score_heuristic_single,
)
from llm_judge_scorer import LLM_WEIGHTS, LLMJudgeScores, call_judge_single
from compute_metrics import (
    LABEL_NAMES,
    SAFETY_WEIGHTS,
    extract_fatigue_level_from_path,
    extract_level_from_text,
    compute_metrics as compute_clf_metrics,
    print_report    as print_clf_report,
)

_DEFAULT_PRED_ROLE = "qwen3_5_0.8B"
_REF_ROLE          = "assistant"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 综合分权重
COMPOSITE_W_LLM = 0.6
COMPOSITE_W_HEU = 0.4


# ===========================================================================
# 数据加载
# ===========================================================================

def detect_model_roles(path: Path) -> list[str]:
    """从预测 JSON 中自动检测所有模型 role（排除 user 和 assistant）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    roles: list[str] = []
    seen: set[str] = set()
    for item in data:
        for m in item.get("messages", []):
            r = m["role"]
            if r not in (_REF_ROLE, "user") and r not in seen:
                seen.add(r)
                roles.append(r)
    return roles


def load_samples(path: Path, pred_role: str, ref_role: str = _REF_ROLE) -> list[dict]:
    """加载预测 JSON，返回每条包含 pred_text / ref_text / gt_level / pred_level 的字典。

    Args:
        path:      预测 JSON 文件路径
        pred_role: 模型预测的 role 名称
        ref_role:  参考答案的 role 名称（默认 "assistant"）
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples: list[dict] = []
    skipped = 0

    for item in data:
        video_path = item.get("videos", [""])[0]
        gt_level   = extract_fatigue_level_from_path(video_path)

        pred_text = next(
            (m["content"] for m in item["messages"] if m["role"] == pred_role), ""
        )
        pred_level = extract_level_from_text(pred_text)

        if gt_level is None or pred_level is None:
            skipped += 1
            continue

        ref_text = next(
            (m["content"] for m in item["messages"] if m["role"] == ref_role), ""
        )

        samples.append({
            "video_path": video_path,
            "gt_level":   gt_level,
            "pred_level": pred_level,
            "pred_text":  pred_text,
            "ref_text":   ref_text,
        })

    if skipped:
        logger.warning(f"[{pred_role}] 跳过 {skipped} 条无法解析等级的样本")

    return samples


# ===========================================================================
# 启发式评分（H1-H5）
# ===========================================================================

def run_heuristic_scoring(samples: list[dict]) -> list[dict]:
    """对所有样本运行 H1-H5，将 HeuristicScores 以字典形式附加到每条样本。"""
    pred_texts = [s["pred_text"] for s in samples]
    ref_texts  = [s["ref_text"]  for s in samples]

    # 批量计算 H5（BERTScore）
    if BERT_SCORE_AVAILABLE and any(ref_texts):
        logger.info(f"计算 BERTScore（H5），共 {len(pred_texts)} 条...")
        h5_results = score_H5_batch(pred_texts, ref_texts)
        h5_scores: list[Optional[float]] = [sc for sc, _ in h5_results]
    else:
        if not BERT_SCORE_AVAILABLE:
            logger.warning("bert-score 未安装，H5 将使用默认分 5.0（权重重分配）")
        h5_scores = [None] * len(samples)

    result: list[dict] = []
    for i, sample in enumerate(samples):
        hs = score_heuristic_single(
            pred_text=          sample["pred_text"],
            gt_level=           sample["gt_level"],
            pred_level=         sample["pred_level"],
            h5_score=           h5_scores[i],
        )
        result.append({**sample, "heuristic": _hs_to_dict(hs)})

    return result


def _hs_to_dict(hs: HeuristicScores) -> dict:
    return {
        "H1_format":        hs.H1_format,
        "H2_evidence":      hs.H2_evidence,
        "H3_temporal":      hs.H3_temporal,
        "H4_intervention":  hs.H4_intervention,
        "H5_semantic":      hs.H5_semantic,
        "weighted_total":   hs.weighted_total,
        "h5_available":     hs.h5_available,
        "details":          hs.details,
    }


# ===========================================================================
# LLM Judge 评分（L1-L5）
# ===========================================================================

def run_llm_judge_scoring(
    samples:      list[dict],
    api_key:      str,
    model:        str,
    base_url:     str,
    video_fps:    int,
    llm_eval_n:   Optional[int] = None,
    max_workers:  int = 4,
) -> list[dict]:
    """对 samples 中前 llm_eval_n 条（或全部）调用 LLM Judge。

    未参与 LLM Judge 的条目 llm_judge 字段为 None（后续自动降级到启发式评分）。
    """
    n_eval       = len(samples) if llm_eval_n is None else min(llm_eval_n, len(samples))
    eval_indices = set(range(n_eval))
    results      = list(samples)

    to_eval = [(i, samples[i]) for i in eval_indices]
    logger.info(f"LLM Judge 评分：{len(to_eval)} 条，并发 {max_workers} 线程")

    def _judge(idx_sample: tuple[int, dict]) -> tuple[int, LLMJudgeScores]:
        idx, sample = idx_sample
        return idx, call_judge_single(
            pred_text=  sample["pred_text"],
            video_path= sample["video_path"],
            api_key=    api_key,
            model=      model,
            base_url=   base_url,
            video_fps=  video_fps,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge, item): item[0] for item in to_eval}
        done = 0
        for future in as_completed(futures):
            idx, scores = future.result()
            results[idx] = {**results[idx], "llm_judge": _lj_to_dict(scores)}
            done += 1
            if done % 10 == 0 or done == len(to_eval):
                logger.info(f"  LLM Judge 进度: {done}/{len(to_eval)}")

    # 未评估的条目设 None
    for i in range(len(results)):
        results[i].setdefault("llm_judge", None)

    return results


def _lj_to_dict(lj: LLMJudgeScores) -> dict:
    return {
        "L1_visual_grounding":       lj.L1_visual_grounding,
        "L2_reasoning_coherence":    lj.L2_reasoning_coherence,
        "L3_counter_evidence":       lj.L3_counter_evidence,
        "L4_intervention_alignment": lj.L4_intervention_alignment,
        "L5_faithfulness":           lj.L5_faithfulness,
        "weighted_total":            lj.weighted_total,
        "raw_response":              lj.raw_response,
        "error":                     lj.error,
    }


# ===========================================================================
# 综合得分计算（样本级）
# ===========================================================================

def compute_composite(sample: dict) -> dict:
    """
    返回包含三类完整分数的字典，供样本级保存和数据集级汇总使用。

    综合分规则：
      - llm_judge 存在且无 error → composite = LLM×0.6 + 启发式×0.4
      - 否则降级 → composite = 启发式分，scoring_mode = "heuristic_only"
    """
    heu  = sample.get("heuristic", {})
    lj   = sample.get("llm_judge")

    heu_total = heu.get("weighted_total", 0.0)
    llm_valid = lj is not None and not lj.get("error")

    if llm_valid:
        llm_total  = lj["weighted_total"]
        composite  = round(llm_total * COMPOSITE_W_LLM + heu_total * COMPOSITE_W_HEU, 4)
        mode       = "composite"
    else:
        llm_total  = None
        composite  = round(heu_total, 4)
        mode       = "heuristic_only"

    return {
        # ── 综合 ──
        "composite_score": composite,
        "scoring_mode":    mode,

        # ── LLM Judge 维度分 ──
        "llm_weighted_total":            round(llm_total, 4) if llm_valid else None,
        "llm_L1_visual_grounding":       lj["L1_visual_grounding"]       if llm_valid else None,
        "llm_L2_reasoning_coherence":    lj["L2_reasoning_coherence"]    if llm_valid else None,
        "llm_L3_counter_evidence":       lj["L3_counter_evidence"]       if llm_valid else None,
        "llm_L4_intervention_alignment": lj["L4_intervention_alignment"] if llm_valid else None,
        "llm_L5_faithfulness":           lj["L5_faithfulness"]           if llm_valid else None,

        # ── 启发式维度分 ──
        "heu_weighted_total":   round(heu_total, 4),
        "heu_H1_format":        heu.get("H1_format"),
        "heu_H2_evidence":      heu.get("H2_evidence"),
        "heu_H3_temporal":      heu.get("H3_temporal"),
        "heu_H4_intervention":  heu.get("H4_intervention"),
        "heu_H5_semantic":      heu.get("H5_semantic"),
    }


# ===========================================================================
# 数据集级汇总
# ===========================================================================

def _avg(vals: list[float]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def build_clf_metrics(samples: list[dict]) -> dict:
    """从样本列表中提取 y_true / y_pred，计算分类指标并返回可序列化的字典。"""
    from collections import Counter as _Counter
    y_true = [s["gt_level"]   for s in samples]
    y_pred = [s["pred_level"] for s in samples]

    m = compute_clf_metrics(y_true, y_pred)

    # confusion_matrix 是 numpy array，转为嵌套列表方便 JSON 序列化
    cm_list = m["confusion_matrix"].tolist()

    # per_class key 是 int，JSON 要求 str
    per_class_str = {str(k): v for k, v in m["per_class"].items()}

    return {
        "n":               m["n"],
        "accuracy":        round(float(m["accuracy"]),        4),
        "macro_precision": round(float(m["macro_precision"]), 4),
        "macro_recall":    round(float(m["macro_recall"]),    4),
        "macro_f1":        round(float(m["macro_f1"]),        4),
        "weighted_f1":     round(float(m["weighted_f1"]),     4),
        "labels":          m["labels"],
        "per_class":       per_class_str,
        "count_true":      {str(k): v for k, v in _Counter(y_true).items()},
        "confusion_matrix": cm_list,
        "safety": {
            k: (round(float(v), 4) if isinstance(v, float) else v)
            for k, v in m["safety"].items()
        },
    }


def build_summary(samples_with_scores: list[dict]) -> dict:
    """对所有样本的 scores 字段做多维度均值汇总。"""
    scores = [s["scores"] for s in samples_with_scores if "scores" in s]
    n_total = len(scores)

    llm_scores  = [s["llm_weighted_total"]  for s in scores if s["llm_weighted_total"]  is not None]
    heu_scores  = [s["heu_weighted_total"]  for s in scores]
    comp_scores = [s["composite_score"]     for s in scores]

    # LLM 各维度
    llm_dims = {
        "L1_visual_grounding":       [s["llm_L1_visual_grounding"]       for s in scores if s["llm_L1_visual_grounding"]       is not None],
        "L2_reasoning_coherence":    [s["llm_L2_reasoning_coherence"]    for s in scores if s["llm_L2_reasoning_coherence"]    is not None],
        "L3_counter_evidence":       [s["llm_L3_counter_evidence"]       for s in scores if s["llm_L3_counter_evidence"]       is not None],
        "L4_intervention_alignment": [s["llm_L4_intervention_alignment"] for s in scores if s["llm_L4_intervention_alignment"] is not None],
        "L5_faithfulness":           [s["llm_L5_faithfulness"]           for s in scores if s["llm_L5_faithfulness"]           is not None],
    }
    # 启发式各维度
    heu_dims = {
        "H1_format":       [s["heu_H1_format"]       for s in scores if s["heu_H1_format"]       is not None],
        "H2_evidence":     [s["heu_H2_evidence"]     for s in scores if s["heu_H2_evidence"]      is not None],
        "H3_temporal":     [s["heu_H3_temporal"]     for s in scores if s["heu_H3_temporal"]      is not None],
        "H4_intervention": [s["heu_H4_intervention"] for s in scores if s["heu_H4_intervention"]  is not None],
        "H5_semantic":     [s["heu_H5_semantic"]     for s in scores if s["heu_H5_semantic"]      is not None],
    }

    # 按 GT 等级分组
    by_level: dict[int, dict[str, list]] = defaultdict(lambda: {"composite": [], "llm": [], "heuristic": []})
    for s_full in samples_with_scores:
        sc  = s_full.get("scores", {})
        lv  = s_full.get("gt_level")
        if lv is None:
            continue
        by_level[lv]["composite"].append(sc.get("composite_score"))
        by_level[lv]["llm"].append(sc.get("llm_weighted_total"))
        by_level[lv]["heuristic"].append(sc.get("heu_weighted_total"))

    level_summary = {}
    for lv in sorted(by_level):
        g = by_level[lv]
        level_summary[str(lv)] = {
            "label":     LABEL_NAMES.get(lv, f"{lv}级"),
            "n":         len(g["composite"]),
            "composite": _avg(g["composite"]),
            "llm":       _avg([v for v in g["llm"] if v is not None]),
            "heuristic": _avg(g["heuristic"]),
        }

    return {
        "n_total":           n_total,
        "n_llm_judged":      len(llm_scores),
        "avg_composite":     _avg(comp_scores),
        "avg_llm_total":     _avg(llm_scores),
        "avg_heu_total":     _avg(heu_scores),
        "avg_llm_by_dim":    {k: _avg(v) for k, v in llm_dims.items()},
        "avg_heu_by_dim":    {k: _avg(v) for k, v in heu_dims.items()},
        "avg_by_gt_level":   level_summary,
        # 分类指标由 main() 调用 build_clf_metrics() 后注入，此处占位
        "classification":    None,
    }


# ===========================================================================
# 报告打印
# ===========================================================================

def _fmt(v: Optional[float], width: int = 8) -> str:
    if v is None:
        return f"{'N/A':>{width}}"
    return f"{v:>{width}.4f}"


def print_quality_report(samples_with_scores: list[dict], summary: dict) -> None:
    W  = 68
    W2 = 72

    print()
    print("=" * W)
    print(f"  驾驶员疲劳分析输出质量综合评分报告")
    print("=" * W)
    print(f"  总样本数: {summary['n_total']}    "
          f"LLM Judge 覆盖: {summary['n_llm_judged']} 条")
    print("=" * W)

    # ── 三类总分均值 ──────────────────────────────────────────────
    print(f"  {'指标':<36} {'均值':>10}  说明")
    print("-" * W)
    print(f"  {'综合总分 (LLM×0.6 + 启发×0.4)':<36} {_fmt(summary['avg_composite']):>10}")
    print(f"  {'  大模型Judge总分 (LLM)':<36} {_fmt(summary['avg_llm_total']):>10}"
          f"  {summary['n_llm_judged']}条有效")
    print(f"  {'  启发式规则总分 (Heuristic)':<36} {_fmt(summary['avg_heu_total']):>10}")
    print("=" * W)

    # ── 各维度均值（对照表）─────────────────────────────────────
    print()
    print(f"  {'维度':<30} {'LLM Judge均值':>14}  {'启发式均值':>10}  权重对照")
    print("-" * W2)

    dim_pairs = [
        ("L1 视觉接地性",       "L1_visual_grounding",       "H1 格式规范性",   "H1_format",       "25% / 20%"),
        ("L2 推理连贯性",       "L2_reasoning_coherence",    "H2 证据词覆盖率", "H2_evidence",     "20% / 25%"),
        ("L3 特例排除完整性",   "L3_counter_evidence",       "H3 时间定位精确性","H3_temporal",    "20% / 15%"),
        ("L4 话术推理一致性",   "L4_intervention_alignment", "H4 等级话术一致性","H4_intervention","20% / 25%"),
        ("L5 解释忠实性",       "L5_faithfulness",           "H5 语义相似度",   "H5_semantic",     "15% / 15%"),
    ]
    llm_d = summary["avg_llm_by_dim"]
    heu_d = summary["avg_heu_by_dim"]

    for lname, lkey, hname, hkey, wt in dim_pairs:
        print(
            f"  {lname:<14} {_fmt(llm_d.get(lkey)):>14}  "
            f"  {hname:<12} {_fmt(heu_d.get(hkey)):>10}  {wt}"
        )
    print("=" * W2)

    # ── 按 GT 等级分组 ──────────────────────────────────────────
    print()
    print(f"  {'GT等级':<18} {'综合均值':>10}  {'LLM均值':>10}  {'启发式均值':>12}  {'样本数':>6}")
    print("-" * W2)
    for lv_str, info in summary["avg_by_gt_level"].items():
        print(
            f"  {info['label']:<18} {_fmt(info['composite']):>10}"
            f"  {_fmt(info['llm']):>10}  {_fmt(info['heuristic']):>12}"
            f"  {info['n']:>6}"
        )
    print("=" * W2)

    # ── 分类指标（来自 compute_metrics.py）────────────────────
    clf = summary.get("classification")
    if clf:
        print()
        print("=" * 58)
        print("  分类指标（Accuracy / F1 / 混淆矩阵 / 安全专项）")
        print("=" * 58)

        # 汇总指标
        print(f"  {'指标':<22} {'值':>10}")
        print("-" * 58)
        print(f"  {'Accuracy':<22} {clf['accuracy']:>10.4f}")
        print(f"  {'Macro-Precision':<22} {clf['macro_precision']:>10.4f}")
        print(f"  {'Macro-Recall':<22} {clf['macro_recall']:>10.4f}")
        print(f"  {'Macro-F1':<22} {clf['macro_f1']:>10.4f}")
        print(f"  {'Weighted-F1':<22} {clf['weighted_f1']:>10.4f}")
        print("=" * 58)

        # 各类别明细
        print()
        print(f"  {'类别':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'样本数':>8}")
        print("-" * 58)
        for lbl in clf["labels"]:
            c    = clf["per_class"][str(lbl)]
            name = LABEL_NAMES.get(lbl, f"{lbl}级")
            n_s  = clf["count_true"].get(str(lbl), 0)
            print(
                f"  {name:<16} {c['precision']:>10.4f} {c['recall']:>10.4f}"
                f" {c['f1']:>10.4f} {n_s:>8}"
            )
        print("=" * 58)

        # 混淆矩阵
        cm   = clf["confusion_matrix"]
        lbls = clf["labels"]
        col_w = 10
        print()
        print("  混淆矩阵（行=真实，列=预测）")
        print()
        hdr = [f"{'':16}"] + [f"{LABEL_NAMES.get(l, str(l)):>{col_w}}" for l in lbls]
        print("  " + "".join(hdr))
        print("  " + "-" * (16 + col_w * len(lbls)))
        for i, lbl in enumerate(lbls):
            row = [f"{LABEL_NAMES.get(lbl, f'{lbl}级'):<16}"] + [f"{cm[i][j]:>{col_w}}" for j in range(len(lbls))]
            print("  " + "".join(row))
        print("=" * 58)

        # 安全专项指标
        s = clf["safety"]
        print()
        print("  安全驾驶专项指标")
        print("=" * 58)
        print(f"  {'指标':<22} {'值':>10}  说明")
        print("-" * 58)
        def _fmt_optional(v):
            return f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
        print(
            f"  {'高危漏报率':<22} {_fmt_optional(s.get('high_risk_miss_rate'))}"
            f"  高危(2/3级)被判低危的比例"
            f"  [{s.get('_high_risk_missed', 0)}/{s.get('_high_risk_total', 0)}]"
        )
        print(
            f"  {'高危召回率':<22} {_fmt_optional(s.get('high_risk_recall'))}"
            f"  高危(2/3级)被正确捕捉的比例"
        )
        print(
            f"  {'风险降级误判率':<22} {_fmt_optional(s.get('risk_downgrade_rate'))}"
            f"  高危被预测等级严格低于真实等级的比例"
            f"  [{s.get('_high_risk_demoted', 0)}/{s.get('_high_risk_total', 0)}]"
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


# ===========================================================================
# JSON 序列化辅助
# ===========================================================================

def _json_default(obj):
    """处理 numpy scalar / dataclass 等不可直接序列化的对象。"""
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ===========================================================================
# CLI 参数
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="驾驶员疲劳分析输出质量综合评分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例：
  # 评估单个模型（启发式）
  python eval/compute_quality_score.py --pred-file data/process/driving_video_test_pred.json \\
      --pred-role qwen3_5_0.8B --heuristic-only

  # 批量评估 JSON 中所有模型（自动检测 role）
  python eval/compute_quality_score.py --pred-file data/process/driving_video_test_pred.json \\
      --all-roles --heuristic-only

  # 仅评估指定的某几个模型
  python eval/compute_quality_score.py --pred-file data/process/driving_video_test_pred.json \\
      --pred-roles qwen3_5_0.8B_base qwen3_5_2B_base qwen3_5_4B_base --heuristic-only

  # 完整评分（LLM Judge + 启发式）
  python eval/compute_quality_score.py --pred-file data/process/driving_video_test_pred.json \\
      --pred-role qwen3_5_0.8B --config config.yaml
        """,
    )
    p.add_argument("--pred-file",       required=True,
                   help="预测结果 JSON 文件路径")
    p.add_argument("--pred-role",       default=None,
                   help=f"要评估的单个模型 role 名称（默认: {_DEFAULT_PRED_ROLE}）；"
                        "与 --all-roles / --pred-roles 互斥")
    p.add_argument("--pred-roles",     nargs="+", default=None,
                   help="指定要评估的多个模型 role，与 --all-roles / --pred-role 互斥")
    p.add_argument("--all-roles",       action="store_true",
                   help="自动检测 JSON 中所有模型 role 并逐一评估，最后输出横向对比表")
    p.add_argument("--ref-role",        default=_REF_ROLE,
                   help=f"参考答案的 role 名称（默认: {_REF_ROLE}）")
    p.add_argument("--config",          default="config.yaml",
                   help="配置文件路径（读取 expert_api 配置）")
    p.add_argument("--heuristic-only",  action="store_true",
                   help="仅运行启发式评分（H1-H5），不调用 LLM Judge")
    p.add_argument("--llm-eval-sample", type=int, default=None,
                   help="仅对前 N 条调用 LLM Judge（其余降级为启发式）")
    p.add_argument("--output-detail",   default=None,
                   help="将每条样本详细评分 + 汇总保存为 JSON 文件；"
                        "批量模式下自动在文件名中插入 role 名")
    p.add_argument("--llm-workers",     type=int, default=4,
                   help="LLM Judge 并发线程数（默认 4）")
    p.add_argument("--judge-model",     default=None,
                   help="Judge 模型名称（覆盖 config.yaml 中的 expert_api.model）")
    p.add_argument("--sample",          type=int, default=None,
                   help="仅取前 N 条样本参与全部评分（用于快速验证）")
    return p.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# main
# ===========================================================================

def _eval_single_role(
    pred_file: Path,
    pred_role: str,
    ref_role: str,
    args: argparse.Namespace,
    llm_cfg: Optional[dict],
) -> tuple[list[dict], dict]:
    """对单个 pred_role 执行完整评估流程，返回 (samples_with_scores, summary)。"""
    logger.info(f"── 评估模型: {pred_role} ──")
    samples = load_samples(pred_file, pred_role=pred_role, ref_role=ref_role)
    if not samples:
        logger.warning(f"[{pred_role}] 未找到有效样本，跳过。")
        return [], {}

    if args.sample is not None:
        samples = samples[: args.sample]
        logger.info(f"[{pred_role}] 截取前 {len(samples)} 条样本")

    # 启发式评分
    samples = run_heuristic_scoring(samples)

    # LLM Judge 评分（按需）
    if llm_cfg is not None:
        samples = run_llm_judge_scoring(
            samples=     samples,
            api_key=     llm_cfg["api_key"],
            model=       llm_cfg["model"],
            base_url=    llm_cfg["base_url"],
            video_fps=   llm_cfg["video_fps"],
            llm_eval_n=  args.llm_eval_sample,
            max_workers= args.llm_workers,
        )

    # 综合得分
    for s in samples:
        s["scores"] = compute_composite(s)

    summary = build_summary(samples)
    summary["classification"] = build_clf_metrics(samples)

    return samples, summary


def _save_detail(
    out_path: Path,
    samples: list[dict],
    summary: dict,
    pred_role: str,
) -> None:
    """将单个模型的详细评分保存为 JSON 文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_samples = []
    for s in samples:
        lj = s.get("llm_judge") or {}
        hs = s.get("heuristic", {})
        output_samples.append({
            "video_path":       s["video_path"],
            "gt_level":         s["gt_level"],
            "pred_level":       s["pred_level"],
            "pred_text":        s["pred_text"],
            "ref_text":         s["ref_text"],
            "scores":           s.get("scores", {}),
            "llm_judge_detail": lj.get("raw_response", {}),
            "heuristic_detail": hs.get("details", {}),
        })
    output_doc = {
        "pred_role": pred_role,
        "samples":   output_samples,
        "summary":   summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info(f"详细评分已保存至: {out_path}")


def _output_path_for_role(base_path: str, role: str) -> Path:
    """在文件名（不含扩展名）后插入 role 名，避免多模型批量评估时文件互相覆盖。"""
    p = Path(base_path)
    return p.parent / f"{p.stem}_{role}{p.suffix}"


def _print_comparison_table(results: dict[str, dict]) -> None:
    """打印多模型横向对比汇总表。"""
    roles = list(results.keys())
    W = 100
    print()
    print("=" * W)
    print("  多模型横向对比汇总")
    print("=" * W)

    # 质量评分对比
    print(f"  {'模型 Role':<28} {'综合均值':>10}  {'启发式均值':>10}  {'LLM均值':>10}"
          f"  {'Accuracy':>9}  {'Macro-F1':>9}  {'安全加权分':>10}  {'样本数':>6}")
    print("-" * W)
    for role in roles:
        s = results[role].get("summary", {})
        clf = s.get("classification") or {}
        safety = clf.get("safety") or {}
        print(
            f"  {role:<28}"
            f" {_fmt(s.get('avg_composite')):>10}"
            f"  {_fmt(s.get('avg_heu_total')):>10}"
            f"  {_fmt(s.get('avg_llm_total')):>10}"
            f"  {clf.get('accuracy', float('nan')):>9.4f}"
            f"  {clf.get('macro_f1', float('nan')):>9.4f}"
            f"  {_fmt(safety.get('safety_weighted_score')):>10}"
            f"  {s.get('n_total', 0):>6}"
        )
    print("=" * W)

    # 安全专项对比
    print()
    print(f"  {'模型 Role':<28} {'高危漏报率':>10}  {'高危召回率':>10}  {'风险降级率':>10}  {'误报率':>8}")
    print("-" * W)
    for role in roles:
        clf    = results[role].get("summary", {}).get("classification") or {}
        safety = clf.get("safety") or {}
        print(
            f"  {role:<28}"
            f" {_fmt(safety.get('high_risk_miss_rate')):>10}"
            f"  {_fmt(safety.get('high_risk_recall')):>10}"
            f"  {_fmt(safety.get('risk_downgrade_rate')):>10}"
            f"  {_fmt(safety.get('false_alarm_rate')):>8}"
        )
    print("=" * W)


def main() -> None:
    args = parse_args()

    mutually_exclusive = sum([bool(args.all_roles), bool(args.pred_role), bool(args.pred_roles)])
    if mutually_exclusive > 1:
        print("错误：--all-roles、--pred-role、--pred-roles 只能使用其一。")
        sys.exit(1)

    pred_file = Path(args.pred_file)

    # 确定要评估的 role 列表
    if args.pred_roles:
        roles = args.pred_roles
        logger.info(f"指定评估 {len(roles)} 个模型 role: {roles}")
    elif args.all_roles:
        roles = detect_model_roles(pred_file)
        if not roles:
            print("未在 JSON 中检测到任何模型 role，退出。")
            sys.exit(1)
        logger.info(f"检测到 {len(roles)} 个模型 role: {roles}")
    else:
        roles = [args.pred_role or _DEFAULT_PRED_ROLE]

    # 预加载 LLM Judge 配置（只在需要时）
    llm_cfg: Optional[dict] = None
    if not args.heuristic_only:
        cfg        = load_yaml_config(args.config)
        expert_cfg = cfg.get("expert_api", {})
        llm_cfg = {
            "api_key":   os.environ.get("DASHSCOPE_API_KEY", expert_cfg.get("api_key", "")),
            "model":     args.judge_model or expert_cfg.get("model", "qwen-vl-max"),
            "base_url":  expert_cfg.get("base_url", "https://dashscope.aliyuncs.com/api/v1"),
            "video_fps": expert_cfg.get("video_fps", 3),
        }

    # 逐模型评估
    all_results: dict[str, dict] = {}

    for role in roles:
        samples, summary = _eval_single_role(
            pred_file=pred_file,
            pred_role=role,
            ref_role=args.ref_role,
            args=args,
            llm_cfg=llm_cfg,
        )
        if not samples:
            continue

        all_results[role] = {"samples": samples, "summary": summary}

        # 单模型报告（批量模式也逐个打印）
        print()
        print(f"{'─' * 20}  模型: {role}  {'─' * 20}")
        print_quality_report(samples, summary)

        # 保存详细 JSON
        if args.output_detail:
            if len(roles) > 1:
                out_path = _output_path_for_role(args.output_detail, role)
            else:
                out_path = Path(args.output_detail)
            _save_detail(out_path, samples, summary, role)

    # 多模型汇总对比表
    if len(all_results) > 1:
        _print_comparison_table(all_results)

        # 保存多模型汇总 JSON
        if args.output_detail:
            summary_path = _output_path_for_role(args.output_detail, "comparison")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_doc = {
                role: {
                    "summary": r["summary"],
                }
                for role, r in all_results.items()
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(comparison_doc, f, ensure_ascii=False, indent=2, default=_json_default)
            logger.info(f"多模型对比汇总已保存至: {summary_path}")


if __name__ == "__main__":
    main()
