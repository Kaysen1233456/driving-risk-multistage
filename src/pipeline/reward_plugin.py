from __future__ import annotations

import json
import re
from typing import Any, Iterable, List

from swift.plugin import ORM, orms


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        items = re.split(r"[，,]\s*", _normalize_text(tags))
    elif isinstance(tags, Iterable):
        items = list(tags)
    else:
        items = [_normalize_text(tags)]

    cleaned: list[str] = []
    for item in items:
        tag = _normalize_text(item)
        if not tag or tag.lower() in {"none", "null", "无", "[]"}:
            continue
        if tag not in cleaned:
            cleaned.append(tag)
    return cleaned


def _extract_answer_block(text: str) -> str:
    text = _normalize_text(text)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    if match:
        return match.group(1).strip()
    return text


def _extract_json(text: str) -> dict[str, Any] | None:
    answer = _extract_answer_block(text)
    if not answer:
        return None

    candidates = [answer]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", answer, flags=re.S)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    brace = re.search(r"(\{.*\})", answer, flags=re.S)
    if brace:
        candidates.insert(0, brace.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


class RankMatchORM(ORM):

    def __call__(self, completions, solution_rank, solution_best_model=None, **kwargs) -> List[float]:
        rewards: list[float] = []
        for idx, content in enumerate(completions):
            target_rank = solution_rank[idx] if idx < len(solution_rank) else []
            target_rank = [_normalize_text(item) for item in target_rank]
            target_best = (
                _normalize_text(solution_best_model[idx])
                if solution_best_model is not None and idx < len(solution_best_model)
                else (target_rank[0] if target_rank else "")
            )

            parsed = _extract_json(content)
            if not parsed:
                rewards.append(0.0)
                continue

            ranking = parsed.get("ranking", [])
            ranking = [_normalize_text(item) for item in ranking] if isinstance(ranking, list) else []
            best_model = _normalize_text(parsed.get("best_model"))

            if not target_rank:
                rewards.append(0.0)
                continue

            position_score = sum(1 for pred, gold in zip(ranking, target_rank) if pred == gold) / len(target_rank)
            set_score = 1.0 if len(ranking) == len(target_rank) and set(ranking) == set(target_rank) else 0.0
            best_score = 1.0 if best_model == target_best else 0.0
            rewards.append(min(1.0, 0.5 * position_score + 0.3 * set_score + 0.2 * best_score))
        return rewards


class CritiqueTagMatchORM(ORM):

    def __call__(self, completions, solution_tags, **kwargs) -> List[float]:
        rewards: list[float] = []
        for idx, content in enumerate(completions):
            gold_tags = set(_normalize_tags(solution_tags[idx] if idx < len(solution_tags) else []))
            parsed = _extract_json(content)
            if not parsed:
                rewards.append(0.0)
                continue

            pred_tags = set(_normalize_tags(parsed.get("abnormal_tags", [])))
            if not gold_tags and not pred_tags:
                rewards.append(1.0)
                continue
            if not gold_tags or not pred_tags:
                rewards.append(0.0)
                continue

            intersection = len(gold_tags & pred_tags)
            precision = intersection / len(pred_tags)
            recall = intersection / len(gold_tags)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            exact_bonus = 0.2 if gold_tags == pred_tags else 0.0
            rewards.append(min(1.0, 0.8 * f1 + exact_bonus))
        return rewards


class FinalLabelMatchORM(ORM):

    def __call__(self, completions, solution_label, **kwargs) -> List[float]:
        rewards: list[float] = []
        for idx, content in enumerate(completions):
            gold_label = _normalize_text(solution_label[idx] if idx < len(solution_label) else "")
            parsed = _extract_json(content)
            if not parsed:
                rewards.append(0.0)
                continue

            pred_label = _normalize_text(parsed.get("transform_label") or parsed.get("label"))
            rewards.append(1.0 if pred_label == gold_label else 0.0)
        return rewards


orms["zxgu_rank_match"] = RankMatchORM
orms["zxgu_critique_tag_match"] = CritiqueTagMatchORM
orms["zxgu_final_label_match"] = FinalLabelMatchORM
