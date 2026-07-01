from __future__ import annotations

import hashlib
import json
import random
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


STUDENT_MODELS: tuple[str, ...] = (
    "qwen3-vl-flash",
    "qwen3-vl-plus",
    "qwen3.5-flash",
    "qwen3.5-plus",
)

RANDOM_SEED = 3407
SPLIT_RATIOS = (0.8, 0.1, 0.1)
IGNORE_TAG_VALUES = {"", "无", "none", "None", "NONE", "null", "NULL", "[]"}


COMMON_RUBRIC = textwrap.dedent(
    """
    你是公交/营运车辆驾驶风险评审专家，需要结合视频画面与平台事件数据完成判断。

    评审原则：
    1. 先看视频证据，再结合平台事件表和聚合统计，避免机械采信平台误报。
    2. 严格区分疲劳、分心、环境干扰、习惯性闭眼/眨眼、强光、干眼症等非疲劳因素。
    3. 对风险分级要保守且一致，只有证据充分时才升级到人工干预或唤醒。
    4. 输出中的模型名、标签字符串必须和输入保持完全一致，不要自造新名称。
    """
).strip()


FINAL_JUDGE_SYSTEM = (
    COMMON_RUBRIC
    + "\n\n你要输出最终专家判断，字段包括 transform_label、risk_description、intervention、risk_decision。"
)

RANK_SYSTEM = (
    COMMON_RUBRIC
    + "\n\n你要评估 4 个学生模型回答的质量，只比较事实一致性、风险等级合理性、分心/疲劳区分、是否幻觉、描述完整度。"
)

CRITIQUE_SYSTEM = (
    COMMON_RUBRIC
    + "\n\n你要评估单个学生模型回答的异常标签，并给出简明不足分析。异常标签尽量沿用训练数据中的原始中文标签。"
)

COT_SYSTEM = (
    COMMON_RUBRIC
    + "\n\n你要先给出思考过程，再输出最终标签与结构化判断。思考过程应围绕视频证据、平台数据、分心/疲劳区分和干预等级。"
)

REASON_SYSTEM = (
    COMMON_RUBRIC
    + "\n\n你会拿到专家排序结果，需要解释每个学生模型为什么处于该位置。重点解释证据充分性、幻觉、等级判断和分心/疲劳区分。"
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    train_mode: str
    description: str
    system_prompt: str
    default_max_length: int
    default_max_completion_length: int
    default_learning_rate: float
    default_num_train_epochs: float
    default_num_generations: int = 0
    reward_name: str | None = None
    warm_start_from: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


STAGE_SPECS: dict[str, StageSpec] = {
    "01_final_judge_sft": StageSpec(
        name="01_final_judge_sft",
        train_mode="sft",
        description="视频 + 平台数据 -> 最终专家描述/干预/标签",
        system_prompt=FINAL_JUDGE_SYSTEM,
        default_max_length=6144,
        default_max_completion_length=1024,
        default_learning_rate=1e-4,
        default_num_train_epochs=3.0,
    ),
    "02_student_rank_sft": StageSpec(
        name="02_student_rank_sft",
        train_mode="sft",
        description="视频 + 4 个学生回答 -> 排序（GRPO 预热）",
        system_prompt=RANK_SYSTEM,
        default_max_length=8192,
        default_max_completion_length=768,
        default_learning_rate=8e-5,
        default_num_train_epochs=3.0,
    ),
    "03_student_rank_grpo": StageSpec(
        name="03_student_rank_grpo",
        train_mode="grpo",
        description="视频 + 4 个学生回答 -> 排序 GRPO 强化",
        system_prompt=RANK_SYSTEM,
        default_max_length=8192,
        default_max_completion_length=768,
        default_learning_rate=5e-6,
        default_num_train_epochs=1.0,
        default_num_generations=4,
        reward_name="zxgu_rank_match",
        warm_start_from="02_student_rank_sft",
    ),
    "04_student_critique_sft": StageSpec(
        name="04_student_critique_sft",
        train_mode="sft",
        description="视频 + 单个学生回答 -> 异常标签/不足（GRPO 预热）",
        system_prompt=CRITIQUE_SYSTEM,
        default_max_length=6144,
        default_max_completion_length=512,
        default_learning_rate=1e-4,
        default_num_train_epochs=3.0,
    ),
    "05_student_critique_grpo": StageSpec(
        name="05_student_critique_grpo",
        train_mode="grpo",
        description="视频 + 单个学生回答 -> 异常标签 GRPO 强化",
        system_prompt=CRITIQUE_SYSTEM,
        default_max_length=6144,
        default_max_completion_length=512,
        default_learning_rate=5e-6,
        default_num_train_epochs=1.0,
        default_num_generations=4,
        reward_name="zxgu_critique_tag_match",
        warm_start_from="04_student_critique_sft",
    ),
    "06_final_label_cot_sft": StageSpec(
        name="06_final_label_cot_sft",
        train_mode="sft",
        description="视频 + 平台数据 -> 思维链 + 最终标签/干预（GRPO 预热）",
        system_prompt=COT_SYSTEM,
        default_max_length=8192,
        default_max_completion_length=1024,
        default_learning_rate=6e-5,
        default_num_train_epochs=2.0,
    ),
    "07_final_label_cot_grpo": StageSpec(
        name="07_final_label_cot_grpo",
        train_mode="grpo",
        description="视频 + 平台数据 -> 思维链 GRPO 强化",
        system_prompt=COT_SYSTEM,
        default_max_length=8192,
        default_max_completion_length=1024,
        default_learning_rate=2e-6,
        default_num_train_epochs=1.0,
        default_num_generations=4,
        reward_name="zxgu_final_label_match",
        warm_start_from="06_final_label_cot_sft",
    ),
    "08_ranking_reason_sft": StageSpec(
        name="08_ranking_reason_sft",
        train_mode="sft",
        description="视频 + 学生回答 + 排序结果 -> 排序原因",
        system_prompt=REASON_SYSTEM,
        default_max_length=8192,
        default_max_completion_length=1024,
        default_learning_rate=6e-5,
        default_num_train_epochs=2.0,
    ),
}


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def dumps_jsonl_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def clip_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()} ...[截断]"


def normalize_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[，,]\s*", normalize_text(raw))
    tags: list[str] = []
    for item in items:
        tag = normalize_text(item)
        if tag in IGNORE_TAG_VALUES:
            continue
        if tag not in tags:
            tags.append(tag)
    return tags


def truncate_markdown_table(text: str, max_rows: int = 16, max_chars: int = 2200) -> str:
    text = normalize_text(text)
    if not text:
        return "无"

    lines = text.splitlines()
    if len(lines) <= 2:
        return clip_text(text, max_chars=max_chars)

    header = lines[:2]
    rows = lines[2:]
    if len(rows) > max_rows:
        rows = rows[:max_rows] + [f"| ... | 共截断 {len(lines) - 2 - max_rows} 行 |"]
    return clip_text("\n".join(header + rows), max_chars=max_chars)


def parse_labeled_sections(text: str) -> dict[str, str]:
    alias_to_field = {
        "风险解读": "risk_description",
        "综合描述": "risk_description",
        "干预话术": "intervention",
        "干预建议": "intervention",
        "风险等级判断（疲劳或者分心）": "risk_decision",
        "风险等级判断": "risk_decision",
        "疲劳等级判断": "risk_decision",
        "疲劳等级判定": "risk_decision",
        "疲劳等级判断为": "risk_decision",
        "疲劳等级判定为": "risk_decision",
    }
    pattern = re.compile(
        r"(" + "|".join(sorted((re.escape(alias) for alias in alias_to_field), key=len, reverse=True)) + r")\s*[:：]"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        alias = match.group(1)
        field_name = alias_to_field[alias]
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = normalize_text(text[start:end].strip("；;。 \n"))
        if value and field_name not in sections:
            sections[field_name] = value
    return sections


def derive_intervention_from_text(text: str) -> str:
    patterns = [
        r"(无需干预[^。；;\n]*[。；;]?)",
        r"(建议[^。；;\n]*[。；;]?)",
        r"(请[^。；;\n]*[。；;]?)",
        r"(系统已通知后台介入[^。；;\n]*[。；;]?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return normalize_text(matches[-1].strip("；;。 "))
    return ""


def derive_risk_decision_from_text(text: str, fallback_label: str) -> str:
    patterns = [
        r"风险等级判断（疲劳或者分心）\s*[:：]\s*([^；;\n]+)",
        r"风险等级判断\s*[:：]\s*([^；;\n]+)",
        r"疲劳等级判断(?:为)?\s*[:：]?\s*([^；;，,\n。]+)",
        r"疲劳等级判定(?:为)?\s*[:：]?\s*([^；;，,\n。]+)",
        r"综合判定司机处于\s*([^；;，,\n。]+)",
        r"符合\s*([^；;，,\n。]*疲劳[^；;，,\n。]*)(?:标准)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        decision = normalize_text(match.group(1).strip("；;。 "))
        if decision:
            return decision.removesuffix("标准").strip()
    return normalize_text(fallback_label)


def parse_final_composite(final_text: str, fallback_label: str) -> dict[str, str]:
    text = normalize_text(final_text)
    if not text:
        return {
            "risk_description": "",
            "intervention": "请结合现场风险谨慎驾驶。",
            "risk_decision": normalize_text(fallback_label),
        }

    sections = parse_labeled_sections(text)
    risk_decision = normalize_text(sections.get("risk_decision")) or derive_risk_decision_from_text(
        text, fallback_label
    )
    intervention = normalize_text(sections.get("intervention")) or derive_intervention_from_text(text)
    if not intervention:
        intervention = "请结合现场风险谨慎驾驶。"

    description = normalize_text(sections.get("risk_description"))
    if not description:
        description = text
        for value in sections.values():
            value = normalize_text(value)
            if not value:
                continue
            description = description.replace(value, " ")
        for alias in (
            "风险解读",
            "综合描述",
            "干预话术",
            "干预建议",
            "风险等级判断（疲劳或者分心）",
            "风险等级判断",
            "疲劳等级判断",
            "疲劳等级判定",
            "疲劳等级判断为",
            "疲劳等级判定为",
        ):
            description = description.replace(f"{alias}：", " ")
            description = description.replace(f"{alias}:", " ")
        if intervention:
            description = description.replace(intervention, " ")
        description = normalize_text(re.sub(r"\s*[；;]\s*", "；", description))
        description = description.strip("；;。 ")
        description = re.sub(r"(建议|请)\s*$", "", description).strip()
    if not description:
        description = text

    return {
        "risk_description": description,
        "intervention": intervention,
        "risk_decision": risk_decision,
    }


def choose_video_path(primary: str, secondary: str) -> str:
    primary = normalize_text(primary)
    secondary = normalize_text(secondary)
    if primary:
        return primary
    return secondary


def normalize_input_dirs(input_dirs: Sequence[str | Path] | str | Path) -> list[Path]:
    if isinstance(input_dirs, (str, Path)):
        raw_dirs: Sequence[str | Path] = [input_dirs]
    else:
        raw_dirs = input_dirs

    resolved: list[Path] = []
    seen: set[str] = set()
    for raw_dir in raw_dirs:
        input_path = Path(raw_dir).resolve()
        input_key = str(input_path)
        if input_key in seen:
            continue
        seen.add(input_key)
        resolved.append(input_path)
    return resolved


def build_pair_key(root_dir: Path, file_path: Path, folder_name: str, file_suffix: str) -> str:
    relative_path = file_path.relative_to(root_dir)
    parent_parts = [part for part in relative_path.parts[:-1] if part != folder_name]
    stem = file_path.stem
    if stem.endswith(file_suffix):
        stem = stem[: -len(file_suffix)]
    return "/".join(parent_parts + [stem])


def make_sample_id(source_root: Path, pair_key: str, image_id: str) -> str:
    digest = hashlib.sha1(f"{source_root.resolve()}::{pair_key}".encode("utf-8")).hexdigest()[:12]
    base = normalize_text(image_id) or Path(pair_key).name or "sample"
    base = re.sub(r"[^\w.\-]+", "_", base).strip("_") or "sample"
    return f"{base}__{digest}"


def infer_plate_number(image_id: str) -> str:
    image_id = normalize_text(image_id)
    if not image_id:
        return ""
    return normalize_text(image_id.split("_", 1)[0])


def infer_alarm_time_ms(image_id: str) -> str:
    image_id = normalize_text(image_id)
    parts = image_id.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return ""


def format_source_ref(root_dir: Path, file_path: Path) -> str:
    return f"{root_dir}::{file_path.relative_to(root_dir)}"


def synthesize_rank_think(record: dict[str, Any]) -> str:
    top_model = record["ranking_order"][0]
    common_errors = []
    for tags in record["abnormal_tags_by_model"].values():
        common_errors.extend(tags)
    common_errors = [tag for tag in common_errors if tag]
    common_errors = list(dict.fromkeys(common_errors))[:4]
    steps = [
        f"先以视频证据和平台事件为基准核对 4 个学生回答，最佳回答是 {top_model}。",
        "重点比较事实是否与视频一致、风险等级是否过高/过低、是否把分心误判成疲劳、是否出现幻觉。",
    ]
    if common_errors:
        steps.append(f"这组样本里常见问题包括：{'、'.join(common_errors)}。")
    steps.append("最后仅输出模型 id 排序，保持 best_model 与 ranking[0] 一致。")
    return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(steps))


def synthesize_critique_think(model_name: str, reason: str, tags: Sequence[str]) -> str:
    parts = [
        f"当前需要检查学生模型 {model_name} 的回答是否与视频和平台数据一致。",
        f"专家排序原因：{normalize_text(reason)}",
    ]
    if tags:
        parts.append(f"专家标注的异常标签：{'、'.join(tags)}。")
    else:
        parts.append("专家未标出明显异常标签，说明该回答整体可信。")
    return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(parts))


def synthesize_cot_think(record: dict[str, Any]) -> str:
    best_model = record["ranking_order"][0]
    steps = [
        f"先结合视频和平台数据确认主要风险类型，平台初判为：{record['risk_assessment'] or '无'}。",
        "再判断视频中的关键行为到底更接近疲劳、分心还是环境干扰，不机械依赖平台告警。",
        f"学生回答中最可信的是 {best_model}，其余回答可用于辅助比对常见误判点。",
        f"最终需要输出精确的 transform_label：{record['transform_label']}，并给出风险解读、干预话术和风险判断。",
    ]
    return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(steps))


def build_common_context(record: dict[str, Any], include_event_rows: int = 16) -> str:
    sections = [
        f"样本 ID: {record['image_id']}",
        f"车牌号: {record['plate_number'] or '未知'}",
        f"报警 ID: {record['alarm_id'] or '未知'}",
        f"报警时间(ms): {record['alarm_time_ms'] or '未知'}",
        f"平台风险初判: {record['risk_assessment'] or '无'}",
        "",
        "告警摘要:",
        truncate_markdown_table(record["alarm_info"], max_rows=4, max_chars=600),
        "",
        "聚合事件统计:",
        truncate_markdown_table(record["aggregate_event_data"], max_rows=12, max_chars=900),
        "",
        "关键事件明细（截断）:",
        truncate_markdown_table(record["event_info"], max_rows=include_event_rows, max_chars=1800),
        "",
        "司机画像:",
        clip_text(record["driver_profile_info"] or "无相关数据", max_chars=300),
    ]
    return "\n".join(sections).strip()


def build_student_answers_block(record: dict[str, Any], only_model: str | None = None, max_chars: int = 1600) -> str:
    blocks: list[str] = []
    model_names = (only_model,) if only_model else STUDENT_MODELS
    for model_name in model_names:
        answer = clip_text(record["student_answers"][model_name], max_chars=max_chars)
        blocks.append(f"[{model_name}]\n{answer}")
    return "\n\n".join(blocks)


def build_messages(system_prompt: str, user_content: str, assistant_content: str | None = None) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def build_final_answer_json(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "transform_label": record["transform_label"],
        "risk_description": record["final_parts"]["risk_description"],
        "intervention": record["final_parts"]["intervention"],
        "risk_decision": record["final_parts"]["risk_decision"],
    }


def build_rank_answer_json(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "ranking": record["ranking_order"],
        "best_model": record["ranking_order"][0],
    }


def build_critique_answer_json(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    reason = record["ranking_reason_by_model"][model_name]
    tags = record["abnormal_tags_by_model"][model_name]
    return {
        "student_model": model_name,
        "abnormal_tags": tags,
        "critique": normalize_text(reason),
    }


def build_reason_answer_json(record: dict[str, Any]) -> dict[str, Any]:
    items = []
    for model_name in record["ranking_order"]:
        items.append(
            {
                "model": model_name,
                "reason": record["ranking_reason_by_model"][model_name],
            }
        )
    return {"ranking_reasons": items}


def build_final_user_prompt(record: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        <video>
        请结合视频和平台事件信息，输出最终专家判断。

        {build_common_context(record)}

        输出要求：
        1. 只输出 JSON。
        2. JSON 字段固定为：transform_label、risk_description、intervention、risk_decision。
        3. transform_label 必须输出精确标签字符串。
        4. risk_description 只写专家综合判断，不要复制整张表。
        5. intervention 必须是可直接播报给司机的话术。
        """
    ).strip()


def build_rank_user_prompt(record: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        <video>
        你将看到同一段视频的 4 个学生模型回答。请按照回答质量从高到低排序。

        {build_common_context(record, include_event_rows=12)}

        学生回答：
        {build_student_answers_block(record, max_chars=1400)}

        输出要求：
        1. 先在 <think> 中简要说明判断逻辑。
        2. 再在 <answer> 中输出 JSON。
        3. JSON 字段固定为：ranking、best_model。
        4. ranking 必须包含且只包含这 4 个模型 id，不能重复。
        5. best_model 必须等于 ranking[0]。
        """
    ).strip()


def build_critique_user_prompt(record: dict[str, Any], model_name: str) -> str:
    return textwrap.dedent(
        f"""
        <video>
        请评估单个学生模型回答是否存在问题，并输出异常标签与简明不足分析。

        {build_common_context(record, include_event_rows=10)}

        当前学生模型：{model_name}
        学生回答：
        {build_student_answers_block(record, only_model=model_name, max_chars=1800)}

        输出要求：
        1. 先在 <think> 中说明判断依据。
        2. 再在 <answer> 中输出 JSON。
        3. JSON 字段固定为：student_model、abnormal_tags、critique。
        4. abnormal_tags 必须是字符串数组；如果没有明显问题，输出空数组 []。
        5. student_model 必须保持为输入中的模型 id。
        """
    ).strip()


def build_cot_user_prompt(record: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        <video>
        请先给出判断过程，再输出最终专家结论。

        {build_common_context(record)}

        学生回答参考：
        {build_student_answers_block(record, max_chars=900)}

        输出要求：
        1. 必须先输出 <think> ... </think>。
        2. 然后输出 <answer> ... </answer>，其中 answer 部分是 JSON。
        3. JSON 字段固定为：transform_label、risk_description、intervention、risk_decision。
        4. transform_label 必须输出精确标签字符串。
        """
    ).strip()


def build_reason_user_prompt(record: dict[str, Any]) -> str:
    ranking_text = "\n".join(
        f"{idx + 1}. {model_name}" for idx, model_name in enumerate(record["ranking_order"])
    )
    return textwrap.dedent(
        f"""
        <video>
        已知专家已经给出了 4 个学生模型的排序结果，请解释每个模型为什么处于该位置。

        {build_common_context(record, include_event_rows=12)}

        学生回答：
        {build_student_answers_block(record, max_chars=1400)}

        专家排序结果：
        {ranking_text}

        输出要求：
        1. 只输出 JSON。
        2. JSON 字段固定为 ranking_reasons。
        3. ranking_reasons 必须是长度为 4 的数组，每项包含 model 和 reason。
        4. model 的顺序必须与专家排序结果一致。
        """
    ).strip()


def build_rank_sft_answer(record: dict[str, Any]) -> str:
    think = synthesize_rank_think(record)
    answer = dumps_json(build_rank_answer_json(record))
    return f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"


def build_critique_sft_answer(record: dict[str, Any], model_name: str) -> str:
    think = synthesize_critique_think(
        model_name=model_name,
        reason=record["ranking_reason_by_model"][model_name],
        tags=record["abnormal_tags_by_model"][model_name],
    )
    answer = dumps_json(build_critique_answer_json(record, model_name))
    return f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"


def build_cot_sft_answer(record: dict[str, Any]) -> str:
    think = synthesize_cot_think(record)
    answer = dumps_json(build_final_answer_json(record))
    return f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"


def load_valid_records(input_dirs: Sequence[str | Path] | str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_paths = normalize_input_dirs(input_dirs)

    label_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "input_dirs": [str(path) for path in input_paths],
        "label_files": 0,
        "ranking_files": 0,
        "valid_records": 0,
        "skipped_missing_label": [],
        "skipped_invalid_ranking": [],
        "skipped_missing_student_model": [],
        "skipped_duplicate_label_pair": [],
    }

    for input_path in input_paths:
        for label_path in sorted(input_path.rglob("*_labels.json")):
            stats["label_files"] += 1
            pair_key = build_pair_key(input_path, label_path, folder_name="labels", file_suffix="_labels")
            pair_id = (str(input_path), pair_key)
            if pair_id in label_by_pair:
                stats["skipped_duplicate_label_pair"].append(format_source_ref(input_path, label_path))
                continue
            label_by_pair[pair_id] = {
                "path": label_path,
                "pair_key": pair_key,
                "root_dir": input_path,
                "data": json.loads(label_path.read_text(encoding="utf-8")),
            }

    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        for ranking_path in sorted(input_path.rglob("*_ranking.json")):
            stats["ranking_files"] += 1
            pair_key = build_pair_key(input_path, ranking_path, folder_name="rankings", file_suffix="_ranking")
            label_entry = label_by_pair.get((str(input_path), pair_key))
            if label_entry is None:
                stats["skipped_missing_label"].append(format_source_ref(input_path, ranking_path))
                continue

            data = json.loads(ranking_path.read_text(encoding="utf-8"))
            label_data = label_entry["data"]
            ranking = data.get("ranking")
            if not isinstance(ranking, dict):
                stats["skipped_invalid_ranking"].append(format_source_ref(input_path, ranking_path))
                continue

            order = ranking.get("排序结果")
            reasons = ranking.get("排序原因")
            abnormal = ranking.get("异常标签")
            final_text = normalize_text(ranking.get("最终综合描述"))
            normalized_order = [normalize_text(model_name) for model_name in order] if isinstance(order, list) else []
            if (
                not isinstance(order, list)
                or not isinstance(reasons, list)
                or not isinstance(abnormal, list)
                or len(normalized_order) != len(STUDENT_MODELS)
                or len(reasons) != len(STUDENT_MODELS)
                or len(abnormal) != len(STUDENT_MODELS)
                or len(set(normalized_order)) != len(STUDENT_MODELS)
                or set(normalized_order) != set(STUDENT_MODELS)
                or not final_text
            ):
                stats["skipped_invalid_ranking"].append(format_source_ref(input_path, ranking_path))
                continue

            student_answers = label_data.get("labels", {})
            if not isinstance(student_answers, dict) or not all(model_name in student_answers for model_name in STUDENT_MODELS):
                stats["skipped_missing_student_model"].append(format_source_ref(input_path, ranking_path))
                continue

            ranking_reason_by_model = {
                model_name: normalize_text(reason) for model_name, reason in zip(normalized_order, reasons)
            }
            abnormal_tags_by_model = {
                model_name: normalize_tags(tag) for model_name, tag in zip(normalized_order, abnormal)
            }

            transform_label = first_non_empty(
                data.get("transform_label"),
                label_data.get("transform_label"),
                data.get("label"),
                label_data.get("label"),
            )
            image_id = first_non_empty(data.get("image_id"), label_data.get("image_id"), Path(pair_key).name)
            sample_id = make_sample_id(input_path, pair_key, image_id)
            final_parts = parse_final_composite(final_text, transform_label)
            record = {
                "sample_id": sample_id,
                "source_root": str(input_path),
                "source_pair_key": pair_key,
                "image_id": image_id,
                "video_path": choose_video_path(
                    first_non_empty(data.get("video_path_absolute"), label_data.get("video_path_absolute")),
                    first_non_empty(data.get("video_path"), label_data.get("video_path")),
                ),
                "student_answers": {model_name: normalize_text(student_answers[model_name]) for model_name in STUDENT_MODELS},
                "ranking_order": normalized_order,
                "ranking_reason_by_model": ranking_reason_by_model,
                "abnormal_tags_by_model": abnormal_tags_by_model,
                "final_text": final_text,
                "final_parts": final_parts,
                "transform_label": transform_label,
                "alarm_info": first_non_empty(data.get("alarmInfo"), label_data.get("alarmInfo")),
                "event_info": first_non_empty(data.get("eventInfo"), label_data.get("eventInfo")),
                "aggregate_event_data": first_non_empty(
                    data.get("aggregateEventData"), label_data.get("aggregateEventData")
                ),
                "driver_profile_info": first_non_empty(
                    data.get("driverProfileInfo"), label_data.get("driverProfileInfo")
                ),
                "risk_assessment": first_non_empty(
                    data.get("RiskAssessment"),
                    label_data.get("RiskAssessment"),
                    data.get("risk_type_post_process"),
                    label_data.get("risk_type_post_process"),
                    transform_label,
                ),
                "alarm_id": first_non_empty(data.get("alarm_id"), label_data.get("alarm_id")),
                "alarm_time_ms": first_non_empty(
                    data.get("alarm_time_ms"),
                    label_data.get("alarm_time_ms"),
                    infer_alarm_time_ms(image_id),
                ),
                "plate_number": first_non_empty(
                    data.get("plate_number"),
                    label_data.get("plate_number"),
                    infer_plate_number(image_id),
                ),
                "ranking_path": str(ranking_path),
                "label_path": str(label_entry["path"]),
            }
            records.append(record)

    records.sort(key=lambda item: item["sample_id"])
    stats["valid_records"] = len(records)
    return records, stats


def build_split_map(records: Sequence[dict[str, Any]], seed: int = RANDOM_SEED) -> dict[str, str]:
    sample_ids = [record["sample_id"] for record in records]
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    total = len(sample_ids)
    train_count = int(total * SPLIT_RATIOS[0])
    val_count = int(total * SPLIT_RATIOS[1])
    test_count = total - train_count - val_count

    if total >= 3:
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
        test_count = max(test_count, 1)
        while train_count + val_count + test_count > total:
            train_count -= 1

    split_map: dict[str, str] = {}
    for sample_id in sample_ids[:train_count]:
        split_map[sample_id] = "train"
    for sample_id in sample_ids[train_count : train_count + val_count]:
        split_map[sample_id] = "val"
    for sample_id in sample_ids[train_count + val_count :]:
        split_map[sample_id] = "test"
    return split_map


def ensure_jsonable_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def stage_names() -> list[str]:
    return list(STAGE_SPECS.keys())


def get_stage_spec(stage_name: str) -> StageSpec:
    return STAGE_SPECS[stage_name]
