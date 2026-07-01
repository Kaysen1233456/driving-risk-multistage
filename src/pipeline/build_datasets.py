from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_lib import (
    STUDENT_MODELS,
    build_cot_sft_answer,
    build_cot_user_prompt,
    build_critique_answer_json,
    build_critique_sft_answer,
    build_critique_user_prompt,
    build_final_answer_json,
    build_final_user_prompt,
    build_messages,
    build_rank_answer_json,
    build_rank_sft_answer,
    build_rank_user_prompt,
    build_reason_answer_json,
    build_reason_user_prompt,
    build_split_map,
    dumps_json,
    dumps_jsonl_line,
    get_stage_spec,
    load_valid_records,
    stage_names,
)


def make_sample_base(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "image_id": record["image_id"],
        "source_root": record["source_root"],
        "source_pair_key": record["source_pair_key"],
        "videos": [record["video_path"]],
        "plate_number": record["plate_number"],
        "alarm_id": record["alarm_id"],
        "alarm_time_ms": record["alarm_time_ms"],
    }


def final_judge_samples(record: dict[str, Any], infer_only: bool = False) -> dict[str, Any]:
    assistant = None if infer_only else dumps_json(build_final_answer_json(record))
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("01_final_judge_sft").system_prompt,
        user_content=build_final_user_prompt(record),
        assistant_content=assistant,
    )
    sample["expected_transform_label"] = record["transform_label"]
    sample["expected_risk_decision"] = record["final_parts"]["risk_decision"]
    return sample


def rank_sft_samples(record: dict[str, Any], infer_only: bool = False) -> dict[str, Any]:
    assistant = None if infer_only else build_rank_sft_answer(record)
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("02_student_rank_sft").system_prompt,
        user_content=build_rank_user_prompt(record),
        assistant_content=assistant,
    )
    sample["expected_ranking"] = record["ranking_order"]
    return sample


def rank_grpo_samples(record: dict[str, Any]) -> dict[str, Any]:
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("03_student_rank_grpo").system_prompt,
        user_content=build_rank_user_prompt(record),
        assistant_content=None,
    )
    sample["solution_rank"] = record["ranking_order"]
    sample["solution_best_model"] = record["ranking_order"][0]
    return sample


def critique_sft_samples(record: dict[str, Any], model_name: str, infer_only: bool = False) -> dict[str, Any]:
    assistant = None if infer_only else build_critique_sft_answer(record, model_name)
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("04_student_critique_sft").system_prompt,
        user_content=build_critique_user_prompt(record, model_name),
        assistant_content=assistant,
    )
    answer_json = build_critique_answer_json(record, model_name)
    sample["student_model"] = model_name
    sample["expected_abnormal_tags"] = answer_json["abnormal_tags"]
    return sample


def critique_grpo_samples(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("05_student_critique_grpo").system_prompt,
        user_content=build_critique_user_prompt(record, model_name),
        assistant_content=None,
    )
    sample["student_model"] = model_name
    sample["solution_tags"] = record["abnormal_tags_by_model"][model_name]
    return sample


def cot_sft_samples(record: dict[str, Any], infer_only: bool = False) -> dict[str, Any]:
    assistant = None if infer_only else build_cot_sft_answer(record)
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("06_final_label_cot_sft").system_prompt,
        user_content=build_cot_user_prompt(record),
        assistant_content=assistant,
    )
    sample["expected_transform_label"] = record["transform_label"]
    return sample


def cot_grpo_samples(record: dict[str, Any]) -> dict[str, Any]:
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("07_final_label_cot_grpo").system_prompt,
        user_content=build_cot_user_prompt(record),
        assistant_content=None,
    )
    sample["solution_label"] = record["transform_label"]
    return sample


def reason_sft_samples(record: dict[str, Any], infer_only: bool = False) -> dict[str, Any]:
    assistant = None if infer_only else dumps_json(build_reason_answer_json(record))
    sample = make_sample_base(record)
    sample["messages"] = build_messages(
        system_prompt=get_stage_spec("08_ranking_reason_sft").system_prompt,
        user_content=build_reason_user_prompt(record),
        assistant_content=assistant,
    )
    sample["expected_ranking"] = record["ranking_order"]
    return sample


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_jsonl_line(row))
            f.write("\n")


def build_stage_rows(records: list[dict[str, Any]], split_map: dict[str, str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    stage_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        stage_name: {"train": [], "val": [], "test": [], "test_infer": []} for stage_name in stage_names()
    }

    for record in records:
        split = split_map[record["sample_id"]]

        stage_rows["01_final_judge_sft"][split].append(final_judge_samples(record, infer_only=False))

        stage_rows["02_student_rank_sft"][split].append(rank_sft_samples(record, infer_only=False))

        stage_rows["03_student_rank_grpo"][split].append(rank_grpo_samples(record))

        stage_rows["06_final_label_cot_sft"][split].append(cot_sft_samples(record, infer_only=False))

        stage_rows["07_final_label_cot_grpo"][split].append(cot_grpo_samples(record))

        stage_rows["08_ranking_reason_sft"][split].append(reason_sft_samples(record, infer_only=False))

        for model_name in STUDENT_MODELS:
            stage_rows["04_student_critique_sft"][split].append(
                critique_sft_samples(record, model_name=model_name, infer_only=False)
            )
            stage_rows["05_student_critique_grpo"][split].append(
                critique_grpo_samples(record, model_name=model_name)
            )

        if split != "test":
            continue

        stage_rows["01_final_judge_sft"]["test_infer"].append(final_judge_samples(record, infer_only=True))
        stage_rows["02_student_rank_sft"]["test_infer"].append(rank_sft_samples(record, infer_only=True))
        stage_rows["03_student_rank_grpo"]["test_infer"].append(rank_grpo_samples(record))
        stage_rows["06_final_label_cot_sft"]["test_infer"].append(cot_sft_samples(record, infer_only=True))
        stage_rows["07_final_label_cot_grpo"]["test_infer"].append(cot_grpo_samples(record))
        stage_rows["08_ranking_reason_sft"]["test_infer"].append(reason_sft_samples(record, infer_only=True))

        for model_name in STUDENT_MODELS:
            stage_rows["04_student_critique_sft"]["test_infer"].append(
                critique_sft_samples(record, model_name=model_name, infer_only=True)
            )
            stage_rows["05_student_critique_grpo"]["test_infer"].append(
                critique_grpo_samples(record, model_name=model_name)
            )

    return stage_rows


def build_manifest(
    input_dirs: list[Path],
    output_dir: Path,
    record_stats: dict[str, Any],
    split_map: dict[str, str],
    stage_rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    split_counts = {"train": 0, "val": 0, "test": 0}
    for split in split_map.values():
        split_counts[split] += 1

    stages: dict[str, Any] = {}
    for stage_name, rows_by_split in stage_rows.items():
        stage_output_dir = output_dir / "datasets" / stage_name
        files = {}
        counts = {}
        for split_name, rows in rows_by_split.items():
            file_name = f"{split_name}.jsonl"
            files[split_name] = str(stage_output_dir / file_name)
            counts[split_name] = len(rows)
        stages[stage_name] = {
            "spec": get_stage_spec(stage_name).to_dict(),
            "files": files,
            "counts": counts,
        }

    return {
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "record_stats": record_stats,
        "split_counts": split_counts,
        "stages": stages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为公交视频风险数据生成 ms-swift 多阶段训练数据集。")
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        help="原始 labels/rankings 数据目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "artifacts"),
        help="转换后数据和 manifest 输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dirs:
        raise SystemExit("请至少通过 --input-dir 指定一个 labels/rankings 数据目录。")
    input_dirs = [Path(raw_path).resolve() for raw_path in args.input_dirs]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, record_stats = load_valid_records(input_dirs)
    if not records:
        raise SystemExit("没有找到有效的 labels/rankings 配对样本，无法继续。")

    split_map = build_split_map(records)
    stage_rows = build_stage_rows(records, split_map)

    for stage_name, rows_by_split in stage_rows.items():
        stage_dir = output_dir / "datasets" / stage_name
        for split_name, rows in rows_by_split.items():
            write_jsonl(stage_dir / f"{split_name}.jsonl", rows)

    manifest = build_manifest(
        input_dirs=input_dirs,
        output_dir=output_dir,
        record_stats=record_stats,
        split_map=split_map,
        stage_rows=stage_rows,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"有效样本数: {record_stats['valid_records']}")
    print(f"输入目录: {[str(path) for path in input_dirs]}")
    print(f"切分统计: {manifest['split_counts']}")
    print(f"manifest: {manifest_path}")
    for stage_name in stage_names():
        counts = manifest["stages"][stage_name]["counts"]
        print(
            f"{stage_name}: train={counts['train']} val={counts['val']} test={counts['test']} "
            f"test_infer={counts['test_infer']}"
        )


if __name__ == "__main__":
    main()
