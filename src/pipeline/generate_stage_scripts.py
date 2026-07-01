from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_lib import get_stage_spec, stage_names

DEFAULT_BASE_MODEL = "$PROJECT_ROOT/../models/qwen3_5_0.8B"
DEFAULT_OUTPUT_ROOT = '$PROJECT_ROOT/../saves/qwen3_5_0.8B/multi_stage_swift'
DEFAULT_TORCH_DTYPE = "float16"


SCRIPT_PREAMBLE = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
SWIFT_ROOT="${SWIFT_ROOT:-$PROJECT_ROOT/../ms-swift}"
OUTPUT_ROOT="${OUTPUT_ROOT:-""" + DEFAULT_OUTPUT_ROOT + """}"
DEFAULT_BASE_MODEL="${DEFAULT_BASE_MODEL:-""" + DEFAULT_BASE_MODEL + """}"

if [ ! -d "$SWIFT_ROOT" ]; then
  echo "未找到 ms-swift 目录: $SWIFT_ROOT"
  exit 1
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
elif command -v python >/dev/null 2>&1; then
  _py_lib="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null)/lib"
  if [ -d "$_py_lib" ]; then
    export LD_LIBRARY_PATH="${_py_lib}:${LD_LIBRARY_PATH:-}"
  fi
fi

export PYTHONPATH="$PROJECT_ROOT:$SWIFT_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_ROOT"

run_swift() {
  if command -v swift >/dev/null 2>&1; then
    swift "$@"
  else
    python "$SWIFT_ROOT/swift/cli/main.py" "$@"
  fi
}

latest_checkpoint() {
  local stage_dir="$1"
  find "$stage_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}
"""


BUILD_DATA_ARGS_SNIPPET = """
BUILD_DATA_ARGS=(--output-dir "$ARTIFACT_ROOT")
if [ -n "${INPUT_DIR:-}" ]; then
  BUILD_DATA_ARGS+=(--input-dir "$INPUT_DIR")
fi
if [ -n "${INPUT_DIRS:-}" ]; then
  read -r -a INPUT_DIR_ARRAY <<< "$INPUT_DIRS"
  for input_dir in "${INPUT_DIR_ARRAY[@]}"; do
    BUILD_DATA_ARGS+=(--input-dir "$input_dir")
  done
fi
"""


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_sft_train_script(stage_name: str, stage_meta: dict[str, Any]) -> str:
    spec = get_stage_spec(stage_name)
    train_data = stage_meta["files"]["train"]
    val_data = stage_meta["files"]["val"]
    output_dir = f'$OUTPUT_ROOT/{stage_name}'
    return (
        SCRIPT_PREAMBLE
        + f"""

STAGE="{stage_name}"
TRAIN_DATA="{train_data}"
VAL_DATA="{val_data}"
OUT_DIR="{output_dir}"

BASE_MODEL="${{BASE_MODEL:-$DEFAULT_BASE_MODEL}}"
TRAIN_TYPE="${{TRAIN_TYPE:-lora}}"
TORCH_DTYPE="${{TORCH_DTYPE:-{DEFAULT_TORCH_DTYPE}}}"
MAX_LENGTH="${{MAX_LENGTH:-{spec.default_max_length}}}"
LEARNING_RATE="${{LEARNING_RATE:-{spec.default_learning_rate}}}"
NUM_TRAIN_EPOCHS="${{NUM_TRAIN_EPOCHS:-{spec.default_num_train_epochs}}}"
PER_DEVICE_TRAIN_BATCH_SIZE="${{PER_DEVICE_TRAIN_BATCH_SIZE:-1}}"
PER_DEVICE_EVAL_BATCH_SIZE="${{PER_DEVICE_EVAL_BATCH_SIZE:-1}}"
GRADIENT_ACCUMULATION_STEPS="${{GRADIENT_ACCUMULATION_STEPS:-8}}"
SAVE_STEPS="${{SAVE_STEPS:-20}}"
EVAL_STEPS="${{EVAL_STEPS:-20}}"
LOGGING_STEPS="${{LOGGING_STEPS:-5}}"
DATALOADER_NUM_WORKERS="${{DATALOADER_NUM_WORKERS:-4}}"
DATASET_NUM_PROC="${{DATASET_NUM_PROC:-4}}"
WARMUP_RATIO="${{WARMUP_RATIO:-0.05}}"
LORA_RANK="${{LORA_RANK:-8}}"
LORA_ALPHA="${{LORA_ALPHA:-32}}"
VIDEO_MAX_PIXELS="${{VIDEO_MAX_PIXELS:-50176}}"
FPS_MAX_FRAMES="${{FPS_MAX_FRAMES:-12}}"

if [ ! -d "$BASE_MODEL" ]; then
  echo "未找到本地底模目录: $BASE_MODEL"
  exit 1
fi

export VIDEO_MAX_PIXELS="$VIDEO_MAX_PIXELS"
export FPS_MAX_FRAMES="$FPS_MAX_FRAMES"

mkdir -p "$OUT_DIR"

run_swift sft \\
  --model "$BASE_MODEL" \\
  --train_type "$TRAIN_TYPE" \\
  --dataset "$TRAIN_DATA" \\
  --val_dataset "$VAL_DATA" \\
  --split_dataset_ratio 0 \\
  --torch_dtype "$TORCH_DTYPE" \\
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \\
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \\
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \\
  --learning_rate "$LEARNING_RATE" \\
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \\
  --eval_steps "$EVAL_STEPS" \\
  --save_steps "$SAVE_STEPS" \\
  --save_total_limit 2 \\
  --logging_steps "$LOGGING_STEPS" \\
  --max_length "$MAX_LENGTH" \\
  --output_dir "$OUT_DIR" \\
  --warmup_ratio "$WARMUP_RATIO" \\
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \\
  --dataset_num_proc "$DATASET_NUM_PROC" \\
  --save_only_model true \\
  --target_modules all-linear \\
  --lora_rank "$LORA_RANK" \\
  --lora_alpha "$LORA_ALPHA" \\
  --freeze_vit true
"""
    ).strip() + "\n"


def build_grpo_train_script(stage_name: str, stage_meta: dict[str, Any]) -> str:
    spec = get_stage_spec(stage_name)
    assert spec.reward_name is not None
    assert spec.warm_start_from is not None
    train_data = stage_meta["files"]["train"]
    val_data = stage_meta["files"]["val"]
    output_dir = f'$OUTPUT_ROOT/{stage_name}'
    warm_dir = f'$OUTPUT_ROOT/{spec.warm_start_from}'
    return (
        SCRIPT_PREAMBLE
        + f"""

STAGE="{stage_name}"
TRAIN_DATA="{train_data}"
VAL_DATA="{val_data}"
OUT_DIR="{output_dir}"
WARM_STAGE="{spec.warm_start_from}"
WARM_DIR="{warm_dir}"

BASE_MODEL="${{BASE_MODEL:-$DEFAULT_BASE_MODEL}}"
TRAIN_TYPE="${{TRAIN_TYPE:-lora}}"
TORCH_DTYPE="${{TORCH_DTYPE:-{DEFAULT_TORCH_DTYPE}}}"
MAX_LENGTH="${{MAX_LENGTH:-{spec.default_max_length}}}"
MAX_COMPLETION_LENGTH="${{MAX_COMPLETION_LENGTH:-{spec.default_max_completion_length}}}"
LEARNING_RATE="${{LEARNING_RATE:-{spec.default_learning_rate}}}"
NUM_TRAIN_EPOCHS="${{NUM_TRAIN_EPOCHS:-{spec.default_num_train_epochs}}}"
NUM_GENERATIONS="${{NUM_GENERATIONS:-{spec.default_num_generations}}}"
PER_DEVICE_TRAIN_BATCH_SIZE="${{PER_DEVICE_TRAIN_BATCH_SIZE:-1}}"
PER_DEVICE_EVAL_BATCH_SIZE="${{PER_DEVICE_EVAL_BATCH_SIZE:-1}}"
GRADIENT_ACCUMULATION_STEPS="${{GRADIENT_ACCUMULATION_STEPS:-1}}"
SAVE_STEPS="${{SAVE_STEPS:-20}}"
EVAL_STEPS="${{EVAL_STEPS:-20}}"
LOGGING_STEPS="${{LOGGING_STEPS:-5}}"
DATALOADER_NUM_WORKERS="${{DATALOADER_NUM_WORKERS:-4}}"
WARMUP_RATIO="${{WARMUP_RATIO:-0.05}}"
LORA_RANK="${{LORA_RANK:-8}}"
LORA_ALPHA="${{LORA_ALPHA:-32}}"
TEMPERATURE="${{TEMPERATURE:-1.0}}"
TOP_P="${{TOP_P:-0.9}}"
BETA="${{BETA:-0.04}}"
VIDEO_MAX_PIXELS="${{VIDEO_MAX_PIXELS:-50176}}"
FPS_MAX_FRAMES="${{FPS_MAX_FRAMES:-12}}"
PLUGIN_PATH="$PROJECT_ROOT/reward_plugin.py"
SFT_CKPT="${{SFT_CKPT:-$(latest_checkpoint "$WARM_DIR")}}"
USE_VLLM="${{USE_VLLM:-false}}"

if [ ! -d "$BASE_MODEL" ]; then
  echo "未找到本地底模目录: $BASE_MODEL"
  exit 1
fi

if [ -z "$SFT_CKPT" ]; then
  echo "未找到预热 SFT checkpoint，请先运行 $WARM_STAGE 对应的 train 脚本。"
  exit 1
fi

export VIDEO_MAX_PIXELS="$VIDEO_MAX_PIXELS"
export FPS_MAX_FRAMES="$FPS_MAX_FRAMES"

mkdir -p "$OUT_DIR"

extra_args=()
if [ "$USE_VLLM" = "true" ]; then
  extra_args+=(
    --use_vllm true
    --vllm_mode "${{VLLM_MODE:-colocate}}"
    --vllm_gpu_memory_utilization "${{VLLM_GPU_MEMORY_UTILIZATION:-0.5}}"
    --vllm_tensor_parallel_size "${{VLLM_TENSOR_PARALLEL_SIZE:-1}}"
  )
fi

run_swift rlhf \\
  --rlhf_type grpo \\
  --model "$BASE_MODEL" \\
  --train_type "$TRAIN_TYPE" \\
  --dataset "$TRAIN_DATA" \\
  --val_dataset "$VAL_DATA" \\
  --split_dataset_ratio 0 \\
  --torch_dtype "$TORCH_DTYPE" \\
  --adapters "$SFT_CKPT" \\
  --ref_adapters "$SFT_CKPT" \\
  --external_plugins "$PLUGIN_PATH" \\
  --reward_funcs {spec.reward_name} format \\
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \\
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \\
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \\
  --learning_rate "$LEARNING_RATE" \\
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \\
  --save_total_limit 2 \\
  --logging_steps "$LOGGING_STEPS" \\
  --save_steps "$SAVE_STEPS" \\
  --eval_steps "$EVAL_STEPS" \\
  --output_dir "$OUT_DIR" \\
  --warmup_ratio "$WARMUP_RATIO" \\
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \\
  --max_length "$MAX_LENGTH" \\
  --max_completion_length "$MAX_COMPLETION_LENGTH" \\
  --num_generations "$NUM_GENERATIONS" \\
  --target_modules all-linear \\
  --lora_rank "$LORA_RANK" \\
  --lora_alpha "$LORA_ALPHA" \\
  --freeze_vit true \\
  --temperature "$TEMPERATURE" \\
  --top_p "$TOP_P" \\
  --beta "$BETA" \\
  "${{extra_args[@]}}"
"""
    ).strip() + "\n"


def build_export_script(stage_name: str) -> str:
    return (
        SCRIPT_PREAMBLE
        + f"""

STAGE="{stage_name}"
STAGE_DIR="$OUTPUT_ROOT/$STAGE"
CKPT="${{CKPT:-$(latest_checkpoint "$STAGE_DIR")}}"
MERGED_DIR="${{MERGED_DIR:-$STAGE_DIR/merged}}"

if [ -z "$CKPT" ]; then
  echo "未找到 checkpoint，请先完成训练。"
  exit 1
fi

run_swift export \\
  --adapters "$CKPT" \\
  --merge_lora true \\
  --output_dir "$MERGED_DIR"
"""
    ).strip() + "\n"


def build_infer_script(stage_name: str, stage_meta: dict[str, Any]) -> str:
    infer_data = stage_meta["files"]["test_infer"]
    spec = get_stage_spec(stage_name)
    max_new_tokens = spec.default_max_completion_length or 1024
    return (
        SCRIPT_PREAMBLE
        + f"""

STAGE="{stage_name}"
INFER_DATA="{infer_data}"
STAGE_DIR="$OUTPUT_ROOT/$STAGE"
MERGED_DIR="${{MERGED_DIR:-$STAGE_DIR/merged}}"
CKPT="${{CKPT:-$(latest_checkpoint "$STAGE_DIR")}}"
RESULT_PATH="${{RESULT_PATH:-$STAGE_DIR/test_predictions.jsonl}}"
MAX_NEW_TOKENS="${{MAX_NEW_TOKENS:-{max_new_tokens}}}"
VIDEO_MAX_PIXELS="${{VIDEO_MAX_PIXELS:-50176}}"
FPS_MAX_FRAMES="${{FPS_MAX_FRAMES:-12}}"
MAX_PIXELS="${{MAX_PIXELS:-1003520}}"

mkdir -p "$STAGE_DIR"
export VIDEO_MAX_PIXELS="$VIDEO_MAX_PIXELS"
export FPS_MAX_FRAMES="$FPS_MAX_FRAMES"
export MAX_PIXELS="$MAX_PIXELS"

if [ -d "$MERGED_DIR" ]; then
  run_swift infer \\
    --model "$MERGED_DIR" \\
    --val_dataset "$INFER_DATA" \\
    --result_path "$RESULT_PATH" \\
    --max_new_tokens "$MAX_NEW_TOKENS" \\
    --load_args false
elif [ -n "$CKPT" ]; then
  run_swift infer \\
    --adapters "$CKPT" \\
    --val_dataset "$INFER_DATA" \\
    --result_path "$RESULT_PATH" \\
    --max_new_tokens "$MAX_NEW_TOKENS" \\
    --load_data_args true
else
  echo "既没找到 merged 模型，也没找到 checkpoint。"
  exit 1
fi
"""
    ).strip() + "\n"


def build_prepare_script(manifest_path: Path, project_root: Path) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${{ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}}"
{BUILD_DATA_ARGS_SNIPPET.strip()}
python "$PROJECT_ROOT/build_datasets.py" "${{BUILD_DATA_ARGS[@]}}"
python "$PROJECT_ROOT/generate_stage_scripts.py" --manifest "$ARTIFACT_ROOT/manifest.json" --output-dir "$SCRIPT_DIR"
echo "准备完成: $ARTIFACT_ROOT"
"""


def build_run_all_script() -> str:
    ordered_stages = stage_names()
    stage_lines = "\n".join(f'  "{stage}"' for stage in ordered_stages)
    return (
        SCRIPT_PREAMBLE
        + f"""

REBUILD_DATA="${{REBUILD_DATA:-0}}"
RUN_EXPORT="${{RUN_EXPORT:-1}}"
RUN_INFER="${{RUN_INFER:-1}}"
FROM_STAGE="${{FROM_STAGE:-}}"
TO_STAGE="${{TO_STAGE:-}}"
STAGES=(
{stage_lines}
)

if [ "$REBUILD_DATA" = "1" ] || [ ! -f "$ARTIFACT_ROOT/manifest.json" ]; then
  bash "$SCRIPT_DIR/prepare_all.sh"
fi

started=1
if [ -n "$FROM_STAGE" ]; then
  started=0
fi

for stage in "${{STAGES[@]}}"; do
  if [ "$started" = "0" ]; then
    if [ "$stage" = "$FROM_STAGE" ]; then
      started=1
    else
      continue
    fi
  fi

  echo ""
  echo "============================================================"
  echo "阶段: $stage"
  echo "底模: ${{BASE_MODEL:-$DEFAULT_BASE_MODEL}}"
  echo "输出: $OUTPUT_ROOT/$stage"
  echo "============================================================"

  bash "$SCRIPT_DIR/train_${{stage}}.sh"

  if [ "$RUN_EXPORT" = "1" ]; then
    bash "$SCRIPT_DIR/export_${{stage}}.sh"
  fi

  if [ "$RUN_INFER" = "1" ]; then
    bash "$SCRIPT_DIR/infer_${{stage}}.sh"
  fi

  if [ -n "$TO_STAGE" ] && [ "$stage" = "$TO_STAGE" ]; then
    break
  fi
done

echo ""
echo "所有阶段执行完成。"
"""
    ).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 manifest 生成多阶段 ms-swift 训练/导出/推理脚本。")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "artifacts" / "manifest.json"),
        help="build_datasets.py 生成的 manifest.json 路径。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "generated_scripts"),
        help="脚本输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    project_root = Path(__file__).resolve().parent

    for stage_name in stage_names():
        stage_meta = manifest["stages"][stage_name]
        train_path = output_dir / f"train_{stage_name}.sh"
        export_path = output_dir / f"export_{stage_name}.sh"
        infer_path = output_dir / f"infer_{stage_name}.sh"

        if get_stage_spec(stage_name).train_mode == "sft":
            train_path.write_text(build_sft_train_script(stage_name, stage_meta), encoding="utf-8")
        else:
            train_path.write_text(build_grpo_train_script(stage_name, stage_meta), encoding="utf-8")

        export_path.write_text(build_export_script(stage_name), encoding="utf-8")
        infer_path.write_text(build_infer_script(stage_name, stage_meta), encoding="utf-8")

    prepare_path = output_dir / "prepare_all.sh"
    prepare_path.write_text(build_prepare_script(manifest_path, project_root), encoding="utf-8")
    run_all_path = output_dir / "run_all_stages.sh"
    run_all_path.write_text(build_run_all_script(), encoding="utf-8")

    for script_path in output_dir.glob("*.sh"):
        script_path.chmod(0o755)

    print(f"脚本已生成到: {output_dir}")


if __name__ == "__main__":
    main()
