#!/usr/bin/env python3
# 将本仓库 K1 起身 checkpoint 中的 actor 导出为 ONNX。
#   python pt2onnx.py --input xxx.pt --output xxx.onnx
# 在仓库根目录运行：
#   uv run python scripts/pt2onnx.py \
#     --input logs/rsl_rl/k1_getup/2026-09-30_12-53-20/model_2000.pt \
#     --output logs/rsl_rl/k1_getup/2026-09-30_12-53-20/k1_getup_mjlab.onnx
# 输入 obs: float32 [1, 72]；输出 action: float32 [1, 22]，为原始动作。
# 部署端执行 q_target = q_current + 0.6 * action，再执行 PD。
# last_action 直接使用上一次原始输出

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn


def export_actor(input_path: Path, output_path: Path) -> None:
  torch.set_num_threads(1)
  checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
  state = checkpoint["actor_state_dict"]
  if any(key.startswith("obs_normalizer.") for key in state):
    raise ValueError("此脚本适用于未启用 actor 观测归一化的 K1 起身 checkpoint")

  actor = nn.Sequential(
    nn.Linear(72, 512),
    nn.ELU(),
    nn.Linear(512, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 22),
  ).eval()
  actor.load_state_dict(
    {
      key.removeprefix("mlp."): value
      for key, value in state.items()
      if key.startswith("mlp.")
    },
    strict=True,
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with torch.no_grad():
    torch.onnx.export(
      actor,
      torch.zeros(1, 72),
      str(output_path),
      input_names=["obs"],
      output_names=["action"],
      opset_version=17,
      dynamo=False,
    )
  onnx.checker.check_model(onnx.load(output_path))
  print(f"已导出：{output_path}")
  print("输入 obs: float32 [1, 72]；输出 action: float32 [1, 22]（原始动作）")
  print("部署端：q_target = q_current + 0.6 * action；last_action = 上一次 action")


def main() -> None:
  parser = argparse.ArgumentParser(description="将 K1 起身 actor 从 PT 导出为 ONNX")
  parser.add_argument("--input", type=Path, required=True, help="训练 checkpoint 路径")
  parser.add_argument("--output", type=Path, required=True, help="输出 ONNX 路径")
  args = parser.parse_args()
  if not args.input.is_file():
    parser.error(f"输入文件不存在：{args.input}")
  if args.input.resolve() == args.output.resolve():
    parser.error("输入和输出不能使用同一个文件")
  export_actor(args.input, args.output)


if __name__ == "__main__":
  main()
