import argparse
from pathlib import Path

import torch

from model_def import GaussianSVDDModel, ONNXExportWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to repo checkpoint, e.g. model_49")
    p.add_argument("--onnx", type=str, default="m_svdd.onnx")
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic-batch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cpu"

    model = GaussianSVDDModel(output_dim=args.feature_dim, feature_dim=args.feature_dim)
    # model.load_checkpoint(args.checkpoint, map_location=device)
    wrapper = ONNXExportWrapper(model).eval()

    x_audio = torch.randn(args.batch_size, 2, 4410, dtype=torch.float32)
    x_imu = torch.randn(args.batch_size, 400, dtype=torch.float32)

    output_path = Path(args.onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["x_audio", "x_imu"]
    output_names = [
        "distance",
        "radius",
        "anomaly_score",
        "audio_recon",
        "imu_recon",
        "embedding",
    ]

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "x_audio": {0: "batch"},
            "x_imu": {0: "batch"},
            "distance": {0: "batch"},
            "radius": {0: "batch"},
            "anomaly_score": {0: "batch"},
            "audio_recon": {0: "batch"},
            "imu_recon": {0: "batch"},
            "embedding": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x_audio, x_imu),
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print(f"Exported ONNX to: {output_path}")
    print("Inputs:")
    print("  x_audio: [B, 2, 4410]")
    print("  x_imu:   [B, 400]")
    print("Outputs:")
    print("  distance:      [B]")
    print("  radius:        [B]")
    print("  anomaly_score: [B]")
    print("  audio_recon:   [B, 2, 4410]")
    print("  imu_recon:     [B, 400]")
    print("  embedding:     [B, feature_dim]")


if __name__ == "__main__":
    main()
