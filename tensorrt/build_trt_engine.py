import argparse
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--engine", type=str, default="m_svdd_fp16.engine")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--workspace", type=int, default=4096, help="Workspace in MiB")
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace}",
        f"--minShapes=x_audio:{args.min_batch}x2x4410,x_imu:{args.min_batch}x400",
        f"--optShapes=x_audio:{args.opt_batch}x2x4410,x_imu:{args.opt_batch}x400",
        f"--maxShapes=x_audio:{args.max_batch}x2x4410,x_imu:{args.max_batch}x400",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.int8:
        cmd.append("--int8")

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved TensorRT engine to: {engine_path}")


if __name__ == "__main__":
    main()
