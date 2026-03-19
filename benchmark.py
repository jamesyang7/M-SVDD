# benchmark_msvdd.py
# 放在 M-SVDD 仓库根目录运行
# 示例：
#   python benchmark_msvdd.py --device cuda
#   python benchmark_msvdd.py --device cuda --batch_size 1 --runs 200
#   python benchmark_msvdd.py --device cpu --num_threads 1

import time
import argparse
import warnings
import numpy as np
import torch

from nets.gaussianNet import GaussianSVDDModel

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser("Benchmark M-SVDD: Params / FLOPs / Inference Speed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--audio_channels", type=int, default=2)
    parser.add_argument("--audio_len", type=int, default=4410)
    parser.add_argument("--imu_len", type=int, default=400)
    parser.add_argument("--output_dim", type=int, default=32,
                        help="repo config/config.json 默认 feature_dim=32")
    parser.add_argument("--feature_dim", type=int, default=32,
                        help="音频/IMU encoder 输出维度，默认与 output_dim 一致")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true",
                        help="仅 CUDA 下有效")
    return parser.parse_args()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_profile_flops(model, audio, imu):
    macs, flops, note = None, None, ""
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(audio, imu, 0), verbose=False)
        flops = macs * 2
        note = "FLOPs 由 THOP 估计；LSTM 与 MultiheadAttention 可能存在低估。"
    except Exception as e:
        note = f"THOP 统计失败: {repr(e)}"
    return macs, flops, note


@torch.no_grad()
def benchmark_speed(model, audio, imu, device, warmup=30, runs=100, use_fp16=False):
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # 首次前向
    if device.type == "cuda":
        with torch.cuda.amp.autocast(enabled=use_fp16):
            out = model(audio, imu, 0)
        torch.cuda.synchronize()
    else:
        out = model(audio, imu, 0)

    # warmup
    if device.type == "cuda":
        for _ in range(warmup):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                _ = model(audio, imu, 0)
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            _ = model(audio, imu, 0)

    times = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(runs):
            starter.record()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                _ = model(audio, imu, 0)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(audio, imu, 0)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    latency_mean = float(times.mean())
    latency_std = float(times.std())
    fps = 1000.0 / latency_mean if latency_mean > 0 else 0.0
    throughput = fps * audio.shape[0]

    if isinstance(out, (tuple, list)):
        output_shapes = [tuple(x.shape) if hasattr(x, "shape") else str(type(x)) for x in out]
    else:
        output_shapes = str(type(out))

    return {
        "latency_mean_ms": latency_mean,
        "latency_std_ms": latency_std,
        "fps": fps,
        "throughput_samples_per_sec": throughput,
        "output_shapes": output_shapes,
    }


def format_number(num):
    if num is None:
        return "N/A"
    num = float(num)
    if num >= 1e12:
        return f"{num / 1e12:.4f} T"
    if num >= 1e9:
        return f"{num / 1e9:.4f} G"
    if num >= 1e6:
        return f"{num / 1e6:.4f} M"
    if num >= 1e3:
        return f"{num / 1e3:.4f} K"
    return f"{num:.4f}"


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA 不可用，自动切换到 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    # 关键：is_train=0，跳过 MCD 更新，做纯前向 benchmark
    model = GaussianSVDDModel(
        output_dim=args.output_dim,
        feature_dim=args.feature_dim,
        is_train=0
    ).to(device).eval()

    audio = torch.randn(
        args.batch_size, args.audio_channels, args.audio_len, device=device
    )
    imu = torch.randn(
        args.batch_size, args.imu_len, device=device
    )

    total_params, trainable_params = count_parameters(model)
    macs, flops, flops_note = try_profile_flops(model, audio, imu)
    speed = benchmark_speed(
        model=model,
        audio=audio,
        imu=imu,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
        use_fp16=args.use_fp16 and device.type == "cuda",
    )

    print("=" * 80)
    print("M-SVDD Benchmark Result")
    print("=" * 80)
    print(f"Device                  : {device}")
    print(f"Audio input             : [{args.batch_size}, {args.audio_channels}, {args.audio_len}]")
    print(f"IMU input               : [{args.batch_size}, {args.imu_len}]")
    print(f"Output dim              : {args.output_dim}")
    print(f"Feature dim             : {args.feature_dim}")
    print(f"Mode                    : is_train=0 (pure inference benchmark)")
    print("-" * 80)
    print(f"Total params            : {total_params:,} ({format_number(total_params)})")
    print(f"Trainable params        : {trainable_params:,} ({format_number(trainable_params)})")
    print(f"MACs (estimated)        : {format_number(macs)}")
    print(f"FLOPs (estimated)       : {format_number(flops)}")
    print("-" * 80)
    print(f"Latency mean            : {speed['latency_mean_ms']:.4f} ms")
    print(f"Latency std             : {speed['latency_std_ms']:.4f} ms")
    print(f"FPS                     : {speed['fps']:.4f}")
    print(f"Throughput              : {speed['throughput_samples_per_sec']:.4f} samples/s")
    print(f"Output shapes           : {speed['output_shapes']}")
    print("-" * 80)
    print(f"FLOPs note              : {flops_note}")
    if device.type == "cuda":
        print(f"Max memory allocated    : {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")
        print(f"Max memory reserved     : {torch.cuda.max_memory_reserved(device)/1024**2:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()