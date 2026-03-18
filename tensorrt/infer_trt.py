import os
import time
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __repr__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}"


def volume(shape):
    v = 1
    for s in shape:
        v *= int(s)
    return v


def trt_dtype_to_np(dtype):
    return np.dtype(trt.nptype(dtype))


def allocate_buffers(engine, context, batch_size):
    """
    Allocate host/device buffers for all IO tensors.
    Supports TensorRT 8.5+ tensor API.
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    num_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]

    for name in tensor_names:
        mode = engine.get_tensor_mode(name)
        dtype = trt_dtype_to_np(engine.get_tensor_dtype(name))

        shape = tuple(context.get_tensor_shape(name))
        if -1 in shape:
            raise RuntimeError(f"Dynamic shape for tensor {name} not fully specified: {shape}")

        size = volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        pair = HostDeviceMem(host_mem, device_mem)

        if mode == trt.TensorIOMode.INPUT:
            inputs.append((name, pair, shape, dtype))
        else:
            outputs.append((name, pair, shape, dtype))

    return inputs, outputs, bindings, stream


def do_inference_v3(context, inputs, outputs, stream):
    """
    Execute one inference with async memcpy.
    """
    # HtoD
    for name, mem, shape, dtype in inputs:
        cuda.memcpy_htod_async(mem.device, mem.host, stream)

    # Execute
    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT inference failed.")

    # DtoH
    for name, mem, shape, dtype in outputs:
        cuda.memcpy_dtoh_async(mem.host, mem.device, stream)

    stream.synchronize()


def get_input_specs(engine):
    specs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = tuple(engine.get_tensor_shape(name))
            dtype = trt_dtype_to_np(engine.get_tensor_dtype(name))
            specs.append((name, shape, dtype))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True, help="Path to TensorRT engine")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--audio-shape", type=int, nargs=2, default=[2, 4410],
                        help="Audio input shape without batch, default: 2 4410")
    parser.add_argument("--imu-shape", type=int, nargs=1, default=[400],
                        help="IMU input shape without batch, default: 400")
    args = parser.parse_args()

    if not os.path.exists(args.engine):
        raise FileNotFoundError(f"Engine not found: {args.engine}")

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)

    with open(args.engine, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context.")

    # -----------------------------
    # Set dynamic input shapes
    # -----------------------------
    input_specs = get_input_specs(engine)
    if len(input_specs) != 2:
        print("Warning: expected 2 inputs, but found:")
        for name, shape, dtype in input_specs:
            print(f"  {name}: shape={shape}, dtype={dtype}")

    # By convention:
    # audio -> (B, 2, 4410)
    # imu   -> (B, 400)
    # If engine input names differ, we assign based on dimensionality.
    for name, shape, dtype in input_specs:
        if len(shape) == 3:
            target_shape = (args.batch_size, args.audio_shape[0], args.audio_shape[1])
        elif len(shape) == 2:
            target_shape = (args.batch_size, args.imu_shape[0])
        else:
            raise RuntimeError(f"Unsupported input rank for {name}: {shape}")

        ok = context.set_input_shape(name, target_shape)
        if not ok:
            raise RuntimeError(f"Failed to set input shape for {name} to {target_shape}")

    # Set tensor addresses after buffers are allocated
    inputs, outputs, bindings, stream = allocate_buffers(engine, context, args.batch_size)

    # Bind tensor addresses
    bind_idx = 0
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        context.set_tensor_address(name, bindings[bind_idx])
        bind_idx += 1

    # -----------------------------
    # Prepare random input
    # -----------------------------
    for name, mem, shape, dtype in inputs:
        if len(shape) == 3:
            data = np.random.randn(*shape).astype(dtype)
        elif len(shape) == 2:
            data = np.random.randn(*shape).astype(dtype)
        else:
            raise RuntimeError(f"Unsupported input shape: {shape}")
        np.copyto(mem.host, data.ravel())

    # -----------------------------
    # Warmup
    # -----------------------------
    for _ in range(args.warmup):
        do_inference_v3(context, inputs, outputs, stream)

    # -----------------------------
    # Benchmark
    # -----------------------------
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        do_inference_v3(context, inputs, outputs, stream)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # ms

    times = np.array(times, dtype=np.float64)
    mean_ms = times.mean()
    std_ms = times.std()
    min_ms = times.min()
    max_ms = times.max()

    fps = args.batch_size / (mean_ms / 1000.0)

    print("=" * 60)
    print(f"Engine: {args.engine}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup: {args.warmup}")
    print(f"Runs: {args.runs}")
    print("-" * 60)
    print("Inputs:")
    for name, mem, shape, dtype in inputs:
        print(f"  {name}: shape={shape}, dtype={dtype}")
    print("Outputs:")
    for name, mem, shape, dtype in outputs:
        print(f"  {name}: shape={shape}, dtype={dtype}")
    print("-" * 60)
    print(f"Mean latency: {mean_ms:.3f} ms")
    print(f"Std latency : {std_ms:.3f} ms")
    print(f"Min latency : {min_ms:.3f} ms")
    print(f"Max latency : {max_ms:.3f} ms")
    print(f"Throughput  : {fps:.2f} samples/s")
    print("=" * 60)


if __name__ == "__main__":
    main()