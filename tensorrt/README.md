# M-SVDD -> ONNX -> TensorRT

This folder contains standalone scripts for exporting the `GaussianSVDDModel` from the M-SVDD repo to ONNX, building a TensorRT engine, and running TensorRT inference.

## Expected checkpoint
Use the training checkpoint saved by the original repo, i.e. the file created by:

- `torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))`

That checkpoint should contain:
- `model_state_dict`
- `mu`
- `sigma_inv`
- `radius`

## 1) Export to ONNX

```bash
cd m_svdd_trt
python export_onnx.py \
  --checkpoint /path/to/model_49 \
  --onnx weights/m_svdd.onnx \
  --feature-dim 32 \
  --dynamic-batch
```

## 2) Build TensorRT engine
Using `trtexec`:

```bash
python build_trt_engine.py \
  --onnx weights/m_svdd.onnx \
  --engine weights/m_svdd_fp16.engine \
  --min-batch 1 \
  --opt-batch 1 \
  --max-batch 8 \
  --fp16
```

Equivalent direct command:

```bash
trtexec \
  --onnx=weights/m_svdd.onnx \
  --saveEngine=weights/m_svdd_fp16.engine \
  --memPoolSize=workspace:4096 \
  --minShapes=x_audio:1x2x4410,x_imu:1x400 \
  --optShapes=x_audio:1x2x4410,x_imu:1x400 \
  --maxShapes=x_audio:8x2x4410,x_imu:8x400 \
  --fp16
```

## 3) TensorRT inference
With random test input:

```bash
python infer_trt.py \
  --engine weights/m_svdd_fp16.engine \
  --batch-size 1 \
  --save-dir outputs
```

With real `.npy` input:

```bash
python infer_trt.py \
  --engine weights/m_svdd_fp16.engine \
  --audio-npy /path/to/audio.npy \
  --imu-npy /path/to/imu.npy \
  --save-dir outputs
```

## Input / output interface
Inputs:
- `x_audio`: `[B, 2, 4410]`, `float32`
- `x_imu`: `[B, 400]`, `float32`

Outputs:
- `distance`: `[B]`
- `radius`: `[B]`
- `anomaly_score`: `[B] = distance / radius`
- `audio_recon`: `[B, 2, 4410]`
- `imu_recon`: `[B, 400]`
- `embedding`: `[B, feature_dim]`

## Notes
- The original repo updates `mu` and `sigma_inv` during training via MCD. For ONNX/TensorRT deployment, they must be treated as fixed checkpoint parameters, which this export path does.
- The original repo has an `eca` module defined but not used in `forward`, so it is omitted from deployment behavior.
- `feature_dim` in the repo config is `32` by default.
python infer_trt.py \
  --engine /home/kemove/yyz/M-SVDD/tensorrt/weights/m_svdd.onnx \
  --batch-size 1 \
  --warmup 50 \
  --runs 200

./trtexec --onnx=/home/kemove/yyz/M-SVDD/tensorrt/weights/m_svdd.onnx --saveEngine=home/kemove/yyz/M-SVDD/tensorrt/weights/m_svdd.engine --fp16 