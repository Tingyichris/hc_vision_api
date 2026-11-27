# TensorRT model deployed on 70

The Go API on port 7000 continues to accept the existing Raspberry Pi model path
`/-v4qph/-j7pjz-4-yolo26n-seg-t1`. The C++ worker uses the direct 640×640 resize
from commit `4364409` before TensorRT inference.

## Model artifacts

| Artifact | SHA-256 |
| --- | --- |
| `weights (1).pt` | `CF9FAF40AE778D249B15549C0C16F762F38A8DB267B6A44824CEEEA957556659` |
| `weights (1).onnx` | `DA52679AABFD44193192CD32035FA1A33539F03D8BC79DCE3A324899DB6B8F68` |

Exported with Ultralytics 8.4.219 using ONNX opset 17, batch 1, static
640×640 input, no graph simplification, and `nms=False, end2end=False`.
Expected float32 outputs are `[1,37,8400]` and `[1,32,160,160]`.
The ONNX and engine files are deployment artifacts stored outside Git.

## Host configuration

The ONNX file is uploaded as `/home/hc/weights-1.onnx`. Build with the
host's TensorRT 10.16.2 tools:

```sh
trtexec --onnx=/home/hc/weights-1.onnx \
  --saveEngine=/home/hc/weights-1-fp32.engine \
  --noTF32 --builderOptimizationLevel=0 --skipInference
```

The FP16 build at optimization level 0 failed with a TensorRT internal
assertion. The active engine uses FP32 on the GPU.

Build the worker with `cmake --build native/build -j4` and stage the
result before replacing `bin/tensorrt_worker`. The Go gateway accesses
`/home/hc/weights_fp16.engine`, which now points to
`/home/hc/weights-1-fp32.engine`. Restart `hc-vision-go.service`
after staging both artifacts. Retain the previous engine and worker
before switching so both can be restored together.

On 2026-11-27, the production API accepted the local 640×640 ROI fixture
and returned seven boxes with segmentation points. Five boxes matched
visible plates, with confidence 0.801–0.963. Two boxes remained possible
false positives. The three Raspberry Pi configurations already used
the same API URL and model path. Live camera results were not verified
as part of this fixture test.
