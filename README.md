# YOLO11 Nuclei Instance Segmentation with MKIR, DDT, and BGFC

This repository provides the network architecture implementation of the proposed YOLO11-seg model, including MKIR, DDT, and BGFC. Datasets, training and evaluation pipelines, and pretrained weights are not included.

MKIR replaces the selected backbone blocks with parallel 1x1, 3x3, and 5x5 depthwise convolutions, channel shuffle, and Coordinate Attention. DDT refines the two top-down neck features with wavelet cross-attention and a prototype-guided path, followed by an FFN, LayerNorm, and SaE. The original implementation name `D2T` is retained; `DDT` is an alias.

BGFC calibrates only the high-resolution P3 feature sent to the segmentation head. Its fixed Sobel gradients and spatial context produce a spatial gate, with a learnable residual scalar. The original P3 feature continues into the bottom-up PAN path. The native Ultralytics segmentation head, mask prototypes, and loss implementation are retained.

## Dependencies

Use Python 3.10 or newer and PyTorch 2.x. Install the architecture dependencies in a suitable environment:

```bash
pip install -r requirements.txt
```

The repository uses the standard layers from `ultralytics==8.4.60`; it does not require the private experiment project or a patched installation. Install the appropriate PyTorch build for the target CPU or CUDA device. The retained MKIR initializer uses timm helpers when available and its original fallback otherwise.

## Build and run

From the repository root:

```python
import torch
from models import build_model

model = build_model().eval()
x = torch.randn(1, 3, 640, 640)
with torch.inference_mode():
    outputs = model(x)
```

`outputs` follows the native Ultralytics segmentation format, which can contain nested tensors and dictionaries. It is not assumed to be a single tensor. `build_model("configs/proposed.yaml")` explicitly selects the complete network. No checkpoint is downloaded.

```bash
python examples/build_network.py --device cpu
python -m unittest discover -s tests -v
python examples/profile_architecture.py
```

`configs/baseline.yaml` and `configs/mkir_ddt.yaml` are architecture references for parameter and routing checks. They contain no training settings. Optional `--counted-flops` in the profiling example reports only the operations supported by PyTorch's FLOP counter; it is not a complete arithmetic count.


