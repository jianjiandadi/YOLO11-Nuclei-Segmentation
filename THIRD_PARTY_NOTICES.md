# Third-party notices

## Ultralytics

This architecture uses Ultralytics YOLO11 standard modules through the `ultralytics==8.4.60` dependency. The model configuration, custom-module source extracted from the existing Ultralytics project fork, and relevant model-construction conventions carry the Ultralytics AGPL-3.0 notice. The complete GNU Affero General Public License version 3 text supplied with that source is retained in `LICENSE`.

Upstream: https://github.com/ultralytics/ultralytics

License information: https://www.ultralytics.com/license

## Retained module provenance

The existing source describes the MKIR inverted residual structure as based on MK-UNet and the DDT components as based on WPFormer. These descriptions are retained as attribution to the source's stated module provenance; this repository preserves the actual existing implementations. It does not include external datasets, papers, weights, or additional upstream code downloads.

## Optional initializer helpers

The retained MKIR initializer can import `named_apply` and `trunc_normal_tf_` from timm when available. timm is an external dependency and its source is not bundled here. The original fallback remains available without timm.

Upstream: https://github.com/huggingface/pytorch-image-models
