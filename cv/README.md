# Computer Vision

8 concepts from image classification to multimodal vision-LLMs. Each concept has a markdown explanation and runnable notebook.

## Prerequisites
- Python, numpy, torch, matplotlib
- Basic linear algebra and calculus

## Concepts
| # | Concept | Key Content | Notebook |
|---|---------|-------------|---------|
| 01 | [Image Classification](concepts/01-image-classification.md) | CNN pipeline, transfer learning, data augmentation | [notebook](notebooks/01-image-classification.ipynb) |
| 02 | [Object Detection](concepts/02-object-detection.md) | YOLO, Faster R-CNN, anchor boxes, NMS, mAP | [notebook](notebooks/02-object-detection.ipynb) |
| 03 | [Image Segmentation](concepts/03-image-segmentation.md) | Semantic vs instance, U-Net, SAM | [notebook](notebooks/03-image-segmentation.ipynb) |
| 04 | [Vision Transformers](concepts/04-vision-transformers.md) | ViT, patch embeddings, CLS token | [notebook](notebooks/04-vision-transformers.ipynb) |
| 05 | [Contrastive Learning](concepts/05-contrastive-learning-vision.md) | CLIP, SimCLR, self-supervised | [notebook](notebooks/05-contrastive-learning-vision.ipynb) |
| 06 | [Diffusion Models](concepts/06-diffusion-models.md) | DDPM, U-Net denoiser, DDIM | [notebook](notebooks/06-diffusion-models.ipynb) |
| 07 | [Video Understanding](concepts/07-video-understanding.md) | Optical flow, 3D convolutions | [notebook](notebooks/07-video-understanding.ipynb) |
| 08 | [Multimodal Vision-LLM](concepts/08-multimodal-vision-llm.md) | LLaVA, visual instruction tuning | [notebook](notebooks/08-multimodal-vision-llm.ipynb) |

## Learning Paths
- **CV Fundamentals:** 01 → 02 → 03 → 04
- **Modern Generative CV:** 04 → 05 → 06
- **Production CV:** 01 → 02 → 07 → 08
