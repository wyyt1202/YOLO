# Enhancing Multi-Scale Fabric Defect Detection with MCF-Net: A Context Fusion Approach ([The Visual Computer](https://link.springer.com/journal/371))

This repository is a PyTorch implementation of our paper: Enhancing Multi-Scale Fabric Defect Detection with MCF-Net: A Context Fusion Approach.


## Comprehensive comparison with other models on the FD6052 dataset
| Models            | Precision (%) | Recall (%) | mAP50 (%) | mAP95 (%) | Params (M) | GFLOPS |
|-------------------|---------------|------------|-----------|-----------|------------|--------|
| SSD-VGG16         | 92.9          | 29.8       | 73.7      | 31.5      | 24.3       | 61.3   |
| Faster RCNN-ResNet50 | 43.4       | 85.3       | 62.0      | 26.1      | 28.3       | 941.0  |
| YOLOv5n          | 87.1          | 77.5       | 85.7      | 44.2      | 1.8        | 4.2    |
| YOLOv5s          | 90.3          | 82.2       | 89.4      | 48.3      | 7.0        | 15.8   |
| YOLOv6n          | 61.7          | 56.1       | 60.0      | 27.5      | 4.6        | 11.3   |
| YOLOv6s          | 66.3          | 57.6       | 62.6      | 29.7      | 18.5       | 45.1   |
| YOLOX-n          | 87.2          | 81.5       | 83.4      | 45.5      | 2.2        | 6.9    |
| YOLOX-s          | 91.4          | 89.7       | 88.9      | 51.0      | 9.0        | 26.8   |
| YOLOv8n          | 62.3          | 58.9       | 64.1      | 36.5      | 3.0        | 8.1    |
| YOLOv8s          | 81.4          | 81.3       | 87.1      | 30.7      | 11.1       | 28.4   |
| YOLO11           | 86.3          | 84.1       | 90.2      | 44.1      | 2.6        | 6.3    |
| YOLO11           | 88.0          | 88.5       | 94.4      | 48.3      | 9.4        | 21.3   |
| IDD-YOLO         | 91.4          | 86.2       | 90.1      | 50.4      | 2.9        | 5.3    |
| LiteYOLO-ID      | 79.7          | 84.0       | 88.3      | 46.2      | 3.8        | 9.6    |
| Ours             | 94.1          | 90.2       | 95.1      | 55.5      | 8.5        | 20.6   |



<details open>
<summary>Install</summary>
  
```bash
git clone https://github.com/wyyt1202/MCF-Net  # clone
cd MCF-Net
pip install -r requirements.txt  # install


Fabric defect detection plays a pivotal role in ensuring product quality in the textile industry. However, the complex morphology, varying sizes, and diverse structures of fabric defects pose significant challenges to existing detection models. To address these issues, we propose MCF-Net, a Multi-Scale Context Fusion Network specifically designed for fabric defect detection. The network comprises several innovative modules: the Multi-Scale Context Aggregation Module (MCAM) leverages multi-scale features to establish long-range dependencies, while the Multi-Scale Context Diffusion Fusion Pyramid Network (MCD-FPN) enriches each detection scale with contextual information. The Cross Stage Partial Bottleneck with 3 convolutions-Latent Space (C3-LS) module enhances feature extraction by mapping features into a high-dimensional space. Furthermore, the Efficient Local Attention (ELA) mechanism dynamically focuses on critical regions, improving detection accuracy. Experimental results on our self-built FD6052 dataset and publicly available TianChi and DAGM2007 datasets demonstrate the effectiveness of MCF-Net, achieving state-of-the-art performance in multi-scale fabric defect detection. To balance model size and accuracy, channel pruning is applied, reducing computational complexity without compromising detection performance.
