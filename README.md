# Enhancing Multi-Scale Fabric Defect Detection with MCF-Net: A Context Fusion Approach ([The Visual Computer](https://link.springer.com/journal/371))

This repository is a PyTorch implementation of our paper: Enhancing Multi-Scale Fabric Defect Detection with MCF-Net: A Context Fusion Approach.

<details open>
<summary>Install</summary>
  
```bash
git clone https://github.com/wyyt1202/MCF-Net  # clone
cd MCF-Net
pip install -r requirements.txt  # install



Fabric defect detection plays a pivotal role in ensuring product quality in the textile industry. However, the complex morphology, varying sizes, and diverse structures of fabric defects pose significant challenges to existing detection models. To address these issues, we propose MCF-Net, a Multi-Scale Context Fusion Network specifically designed for fabric defect detection. The network comprises several innovative modules: the Multi-Scale Context Aggregation Module (MCAM) leverages multi-scale features to establish long-range dependencies, while the Multi-Scale Context Diffusion Fusion Pyramid Network (MCD-FPN) enriches each detection scale with contextual information. The Cross Stage Partial Bottleneck with 3 convolutions-Latent Space (C3-LS) module enhances feature extraction by mapping features into a high-dimensional space. Furthermore, the Efficient Local Attention (ELA) mechanism dynamically focuses on critical regions, improving detection accuracy. Experimental results on our self-built FD6052 dataset and publicly available TianChi and DAGM2007 datasets demonstrate the effectiveness of MCF-Net, achieving state-of-the-art performance in multi-scale fabric defect detection. To balance model size and accuracy, channel pruning is applied, reducing computational complexity without compromising detection performance.
