# Awesome-Prompt-Adapter-Learning-for-VLMs
A curated list of prompt/adapter learning methods for vision-language models.

# Table of Contents

- [Papers](#papers)
    - [Surveys](#surveys)
    - [Prompt Learning](#prompt-learning)
    - [Test-time Prompt Tuning](#test-time-prompt-tuning)
    - [Video Prompting](#video-prompting-learning)

## Keywords

![](https://img.shields.io/badge/Text-green) Use text-based learnable prompts/adapters.

![](https://img.shields.io/badge/Image-orange) Use image-based learnable prompts/adapters.

![](https://img.shields.io/badge/Image--Text-blue) Use text- and image-based learnable prompts/adapters.

# Papers

## Surveys

- A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models. [[Paper](https://arxiv.org/abs/2307.12980)]
- Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey. [[Paper](https://arxiv.org/abs/2402.02242)]

## Prompt Learning
### Experimental Comparison

Base-to-Novel Generalization. (ViT-B/16 CLIP)

| Methods    | Pub      | Base   | Novel  | HM (main)     | Code |
| ---        | ---      | ---    | ---    | :---:  | ---  |
| CLIP       | ICML 21  | 69.34  | 74.22  | 71.70  | [Link](https://github.com/openai/CLIP)  |
| CoOp       | IJCV 22  | 82.69  | 63.22  | 71.66  | [Link](https://github.com/kaiyangzhou/coop)  |
| CoCoOp     | CVPR 22  | 80.47  | 71.69  | 75.83  | [Link](https://github.com/KaiyangZhou/CoOp)  |
| ProDA      | CVPR 22  | 81.56  | 72.30  | 76.65  | [Link](https://github.com/bbbdylan/proda) |
| RPO        | ICCV 23  | 81.13  | 75.00  | 77.78  | [Link](https://github.com/mlvlab/RPO)  |
| MaPLe      | CVPR 23  | 82.28  | 75.14  | 78.55  | [Link](https://github.com/muzairkhattak/multimodal-prompt-learning)  |
| MetaPrompt | TIP 24   | 83.65  | 75.48  | 79.09  | ---  |
| DePT       | CVPR 24  | 83.62  | 75.04  | 79.10  | [Link](https://github.com/Koorye/DePT) |
| LASP       | CVPR 23  | 83.18  | 76.11  | 79.48  | ---  |
| TCP        | CVPR 24  | 84.13  | 75.36  | 79.51  | [Link](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning) |
| MMA        | CVPR 24  | 83.20  | 76.80  | 79.87  | [Link](https://github.com/ZjjConan/Multi-Modal-Adapter) |
| PromptSRC  | ICCV 23  | 84.26  | 76.10  | 79.97  | [Link](https://github.com/muzairkhattak/PromptSRC)  |
| HPT        | AAAI 24  | 84.32  | 76.86  | 80.23  | [Link](https://github.com/vill-lab/2024-aaai-hpt)  |
| CoPrompt   | ICLR 24  | 84.00  | 77.23  | 80.48  | [Link](https://github.com/shuvenduroy/coprompt)  |
| PromptKD   | CVPR 24  | 86.96  | 80.73  | 83.73  | [Link](https://github.com/zhengli97/promptkd)  |

Table 1. Average results on 11 datasets.


<!-- | Methods    | Pub      | Base   | Novel  | HM      | Code |
| ---        | ---      | ---    | ---    | ---     | ---  | 
| CLIP       |          | 72.43  | 68.14  | 70.22   | [Link]()  |
| CoOp       | IJCV 22  | 76.47  | 67.88  |         | [Link]()  |
| CoCoOp     | CVPR 22  | 75.98  | 70.43  | 73.10   | [Link]()  |
| MaPLe      |          | 76.66  | 70.54  | 73.47   | [Link]()  |
| RPO        | ICCV 23  |        |        | 74.00   | ---  |
| PromptSRC  | ICCV 23  | 77.60  | 70.73  | 74.01   | [Link]()  |
| MetaPrompt |          |        |        | 74.02   | ---  |
| HPT        |          |        |        | 74.17   | ---  |
| CoPrompt   | ICLR 24  | 77.67  | 71.27  | 74.33   | ---  |
| CE         |          |        |        | 75.49   | ---  |
| PromptKD   | CVPR 24  | 80.83  | 74.66  | 77.62   | [Link]()  |

Table 2. Experimental results on ImageNet-1K. -->

### Paper List

#### 2022 
- `CoOp` **Learning to Prompt for Vision-Language Models.** IJCV 2022.  
  [[Paper](https://arxiv.org/abs/2203.05557)] [[Code](https://github.com/KaiyangZhou/CoOp)] ![](https://img.shields.io/badge/Text-green)
- `CoCoOp` **Conditional Prompt Learning for Vision-Language Models.** CVPR 2022.   
[[Paper](https://arxiv.org/abs/2203.05557)] [[Code](https://github.com/KaiyangZhou/CoOp)] ![](https://img.shields.io/badge/Text-green)
- `ProDA` **Prompt Distribution Learning.** CVPR 2022.  
[[Paper](https://arxiv.org/abs/2205.03340)] [[Code](https://github.com/bbbdylan/proda)] ![](https://img.shields.io/badge/Text-green)
- `VPT` **Visual Prompt Tuning**. ECCV 2022.  
[[Paper](https://arxiv.org/abs/2203.12119)] [[Code](https://github.com/kmnp/vpt)] ![](https://img.shields.io/badge/Image-orange)

#### 2023
- `MaPLe` **MaPLe: Multi-modal Prompt Learning.** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2210.03117)] [[Code](https://github.com/muzairkhattak/multimodal-prompt-learning)] ![](https://img.shields.io/badge/Image--Text-blue)
- `KgCoOp` **Visual-Language Prompt Tuningx with Knowledge-guided Context Optimization.** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2303.13283)] [[Code](https://github.com/htyao89/KgCoOp)] ![](https://img.shields.io/badge/Text-green)
- `LASP` **LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision & Language Models** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2210.01115)] ![](https://img.shields.io/badge/Text-green)
- `DAM-VP` **Diversity-Aware Meta Visual Prompting** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2303.08138)]  [[Code](https://github.com/shikiw/DAM-VP)] ![](https://img.shields.io/badge/Image-orange) ![](https://img.shields.io/badge/Text-green)
- `TaskRes` **Task Residual for Tuning Vision-Language Models** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2211.10277)] [[Code](https://github.com/geekyutao/TaskRes)]
- `RPO` **Read-only Prompt Optimization for Vision-Language Few-shot Learning.** ICCV 2023.  
[[Paper](https://arxiv.org/abs/2308.14960)] [[Code](https://github.com/mlvlab/rpo)] ![](https://img.shields.io/badge/Image--Text-blue)
- `KAPT` **Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models.** ICCV 2023.  
[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kan_Knowledge-Aware_Prompt_Tuning_for_Generalizable_Vision-Language_Models_ICCV_2023_paper.pdf)] ![](https://img.shields.io/badge/Text-green) <!-- Text-based ViT-B/32 -->
- `ProGrad` **Prompt-aligned Gradient for Prompt Tuning.** ICCV 2023.  
[[Paper](https://arxiv.org/abs/2205.14865)][[Code](https://github.com/BeierZhu/Prompt-align)] ![](https://img.shields.io/badge/Text-green) <!-- ViT-B/32 -->
- `PromptSRC` **Self-regulating Prompts: Foundational Model Adaptation without Forgetting.** ICCV 2023.  
[[Paper](https://openaccess.thecvf.com//content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf)] [[Code](https://github.com/muzairkhattak/PromptSRC)] ![](https://img.shields.io/badge/Image--Text-blue)
- `DeFo` **Learning to Decompose Visual Features with Latent Textual Prompts.** ICLR 2023.  
[[Paper](https://arxiv.org/abs/2210.04287)] ![](https://img.shields.io/badge/Text-green)
- `POMP` **Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition** NeurIPS 2023.  
[[Paper](https://arxiv.org/abs/2304.04704)] [[Code](https://github.com/amazon-science/prompt-pretraining)]  ![](https://img.shields.io/badge/Text-green)

#### 2024
- `MetaPrompt` **Learning Domain Invariant Prompt for Vision-Language Models.** TIP 2024.  
[[Paper](https://arxiv.org/abs/2212.04196)] ![](https://img.shields.io/badge/Image--Text-blue)
- `SA2VP` **SA2VP: Spatially Aligned-and-Adapted Visual Prompt.** AAAI 2024.  
[[Paper](https://arxiv.org/abs/2312.10376)] [[Code](https://github.com/tommy-xq/SA2VP)] ![](https://img.shields.io/badge/Image-orange)
- `HPT` **Learning Hierarchical Prompt with Structured Linguistic Knowledge for Vision-Language Models.** AAAI 2024.  
[[Paper](https://arxiv.org/abs/2312.06323)] [[Code](https://github.com/Vill-Lab/2024-AAAI-HPT)] ![](https://img.shields.io/badge/Image--Text-blue)
- `LaViP` **LaViP: Language-Grounded Visual Prompts**. AAAI 2024.  
[[Paper](https://arxiv.org/abs/2312.10945)] ![](https://img.shields.io/badge/Image-orange) <!-- 没imagenet结果 -->
- `CoPrompt` **Consistency-guided Prompt Learning for Vision-Language Models.** ICLR 2024.  
[[Paper](https://arxiv.org/abs/2306.01195)] [[Code](https://github.com/ShuvenduRoy/CoPrompt)] ![](https://img.shields.io/badge/Image--Text-blue) 
- `ProText` **Learning to Prompt with Text Only Supervision for Vision-Language Models.** arxiv 24.  
[[Paper](https://arxiv.org/abs/2401.02418)] [[Code](https://github.com/muzairkhattak/ProText)] ![](https://img.shields.io/badge/Text-green) 
- `PromptKD` **Unsupervised Prompt Distillation for Vision Language Models.** CVPR 2024.  
[[Paper](https://arxiv.org/abs/2403.02781)] [[Code](https://github.com/zhengli97/PromptKD)] ![](https://img.shields.io/badge/Image--Text-blue) 
- `DePT` **DePT: Decoupled Prompt Tuning**. CVPR 2024.  
[[Paper](https://arxiv.org/abs/2309.07439)] [[Code](https://github.com/Koorye/DePT)] ![](https://img.shields.io/badge/Text-green)
- `ArGue` **ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models**. CVPR 2024.  
[[Paper](https://arxiv.org/abs/2311.16494)] ![](https://img.shields.io/badge/Text-green)
- `TCP` **TCP:Textual-based Class-aware Prompt tuning for Visual-Language Model**. CVPR 2024.  
[[Paper](https://arxiv.org/abs/2311.18231)] [[Code](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning)] ![](https://img.shields.io/badge/Text-green)
- `MMA` **MMA: Multi-Modal Adapter for Vision-Language Models**. CVPR 2024.  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf)] [[Code](https://github.com/ZjjConan/Multi-Modal-Adapter)] ![](https://img.shields.io/badge/Image--Text-blue)


## Test-time Prompt Tuning

### Experimental Comparison

| Methods     | Pub        | ImageNet | -A    | -V2   | -R    | -S    | Avg. (main)  | Code |
|-------------|------------|----------| ---   | ---   |  ---  |  ---  |  :---:  | ---  |
| CoOp        | IJCV 22    | 71.51    | 49.71 | 64.20 | 75.21 | 47.99 | 59.28 | [Link](https://github.com/kaiyangzhou/coop) |
| CoCoOp      | CVPR 22    | 71.02    | 50.63 | 64.07 | 76.18 | 48.75 | 59.91 | [Link](https://github.com/kaiyangzhou/coop) |
| TPT         | NeurIPS 22 | 68.98    | 54.77 | 63.45 | 77.06 | 47.94 | 60.81 | [Link](https://github.com/azshue/TPT) |
| TPT+CoOp    | NeurIPS 22 | 73.61    | 57.95 | 66.83 | 77.27 | 49.29 | 62.84 | [Link](https://github.com/azshue/TPT) |
| PromptAlign | NeurIPS 23 | ---      | 59.37 | 65.29 | 79.33 | 59.37 | 63.55 | [Link](https://github.com/jameelhassan/PromptAlign) |
| TPS+CoOp    | Arxiv 24   | 73.73    | 60.49 | 66.84 | 77.44 | 49.08 | 65.52 | [Link](https://github.com/elaine-sui/TPS) | 
| RLCF        | ICLR 24    | 73.23    | 65.45 | 69.77 | 83.35 | 54.74 | 68.33 | [Link](https://github.com/mzhaoshuai/RLCF) |
| RLCF+CoOp   | ICLR 24    | 76.05    | 69.74 | 70.62 | 84.51 | 56.49 | 70.34 | [Link](https://github.com/mzhaoshuai/RLCF) | 


Table 3. Test-time prompt tuning methods on OOD data.

### Paper List

- `TPT` **Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models.** NeurIPS 2022.  
[[Paper](https://arxiv.org/abs/2209.07511)] [[Code](https://github.com/azshue/TPT)]
- `SwapPrompt` **SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models.** NeurIPS 2023.  
[[Paper](https://openreview.net/forum?id=EhdNQiOWgQ&referrer=%5Bthe%20profile%20of%20Song%20Guo%5D(%2Fprofile%3Fid%3D~Song_Guo5))]
- `PrompAlign` **Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization.** NeurIPS 2023.  
[[Paper](https://arxiv.org/abs/2311.01459)] [[Code](https://github.com/jameelhassan/PromptAlign)]
- `TPS` **Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models.** Arxiv 2024.  
[[Paper](https://arxiv.org/pdf/2403.12952v1.pdf)] [[Code](https://github.com/elaine-sui/TPS)]
- `RLCF` **Test-time Adaptation with CLIP reward for zero-shot generalization in Vision-Language Models.** ICLR 2024.  
[[Paper](https://openreview.net/forum?id=kIP0duasBb)] [[Code](https://github.com/mzhaoshuai/RLCF)]
- `InTTA` **Invariant Test-Time Adaptation for Vision-Language Model Generalization.** Arxiv 2024.  
[[Paper](https://arxiv.org/pdf/2403.00376.pdf)] [[Code](https://github.com/MaHuanAAA/InTTA)]

## Video Prompting Learning

### Experimental Comparison

### Paper List
- `Efficient-Prompt` **Prompting visual-language models for efficient video understanding.** ECCV 2022.  
[[Paper](https://arxiv.org/pdf/2112.04478.pdf)] [[Code](https://github.com/ju-chen/Efficient-Prompt)]
- `InTTA` **Expanding Language-Image Pretrained Models for General Video Recognition.** ECCV 2022.  
[[Paper](https://arxiv.org/pdf/2208.02816.pdf)] [[Code](https://github.com/microsoft/VideoX/tree/master/X-CLIP)]
- `RePro` **Compositional Prompt Tuning with Motion Cues for Open-vocabulary Video Relation Detection.** ICLR 2023.  
[[Paper](https://arxiv.org/pdf/2302.00268.pdf)] [[Code](https://github.com/Dawn-LX/OpenVoc-VidVRD)]

