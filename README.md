# Awesome-Prompt-Adapter-Learning-for-VLMs
A curated list of prompt/adapter learning methods for vision-language models (e.g., CLIP).

# Table of Contents

- [Papers](#papers)
    - [Surveys](#surveys)
    - [Prompt Learning](#general-prompt-learning)
    - [Test-time Prompt Learning](#general-test-time-prompt-learning)
    - [Adapter Learning](#general-adapter-learning)
    - [Video Understanding](#video-understanding)
    - [Continual Learning](#continual-learning)

## 💡Tips:

- If you know that some papers published in top conferences (CVPR, ICCV, ECCV, ICML, NeurlPS, ICLR) or journals (TPAMI, IJCV, TIP) have not been included in this list, please feel free to contact me at any time, either by sending an email (zhengli97[at]qq.com) or submitting an issue.
- We would appreciate more people joining us in maintaining this list of papers.  
- Note that papers without open-source code are not recommended.

## Keywords

![](https://img.shields.io/badge/Text-green) Use text-based prompts/adapters.

![](https://img.shields.io/badge/Image-orange) Use image-based prompts/adapters.

![](https://img.shields.io/badge/Image--Text-blue) Use text- and image-based prompts/adapters.

## Surveys

- A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models. [[Paper](https://arxiv.org/abs/2307.12980)]
- Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey. [[Paper](https://arxiv.org/abs/2402.02242)]

## Foundation Models

- `CLIP` **Learning Transferable Visual Models From Natural Language Supervision.** ICML 2021.  
  [[Paper](https://arxiv.org/abs/2103.00020)] [[Code](https://github.com/OpenAI/CLIP)]  
- `ALIGN` **Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision.** ICML 2021.  
  [[Paper](https://arxiv.org/abs/2102.05918)]
- `LiT` **LiT: Zero-Shot Transfer with Locked-image text Tuning.** CVPR 2022.   
  [[Paper](https://arxiv.org/abs/2111.07991)] [[Code](https://github.com/google-research/vision_transformer#lit-models)]
- `EVA-CLIP` **EVA-CLIP: Improved Training Techniques for CLIP at Scale.** 2023.   
  [[Paper](https://arxiv.org/abs/2303.15389)] [[Code](https://github.com/baaivision/EVA)]   
- `SigLIP` **Sigmoid Loss for Language Image Pre-Training.** ICCV 2023.  
  [[Paper](https://arxiv.org/abs/2303.15343)] [[Code](https://github.com/google-research/big_vision)]  
- `AlphaCLIP` **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want.** CVPR 2024.  
  [[Paper](https://arxiv.org/abs/2312.03818)] [[Code](https://github.com/SunzeY/AlphaCLIP)] 
- `LongCLIP` **Long-CLIP: Unlocking the Long-Text Capability of CLIP** ECCV 2024.  
  [[Paper](https://arxiv.org/abs/2403.15378)] [[Code](https://github.com/beichenzbc/Long-CLIP)]

## General Prompt Learning
### Experimental Comparison

Base-to-Novel Generalization. (ViT-B/16 CLIP)

| Methods    | Paper    | Pub      | Base   | Novel  | HM (main) | Code | Type |
| :---:      | :---:    | :---:    | ---    | ---    | :---:     | :--: | :--: |
| CLIP       | [Link](https://arxiv.org/abs/2103.00020) | ICML 21  | 69.34 | 74.22 | 71.70  | [Link](https://github.com/openai/CLIP)  | Model |
| CoOp       | [Link](https://arxiv.org/abs/2203.05557) | IJCV 22  | 82.69 | 63.22 | 71.66  | [Link](https://github.com/kaiyangzhou/coop)  | - |
| CoCoOp     | [Link](https://arxiv.org/abs/2203.05557) | CVPR 22  | 80.47 | 71.69 | 75.83  | [Link](https://github.com/KaiyangZhou/CoOp)  | - |
| DPC        | [Link](https://arxiv.org/abs/2503.13443) | CVPR 25  | 85.15 | 68.84 | 76.13 | [Link](https://github.com/JREion/DPC) | - |
| DPC+PromptKD | - | - | 87.55 | 80.55 | **83.91** | - | Plugin |
| ProDA      | [Link](https://arxiv.org/abs/2205.03340) | CVPR 22  | 81.56 | 72.30 | 76.65  | [Link](https://github.com/bbbdylan/proda) | - |
| TextRefiner | [Link](https://arxiv.org/abs/2412.08176) | AAAI 25 | 79.74 | 74.32 | 76.94  | [Link](https://github.com/xjjxmu/TextRefiner) | - |
| TextRefiner+PromptKD | - | - | 85.22 | 79.64 | **82.33** | - | Plugin |
| KgCoOp     | [Link](https://arxiv.org/abs/2303.13283) | CVPR 23  | 80.73 | 73.60 | 77.00  | [Link](https://github.com/htyao89/KgCoOp) | - |
| RPO        | [Link](https://arxiv.org/abs/2308.14960) | ICCV 23  | 81.13 | 75.00 | 77.78  | [Link](https://github.com/mlvlab/RPO)  | -
| DePT       | [Link](https://arxiv.org/abs/2309.07439) | CVPR 24  | 83.80 | 72.89 | 77.97  | [Link](https://github.com/Koorye/DePT) | - |
| DePT+PromptSRC | - | - | 85.19 | 76.17 | **80.43** | - | Plugin |
| MaPLe      | [Link](https://arxiv.org/abs/2210.03117) | CVPR 23  | 82.28 | 75.14 | 78.55  | [Link](https://github.com/muzairkhattak/multimodal-prompt-learning) | - |
| QNet       | [Link](https://openreview.net/forum?id=dKlxDx2SoS) | ICLR 24  | 83.32  | 75.65  | 79.30 | [Link](https://github.com/SHIBOYA/QNet) | - |
| CasPL      | [Link](https://arxiv.org/abs/2409.17805) | ECCV 24  | 84.78 | 74.49 | 79.30  | [Link](https://github.com/megvii-research/CasPL) | - |
| CasPL+PromptSRC | - | - | 86.11  | 79.54  | **82.69** | - | Plugin |
| TCP        | [Link](https://arxiv.org/abs/2311.18231) | CVPR 24  | 84.13  | 75.36  | 79.51   | [Link](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning) | - |
| MMA        | [Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf) | CVPR 24  | 83.20  | 76.80  | 79.87     | [Link](https://github.com/ZjjConan/Multi-Modal-Adapter) | - |
| PromptSRC  | [Link](https://arxiv.org/abs/2307.06948) | ICCV 23  | 84.26  | 76.10  | 79.97  | [Link](https://github.com/muzairkhattak/PromptSRC) | - |
| HPT        | [Link](https://arxiv.org/abs/2312.06323) | AAAI 24  | 84.32  | 76.86  | 80.23  | [Link](https://github.com/vill-lab/2024-aaai-hpt) | - |
| CoPrompt   | [Link](https://arxiv.org/abs/2311.18231) | ICLR 24  | 84.00  | 77.23  | 80.48  | [Link](https://github.com/shuvenduroy/coprompt) | - |
| MMRL       | [Link](https://arxiv.org/abs/2503.08497) | CVPR 25  | 85.68  | 77.16  | 81.20  | [Link](https://github.com/yunncheng/MMRL) | - |
| LLaMP      | [Link](https://arxiv.org/abs/2312.04076) | CVPR 24  | 85.16  | 77.71  | 81.27  | [Link](https://github.com/zhaohengz/LLaMP) | - |
| PromptKD   | [Link](https://arxiv.org/abs/2403.02781) | CVPR 24  | 86.96  | 80.73  | 83.73  | [Link](https://github.com/zhengli97/promptkd) | - |

Table 1. Average results on 11 datasets. (Only works with open-source code will be listed.)

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
- `VP` **Exploring Visual Prompts for Adapting Large-Scale Models.** Arxiv 2022.   
[[Paper](https://arxiv.org/abs/2203.17274)] [[Code](https://github.com/hjbahng/visual_prompting)] ![](https://img.shields.io/badge/Image-orange)    

#### 2023
- `MaPLe` **MaPLe: Multi-modal Prompt Learning.** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2210.03117)] [[Code](https://github.com/muzairkhattak/multimodal-prompt-learning)] ![](https://img.shields.io/badge/Image--Text-blue)
- `KgCoOp` **Visual-Language Prompt Tuningx with Knowledge-guided Context Optimization.** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2303.13283)] [[Code](https://github.com/htyao89/KgCoOp)] ![](https://img.shields.io/badge/Text-green)
- `LASP` **LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision & Language Models.** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2210.01115)] [No Code Found] ![](https://img.shields.io/badge/Text-green)
- `DAM-VP` **Diversity-Aware Meta Visual Prompting.** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2303.08138)] [[Code](https://github.com/shikiw/DAM-VP)] ![](https://img.shields.io/badge/Image-orange) 
- `TaskRes` **Task Residual for Tuning Vision-Language Models.** CVPR 2023.  
[[Paper](https://arxiv.org/abs/2211.10277)] [[Code](https://github.com/geekyutao/TaskRes)] ![](https://img.shields.io/badge/Text-green)   
- `RPO` **Read-only Prompt Optimization for Vision-Language Few-shot Learning.** ICCV 2023.  
[[Paper](https://arxiv.org/abs/2308.14960)] [[Code](https://github.com/mlvlab/rpo)] ![](https://img.shields.io/badge/Image--Text-blue)
- `KAPT` **Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models.** ICCV 2023.  
[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kan_Knowledge-Aware_Prompt_Tuning_for_Generalizable_Vision-Language_Models_ICCV_2023_paper.pdf)] [No Code] ![](https://img.shields.io/badge/Text-green)
- `CuPL` **What does a platypus look like? Generating customized prompts for zero-shot image classification.** ICCV 2023.   
[[Paper](https://arxiv.org/pdf/2209.03320)] [[Code](https://github.com/sarahpratt/CuPL)] ![](https://img.shields.io/badge/Text-green)    
- `ProGrad` **Prompt-aligned Gradient for Prompt Tuning.** ICCV 2023.  
[[Paper](https://arxiv.org/abs/2205.14865)][[Code](https://github.com/BeierZhu/Prompt-align)] ![](https://img.shields.io/badge/Text-green) <!-- ViT-B/32 -->
- `PromptSRC` **Self-regulating Prompts: Foundational Model Adaptation without Forgetting.** ICCV 2023.  
[[Paper](https://arxiv.org/abs/2307.06948)] [[Code](https://github.com/muzairkhattak/PromptSRC)] ![](https://img.shields.io/badge/Image--Text-blue)
- `LFA` **Black Box Few-Shot Adaptation for Vision-Language models.** ICCV 2023.   
[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.pdf)] [[Code](https://github.com/saic-fi/LFA)]   
- `DeFo` **Learning to Decompose Visual Features with Latent Textual Prompts.** ICLR 2023.  
[[Paper](https://arxiv.org/abs/2210.04287)] [No Code Found] ![](https://img.shields.io/badge/Text-green)
- `PLOT` **PLOT: Prompt Learning with Optimal Transport for Vision-Language Models.** ICLR 2023.   
[[Paper](https://arxiv.org/pdf/2210.01253)] [[Code](https://github.com/CHENGY12/PLOT)] ![](https://img.shields.io/badge/Text-green)   
- `POMP` **Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition.** NeurIPS 2023.  
[[Paper](https://arxiv.org/abs/2304.04704)] [[Code](https://github.com/amazon-science/prompt-pretraining)]  ![](https://img.shields.io/badge/Text-green)

#### 2024
- `MetaPrompt` **Learning Domain Invariant Prompt for Vision-Language Models.** TIP 2024.  
[[Paper](https://arxiv.org/abs/2212.04196)] [No Code] ![](https://img.shields.io/badge/Image--Text-blue)
- `ProVP` **Progressive Visual Prompt Learning with Contrastive Feature Re-formation.** IJCV 2024.  
[[Paper](https://arxiv.org/abs/2304.08386)] [[Code](https://github.com/MCG-NJU/ProVP)] ![](https://img.shields.io/badge/Image-orange)
- `CoPL` **CoPL: Contextual Prompt Learning for Vision-Language Understanding.**  AAAI 2024.    
[[Paper](https://arxiv.org/abs/2307.00910)] [No Code Found] ![](https://img.shields.io/badge/Text-green)    
- `SA2VP` **SA2VP: Spatially Aligned-and-Adapted Visual Prompt.** AAAI 2024.  
[[Paper](https://arxiv.org/abs/2312.10376)] [[Code](https://github.com/tommy-xq/SA2VP)] ![](https://img.shields.io/badge/Image-orange)
- `HPT` **Learning Hierarchical Prompt with Structured Linguistic Knowledge for Vision-Language Models.** AAAI 2024.  
[[Paper](https://arxiv.org/abs/2312.06323)] [[Code](https://github.com/Vill-Lab/2024-AAAI-HPT)] ![](https://img.shields.io/badge/Image--Text-blue)
- `LaViP` **LaViP: Language-Grounded Visual Prompts.** AAAI 2024.  
[[Paper](https://arxiv.org/abs/2312.10945)] [No Code Found] ![](https://img.shields.io/badge/Image-orange) <!-- 没imagenet结果 -->
- `CoPrompt` **Consistency-guided Prompt Learning for Vision-Language Models.** ICLR 2024.  
[[Paper](https://arxiv.org/abs/2306.01195)] [[Code](https://github.com/ShuvenduRoy/CoPrompt)] ![](https://img.shields.io/badge/Image--Text-blue) 
- `PromptKD` **PromptKD: Unsupervised Prompt Distillation for Vision Language Models.** CVPR 2024.  
[[Paper](https://arxiv.org/abs/2403.02781)] [[Code](https://github.com/zhengli97/PromptKD)] ![](https://img.shields.io/badge/Image--Text-blue) 
- `DePT` **DePT: Decoupled Prompt Tuning.** CVPR 2024.  
[[Paper](https://arxiv.org/abs/2309.07439)] [[Code](https://github.com/Koorye/DePT)] ![](https://img.shields.io/badge/Image--Text-blue) 
- `ArGue` **ArGue: Attribute-Guided Prompt Tuning for Vision-Language Models.** CVPR 2024.  
[[Paper](https://arxiv.org/abs/2311.16494)] [No Code Found] ![](https://img.shields.io/badge/Text-green)
- `TCP` **TCP: Textual-based Class-aware Prompt tuning for Visual-Language Model.** CVPR 2024.  
[[Paper](https://arxiv.org/abs/2311.18231)] [[Code](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning)] ![](https://img.shields.io/badge/Text-green)
- `MMA` **MMA: Multi-Modal Adapter for Vision-Language Models.** CVPR 2024.  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf)] [[Code](https://github.com/ZjjConan/Multi-Modal-Adapter)] ![](https://img.shields.io/badge/Image--Text-blue)
- `LLaMP` **Large Language Models are Good Prompt Learners for Low-Shot Image Classification.** CVPR 24.   
[[Paper](https://arxiv.org/abs/2312.04076)] [[Code](https://github.com/zhaohengz/LLaMP)] ![](https://img.shields.io/badge/Image--Text-blue)   
- `KDPL` **Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2407.03056)] [[Code](https://github.com/miccunifi/KDPL)] ![](https://img.shields.io/badge/Image--Text-blue)
- `CoCoLe` **Conceptual Codebook Learning for Vision-Language Models.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2407.02350)] [No Code Found] ![](https://img.shields.io/badge/Image--Text-blue)
- `CasPL` **Cascade Prompt Learning for Vision-Language Model Adaptation** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2409.17805)] [[Code](https://github.com/megvii-research/CasPL)] ![](https://img.shields.io/badge/Image--Text-blue)
- `GalLoP` **GalLoP: Learning Global and Local Prompts for Vision-Language Models.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2407.01400)] [No Code Found] ![](https://img.shields.io/badge/Image--Text-blue)   
- `AWT` **AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation.** NeurIPS 2024.  
[[Paper](https://arxiv.org/abs/2407.04603)] [[Code](https://github.com/MCG-NJU/AWT)] ![](https://img.shields.io/badge/Image--Text-blue)   
- `ATPrompt` **ATPrompt: Textual Prompt Learning with Embedded Attributes.** arxiv 2024.   
[[Paper](https://arxiv.org/abs/2412.09442)] [[Code](https://github.com/zhengli97/ATPrompt)] ![](https://img.shields.io/badge/Text-green)
- `QNet` **Prompt Learning with Quaternion Networks.** ICLR 2024.   
[[Paper](https://openreview.net/forum?id=dKlxDx2SoS)] [[Code](https://github.com/SHIBOYA/QNet)(Empty)] ![](https://img.shields.io/badge/Image--Text-blue)
- `QMaPLe` **Quantized Prompt for Efficient Generalization of Vision-Language Models.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2407.10704)] [[Code](https://github.com/beyondhtx/QPrompt)(Empty)]

#### 2025
- `TextRefiner` **TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning.** AAAI 2025.     
[[Paper](https://arxiv.org/abs/2412.08176)] [[Code](https://github.com/xjjxmu/TextRefiner)] ![](https://img.shields.io/badge/Text-green)   
- `ProText` **Learning to Prompt with Text Only Supervision for Vision-Language Models.** AAAI 2025.  
[[Paper](https://arxiv.org/abs/2401.02418)] [[Code](https://github.com/muzairkhattak/ProText)] ![](https://img.shields.io/badge/Text-green) 
- `MMRL` **MMRL: Multi-Modal Representation Learning for Vision-Language Models.** CVPR 2025.  
[[Paper](https://arxiv.org/abs/2503.08497)] [[Code](https://github.com/yunncheng/MMRL)] ![](https://img.shields.io/badge/Image--Text-blue)
- `DPC` **DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2503.13443)] [[Code](https://github.com/JREion/DPC)] ![](https://img.shields.io/badge/Text-green)   

## Another form of Prompt

### Paper List

- `CPT` **CPT: Colorful Prompt Tuning for pre-trained vision-language models** Arxiv 2021.   
[[Paper](https://arxiv.org/abs/2109.11797)] [[Code](https://github.com/thunlp/CPT)] ![](https://img.shields.io/badge/Image--Text-blue)
- `DetPro` **Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model.** CVPR 2022.   
[[Paper](https://arxiv.org/abs/2203.14940)] [[Code](https://github.com/dyabel/detpro)] ![](https://img.shields.io/badge/Text-green)
- `PromptDet` **PromptDet: Towards Open-vocabulary Detection using Uncurated Images.** ECCV 2022.   
[[Paper](https://arxiv.org/abs/2203.16513)] [[Code](https://github.com/fcjian/PromptDet)] ![](https://img.shields.io/badge/Text-green)
- **Visual Prompting via Image Inpainting**. NeurIPS 2022.   
[[Paper](https://arxiv.org/abs/2209.00647)] ![](https://img.shields.io/badge/Image-orange)
- `OVSeg` **Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP.** CVPR 2023.   
[[Paper](https://arxiv.org/abs/2210.04150)] [[Code](https://github.com/facebookresearch/ov-seg)] ![](https://img.shields.io/badge/Image-orange)
- `LoGoPrompt` **LoGoPrompt: Synthetic Text Images Can Be Good Visual Prompts for Vision-Language Models.** ICCV 2023.    
[[Paper](https://arxiv.org/abs/2309.01155)] ![](https://img.shields.io/badge/Image-orange)    
- `RedCircle` **What does CLIP know about a red circle? Visual prompt engineering for VLMs.** ICCV 2023.   
[[Paper](http://arxiv.org/abs/2304.06712)]] ![](https://img.shields.io/badge/Image-orange)   
- `FGVP` **Fine-Grained Visual Prompting.** NeurIPS 2023.   
[[Paper](https://arxiv.org/abs/2306.04356)] [[Code](https://github.com/ylingfeng/FGVP)] ![](https://img.shields.io/badge/Image-orange)
- `SoM` **Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v.** Arxiv 2023.  
[[Paper](https://arxiv.org/abs/2310.11441)] [[Code](https://github.com/microsoft/SoM)] ![](https://img.shields.io/badge/Image-orange)
- `Alpha-CLIP` **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want.** CVPR 2024.   
[[Paper](https://arxiv.org/abs/2312.03818)] [[Code](https://github.com/SunzeY/AlphaCLIP)] ![](https://img.shields.io/badge/Image-orange)
- `ViP-LLaVA` **Making Large Multimodal Models Understand Arbitrary Visual Prompts.** CVPR 2024.   
[[Paper](https://arxiv.org/abs/2312.00784)] [[Code](https://github.com/WisconsinAIVision/ViP-LLaVA)] ![](https://img.shields.io/badge/Image-orange)
- `SSC` **Segment, Select, Correct: A Framework for Weakly-Supervised Referring Segmentation.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2310.13479)] [[Code](https://github.com/fgirbal/segment-select-correct)] ![](https://img.shields.io/badge/Image-orange)


## General Test-time Prompt Learning

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

Table 2. Test-time prompt tuning methods on OOD data.

### Paper List

- `TPT` **Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models.** NeurIPS 2022.  
[[Paper](https://arxiv.org/abs/2209.07511)] [[Code](https://github.com/azshue/TPT)]
- `SwapPrompt` **SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models.** NeurIPS 2023.  
[[Paper](https://openreview.net/forum?id=EhdNQiOWgQ&referrer=%5Bthe%20profile%20of%20Song%20Guo%5D(%2Fprofile%3Fid%3D~Song_Guo5))]
- `PrompAlign` **Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization.** NeurIPS 2023.  
[[Paper](https://arxiv.org/abs/2311.01459)] [[Code](https://github.com/jameelhassan/PromptAlign)]
- `TPS` **Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models.** Arxiv 2024.  
[[Paper](https://arxiv.org/abs/2403.12952)] [[Code](https://github.com/elaine-sui/TPS)]
- `RLCF` **Test-time Adaptation with CLIP reward for zero-shot generalization in Vision-Language Models.** ICLR 2024.  
[[Paper](https://openreview.net/forum?id=kIP0duasBb)] [[Code](https://github.com/mzhaoshuai/RLCF)]
- `InTTA` **Invariant Test-Time Adaptation for Vision-Language Model Generalization.** Arxiv 2024.  
[[Paper](https://arxiv.org/abs/2403.00376)] [[Code](https://github.com/MaHuanAAA/InTTA)]
- `TDA` **Efficient Test-Time Adaptation of Vision-Language Models.** CVPR 2024.   
[[Paper](https://arxiv.org/abs/2403.18293)] [[Code](https://github.com/kdiAAA/TDA?tab=readme-ov-file)]   
- `C-TPT` **C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion.** ICLR 2024.   
[[Paper](https://arxiv.org/abs/2403.14119)] [[Code](https://github.com/hee-suk-yoon/C-TPT)]   
- `DynaPrompt` **DynaPrompt: Dynamic Test-Time Prompt Tuning.** ICLR 2025.   
[[Paper](https://openreview.net/forum?id=EFZEdHB3Mp)]   
- `R-TPT` **R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning.** CVPR 25.   
[[Paper](https://arxiv.org/abs/2504.11195)] [[Code](https://github.com/TomSheng21/R-TPT)]   

## General Adapter Learning

### Paper List

- `CLIP-Adapter` **CLIP-Adapter: Better Vision-Language Models with Feature Adapters.** Arxiv 2021.  
[[Paper](https://arxiv.org/abs/2110.04544)] [[Code](https://github.com/gaopengcuhk/CLIP-Adapter)] ![](https://img.shields.io/badge/Image--Text-blue)  
- `Tip-Adapter` **Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification.** ECCV 2022.   
[[Paper](https://arxiv.org/abs/2207.09519)] [[Code](https://github.com/gaopengcuhk/Tip-Adapter)]
- `APE` **Not All Features Matter: Enhancing Few-shot CLIP with Adaptive Prior Refinement.** ICCV 2023.   
[[Paper](https://arxiv.org/abs/2304.01195)] [[Code](https://github.com/yangyangyang127/APE)]
- `CaFo`**Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners.** CVPR 2023.   
[[Paper](https://arxiv.org/abs/2303.02151)] [[Code](https://github.com/ZrrSkywalker/CaFo)]   
- `Meta-Adapter` **Meta-Adapter: An Online Few-shot Learner for Vision-Language Model.** NeurIPS 2023.   
[[Paper](https://arxiv.org/abs/2311.03774)] [[Code](https://github.com/ArsenalCheng/Meta-Adapter)]

## Video Understanding

### Prompt Learning
- `Efficient-Prompt` **Prompting visual-language models for efficient video understanding.** ECCV 2022.  
[[Paper](https://arxiv.org/pdf/2112.04478.pdf)] [[Code](https://github.com/ju-chen/Efficient-Prompt)]
- `InTTA` **Expanding Language-Image Pretrained Models for General Video Recognition.** ECCV 2022.  
[[Paper](https://arxiv.org/pdf/2208.02816.pdf)] [[Code](https://github.com/microsoft/VideoX/tree/master/X-CLIP)]
- `RePro` **Compositional Prompt Tuning with Motion Cues for Open-vocabulary Video Relation Detection.** ICLR 2023.  
[[Paper](https://arxiv.org/pdf/2302.00268.pdf)] [[Code](https://github.com/Dawn-LX/OpenVoc-VidVRD)]

## Continual Learning

### Prompt Learning
- `L2P` **Learning to Prompt for Continual Learning.** CVPR 2022.  
[[Paper](https://arxiv.org/pdf/2112.08654)] [[Code](https://github.com/google-research/l2p)]
- `DualPrompt` **DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning.** ECCV 2022.  
[[Paper](https://arxiv.org/pdf/2204.04799)] [[Code](https://github.com/google-research/l2p)]
- `EvoPrompt` **Evolving Parameterized Prompt Memory for Continual Learning.** AAAI 2024.  
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29231)]  
- `CPrompt` **Consistent Prompting for Rehearsal-Free Continual Learning.** CVPR 2024.  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2024_paper.pdf)] [[Code](https://github.com/Zhanxin-Gao/CPrompt)]
- `DIKI` **Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models.** ECCV 2024.  
[[Paper](https://arxiv.org/pdf/2407.05342)] [[Code](https://github.com/lloongx/DIKI)]

### Adapter Learning
- `MoE-Adapters4CL` **Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters.** CVPR 2024.  
[[Paper](https://arxiv.org/pdf/2403.11549)] [[Code](https://github.com/JiazuoYu/MoE-Adapters4CL)]
- `SSIAT` **Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer.** CVPR 2024.  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_Semantically-Shifted_Incremental_Adapter-Tuning_is_A_Continual_ViTransformer_CVPR_2024_paper.pdf)]
<!--- `RAIL` **Advancing Cross-domain Discriminability in Continual Learning of Vison-Language Models.** Arxiv 2024.  
[[Paper](https://arxiv.org/pdf/2406.18868)]
- `SEMA` **Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning.** Arxiv 2024.  
[[Paper](https://arxiv.org/pdf/2403.18886)] -->

## Others

### OOD
- `LoCoOp` **LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning.** NeurIPS 2023.   
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f0606b882692637835e8ac981089eccd-Paper-Conference.pdf)] [[Code](https://github.com/AtsuMiyai/LoCoOp)]

### Point Cloud
- `PPT` **Parameter-efficient Prompt Learning for 3D Point Cloud Understanding.** ICRA 2024.    
[[Paper](https://arxiv.org/abs/2402.15823)] [[Code](https://github.com/auniquesun/PPT)]
- `Point-PRC` **Point-PRC: A Prompt Learning Based Regulation Framework for Generalizable Point Cloud Analysis.** NeurIPS 2024.   
[[Paper](https://arxiv.org/abs/2410.20406)] [[Code](https://github.com/auniquesun/Point-PRC)]

### BioMedical
- `BiomedCoOp` **BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2411.15232)] [[Code](https://github.com/HealthX-Lab/BiomedCoOp)]   

### Robot
- `PPL` **Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation.** CVPR 2025.   
[[Paper](https://www.arxiv.org/abs/2504.00420)]


