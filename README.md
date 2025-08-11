# Awesome-Prompt-Adapter-Learning-for-VLMs
A curated list of prompt/adapter learning methods for vision-language models (e.g., CLIP, ALIGN).

# Table of Contents

- [Papers](#papers)
    - [Surveys](#surveys)
    - [Foundation Models](#foundation-models)
    - [General Prompt Learning](#general-prompt-learning)
    - [Another form of Prompt](#another-form-of-prompt)
    - [General Test-time Prompt Learning](#general-test-time-prompt-learning)
    - [General Adapter Learning](#general-adapter-learning)
    - [Video Understanding](#video-understanding)
    - [Continual Learning](#continual-learning)
    - [Others](#others)

## 💡Tips:

- If you know that some papers published in top conferences (CVPR, ICCV, ECCV, ICML, NeurlPS, ICLR) or journals (TPAMI, IJCV, TIP) have not been included in this list, please feel free to contact me at any time, either by sending an email (zhengli97[at]qq.com) or submitting an issue.
- We would appreciate more people joining us in maintaining this list of papers.  
- Note that papers without open-source code are not recommended.
- We sincerely thank the following people for contributing to this list: [Lingfeng Yang](https://scholar.google.com/citations?user=RLhH0jwAAAAJ&hl=en), [Ge Wu](https://github.com/Martinser), [Jiazuo Yu](https://scholar.google.com/citations?user=FPoIE0UAAAAJ&hl=zh-CN), [Clayne](https://github.com/StellarOdys2ey)[[List](https://github.com/StellarOdys2ey/Generalizable-Prompt-Learning-for-VLMs)]


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
- `CLIP-KD` **CLIP-KD: An Empirical Study of CLIP Model Distillation.** CVPR 2024.   
  [[Paper](https://arxiv.org/abs/2307.12732)] [[Code](https://github.com/winycg/CLIP-KD)] [[论文解读](https://www.zhihu.com/question/646919153/answer/3553439426)]  
- `LongCLIP` **Long-CLIP: Unlocking the Long-Text Capability of CLIP.** ECCV 2024.  
  [[Paper](https://arxiv.org/abs/2403.15378)] [[Code](https://github.com/beichenzbc/Long-CLIP)]

## General Prompt Learning
### Experimental Comparison

Base-to-Novel Generalization. (ViT-B/16 CLIP)

| Methods    | Paper    | Pub      | Base   | Novel  | HM (main) | Code | Type |
| :---:      | :---:    | :---:    | ---    | ---    | :---:     | :--: | :--: |
| CLIP       | [Link](https://arxiv.org/abs/2103.00020) | ICML 21 | 69.34 | 74.22 | 71.70 | [Link](https://github.com/openai/CLIP)  | Model |
| CoOp       | [Link](https://arxiv.org/abs/2203.05557) | IJCV 22 | 82.69 | 63.22 | 71.66 | [Link](https://github.com/kaiyangzhou/coop)  | - |
| ATPrompt   | [Link](https://arxiv.org/abs/2412.09442) | ICCV 25 | 82.68 | 68.04 | 74.65 | [Link](https://github.com/zhengli97/ATPrompt) | - |
| ATPrompt+PromptKD | - | - | 87.05 | 81.82 | **84.35** | - | Plugin |
| CoCoOp     | [Link](https://arxiv.org/abs/2203.05557) | CVPR 22 | 80.47 | 71.69 | 75.83 | [Link](https://github.com/KaiyangZhou/CoOp)  | - |
| DPC        | [Link](https://arxiv.org/abs/2503.13443) | CVPR 25 | 85.15 | 68.84 | 76.13 | [Link](https://github.com/JREion/DPC) | - |
| DPC+PromptKD | - | - | 87.55 | 80.55 | **83.91** | - | Plugin |
| ProDA       | [Link](https://arxiv.org/abs/2205.03340) | CVPR 22 | 81.56 | 72.30 | 76.65 | [Link](https://github.com/bbbdylan/proda) | - |
| TextRefiner | [Link](https://arxiv.org/abs/2412.08176) | AAAI 25 | 79.74 | 74.32 | 76.94 | [Link](https://github.com/xjjxmu/TextRefiner) | - |
| TextRefiner+PromptKD | - | - | 85.22 | 79.64 | **82.33** | - | Plugin |
| KgCoOp     | [Link](https://arxiv.org/abs/2303.13283) | CVPR 23 | 80.73 | 73.60 | 77.00 | [Link](https://github.com/htyao89/KgCoOp) | - |
| RPO        | [Link](https://arxiv.org/abs/2308.14960) | ICCV 23 | 81.13 | 75.00 | 77.78 | [Link](https://github.com/mlvlab/RPO)  | -
| DePT       | [Link](https://arxiv.org/abs/2309.07439) | CVPR 24 | 83.80 | 72.89 | 77.97 | [Link](https://github.com/Koorye/DePT) | - |
| DePT+PromptSRC | - | - | 85.19 | 76.17 | **80.43** | - | Plugin |
| MaPLe      | [Link](https://arxiv.org/abs/2210.03117) | CVPR 23 | 82.28 | 75.14 | 78.55 | [Link](https://github.com/muzairkhattak/multimodal-prompt-learning) | - |
| QNet       | [Link](https://openreview.net/forum?id=dKlxDx2SoS) | ICLR 24  | 83.32  | 75.65  | 79.30 | [Link](https://github.com/SHIBOYA/QNet) | - |
| CasPL      | [Link](https://arxiv.org/abs/2409.17805) | ECCV 24 | 84.78 | 74.49 | 79.30 | [Link](https://github.com/megvii-research/CasPL) | - |
| CasPL+PromptSRC | - | - | 86.11  | 79.54  | **82.69** | - | Plugin |
| TCP        | [Link](https://arxiv.org/abs/2311.18231) | CVPR 24 | 84.13 | 75.36 | 79.51 | [Link](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning) | - |
| MMA        | [Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf) | CVPR 24  | 83.20 | 76.80 | 79.87 | [Link](https://github.com/ZjjConan/Multi-Modal-Adapter) | - |
| PromptSRC  | [Link](https://arxiv.org/abs/2307.06948) | ICCV 23 | 84.26 | 76.10 | 79.97 | [Link](https://github.com/muzairkhattak/PromptSRC) | - |
| 2SFS       | [Link](https://arxiv.org/abs/2503.11609) | CVPR 25 | 85.55 | 75.48 | 80.20 | [Link](https://github.com/FarinaMatteo/rethinking_fewshot_vlms) | - |
| HPT        | [Link](https://arxiv.org/abs/2312.06323) | AAAI 24 | 84.32 | 76.86 | 80.23 | [Link](https://github.com/vill-lab/2024-aaai-hpt) | - |
| CoPrompt   | [Link](https://arxiv.org/abs/2306.01195) | ICLR 24 | 84.00 | 77.23 | 80.48 | [Link](https://github.com/shuvenduroy/coprompt) | - |
| SkipT      | [Link](https://arxiv.org/abs/2412.11509) | CVPR 25 | 85.04 | 77.53 | 81.11 | [Link](https://github.com/Koorye/SkipTuning) | - | 
| MMRL       | [Link](https://arxiv.org/abs/2503.08497) | CVPR 25 | 85.68 | 77.16 | 81.20 | [Link](https://github.com/yunncheng/MMRL) | - |
| LLaMP      | [Link](https://arxiv.org/abs/2312.04076) | CVPR 24 | 85.16 | 77.71 | 81.27 | [Link](https://github.com/zhaohengz/LLaMP) | - |
| PromptKD   | [Link](https://arxiv.org/abs/2403.02781) | CVPR 24 | 86.96 | 80.73 | 83.73 | [Link](https://github.com/zhengli97/promptkd) | - |

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
[[Paper](https://arxiv.org/abs/2403.02781)] [[Code](https://github.com/zhengli97/PromptKD)] [[中文版](https://github.com/zhengli97/PromptKD/blob/main/docs/PromptKD_chinese_version.pdf)] [[论文解读](https://zhuanlan.zhihu.com/p/684269963)] [[视频解读](https://www.techbeat.net/talk-info?id=915)] ![](https://img.shields.io/badge/Image--Text-blue) 
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
- `CasPL` **Cascade Prompt Learning for Vision-Language Model Adaptation.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2409.17805)] [[Code](https://github.com/megvii-research/CasPL)] [[论文解读](https://zhuanlan.zhihu.com/p/867291664)] ![](https://img.shields.io/badge/Image--Text-blue)
- `GalLoP` **GalLoP: Learning Global and Local Prompts for Vision-Language Models.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2407.01400)] [No Code Found] ![](https://img.shields.io/badge/Image--Text-blue)   
- `AWT` **AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation.** NeurIPS 2024.  
[[Paper](https://arxiv.org/abs/2407.04603)] [[Code](https://github.com/MCG-NJU/AWT)] ![](https://img.shields.io/badge/Image--Text-blue)   
- `QNet` **Prompt Learning with Quaternion Networks.** ICLR 2024.   
[[Paper](https://openreview.net/forum?id=dKlxDx2SoS)] [[Code](https://github.com/SHIBOYA/QNet)(Empty)] ![](https://img.shields.io/badge/Image--Text-blue)
- `QMaPLe` **Quantized Prompt for Efficient Generalization of Vision-Language Models.** ECCV 2024.   
[[Paper](https://arxiv.org/abs/2407.10704)] [[Code](https://github.com/beyondhtx/QPrompt)(Empty)]

#### 2025
- `TextRefiner` **TextRefiner: Internal Visual Feature as Efficient Refiner for Vision-Language Models Prompt Tuning.** AAAI 2025.     
[[Paper](https://arxiv.org/abs/2412.08176)] [[Code](https://github.com/xjjxmu/TextRefiner)] [[论文解读](https://zhuanlan.zhihu.com/p/15940023585)] ![](https://img.shields.io/badge/Text-green)  
- `ProText` **Learning to Prompt with Text Only Supervision for Vision-Language Models.** AAAI 2025.  
[[Paper](https://arxiv.org/abs/2401.02418)] [[Code](https://github.com/muzairkhattak/ProText)] ![](https://img.shields.io/badge/Text-green)
- `FATE` **FATE: Feature-Adapted Parameter Tuning for Vision-Language Models.** AAAI 2025.  
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32975)] [No Code Found] ![](https://img.shields.io/badge/Text-green)
- `TAP` **Tree of Attributes Prompt Learning For Vision Language Models.** ICLR 2025.   
[[Paper](https://arxiv.org/abs/2410.11201)] [[Code](https://github.com/HHenryD/TAP)]![](https://img.shields.io/badge/Image--Text-blue)
- `DeKg` **Divergence-enhanced Knowledge-guided Context Optimization for Visual-Language Prompt Tuning.** ICLR 2025.   
[[Paper](https://openreview.net/pdf?id=6wOmHdwCC4)] [[Code](https://github.com/cnunlp/DeKg)]![](https://img.shields.io/badge/Text-Green)   
- `MMRL` **MMRL: Multi-Modal Representation Learning for Vision-Language Models.** CVPR 2025.  
[[Paper](https://arxiv.org/abs/2503.08497)] [[Code](https://github.com/yunncheng/MMRL)] ![](https://img.shields.io/badge/Image--Text-blue)
- `DPC` **DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2503.13443)] [[Code](https://github.com/JREion/DPC)] [[论文解读]](https://mp.weixin.qq.com/s/reOUIzVdIpNDcnX4p5ArOg) ![](https://img.shields.io/badge/Text-green)   
- `2SFS` **Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages.** CVPR 2025.    
[[Paper](https://arxiv.org/abs/2503.11609)] [[Code](https://github.com/FarinaMatteo/rethinking_fewshot_vlms)] ![](https://img.shields.io/badge/Image--Text-blue)     
- `SkipT` **Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2412.11509)] [[Code](https://github.com/Koorye/SkipTuning)] ![](https://img.shields.io/badge/Image--Text-blue)     
- `NLPrompt` **NLPrompt: Noise-Label Prompt Learning for Vision-Language Models.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2412.01256)] [[Code](https://github.com/qunovo/NLPrompt)] ![](https://img.shields.io/badge/Text-green)   
- `TAC` **Task-Aware Clustering for Prompting Vision-Language Models**. CVPR 2025.   
[[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Hao_Task-Aware_Clustering_for_Prompting_Vision-Language_Models_CVPR_2025_paper.pdf)] [[Code](https://github.com/FushengHao/TAC)] ![](https://img.shields.io/badge/Image--Text-blue)   
- `OpenworldAUC` **OpenworldAUC: Towards Unified Evaluation and Optimization for Open-world Prompt Tuning.** ICML 2025.   
[[Paper](https://arxiv.org/abs/2505.05180)] [[Code](https://github.com/huacong/OpenworldAUC)] ![](https://img.shields.io/badge/Text-green)
- `FM` **Enhancing Target-unspecific Tasks through a Features Matrix.** ICML 2025.   
[[Paper](https://arxiv.org/abs/2505.03414)] [No Code Found]![](https://img.shields.io/badge/Text-Green)   
- `SurPL` **Surrogate Prompt Learning: Towards Efficient and Diverse Prompt Learning for Vision-Language Models.** ICML 2025.   
[[Paper](https://openreview.net/pdf?id=zjG9GRG462)] [[Code](https://github.com/llcllc1997/SurPL)] ![](https://img.shields.io/badge/Image--Text-blue)   
- `ATPrompt` **Advancing Textual Prompt Learning with Anchored Attributes.** ICCV 2025.   
[[Paper](https://arxiv.org/abs/2412.09442)] [[Code](https://github.com/zhengli97/ATPrompt)] [[论文解读](https://zhuanlan.zhihu.com/p/11787739769)] [[中文版](https://github.com/zhengli97/ATPrompt/blob/main/docs/ATPrompt_chinese_version.pdf)] ![](https://img.shields.io/badge/Text-green)   
- `HicroPL` **Hierarchical Cross-modal Prompt Learning for Vision-Language Models.** ICCV 2025.   
[[Paper](https://arxiv.org/abs/2507.14976)] [[Code](https://github.com/zzeoZheng/HiCroPL)(empty)] ![](https://img.shields.io/badge/Image--Text-blue)   
- `LwEIB` **Learning with Enriched Inductive Biases for Vision-Language Models** IJCV 2025.   
[[Paper](https://link.springer.com/article/10.1007/s11263-025-02354-1)] [[Code](https://github.com/ZjjConan/VLM-LwEIB)]![](https://img.shields.io/badge/Image--Text-blue)  
- `BIP` **Bi-modality Individual-aware Prompt tuning for Visual-Language Model.** TPAMI 2025.   
[[Paper](https://ieeexplore.ieee.org/abstract/document/10949734)] [[Code](https://github.com/htyao89/BIP)] ![](https://img.shields.io/badge/Image--Text-blue)    
- `DAPT` **Decouple before Align: Visual Disentanglement Enhances Prompt Tuning.** TPAMI 2025.   
[[Paper](https://ieeexplore.ieee.org/abstract/document/11106768)] [[Code](https://github.com/Ferenas/DAPT)(empty)] ![](https://img.shields.io/badge/Image--Text-blue)     
 
## Another form of Prompt

### Paper List

- `CPT` **CPT: Colorful Prompt Tuning for pre-trained vision-language models.** Arxiv 2021.   
[[Paper](https://arxiv.org/abs/2109.11797)] [[Code](https://github.com/thunlp/CPT)] ![](https://img.shields.io/badge/Image--Text-blue)
- `DetPro` **Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model.** CVPR 2022.   
[[Paper](https://arxiv.org/abs/2203.14940)] [[Code](https://github.com/dyabel/detpro)] ![](https://img.shields.io/badge/Text-green)
- `PromptDet` **PromptDet: Towards Open-vocabulary Detection using Uncurated Images.** ECCV 2022.   
[[Paper](https://arxiv.org/abs/2203.16513)] [[Code](https://github.com/fcjian/PromptDet)] ![](https://img.shields.io/badge/Text-green)
- **Visual Prompting via Image Inpainting.** NeurIPS 2022.   
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
| DiffTPT     | ICCV 23    | 70.30    | 55.68 | 65.10 | 75.00 | 46.80 | 60.65 | [Link](https://github.com/chunmeifeng/DiffTPT) |
| TPT         | NeurIPS 22 | 68.98    | 54.77 | 63.45 | 77.06 | 47.94 | 60.81 | [Link](https://github.com/azshue/TPT) |
| TPT+CoOp    | NeurIPS 22 | 73.61    | 57.95 | 66.83 | 77.27 | 49.29 | 62.84 | [Link](https://github.com/azshue/TPT) |
| PromptAlign | NeurIPS 23 | ---      | 59.37 | 65.29 | 79.33 | 59.37 | 63.55 | [Link](https://github.com/jameelhassan/PromptAlign) |
| TPS+CoOp    | Arxiv 24   | 73.73    | 60.49 | 66.84 | 77.44 | 49.08 | 65.52 | [Link](https://github.com/elaine-sui/TPS) | 
| RLCF        | ICLR 24    | 73.23    | 65.45 | 69.77 | 83.35 | 54.74 | 68.33 | [Link](https://github.com/mzhaoshuai/RLCF) |
| RLCF+CoOp   | ICLR 24    | 76.05    | 69.74 | 70.62 | 84.51 | 56.49 | 70.34 | [Link](https://github.com/mzhaoshuai/RLCF) | 
| COSMIC      | CVPR 25    | 78.19    | 73.32 | 69.62 | 85.60 | 62.79 | 72.83 | [Link](https://github.com/hf618/COSMIC) | 

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
- `DMN` **Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models.**  CVPR 2024.   
[[Paper](https://arxiv.org/abs/2403.17589)] [[Code](https://github.com/YBZh/DMN)]    
- `C-TPT` **C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion.** ICLR 2024.   
[[Paper](https://arxiv.org/abs/2403.14119)] [[Code](https://github.com/hee-suk-yoon/C-TPT)]   
- `DynaPrompt` **DynaPrompt: Dynamic Test-Time Prompt Tuning.** ICLR 2025.   
[[Paper](https://openreview.net/forum?id=EFZEdHB3Mp)]   
- `R-TPT` **R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2504.11195)] [[Code](https://github.com/TomSheng21/R-TPT)]   
- `StatA` **Realistic Test-Time Adaptation of Vision-Language Models.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2501.03729)] [[Code](https://github.com/MaxZanella/StatA)]
- `O-TPT` **O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2503.12096)] [[Code](https://github.com/ashshaksharifdeen/O-TPT)]   
- `COSMIC` **COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation.** CVPR 2025.   
[[Paper](https://arxiv.org/abs/2503.23388)] [[Code](https://github.com/hf618/COSMIC)]   

## General Adapter Learning

### Paper List

- `CLIP-Adapter` **CLIP-Adapter: Better Vision-Language Models with Feature Adapters.** Arxiv 2021.  
[[Paper](https://arxiv.org/abs/2110.04544)] [[Code](https://github.com/gaopengcuhk/CLIP-Adapter)]
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
- `ActionCLIP` **Actionclip: A new paradigm for video action recognition.** arxiv 21.   
[[Paper](https://arxiv.org/abs/2109.08472)] [[Code](https://github.com/sallymmx/ActionCLIP)]   
- `VideoPrompt` **Prompting visual-language models for efficient video understanding.** ECCV 2022.  
[[Paper](https://arxiv.org/pdf/2112.04478.pdf)] [[Code](https://github.com/ju-chen/Efficient-Prompt)]
- `InTTA` **Expanding Language-Image Pretrained Models for General Video Recognition.** ECCV 2022.  
[[Paper](https://arxiv.org/pdf/2208.02816.pdf)] [[Code](https://github.com/microsoft/VideoX/tree/master/X-CLIP)]
- `RePro` **Compositional Prompt Tuning with Motion Cues for Open-vocabulary Video Relation Detection.** ICLR 2023.   
[[Paper](https://arxiv.org/pdf/2302.00268.pdf)] [[Code](https://github.com/Dawn-LX/OpenVoc-VidVRD)]   
- `Vita-CLIP` **Vita-CLIP: Video and text adaptive CLIP via Multimodal Prompting.** CVPR 2023.   
[[Paper](https://arxiv.org/abs/2304.03307)] [[Code](https://github.com/TalalWasim/Vita-CLIP)]   
- `ViFi-CLIP` **Fine-tuned CLIP Models are Efficient Video Learners.** CVPR 2023.   
[[Paper](https://arxiv.org/abs/2212.03640)] [[Code](https://github.com/muzairkhattak/ViFi-CLIP)]   
- `OpenVCLIP` **Open-VCLIP: Transforming CLIP to an Open-vocabulary Video Model via Interpolated Weight Optimization.** ICML 2023.   
[[Paper](https://arxiv.org/abs/2302.00624)] [[Code](https://github.com/wengzejia1/Open-VCLIP)]    
- `M2-CLIP` **M2-CLIP: A Multimodal, Multi-task Adapting Framework for Video Action Recognition.** AAAI 2024.   
[[Paper](https://arxiv.org/abs/2401.11649)] [[Code](https://github.com/sallymmx/m2clip)]   
- `ViLT-CLIP` **ViLT-CLIP: Video and Language Tuning CLIP with Multimodal Prompt Learning and Scenario-Guided Optimization.** AAAI 2024.   
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28347)] [Code(None)]   
- `FROSTER` **FROSTER: Frozen CLIP Is A Strong Teacher for Open-Vocabulary Action Recognition.** ICLR 2024.   
[[Paper](https://arxiv.org/abs/2402.03241)] [[Code](https://github.com/Visual-AI/FROSTER)]

### Adapter Learning
- `BT-Adapter` **BT-Adapter: Video Conversation is Feasible Without Video Instruction Tuning.** CVPR 2024.   
[[Paper](https://arxiv.org/abs/2309.15785)] [[Code](https://github.com/farewellthree/BT-Adapter)]   


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
[[Paper](https://arxiv.org/abs/2403.19979)] [[Code](https://github.com/HAIV-Lab/SSIAT)]   
<!--- `RAIL` **Advancing Cross-domain Discriminability in Continual Learning of Vison-Language Models.** Arxiv 2024.  
[[Paper](https://arxiv.org/pdf/2406.18868)]
- `SEMA` **Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning.** Arxiv 2024.  
[[Paper](https://arxiv.org/pdf/2403.18886)] -->

## Others

### OOD
- `LoCoOp` **LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning.** NeurIPS 2023.   
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f0606b882692637835e8ac981089eccd-Paper-Conference.pdf)] [[Code](https://github.com/AtsuMiyai/LoCoOp)]
- `DeCoOp` **DeCoOp: Robust Prompt Tuning with Out-of-Distribution Detection.** ICML 2024.   
[[Paper](https://arxiv.org/abs/2406.00345)] [[Code](https://github.com/WNJXYK/DeCoOp)]   

### Point Cloud
- `IDPT` **Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models.** ICCV 2023.
[[Paper](https://arxiv.org/abs/2304.07221)] [[Code](https://github.com/zyh16143998882/ICCV23-IDPT)]   
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

### Retrival
- `CLIP4clip` **Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning.** Neurocomputing 2022.   
[[Paper](https://arxiv.org/abs/2104.08860)] [[Code](https://github.com/ArrowLuo/CLIP4Clip)]   
- `VoP` **VoP: Text-Video Co-Operative Prompt Tuning for Cross-Modal Retrieval.** CVPR 2023.   
[[Paper](https://arxiv.org/abs/2211.12764)] [[Code](https://github.com/bighuang624/VoP)]   
- `DGL` **DGL: Dynamic Global-Local Prompt Tuning for Text-Video Retrieval.** AAAI 2024.   
[[Paper](https://arxiv.org/abs/2401.10588)] [[Code](https://github.com/knightyxp/DGL)]
