# Awesome-Prompt-Learning-for-VLMs
A curated list of prompt learning methods for vision-language models.

# Table of Contents

- [Papers](#papers)
    - [Surveys](#surveys)
    - [Prompt Learning](#prompt-learning)
    - [Test-time Prompt Tuning](#test-time-prompt-tuning)


# Papers

## Surveys


## Prompt Learning

### Experimental Comparison

Base-to-Novel Generalization

| Methods    | Pub      | Base   | Novel  | HM (main)     | Code |
| ---        | ---      | ---    | ---    | :---:  | ---  |
| CLIP       |          | 69.34  | 74.22  | 71.70  | [Link](https://github.com/openai/CLIP)  |
| CoOp       | IJCV 22  | 82.69  | 63.22  | 71.66  | [Link](https://github.com/kaiyangzhou/coop)  |
| CoCoOp     | CVPR 22  | 80.47  | 71.69  | 75.83  | [Link](https://github.com/KaiyangZhou/CoOp)  |
| ProDA      | CVPR 22  | 81.56  | 72.30  | 76.65  | [Link](https://github.com/bbbdylan/proda) |
| RPO        | ICCV 23  | 81.13  | 75.00  | 77.78  | [Link](https://github.com/mlvlab/RPO)  |
| MaPLe      | CVPR 23  | 82.28  | 75.14  | 78.55  | [Link](https://github.com/muzairkhattak/multimodal-prompt-learning)  |
| MetaPrompt | TIP 24   | 83.65  | 75.48  | 79.09  | ---  |
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

- `CoOp` **Learning to Prompt for Vision-Language Models.** IJCV 2022. [[Paper](https://arxiv.org/abs/2203.05557)] [[Code](https://github.com/KaiyangZhou/CoOp)]
- `CoCoOp` **Conditional Prompt Learning for Vision-Language Models.** CVPR 2022. [[Paper](https://arxiv.org/abs/2203.05557)] [[Code](https://github.com/KaiyangZhou/CoOp)]
- `ProDA` **Prompt Distribution Learning.** CVPR 2022. [[Paper](https://arxiv.org/abs/2205.03340)] [[Code](https://github.com/bbbdylan/proda)]
- `MaPLe` **MaPLe: Multi-modal Prompt Learning.** CVPR 2023. [[Paper](https://arxiv.org/abs/2210.03117)] [[Code]()]
- `RPO` **Read-only Prompt Optimization for Vision-Language Few-shot Learning.** ICCV 2023. [[Paper](https://arxiv.org/abs/2308.14960)] [[Code](https://github.com/mlvlab/rpo)]
- `PromptSRC` **Self-regulating Prompts: Foundational Model Adaptation without Forgetting.** ICCV 2023. [[Paper](https://openaccess.thecvf.com//content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf)] [[Code](https://github.com/muzairkhattak/PromptSRC)]
- `MetaPrompt` **Learning Domain Invariant Prompt for Vision-Language Models.** TIP 2024. [[Paper](https://arxiv.org/abs/2212.04196)]
- `HPT` **Learning Hierarchical Prompt with Structured Linguistic Knowledge for Vision-Language Models.** AAAI 2024. [[Paper](https://arxiv.org/abs/2312.06323)] [[Code](https://github.com/Vill-Lab/2024-AAAI-HPT)]
- `CoPrompt` **Consistency-guided Prompt Learning for Vision-Language Models.** ICLR 2024. [[Paper](https://arxiv.org/abs/2306.01195)] [[Code](https://github.com/ShuvenduRoy/CoPrompt)]
- `PromptKD` **Unsupervised Prompt Distillation for Vision Language Models.** CVPR 2024. [[Paper](https://arxiv.org/abs/2403.02781)] [[Code](https://github.com/zhengli97/PromptKD)]



## Test-time Prompt Tuning





