# Paper Reading List

> (Now) A personal paper / blog reading list of Kinnari. Welcome to contribute together! See [template](./template.md) for more format requirements.

## 2025-05-15

### Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl

- **Tags**: LLM Reasoning, RL Finetune
- **Authors**: Michael Luo*, Sijun Tan*, Justin Wong†, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo
- **TL;DR**:
  <details>
  RL magic is in the air! We introduce DeepScaleR-1.5B-Preview, a language model finetuned from Deepseek-R1-Distilled-Qwen-1.5B using simple reinforcement learning (RL). It achieves an impressive 43.1% Pass@1 accuracy on AIME2024 (+14.3% improvement over the base model), surpassing the performance of OpenAI’s o1-preview with just 1.5B parameters. We open sourced our dataset, code and training logs for everyone to progress on scaling intelligence with RL.
  </details>
- **Links**:
  - HuggingFace: https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview
  - GitHub: https://github.com/agentica-project/rllm
  - WebSite: https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2
- **Progress**: not started / reading / done
- **Notes**: (if any)

### FastCuRL: Curriculum Reinforcement Learning with Progressive Context Extension for Efficient Training R1-like Reasoning Models

- **Tags**: Curriculum Learning, RL Finetune, LLM, 
- **Authors**: Mingyang Song, Mao Zheng, Zheng Li, Wenjie Yang, Xuan Luo, Yue Pan, Feng Zhang
- **Abstract**:
  <details>
  Improving the training efficiency remains one of the most significant challenges in large-scale reinforcement learning. In this paper, we investigate how the model's context length and the complexity of the training dataset influence the training process of R1-like models. Our experiments reveal three key insights: (1) adopting longer context lengths may not necessarily result in better performance; (2) selecting an appropriate context length helps mitigate entropy collapse; and (3) appropriately controlling the model's context length and curating training data based on input prompt length can effectively improve RL training efficiency, achieving better performance with shorter thinking length. Inspired by these insights, we propose FastCuRL, a curriculum reinforcement learning framework with the progressive context extension strategy, and successfully accelerate the training process of RL models. Experimental results demonstrate that FastCuRL-1.5B-Preview surpasses DeepScaleR-1.5B-Preview across all five benchmarks while only utilizing 50\% of training steps. Furthermore, all training stages for FastCuRL-1.5B-Preview are completed using a single node with 8 GPUs.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2503.17287
  - HuggingFace: https://huggingface.co/papers/2503.17287
  - GitHub: https://github.com/nick7nlp/FastCuRL
- **Progress**: not started

### Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond

- **Tags**: Curriculum Learning, LLM, RL Finetune, CoT
- **Authors**: Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, Xiangzheng Zhang
- **Abstract**:
  <details>
  This paper introduces Light-R1, an open-source suite for training long reasoning models using reproducible and cost-effective methodology. Given the proprietary nature of data used in the DeepSeek-R1 series, we develop an alternative approach leveraging exclusively public data and models. Our curriculum training progressively increases data difficulty, combined with multi-staged post-training. Our Light-R1-32B model, trained from Qwen2.5-32B-Instruct, outperforms DeepSeek-R1-Distill-Qwen-32B in math reasoning.
  Experimental results show that this curriculum approach becomes more effective when distinct, diverse datasets are available for different training stages: fine-tuning DeepSeek-R1-Distilled models (pre-tuned by DeepSeek team on proprietary data) with 3,000 challenging examples from our curriculum dataset yielded state-of-the-art 7B and 14B models, while the 32B model, Light-R1-32B-DS performed comparably to QwQ-32B and DeepSeek-R1.
  Furthermore, we extend our work by applying GRPO on long reasoning models. Our final Light-R1-14B-DS achieves SOTA performance among 14B models in math, with AIME24 \& 25 scores of 74.0 and 60.2 respectively, surpassing many 32B models and DeepSeek-R1-Distill-Llama-70B. Despite math-focused training, Light-R1-14B-DS demonstrates strong cross-domain generalization.
  Light-R1 represents a significant advancement in making sophisticated reasoning models more accessible and implementable in real-world applications. Our models, training data and code have been made available at [this https URL](https://github.com/Qihoo360/Light-R1).
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2503.10460
  - GitHub: https://github.com/Qihoo360/Light-R1
- **Progress**: not started

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale

- **Tags**: LLM Reasoning, RL Finetune
- **Authors**: ByteDance Seed, Institute for AI Industry Research (AIR), Tsinghua University, The University of Hong Kong, SIA-Lab of Tsinghua AIR and ByteDance Seed
- **Abstract**:
  <details>
  Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results. We propose the \textbf{D}ecoupled Clip and \textbf{D}ynamic s\textbf{A}mpling \textbf{P}olicy \textbf{O}ptimization (\textbf{DAPO}) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model. Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2503.14476
  - Project Homepage: https://dapo-sia.github.io/
  - GitHub: https://github.com/BytedTsinghua-SIA/DAPO
  - HuggingFace: https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B
- **Progress**: reading

### Llama-Nemotron: Efficient Reasoning Models

- **Tags**: LLM Reasoning, RL Finetune, Curriculum Learning, Test Time Scaling, NAS
- **Authors**: NVIDIA
- **Abstract**:
  <details>
  We introduce the Llama-Nemotron series of models, an open family of heterogeneous reasoning models that deliver exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use. The family comes in three sizes -- Nano (8B), Super (49B), and Ultra (253B) -- and performs competitively with state-of-the-art reasoning models such as DeepSeek-R1 while offering superior inference throughput and memory efficiency. In this report, we discuss the training procedure for these models, which entails using neural architecture search from Llama 3 models for accelerated inference, knowledge distillation, and continued pretraining, followed by a reasoning-focused post-training stage consisting of two main parts: supervised fine-tuning and large scale reinforcement learning. Llama-Nemotron models are the first open-source models to support a dynamic reasoning toggle, allowing users to switch between standard chat and reasoning modes during inference. To further support open research and facilitate model development, we provide the following resources: 1. We release the Llama-Nemotron reasoning models -- LN-Nano, LN-Super, and LN-Ultra -- under the commercially permissive NVIDIA Open Model License Agreement. 2. We release the complete post-training dataset: Llama-Nemotron-Post-Training-Dataset. 3. We also release our training codebases: NeMo, NeMo-Aligner, and Megatron-LM.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2505.00949
  - HuggingFace: https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
- **Progress**: done
- **Note**: not open-source code now, only know that they use [NeMo-RL](https://github.com/NVIDIA/NeMo-RL) for training.

### Why We Think

- **Tags**: LLM Reasoning, CoT, RL, Scaling Law, Overview, Test Time Scaling
- **Author(s)**: Lilian Weng
- **Abstract**:
  <details>
  This article explores the effective use of computing resources (i.e., "thinking time") during test time and why it improves model performance. By introducing techniques such as Chain-of-Thought (CoT), the model is able to think for longer periods of time, similar to the slow thinking mode of humans, which significantly improves the ability to solve complex problems. The article reviews the latest advances in test-time computing, including parallel sampling, sequence revision, the application of reinforcement learning, and the use of external tools, and discusses how to evaluate and improve the model's thinking process, as well as the expansion rules and future research directions of test-time computing. (by Gemini 2.5 Flash)
  </details>
- **Links**: https://lilianweng.github.io/posts/2025-05-01-thinking/
- **Progress**: reading

### Parallel Scaling Law for Language Models

- **Tags**: LLM, Scaling Law, Parallelism
- **Authors**: Mouxiang Chen, Binyuan Hui, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Jianling Sun, Junyang Lin, Zhongxin Liu
- **Abstract**:
  <details>It is commonly believed that scaling language models should commit a significant space or time cost, by increasing the parameters (parameter scaling) or output tokens (inference-time scaling). We introduce the third and more inference-efficient scaling paradigm: increasing the model's parallel computation during both training and inference time. We apply $P$ diverse and learnable transformations to the input, execute forward passes of the model in parallel, and dynamically aggregate the outputs. This method, namely parallel scaling (ParScale), scales parallel computation by reusing existing parameters and can be applied to any model structure, optimization procedure, data, or task. We theoretically propose a new scaling law and validate it through large-scale pre-training, which shows that a model with parallel streams is similar to scaling the parameters by while showing superior inference efficiency. For example, ParScale can use up to 22 less memory increase and 6 less latency increase compared to parameter scaling that achieves the same performance improvement. It can also recycle an off-the-shelf pre-trained model into a parallelly scaled one by post-training on a small amount of tokens, further reducing the training budget. The new scaling law we discovered potentially facilitates the deployment of more powerful models in low-resource scenarios, and provides an alternative perspective for the role of computation in machine learning. 
  </details>
- **Links**: https://arxiv.org/abs/2505.10475
- **Progress**: not started

### Efficient Reinforcement Finetuning via Adaptive Curriculum Learning

- **Tags**: RL Finetune, LLM Reasoning, Dynamic Curriculum Learning
- **Authors**: Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, Jieyu Zhao
- **Abstract**:
  <details>Reinforcement finetuning (RFT) has shown great potential for enhancing the mathematical reasoning capabilities of large language models (LLMs), but it is often sample- and compute-inefficient, requiring extensive training. In this work, we introduce AdaRFT (Adaptive Curriculum Reinforcement Finetuning), a method that significantly improves both the efficiency and final accuracy of RFT through adaptive curriculum learning. AdaRFT dynamically adjusts the difficulty of training problems based on the model's recent reward signals, ensuring that the model consistently trains on tasks that are challenging but solvable. This adaptive sampling strategy accelerates learning by maintaining an optimal difficulty range, avoiding wasted computation on problems that are too easy or too hard. AdaRFT requires only a lightweight extension to standard RFT algorithms like Proximal Policy Optimization (PPO), without modifying the reward function or model architecture. Experiments on competition-level math datasets-including AMC, AIME, and IMO-style problems-demonstrate that AdaRFT significantly improves both training efficiency and reasoning performance. We evaluate AdaRFT across multiple data distributions and model sizes, showing that it reduces training time by up to 2x and improves accuracy by a considerable margin, offering a more scalable and effective RFT framework.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2504.05520
  - GitHub: https://github.com/limenlp/verl
- **Progress**: reading
- **Notes**:
  <details>
  - precomputed difficulty score
  </details>

### Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation

- **Tags**: LLM Reasoning, Dynamic Curriculum Learning, RL Finetune
- **Authors**: Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, Qianchun Lu
- **Abstract**:
  <details>
  Despite impressive progress in areas like mathematical reasoning, large language models still face significant challenges in consistently solving complex problems. Drawing inspiration from key human learning strategies, we propose two novel strategies to enhance the capability of large language models to solve these complex problems. First, Adaptive Difficulty Curriculum Learning (ADCL) is a novel curriculum learning strategy that tackles the Difficulty Shift phenomenon (i.e., a model's perception of problem difficulty dynamically changes during training) by periodically re-estimating difficulty within upcoming data batches to maintain alignment with the model's evolving capabilities. Second, Expert-Guided Self-Reformulation (EGSR) is a novel reinforcement learning strategy that bridges the gap between imitation learning and pure exploration by guiding models to reformulate expert solutions within their own conceptual framework, rather than relying on direct imitation, fostering deeper understanding and knowledge assimilation. Extensive experiments on challenging mathematical reasoning benchmarks, using Qwen2.5-7B as the base model, demonstrate that these human-inspired strategies synergistically and significantly enhance performance. Notably, their combined application improves performance over the standard Zero-RL baseline by 10% on the AIME24 benchmark and 16.6% on AIME25.
  </details>
- **Links**: https://arxiv.org/abs/2505.08364
- **Progress**: not started