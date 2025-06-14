# Paper Reading List

> (Now) A personal paper / blog reading list of Kinnari. Welcome to contribute together! See [template](./template.md) for more format requirements.

## 2025-06-06

### Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling

- **Tags**: dLM
- **Authors**: Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, Qinsheng Zhang
- **Abstract**:
  <details>
  Masked diffusion models (MDMs) have emerged as a popular research topic for generative modeling of discrete data, thanks to their superior performance over other discrete diffusion models, and are rivaling the auto-regressive models (ARMs) for language modeling tasks. The recent effort in simplifying the masked diffusion framework further leads to alignment with continuous-space diffusion models and more principled training and sampling recipes. In this paper, however, we reveal that both training and sampling of MDMs are theoretically free from the time variable, arguably the key signature of diffusion models, and are instead equivalent to masked models. The connection on the sampling aspect is drawn by our proposed first-hitting sampler (FHS). Specifically, we show that the FHS is theoretically equivalent to MDMs' original generation process while significantly alleviating the time-consuming categorical sampling and achieving a 20 speedup. In addition, our investigation raises doubts about whether MDMs can truly beat ARMs in text generation. We identify, for the first time, an underlying numerical issue, even with the commonly used 32-bit floating-point precision, which results in inaccurate categorical sampling. We show that it lowers the effective temperature both theoretically and empirically, and the resulting decrease in token diversity makes previous evaluations, which assess the generation quality solely through the incomplete generative perplexity metric, somewhat unfair.
  </details>
- **Links**:
  - arxiv: http://arxiv.org/abs/2409.02908
- **Progress**: not started

### When Do LLMs Help With Node Classification? A Comprehensive Analysis

- **Tags**: LLM, Graph NN
- **Authors**: Xixi Wu, Yifei Shen, Fangzhou Ge, Caihua Shan, Yizhu Jiao, Xiangguo Sun, Hong Cheng
- **Abstract**:
  <details>
  Node classification is a fundamental task in graph analysis, with broad applications across various fields. Recent breakthroughs in Large Language Models (LLMs) have enabled LLM-based approaches for this task. Although many studies demonstrate the impressive performance of LLM-based methods, the lack of clear design guidelines may hinder their practical application. In this work, we aim to establish such guidelines through a fair and systematic comparison of these algorithms. As a first step, we developed LLMNodeBed, a comprehensive codebase and testbed for node classification using LLMs. It includes 10 homophilic datasets, 4 heterophilic datasets, 8 LLM-based algorithms, 8 classic baselines, and 3 learning paradigms. Subsequently, we conducted extensive experiments, training and evaluating over 2,700 models, to determine the key settings (e.g., learning paradigms and homophily) and components (e.g., model size and prompt) that affect performance. Our findings uncover 8 insights, e.g., (1) LLM-based methods can significantly outperform traditional methods in a semi-supervised setting, while the advantage is marginal in a supervised setting; (2) Graph Foundation Models can beat open-source LLMs but still fall short of strong LLMs like GPT-4o in a zero-shot setting. We hope that the release of LLMNodeBed, along with our insights, will facilitate reproducible research and inspire future studies in this field. Codes and datasets are released at \href{this https URL}{\texttt{this https URL}}.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2502.00829
  - HuggingFace: https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/xxwu/LLMNodeBed
  - GitHub: https://link.zhihu.com/?target=https%3A//github.com/WxxShirley/LLMNodeBed
- **Progress**: not started

## 2025-05-31

### ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT

- **Tags**: IR, LLM, BERT, Efficient Retrieval
- **Authors**: Omar Khattab, Matei Zaharia
- **Abstract**:
  <details>
  Recent progress in Natural Language Understanding (NLU) is driving fast-paced advances in Information Retrieval (IR), largely owed to fine-tuning deep language models (LMs) for document ranking. While remarkably effective, the ranking models based on these LMs increase computational cost by orders of magnitude over prior approaches, particularly as they must feed each query-document pair through a massive neural network to compute a single relevance score. To tackle this, we present ColBERT, a novel ranking model that adapts deep LMs (in particular, BERT) for efficient retrieval. ColBERT introduces a late interaction architecture that independently encodes the query and the document using BERT and then employs a cheap yet powerful interaction step that models their fine-grained similarity. By delaying and yet retaining this fine-granular interaction, ColBERT can leverage the expressiveness of deep LMs while simultaneously gaining the ability to pre-compute document representations offline, considerably speeding up query processing. Beyond reducing the cost of re-ranking the documents retrieved by a traditional model, ColBERT's pruning-friendly interaction mechanism enables leveraging vector-similarity indexes for end-to-end retrieval directly from a large document collection. We extensively evaluate ColBERT using two recent passage search datasets. Results show that ColBERT's effectiveness is competitive with existing BERT-based models (and outperforms every non-BERT baseline), while executing two orders-of-magnitude faster and requiring four orders-of-magnitude fewer FLOPs per query.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2004.12832
- **Progress**: not started

### Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

- **Tags**: Diffusion LLM, Parallel Decoding, KV Cache, Efficiency
- **Authors**: Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie
- **Abstract**:
  <details>
  Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6 throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2505.22618v1
  - GitHub: https://github.com/NVlabs/Fast-dLLM
- **Progress**: not started

## 2025-05-26

### One RL to See Them All: Visual Triple Unified Reinforcement Learning

- **Tags**: RL Finetune, VLM
- **Authors**: Yan Ma, Linge Du, Xuyang Shen, Shaoxiang Chen, Pengfei Li, Qibing Ren, Lizhuang Ma, Yuchao Dai, Pengfei Liu, Junjie Yan
- **Abstract**:
  <details>
  Reinforcement learning (RL) has significantly advanced the reasoning capabilities of vision-language models (VLMs). However, the use of RL beyond reasoning tasks remains largely unexplored, especially for perceptionintensive tasks like object detection and grounding. We propose V-Triune, a Visual Triple Unified Reinforcement Learning system that enables VLMs to jointly learn visual reasoning and perception tasks within a single training pipeline. V-Triune comprises triple complementary components: Sample-Level Data Formatting (to unify diverse task inputs), Verifier-Level Reward Computation (to deliver custom rewards via specialized verifiers) , and Source-Level Metric Monitoring (to diagnose problems at the data-source level). We further introduce a novel Dynamic IoU reward, which provides adaptive, progressive, and definite feedback for perception tasks handled by V-Triune. Our approach is instantiated within off-the-shelf RL training framework using open-source 7B and 32B backbone models. The resulting model, dubbed Orsta (One RL to See Them All), demonstrates consistent improvements across both reasoning and perception tasks. This broad capability is significantly shaped by its training on a diverse dataset, constructed around four representative visual reasoning tasks (Math, Puzzle, Chart, and Science) and four visual perception tasks (Grounding, Detection, Counting, and OCR). Subsequently, Orsta achieves substantial gains on MEGA-Bench Core, with improvements ranging from +2.1 to an impressive +14.1 across its various 7B and 32B model variants, with performance benefits extending to a wide range of downstream tasks. These results highlight the effectiveness and scalability of our unified RL approach for VLMs. The V-Triune system, along with the Orsta models, is publicly available at https://github.com/MiniMax-AI.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2505.18129
  - HuggingFace: https://huggingface.co/papers/2505.18129
  - GitHub: https://github.com/MiniMax-AI/One-RL-to-See-Them-All
- **Progress**: not started

### Scaling Llama 3 Training with Efficient Parallelism Strategies

- **Tags**: LlaMa, Distributed Training, Parallelism, MLsys
- **Authors**: Weiwei Chu, Xinfeng Xie, Jiecao Yu, Jie Wang, Amar Phanishayee, Chunqiang Tang, Yuchen Hao, Jianyu Huang, Mustafa Ozdal, Jun Wang, Vedanuj Goswami, Naman Goyal, Abhishek Kadian, Andrew Gu, Chris Cai, Feng Tian, Xiaodong Wang, Min Si, Pavan Balaji, Ching-Hsiang Chu, and Jongsoo Park, Meta Platforms, Inc.
- **Abstract**:
  <details>
  Llama is a widely used open-source large language model. This paper presents the design and implementation of the parallelism techniques used in Llama 3 pre-training. To achieve efficient training on tens of thousands of GPUs, Llama 3 employs a combination of four-dimensional parallelism: fully sharded data parallelism, tensor parallelism, pipeline parallelism, and context parallelism. Beyond achieving efficiency through parallelism and model co-design, we also address other equally critical aspects. First, we enhance flexibility—for example, through novel pipeline parallelism that supports evolving batch sizes and heterogeneous model architectures, and innovative context parallelism that enables model innovations such as document-mask attention. Second, we prioritize practicality—for example, by enabling the diagnosis of performance and numerical issues at scale. Finally, drawing on our experience with large-scale training, we provide recommendations for future hardware design.
  </details>
- **Links**: https://aisystemcodesign.github.io/papers/Llama3-ISCA25.pdf
- **Progress**: not started

## 2025-05-21

### SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs

- **Tags**: LLM Reasoning, CoT
- **Authors**:
- **Abstract**:
  <details>
  Chain-of-Thought (CoT) reasoning enables Large Language Models (LLMs) to solve complex reasoning tasks by generating intermediate reasoning steps. However, most existing approaches focus on hard token decoding, which constrains reasoning within the discrete vocabulary space and may not always be optimal. While recent efforts explore continuous-space reasoning, they often suffer from catastrophic forgetting, limiting their applicability to state-of-the-art LLMs that already perform well in zero-shot settings with a proper instruction. To address this challenge, we propose a novel approach for continuous-space reasoning that does not require modifying the underlying LLM. Specifically, we employ a lightweight assistant model to generate instance-specific soft thought tokens speculatively as the initial chain of thoughts, which are then mapped into the LLM's representation space via a projection module. Experimental results on five reasoning benchmarks demonstrate that our method enhances LLM reasoning performance through supervised, parameter-efficient fine-tuning.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2502.12134
- **Progress**: not started

## 2025-05-16

### Monarch: Expressive Structured Matrices for Efficient and Accurate Training

- **Tags**: Finetune, MLsys, Efficiency, Sparse Training
- **Authors**: Tri Dao, Beidi Chen, Nimit Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, Christopher Ré
- **Abstract**:
  <details>
  Large neural networks excel in many domains, but they are expensive to train and fine-tune. A popular approach to reduce their compute or memory requirements is to replace dense weight matrices with structured ones (e.g., sparse, low-rank, Fourier transform). These methods have not seen widespread adoption (1) in end-to-end training due to unfavorable efficiency--quality tradeoffs, and (2) in dense-to-sparse fine-tuning due to lack of tractable algorithms to approximate a given dense weight matrix. To address these issues, we propose a class of matrices (Monarch) that is hardware-efficient (they are parameterized as products of two block-diagonal matrices for better hardware utilization) and expressive (they can represent many commonly used transforms). Surprisingly, the problem of approximating a dense weight matrix with a Monarch matrix, though nonconvex, has an analytical optimal solution. These properties of Monarch matrices unlock new ways to train and fine-tune sparse and dense models. We empirically validate that Monarch can achieve favorable accuracy-efficiency tradeoffs in several end-to-end sparse training applications: speeding up ViT and GPT-2 training on ImageNet classification and Wikitext-103 language modeling by 2x with comparable model quality, and reducing the error on PDE solving and MRI reconstruction tasks by 40%. In sparse-to-dense training, with a simple technique called "reverse sparsification," Monarch matrices serve as a useful intermediate representation to speed up GPT-2 pretraining on OpenWebText by 2x without quality drop. The same technique brings 23% faster BERT pretraining than even the very optimized implementation from Nvidia that set the MLPerf 1.1 record. In dense-to-sparse fine-tuning, as a proof-of-concept, our Monarch approximation algorithm speeds up BERT fine-tuning on GLUE by 1.7x with comparable accuracy.
  </details>
- **Links**:
  - arxiv: https://arxiv.org/abs/2204.00595
- **Progress**: not started

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
- **Progress**: not started

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
