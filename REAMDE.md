# Paper Reading List

> (Now) A personal paper / blog reading list of Kinnari. Welcome to contribute together! See [template](./template.md) for more format requirement.

## 2025-05-15

###

- **Tags**: LLM Reasoning, CoT, RL, Scaling Law, Overview, Test Time Scaling
- **Author(s)**: Lilian Weng
- **Abstract**:
  <details>
  This article explores the effective use of computing resources (i.e., "thinking time") during test time and why it improves model performance. By introducing techniques such as Chain-of-Thought (CoT), the model is able to think for longer periods of time, similar to the slow thinking mode of humans, which significantly improves the ability to solve complex problems. The article reviews the latest advances in test-time computing, including parallel sampling, sequence revision, the application of reinforcement learning, and the use of external tools, and discusses how to evaluate and improve the model's thinking process, as well as the expansion rules and future research directions of test-time computing. (by Gemini 2.5 Flash)
  </details>
- **Links**: https://lilianweng.github.io/posts/2025-05-01-thinking/
- **Progress**: reading
- **Notes**: (if any)

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
- **Links**: https://arxiv.org/abs/2504.05520, https://github.com/limenlp/verl
- **Progress**: not started

### Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation

- **Tags**: LLM Reasoning, Dynamic Curriculum Learning, RL Finetune
- **Authors**: Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, Qianchun Lu
- **Abstract**:
  <details>
  Despite impressive progress in areas like mathematical reasoning, large language models still face significant challenges in consistently solving complex problems. Drawing inspiration from key human learning strategies, we propose two novel strategies to enhance the capability of large language models to solve these complex problems. First, Adaptive Difficulty Curriculum Learning (ADCL) is a novel curriculum learning strategy that tackles the Difficulty Shift phenomenon (i.e., a model's perception of problem difficulty dynamically changes during training) by periodically re-estimating difficulty within upcoming data batches to maintain alignment with the model's evolving capabilities. Second, Expert-Guided Self-Reformulation (EGSR) is a novel reinforcement learning strategy that bridges the gap between imitation learning and pure exploration by guiding models to reformulate expert solutions within their own conceptual framework, rather than relying on direct imitation, fostering deeper understanding and knowledge assimilation. Extensive experiments on challenging mathematical reasoning benchmarks, using Qwen2.5-7B as the base model, demonstrate that these human-inspired strategies synergistically and significantly enhance performance. Notably, their combined application improves performance over the standard Zero-RL baseline by 10% on the AIME24 benchmark and 16.6% on AIME25.
  </details>
- **Links**: https://arxiv.org/abs/2505.08364
- **Progress**: not started