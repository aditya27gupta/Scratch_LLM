# Learning Path

A structured guide through all 44 Scratch LLM implementations from Github Repo [No Magic AI](https://github.com/no-magic-ai/no-magic/blob/main/README.md) using numpy.

### Milestone 1: Text Representation (1.5 hrs)

The foundation — how raw text becomes numbers a model can process.

| #   | Script                                         | Time   | Checkbox |
| --- | ---------------------------------------------- | ------ | -------- |
| 1   | [Tokenizer](src/foundations/microtokenizer.py) | 30 min | [x]      |
| 2   | [Embedding](src/foundations/microembedding.py) | 40 min | [x]      |

### Milestone 2: Training Fundamentals (1.5 hrs)

Core optimization and regularization techniques that every neural network uses.

| #   | Script                                                 | Time   | Checkbox |
| --- | ------------------------------------------------------ | ------ | -------- |
| 3   | [Optimizer](src/foundations/microoptimizer.py)         | 45 min | [x]      |
| 4   | [Batch Normalization](src/alignment/microbatchnorm.py) | 25 min | [x]      |
| 5   | [DropOut](src/alignment/microdropout.py)               | 25 min | [ ]      |

### Milestone 3: Sequence Models (2 hrs)

From recurrence to attention — the architectural evolution that produced modern LLMs.

| #   | Script                             | Time   | Checkbox |
| --- | ---------------------------------- | ------ | -------- |
| 6   | [RNN](src/foundations/micrornn.py) | 45 min |  [ ]    |
| 7   | [GPT](src/foundations/microgpt.py) | 60 min | [ ]    |

### Milestone 4: Transformer Variants (1.5 hrs)

Bidirectional models and convolution — alternative architectures built on the same principles.

| #   | Script                                      | Time   | Checkbox |
| --- | ------------------------------------------- | ------ | -------- |
| 8   | [BERT](src/foundations/microbert.py)        | 45 min |  [ ]    |
| 9   | [Convolution](src/foundations/microconv.py) | 45 min |  [ ]    |

### Milestone 5: Retrieval & Grounding (1 hr)

Connecting language models to external knowledge stores.

| #   | Script                             | Time   | Checkbox |
| --- | ---------------------------------- | ------ | -------- |
| 10  | [RAG](src/foundations/microrag.py) | 50 min |  [ ]    |

### Milestone 6: Generative Models (2.5 hrs)

Three paradigms for generating new data — reconstruction, adversarial, and denoising.

| #   | Script                                         | Time   | Checkbox |
| --- | ---------------------------------------------- | ------ | -------- |
| 11  | [VAE](src/foundations/microvae.py)             | 50 min |  [ ]    |
| 12  | [GAN](src/foundations/microgan.py)             | 50 min |  [ ]    |
| 13  | [Diffusion](src/foundations/microdiffusion.py) | 60 min |  [ ]    |

### Milestone 7: Parameter-Efficient Fine-Tuning (1 hr)

Adapting large models without retraining all parameters.

| #   | Script                               | Time   | Checkbox |
| --- | ------------------------------------ | ------ | -------- |
| 14  | [LoRA](src/alignment/microlora.py)   | 35 min |  [ ]    |
| 15  | [QLoRA](src/alignment/microqlora.py) | 35 min |  [ ]    |

### Milestone 8: Alignment & RL (2 hrs)

Teaching models to follow human preferences through optimization and reinforcement learning.

| #   | Script                                       | Time   | Checkbox |
| --- | -------------------------------------------- | ------ | -------- |
| 16  | [DPO](src/alignment/microdpo.py)             | 40 min |  [ ]    |
| 17  | [Reinforce](src/alignment/microreinforce.py) | 35 min |  [ ]    |
| 18  | [PPO](src/alignment/microppo.py)             | 35 min |  [ ]    |
| 19  | [GRPO](src/alignment/microgrpo.py)           | 35 min |  [ ]    |

### Milestone 9: Mixture of Experts (0.5 hrs)

Conditional computation — activating only a subset of parameters per input.

| #   | Script                           | Time   | Checkbox |
| --- | -------------------------------- | ------ | -------- |
| 20  | [MOE](src/alignment/micromoe.py) | 35 min |  [ ]    |

### Milestone 10: Attention Optimization (2 hrs)

Efficient attention patterns, positional encoding, and memory-aware computation.

| #   | Script                                       | Time   | Checkbox |
| --- | -------------------------------------------- | ------ | -------- |
| 21  | [Attention](src/systems/microattention.py)   | 40 min |  [ ]    |
| 22  | [Flash Attention](src/systems/microflash.py) | 40 min |  [ ]    |
| 23  | [ROPE](src/systems/microrope.py)             | 35 min |  [ ]    |

### Milestone 11: Inference Systems (2.5 hrs)

KV caching, memory management, quantization, and decoding strategies.

| #   | Script                                       | Time   | Checkbox |
| --- | -------------------------------------------- | ------ | -------- |
| 24  | [KV Cache](src/systems/microkv.py)           | 35 min |  [ ]    |
| 25  | [Paged Attention](src/systems/micropaged.py) | 40 min |  [ ]    |
| 26  | [Quantization](src/systems/microquant.py)    | 40 min |  [ ]    |
| 27  | [Beam](src/systems/microbeam.py)             | 35 min |  [ ]    |

### Milestone 12: Advanced Systems (1.5 hrs)

State-space models, gradient checkpointing, and parallelism — the frontier of efficient training and inference.

| #   | Script                                                | Time   | Checkbox |
| --- | ----------------------------------------------------- | ------ | -------- |
| 28  | [State Space Model](src/systems/microssm.py)          | 35 min |  [ ]    |
| 29  | [Gradient Checkpoint](src/systems/microcheckpoint.py) | 30 min |  [ ]    |
| 30  | [Parallelism](src/systems/microparallel.py)           | 30 min |  [ ]    |

### Milestone 13: Mamba-3 Deep Dive (2 hrs)

The SSM frontier — discretization methods, complex eigenvalue dynamics, and hardware-aware algorithm design from the Mamba-3 paper.

| #   | Script                                           | Time   | Checkbox |
| --- | ------------------------------------------------ | ------ | -------- |
| 31  | [Discretization](src/systems/microdiscretize.py) | 40 min |  [ ]    |
| 32  | [Complex SSM](src/systems/microcomplexssm.py)    | 40 min |  [ ]    |
| 33  | [Roofline](src/systems/microroofline.py)         | 40 min |  [ ]    |

### Milestone 14: Agent Algorithms (3 hrs)

How agents search and reason — tree search for planning and tool-augmented reasoning for language agents.

| #   | Script                                   | Time   | Checkbox |
| --- | ---------------------------------------- | ------ | -------- |
| 34  | [MCTS](src/agents/micromcts.py)          | 90 min |  [ ]    |
| 35  | [React Agents](src/agents/microreact.py) | 90 min |  [ ]    |

---
