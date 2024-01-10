# Reproducing GPT2



**This repo is still very much WIP.**

Reproduce GPT2 (124M) from scratch:
* Single 8xA100 node in 12 hours.
* XX lines of code in three source files.

## How to Use

You will need access to a single node of 8xA100 (40GB or 80GB memory) with CUDA drivers installed.

```
pip install -r requirements.txt
torchrun --standalone --nproc_per_node=8 train.py
```

## Features

* Automatic Mixed Precision (AMP) using BFloat16.
* Distributed Data Parallelism.
* Gradient Accumulation.
* Logging in Tensorboard.
* Runtime instructions for Amazon SageMaker.


## The Journey

### Development process

1. Read through foundational [GPT1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) papers by @AlecRad et al...
2. Went through @karpathy's [video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY&pp=ygUMa2FycGF0aHkgZ3B0).
3. Looked up variety of online implementations: Harvard's [annotated transformer](https://nlp.seas.harvard.edu/annotated-transformer/), [openAI](https://github.com/openai/gpt-2?tab=readme-ov-file), [nanogpt](https://github.com/karpathy/nanoGPT/), [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py), [@benjamin_warner](https://github.com/warner-benjamin/commented-transformers), etc...
4. Reproduced the attention function from PyTorch with vanilla matrix multiplication.
5. Created a script to poll for a p4 instance/8xA100 (~19 days 😓).
6. Implemented training engineering stuff (e.g., DDP, AMP, gradient accumulation, logging, etc...)
7. Ran training multiple times, debugging issues including: learning rate scheduler bug, initialization issue, removing dropout, fixing gradient explosion (see below).

### Training process

1. First run: loss never got below 3. I realized I had a in the learning rate scheduler.
2. Second run: loss never dropped below 3 again, but I realized I didn't use the paper's initialization.
3. Third run: I got a much better result of 2.89 with the new correct initialization, but still not quite there.
4. Fourth run: I discovered that the original author's didn't use dropout, so I turned it off, but then I started getting gradient explosion.
5. Fifth run: I turned off automatic mixed precision (AMP), and I was able to get to the goal of ~2.85 validation loss.
6. I did a bunch of debugging with mixed precision trying to figure out what was causing the gradient explosion. As it turned out, when using the bfloat16 data format, you don't need to perform loss scaling, which fixed it.
7. After getting AMP to work, I was able to reproduce the same training result in roughly half the time.
8. Then I tried some light-weight hyperparameter optimization, increasing the learning rate and weight decay and got the training down to 24 hours [link](https://twitter.com/josiahjdavis/status/1722073528508223667).
9. I did some additional experimentation, and got the training down to 12 hours, which is where I stopped [link](https://twitter.com/josiahjdavis/status/1727523774503756217).

For more details, check out my ([tweet thread](https://twitter.com/josiahjdavis/status/1686204521255432193)).

## References

* OpenAI GPT2: https://github.com/openai/gpt-2
* Andrej Karpathy: https://github.com/karpathy/nanoGPT
* HuggingFace: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/