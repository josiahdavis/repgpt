import argparse
from datetime import datetime
import os
import sys
import math
import time
import uuid

from loguru import logger
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPT2, LRScheduler


# Model arguments
class GPT2Config:
    n_layer = 12
    n_embd = 768
    n_head = 12
    context_size = 1024
    dropout = 0.0  # radford18etal is 0.1, however, gpt2 source code has no dropout layer
    vocab_size = 50_304  # 50_257 is used in the paper, but 50_304 is divisible by 8
    bias = False


_MODEL_CONFIGS = {"124M": GPT2Config}

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=300, help="training steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="perform eval after this many steps")
    parser.add_argument("--eval_steps", type=int, default=20, help="perform eval on this many steps")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accum", type=int, default=32, help="n gradient accumulation steps across all devices")
    parser.add_argument("--model_size", type=str, default="124M", choices=["124M", "355M", "774M", "1.5B"])
    parser.add_argument("--tensorboard", type=int, default=1, choices=[0, 1], help="perform tensorboard logging")

    # Sagemaker copies data from S3 to /opt/ml/input/data/{train,eval}
    args, _ = parser.parse_known_args()
    logger.info(f"Got args {args}")

    # Training arguments
    max_steps = args.max_steps
    eval_interval = args.eval_interval
    eval_steps = args.eval_steps
    log_interval = 10
    tensorboard = bool(args.tensorboard)

    # radford18 is 2.5e-4, learning rate should be tune on perplexity on 5% held-out.
    max_learning_rate = 1e-2  # wortsman23b LRs range tested is {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1}
    warmup_steps = 2_000  # min(2_000, max_steps // 10)  # radford18 is 2_000, alternative is max_steps // 10
    min_learning_rate = 0  # ~= max_learning_rate/10 per Chinchilla, but original paper is zero.
    lr_decay_steps = max_steps  # ~= max_steps per Chinchilla
    batch_size = args.batch_size  # Should be 512
    weight_decay = 0.125  # In radford18 it is set to 0.01 for all non bias or gain weights
    gradient_accumulation_steps = args.gradient_accum
    grad_clip = 1
    beta1 = 0.9  # Default of 0.9
    beta2 = 0.95  # Default of 0.999 can lead to instability https://arxiv.org/pdf/2304.13013.pdf

    lr_scheduler = LRScheduler(max_steps, warmup_steps, lr_decay_steps, max_learning_rate, min_learning_rate)

    assert args.model_size in _MODEL_CONFIGS.keys(), f"{args.model_size} not implemented, please use one of {_MODEL_CONFIGS.keys()}"
    config = _MODEL_CONFIGS[args.model_size]()
    logger.info(
        f"Training {args.model_size} model with {config.n_layer=}, {config.n_embd=}"
        f"{config.n_head=}, {config.context_size=}, {config.dropout=}, {config.vocab_size=}"
    )

    context_size = config.context_size
    n_embd = config.n_embd
    n_layer = config.n_layer
    n_head = config.n_head
    dropout = config.dropout
    vocab_size = config.vocab_size

    # Check if it's a SageMaker training run or a local run (e.g., EC2 instance)
    local_run = os.environ.get("SM_CHANNEL_TRAIN") is None

    # Setup directories
    data_dir = "/tmp/data/" if local_run else "/opt/ml/"
    # Create a sensible job name unless running on SageMaker, which will handle this on it's own.
    job_name = f"gpt2-training-{args.model_size}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" if local_run else ""
    train_dir = os.path.join(data_dir, "input/data/train")
    eval_dir = os.path.join(data_dir, "input/data/eval")
    model_dir = os.path.join(data_dir, "model")
    log_dir = os.path.join(data_dir, "output/tensorboard/nov", job_name)

    compile_model = True
    save = True
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "nccl"  # use nccl for distributed GPU per https://pytorch.org/docs/stable/distributed.html
    ddp = int(os.environ.get("RANK", -1)) != -1
    torch.set_float32_matmul_precision("high")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Automatic mixed precision (AMP) training
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    # context manager, allow regions of the script to run in mixed precision
    # CUDA ops run in a dtype chosen by autocast
    ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)

    # Distribution across devices
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        main_process = ddp_rank == 0
        seed_offset = ddp_rank
        logger.debug(f"{ddp_rank=} {ddp_local_rank=}")
        assert gradient_accumulation_steps % ddp_world_size == 0
        # When we say e.g., we want grad accum of 32 what we really mean is we want each of 8 GPUs to accumulate 4 times.
        gradient_accumulation_steps_per_gpu = gradient_accumulation_steps // ddp_world_size
    else:
        main_process = True
        gradient_accumulation_steps_per_gpu = gradient_accumulation_steps
        ddp_world_size = 1
        seed_offset = 0
        device = device_type

    # Loading data
    train_data = np.memmap(os.path.join(train_dir, "train.bin"), dtype=np.uint16, mode="r")
    eval_data = np.memmap(os.path.join(eval_dir, "eval.bin"), dtype=np.uint16, mode="r")
    logger.info(f"Training data is {len(train_data):,} tokens")
    logger.info(f"Evaluation data is {len(eval_data):,} tokens")
    total_training_tokens_unique = len(train_data)

    tokens_per_step = batch_size * context_size * gradient_accumulation_steps_per_gpu * ddp_world_size
    total_training_tokens = tokens_per_step * max_steps

    if main_process:
        logger.info(f"{job_name=}")
        logger.info(f"Tokens / step: {tokens_per_step:,}")
        logger.info(f"Total training tokens: {total_training_tokens:,}")
        logger.info(f"Effective batch size with grad accumulation: {batch_size * gradient_accumulation_steps=}")
        logger.debug(f"{gradient_accumulation_steps_per_gpu=}")
        logger.debug(f"Directories: {train_dir=}, {eval_dir=} {model_dir=} {log_dir=}")
        logger.info(f"Loaded dataset {time_elapsed()}")
        if tensorboard:
            writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(20230821 + seed_offset)

    # Define Model
    model = GPT2(config)
    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Is it even important to keep the raw model?
    raw_model = model.module if ddp else model  # Is it important whether compilation happens after this assignment?

    if compile_model:
        # I get a warning:
        #       torch._inductor.utils: [WARNING] using triton random, expect difference from eager
        # I believe this is warning me that I may not get the same weights in compile vs. non-compile.
        # Not sure how to disable it.
        model = torch.compile(model)

    # Define optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]  # matmuls, embeddings
    nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]  # biases, layernorms
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=max_learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True)

    if main_process:
        logger.info(f"{model}")
        n_params = sum(p.numel() for p in raw_model.parameters())
        n_params -= raw_model.position_embedding_table.weight.numel()  # Subtracting, becuase we use weight-tying for the final layer.
        logger.info(f"Training model with {n_params:,} parameters for {max_steps=:,} on {total_training_tokens=:,}")
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"Decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"Non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    start_training = time.time()

    best_eval_loss = 100
    explosion_step = 0

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "eval"]:
            losses = torch.zeros(eval_steps)
            for k in range(eval_steps):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_batch(split):
        data = train_data if split == "train" else eval_data

        # Take random positions within the text
        ix = torch.randint(len(data) - context_size, (batch_size,))

        # Stack independent rows on top of each other for the batch dimension
        x = torch.stack([torch.from_numpy((data[i : i + context_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + context_size + 1]).astype(np.int64)) for i in ix])

        if "cuda" in device:
            # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
            # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    #  ----- 1. Begin Main Loop -----
    xb, yb = get_batch("train")

    for step in range(max_steps):
        start_step = time.time()

        lr = lr_scheduler.step(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ----- 1.0 Begin evaluation -----
        if (step % eval_interval == 0) & main_process:
            losses = estimate_loss(model)
            if tensorboard:
                writer.add_scalar("Loss/train", losses["train"], step)
                writer.add_scalar("Loss/eval", losses["eval"], step)
                writer.add_scalar("learning_rate", lr, step)
            logger.info(f"Step {step:,}/{max_steps:,} loss: {losses['train']:.4f} (T) {losses['eval']:.4f} (V) | {lr=:.1e}")

            if losses["eval"] < best_eval_loss:
                best_eval_loss = losses["eval"]
                if save:
                    model_args = dict(
                        n_layer=n_layer,
                        n_head=n_head,
                        n_embd=n_embd,
                        context_size=context_size,
                        vocab_size=vocab_size,
                        dropout=dropout,
                    )
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "step_num": step,
                        "best_eval_loss": best_eval_loss,
                        "config": config,
                    }
                    os.makedirs(model_dir, exist_ok=True)
                    logger.info(f"Saving model to {model_dir}... ")
                    torch.save(checkpoint, os.path.join(model_dir, "ckpt.pt"))

        # ----- 1.1 Begin Forward, Backward Pass -----
        for sub_step in range(gradient_accumulation_steps_per_gpu):
            # Sync gradients on the final sub-step
            model.require_backward_grad_sync = (sub_step + 1) == gradient_accumulation_steps_per_gpu
            # Use context manager for forward pass and loss calculation
            # logits is bfloat16 because linear layers ``autocast`` to bfloat16.
            # loss is float32 because linear layers ``autocast`` to bfloat16.
            with ctx:
                logits, loss = model(xb, yb)
                loss = loss / gradient_accumulation_steps_per_gpu
            # assert logits.dtype is torch.bfloat16, f"logits is not torch.bfloat16, it's {logits.dtype}"
            xb, yb = get_batch("train")
            loss.backward()

        # ----- 1.2 Update everything -----
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        training_step_time = time.time() - start_step
        if (step % log_interval == 0) & main_process:
            approx_loss = loss.item() * gradient_accumulation_steps_per_gpu
            logger.debug(
                f"Training step {step}: loss = {approx_loss:.4f} | {training_step_time*1000:.2f}ms | Tokens/s = {tokens_per_step/training_step_time:,.1f}"
            )

    #  ----- Training is over  -----
    if main_process:
        logger.info(f"{job_name=} finished in {time.time() - start_training:.2f}s")
        logger.info(f"Trained for {max_steps:,} steps {total_training_tokens=:,} and achieved  best eval loss={best_eval_loss.item()}")
        losses = estimate_loss(model)

        hparam_dict = {
            "max_learning_rate": max_learning_rate,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "min_learning_rate": min_learning_rate,
            "lr_decay_steps": lr_decay_steps,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "grad_clip": grad_clip,
            "beta1": beta1,
            "beta2": beta2,
            "n_layer": n_layer,
            "n_embd": n_embd,
            "n_head": n_head,
            "context_size": context_size,
            "vocab_size": vocab_size,
            "total_training_tokens": total_training_tokens,
            "n_times_through_data": total_training_tokens / total_training_tokens_unique,
            "n_params": n_params,
        }
        metric_dict = {"hparam/loss": best_eval_loss.item()}
        logger.info(f"Loss: {losses['train']:.4f} (T) {losses['eval']:.4f} (V) | {time_elapsed()}s")
        if tensorboard:
            writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
            writer.flush()

    if ddp:
        destroy_process_group()
