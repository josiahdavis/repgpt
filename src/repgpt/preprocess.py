# saves huggingface openwebtext to a binary file for training. Modified from:
# https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

import argparse
import time
import os
from loguru import logger
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, dataset_dict

def read_data(num_proc: int) -> dataset_dict.DatasetDict:
    # 54GB in huggingface .cache dir, ~ 8M documents
    dataset = load_dataset("openwebtext", num_proc=num_proc)
    logger.info(f"Loaded dataset {len(dataset)=} {time_elapsed()}")
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")
    logger.info(f"Split dataset into training and validation sets {split_dataset=} {time_elapsed()}.")
    logger.info(f"Here is a single example {split_dataset['train'][11]['text'][:1000]=}")
    return split_dataset


def process(example: dict) -> dict:
    ids = enc.encode_ordinary(example["text"])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    out = {"ids": ids, "len": len(ids)}
    return out

def time_elapsed(prev_start=None) -> str:
    total_min = (time.time() - start)/60
    if prev_start is None:
        return print(f"({total_min:.2f}m)")
    else:
        return f"({(time.time() - prev_start)/60:.2f}, {total_min:.2f})"

start = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=8, help="number of workers in .map() call.")
    parser.add_argument("--data_dir", default="/tmp/data/", help="Where to write the tokenized data to")
    parser.add_argument("--total_shards", type=int, default=1024, help="number of data shards for faster writing.")
    args = parser.parse_args()
    logger.info(f"{args=}")

    total_shards = args.total_shards
    num_proc_load = args.num_proc
    num_proc_tokenize = args.num_proc
    data_dir = args.data_dir
    base_dir = os.path.join(data_dir, "/input/data/")
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "eval"), exist_ok=True)

    dataset = read_data(num_proc_load)
    enc = tiktoken.get_encoding("gpt2")

    # "The primary purpose of map() is to speed up processing functions."
    # https://huggingface.co/docs/datasets/process#map
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing data",
        num_proc=num_proc_tokenize,
    )
    logger.info(f"Tokenized data: {tokenized}")
    logger.info(f"Tokenized training and validation sets {time_elapsed()}.")

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(base_dir, f"{split}/{split}.bin")
        logger.info(f"Preparing to write {split} tokens to a vector with dimension=({arr_len:,},) to {filename}")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)

        # Memory-mapped files are used for accessing small segments of large files on disk,
        # without reading the entire file into memory.
        # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        logger.info(f"Writing data batch-by-batch... {time_elapsed()}")

        # This step is very slow, and CPUs are not actually that busy, takes about 12 minutes.
        # I wonder if numpy memmap could be combined with joblib parallel.
        # Or perhaps I should just increase the shard size.
        idx = 0
        for batch_idx in tqdm(range(total_shards), desc=f"writing {filename}"):
            # Split data into shards for faster writing
            # https://huggingface.co/docs/datasets/process#shard
            batch = dset.shard(num_shards=total_shards, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            if idx == 0:
                logger.info(f"Sample batch {arr_batch} ({arr_batch.shape}) {time_elapsed()}")
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        # Flush the memmap instance to write the changes to the file.
        arr.flush()
        logger.info(f"Completed writing {split} to {filename} {time_elapsed()}")
    logger.info(f"Data preparation complete {time_elapsed()}")