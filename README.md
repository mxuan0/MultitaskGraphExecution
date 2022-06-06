# MultitaskGraphExecution

We make use of a scheduling approach that has been shown to yield improved results in a multi-task setting by aligning dissimilar task gradients. We demonstrate that a novel approach SEGA: Sequential neural Execution of Graph Algorithms that incorporates a scheduling approach in the training process in addition to the modulating the availability of training samples substantively improves results over previous works.

This repository is adapted from the source code for [How to transfer algorithmic reasoning knowledge to learn new algorithms?](https://arxiv.org/abs/2110.14056)

## Files

#### baselines.py 
training and evaluation for different baselines.

#### get_stream.py
get train, val, test dataloader for different baselines.

#### logger.py
get logger

#### playground.py
define training parameters; call get_stream and baseline.


## Running 
1. Instantiate model from model.py
2. Setup data streams for training, validation, and testing using either `multi_stream` or `seq_reptile_stream` from `get_streams.py` 
3. Train model with `run_multi` (Neural Executor baseline) or `train_seq_reptile` from `train.py`

