# MultitaskGraphExecution

We make use of a scheduling approach that has been shown to yield improved results in a multi-task setting by aligning dissimilar task gradients. We demonstrate that a novel approach SEGA: Sequential neural Execution of Graph Algorithms that incorporates a scheduling approach in the training process in addition to the modulating the availability of training samples substantively improves results over previous works.

This repository is adapted from the source code for [How to transfer algorithmic reasoning knowledge to learn new algorithms?](https://arxiv.org/abs/2110.14056)

## Environment setup
1. create conda environment
```
conda create -n gne python=3.8
```
2. activate conda environment and install pytorch
```
conda activate gne
conda install pytorch torchvision torchaudio -c pytorch
```
3. install other libs
```
pip install -r requirements.txt
```
## Files

#### baselines.py 
training and evaluation for different baselines.

#### get_stream.py
get train, val, test dataloader for different baselines.

#### logger.py
get experiment logger.

#### playground.py
define training parameters; call get_stream and baseline, example training parameter settings can be found here.

## Running 
0. Prepare data by first generating graphs with gen_\[graph name] functions in datagen/graphgen.py and generate execution trajectories of each algorithm with gen_multi_algo_data in datagen/algorithm.py
1. Instantiate model `NeuralExecutor3` (Neural Executor baseline) or `NeuralExecutor3_` (SEGA) from model.py
2. Setup data streams for training, validation, and testing using either `multi_stream` or `seq_reptile_stream` from `get_streams.py` 
3. Train model with `run_multi` (Neural Executor baseline) or `run_seq_reptile` (SEGA) from `baselines.py`

