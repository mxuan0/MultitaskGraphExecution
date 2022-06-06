# MultitaskGraphExecution

We make use of a scheduling approach that has been shown to yield improved results in a multi-task setting by aligning dissimilar task gradients. We demonstrate that a novel approach SEGA: Sequential neural Execution of Graph Algorithms that incorporates a scheduling approach in the training process in addition to the modulating the availability of training samples substantively improves results over previous works.

## baselines.py 
training and evaluation for different baselines.

## get_stream.py
get train, val, test dataloader for different baselines.

## logger.py
get logger

## playground.py
define trainning parameters; call get_stream and baseline.
