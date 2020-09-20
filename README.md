# Towards Locality-Aware Meta-Learning of Tail Node Embeddings on Networks

This repository is the official implementation of [Towards Locality-Aware Meta-Learning of Tail Node Embeddings on Networks](). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To download the original datasets:
[Wiki](https://github.com/thunlp/OpenNE/tree/master/data/wiki).
[Flickr](http://socialcomputing.asu.edu/datasets/Flickr).
[Email](https://snap.stanford.edu/data/email-Eu-core.html).

## Repository Structure
- meta-tail2vec/:
	- dataset/: original dataset without any processing
	- data/: processed data, including train / test data splitting 
	- prep_dataset.py: Prepare necessary data for data_generator.py. When you use different datasets, remember to change the dataset name in line 3.
	- data_generator.py: Generate pipeline for the model, specifically the meta-training and meta-testing tasks.
	- main.py: The main entrance of the model. You can adjust training batch number and pipeline data batch number here.
	- maml.py: The MAML framework.
	- multiclass_task.py: example code for the downstream task of node classification (multi-class setting) and evaluation
	- multilabel_task.py: example code for the downstream task of node classification (multi-label setting) and evaluation
	- prediction/: data processing code for link prediction. It requires different processing of the original datasets, as we need to remove some links from the original graph for testing.

## Train

To train the model in the paper:

First please run deepwalk or other method as base embedding model, the embedding format is the same as deepwalk output.

```
python prep_dataset.py
python main.py
```

## Citation
If you find this useful for your research, we would be appreciated if you cite the following paper:
```
Zemin Liu, Wentao Zhang, Yuan Fang, Xinming Zhang, Steven C.H. Hoi. 2020. Towards Locality-Aware Meta-Learning of Tail Node Embeddings on Networks. In The 29th ACM International Conference on Information and Knowledge Management (CIKM’20), October 19–23, 2020, Virtual Event, Ireland. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3340531.3411910
```
