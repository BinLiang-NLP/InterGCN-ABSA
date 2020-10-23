# Introduction
This repository was used in our paper:  
  
**“Jointly Learning Aspect-Focused and Inter-Aspect Relations with Graph Convolutional Networks for Aspect Sentiment Analysis”**  
Bin Liang, Rongdi Yin, Lin Gui, Jiachen Du, Ruifeng Xu. COLING 2020
  
Please cite our paper if you use this code. 

## Requirements

* Python 3.6
* PyTorch 1.0.0
* SpaCy 2.0.18
* numpy 1.15.4

## Usage

* Install [SpaCy](https://spacy.io/) package and language models with
```bash
pip install spacy
```
and
```bash
python -m spacy download en
```
* Generate aspect-focused graph with
```bash
python generate_graph_for_aspect.py
```
* Generate inter-aspect graph with
```bash
python generate_position_con_graph.py
```

## Training
* Train with command, optional arguments could be found in [train.py](/train.py)

* Run intergcn: ```./run_intergcn.sh```

* Run afgcn: ```./run_afgcn.sh```
