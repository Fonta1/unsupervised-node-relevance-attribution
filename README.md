# Explaining graph classifiers by unsupervised node relevance attribution
<div align="justify"> This repository contains the code to perform the experiments published in the paper:   <br /> <br />

Fontanesi, M., Micheli, A., Podda, M. (2024). Explaining Graph Classifiers by Unsupervised Node Relevance Attribution. In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2154. Springer, Cham. https://doi.org/10.1007/978-3-031-63797-1_4 <br />

The repository provides a tool to train, assess, and select a GIN-based multilayer Deep Graph Network (DGN) over multiple XAI-ready datasets. In addition, it allows for an easy application of multiple XAI algorithms (Random baseline, CAM, Integrated Gradient, GNNExplainer) and an easy testing of multiple relevance attribution strategies.   </div>

## Setup
First, create an environment with all the packages listed in environment.yaml and install the [GraphXAI](https://github.com/mims-harvard/GraphXAI/tree/main) library  

Download the raw dataset files required by each dataloader from the following github repositories:
* AlkaneCarbonyl, Mutagenicity, and Benzene from https://github.com/mims-harvard/GraphXAI/tree/main
* BA2grid, GridHouse and HouseColors from https://github.com/AntonioLonga/Explaining-the-Explainers-in-Graph-Neural-Networks/tree/main
* BA2Motif from https://github.com/flyingdoog/PGExplainer/tree/master

Put the downloaded files into the folder "./data/datasets/[DatasetName]/raw"

## Usage
1. Check the files in config_files to specify the cross-validation hyperparameters of the grid-search and the metrics to evaluate the DGN performance.
2. Train DGNs with the following command line:  <br /> ```python main.py trainGCN --name [ExperimentName] --dataset [DatasetName] --cuda --exp_config [path to config file] --cpu [number of available cpus] --gpu [fraction of gpu to use for each concurrent experiment] --max_c [maximum number of concurrent experiments]```
3. Apply the XAI algorithms and relevance attribution strategies on the selected model of a previously performed experiment: <br /> ```python.exe main.py explain --name [ExperimentName] --dataset [DatasetName] --thStrategy [AttributionStrategyName] ``` <br /> When explaining synthetic datasets and using the Strategy ```--thStrategy Top ```, add the option ```--k ``` to specify the desired explanation size. When explaining chemical datasets use  ```--thStrategy BestK ``` instead of ```--thStrategy Top ``` and add the option ```--last ``` to every explain command.
4. Use the notebook "analysis.ipynb" to analyze results and compute the metrics used in the paper.
