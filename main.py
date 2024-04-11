import torch
import argparse
import importlib
import pickle
import os
import random
import numpy as np
from data.dataloaders import ChemXAI
from utils.osHandler import loadyaml
from XAI.assessAlgorithm import  Explain
from train.validation import train_crossvalidation_val
from XAI.metrics import StrategyKmeans, StrategyFidelityTh, StrategyTop, StrategyBestK

def arg_parse():
    parser = argparse.ArgumentParser(prog='Expl',
                                     description='This application let you train and explain GIN-based DGNs on different graph classification tasks datasets')
    
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('trainGCN', help='a help')
    parser_a.add_argument('--name', type=str, dest="exp_name", required=True, help="Experiment name")
    parser_a.add_argument('--dataset', type=str, dest="dataset", default='Mutagenicity', help="Input dataset")
    parser_a.add_argument('--hyper_file', type=str, dest="hyper_file", default='GCNhypers.yaml', help="File containing the hyperparameters for the gridsearch")
    parser_a.add_argument('--cuda', action="store_const", const=True, default=False, help="Enable training on GPU")
    parser_a.add_argument('--k_fold', type=int, dest="k_fold", default=5, help="Number of folds for crossvalidation")
    parser_a.add_argument('--metrics', type=str, dest="task_info", default='metrics.yaml', help="File containing the problem paramenters")
    parser_a.set_defaults(command='trainGCN')

    parser_c = subparsers.add_parser('explain', help='define the explanation experiment')
    parser_c.add_argument('--name', type=str, dest="exp_name", required=True, help="Experiment name")
    parser_c.add_argument('--dataset', type=str, dest="dataset", default='Mutagenicity', help="Input dataset")
    parser_c.add_argument('--classes', nargs="+", type=int, default=[1], help="Chose which GT class to explain, default [1]")
    parser_c.add_argument('--seed', type=int, dest="seed", default=109382, help="Define random seed for Python, PyTorch and Numpy")
    parser_c.add_argument('--thStrategy', type=str, dest="thStrategy", default='TopK', help="Relevance attribution strategy (Kmeans, FidelityTh, Top, BestK)")
    parser_c.add_argument('--k', type=float, dest="k", default=9, help="K for Top strategy")
    parser_c.add_argument('--fold', type=int, dest="fold", default=-1, help="Fold to explain (-1 for the best model in the experiment folder)")
    parser_c.add_argument('--last', action="store_const", const=True, default=False, help="Evaluate only on complete GT explanations (for real world datasets)")
    parser_c.add_argument('--model', type=str, dest="model", default = 'BestModel', help="Model to explain")
    parser_c.set_defaults(command='explain')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    if args.command == 'trainGCN':
        exp_name = args.exp_name
        dataset_name = args.dataset
        result_path = f'./experiments/{dataset_name}/{exp_name}'
        hypers_a = loadyaml(args.hyper_file)
        task_info = loadyaml(args.task_info)

        device = torch.device('cpu')
        if args.cuda:
            device = torch.device('cuda')

        if dataset_name in ['Mutagenicity', 'AlkaneCarbonyl', 'FluorideCarbonyl', 'Benzene']:
            dataset = ChemXAI(name = dataset_name)
        elif dataset_name in ['BA2grid', 'GridHouse', 'HouseColors', 'BA2Motif']:
            module = importlib.import_module("data.dataloaders")
            dataset_class = getattr(module, dataset_name)
            dataset = dataset_class()
        else:
            raise Exception('Dataset not found')

        model, model_metrics, test_metrics = train_crossvalidation_val(dataset, hypers_a, task_info, device, result_path, k_fold=args.k_fold)
    
    elif args.command == 'explain':

        name = args.exp_name
        dataset_name = args.dataset
        classes = args.classes
        seed = args.seed
        thstrat = args.thStrategy
        fold = args.fold
        last = args.last
        modelName = args.model

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_details_path = f'./experiments/{dataset_name}/{name}/BestModel/statistics.yaml'
        if os.path.exists(train_details_path):
            train_details = loadyaml(train_details_path)

        if fold == -1:
            model_path = f'./experiments/{dataset_name}/{name}/BestModel/'
            fold = train_details['best_fold_index']
        else:
            model_path = f'./experiments/{dataset_name}/{name}/fold{fold}/{modelName}'

        print('Loading Model...')

        model = torch.load(f'{model_path}/model.pt')
        model.eval()

        print(f'Loaded model: {model_path}/model.pt')

        print('Loading Splits...')

        if train_details["crossVal"] == 'val' or train_details["crossVal"] == 'test':
            split_path = f'./data/datasets/{dataset_name}/splits/cross_validation_{train_details["crossVal"]}_{train_details["num_folders"]}_splits.pkl'
            with open(split_path, 'rb') as f:
                splits = pickle.load(f)
            splits = splits[fold]
        else:
            split_path = f'./data/datasets/{dataset_name}/splits/holdout.pkl'
            with open(split_path, 'rb') as f:
                splits = pickle.load(f)

        print(f'Loaded splits: {split_path}, fold: {fold}')

        if dataset_name in ['Mutagenicity', 'AlkaneCarbonyl', 'FluorideCarbonyl', 'Benzene']:
            dataset = ChemXAI(name = dataset_name)
        elif dataset_name in ['BA2grid', 'GridHouse', 'HouseColors', 'BA2Motif']:
            module = importlib.import_module("data.dataloaders")
            dataset_class = getattr(module, dataset_name)
            dataset = dataset_class()
        else:
            raise Exception('Dataset not found')

        result_path = f'{model_path}/Explanations'

        if thstrat == 'Top':
            filterStrategy = StrategyTop
            filter_kwargs = {'top_k': int(args.k)}
        elif thstrat == 'BestK':
            filterStrategy = StrategyBestK
            filter_kwargs = {}
        elif thstrat == 'Kmeans':
            filterStrategy =  StrategyKmeans
            filter_kwargs = {}
        elif thstrat == 'FidelityTh':
            filterStrategy =  StrategyFidelityTh
            filter_kwargs = {}
        else:
            raise Exception('Specify a correct filter strategy')

        explain = Explain(model, dataset, splits, result_path, last)
        allMetrics = ['Accuracy','F1Fidelity','Precision','Recall','F1','K','Ksilhouette']
        explain.assessAllAlgorithms(metrics = allMetrics, filterStrategy = filterStrategy, filter_kwargs = filter_kwargs, classes = classes)