import torch
import argparse
import importlib
import os
import random
import numpy as np
from os.path import join
from data.dataloaders import ChemXAI
from utils.osHandler import loadyaml
from XAI.assessAlgorithm import  Explain
from train.Kvalidation import KfoldCV
from data.split_generator import split_data
from XAI.metrics import StrategyKmeans, StrategyFidelityTh, StrategyTop, StrategyBestK

os.environ["WANDB_SILENT"] = "true"
os.environ["NUMEXPR_MAX_THREADS"] = "30"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["RAY_AIR_NEW_OUTPUT"] = "1"

def arg_parse():
    parser = argparse.ArgumentParser(prog='Expl',
                                     description='This application let you train and explain GIN-based DGNs on different graph classification tasks datasets')

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('trainGCN', help='a help')
    parser_a.add_argument('--name', type=str, dest="exp_name", required=True, help="Type of validation technique")
    parser_a.add_argument('--dataset', type=str, dest="dataset", default='Mutagenicity', help="Input dataset.")
    parser_a.add_argument('--exp_config', type=str, dest="exp_config", default='GINhypers.yaml', help="File containing the hyperparameters for the gridsearch")
    parser_a.add_argument('--cuda', action="store_const", const=True, default=False, help="Enable training on GPU")
    parser_a.add_argument('--cpu', type=float, dest="cpu", default=1, help="Number of workers for each concurrent processes")
    parser_a.add_argument('--gpu', type=float, dest="gpu", default=0.25, help="Splits of the gpu resources across concurrent processes")
    parser_a.add_argument('--max_c', type=int, dest="max_concurrency", default=5, help="Maximum number of concurrent processes")
    parser_a.set_defaults(command='trainGCN')

    parser_c = subparsers.add_parser('explain', help='define the explanation experiment')
    parser_c.add_argument('--name', type=str, dest="exp_name", required=True, help="Experiment name")
    parser_c.add_argument('--dataset', type=str, dest="dataset", default='Mutagenicity', help="Input dataset")
    parser_c.add_argument('--classes', nargs="+", type=int, default=[1], help="Chose which GT class to explain, default [1]")
    parser_c.add_argument('--seed', type=int, dest="seed", default=109382, help="Define random seed for Python, PyTorch and Numpy")
    parser_c.add_argument('--thStrategy', type=str, dest="thStrategy", default='TopK', help="Relevance attribution strategy (Kmeans, FidelityTh, Top, BestK)")
    parser_c.add_argument('--k', type=float, dest="k", default=9, help="K for Top strategy")
    parser_c.add_argument('--last', action="store_const", const=True, default=False, help="Evaluate only on complete GT explanations (for real world datasets)")
    parser_c.set_defaults(command='explain')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()

    if args.command == 'trainGCN':
        dataset_name = args.dataset
        cpu = args.cpu
        gpu = args.gpu
        max_concurrent = args.max_concurrency
        kfold = 5

        exp_config = loadyaml(args.exp_config)

        device = torch.device('cpu')
        if args.cuda:
            device = torch.device('cuda')

        if dataset_name in ['Mutagenicity', 'AlkaneCarbonyl', 'FluorideCarbonyl', 'Benzene']:
            _ = ChemXAI(name = dataset_name)
        elif dataset_name in ['BA2grid', 'GridHouse', 'HouseColors', 'BA2Motif']:
            module = importlib.import_module("data.dataloaders")
            dataset_class = getattr(module, dataset_name)
            _ = dataset_class()
        else:
            raise Exception('Dataset not found')

        exp_name = f'{args.exp_name}_{dataset_name}_CV'
        result_path = join(os.getcwd(), 'experiments', dataset_name, exp_name)
        kfold = KfoldCV(exp_name, dataset_name, exp_config, device, result_path, kfold)
        results = kfold.parallel_fit(max_concurrent, cpu, gpu)
        metrics, best_model = kfold.select_assess()
    
    elif args.command == 'explain':

        name = args.exp_name
        dataset_name = args.dataset
        classes = args.classes
        seed = args.seed
        thstrat = args.thStrategy
        last = args.last

        exp_name = f'{args.exp_name}_{dataset_name}_CV'

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        print('Loading Model...')
        model_path = f'./experiments/{dataset_name}/{exp_name}'
        model = torch.load(join(model_path, 'model.pt'))
        model.eval()
        print(f'Loaded model: {model_path}')

        if dataset_name in ['Mutagenicity', 'AlkaneCarbonyl', 'FluorideCarbonyl', 'Benzene']:
            dataset = ChemXAI(name = dataset_name)
        elif dataset_name in ['BA2grid', 'GridHouse', 'HouseColors', 'BA2Motif']:
            module = importlib.import_module("data.dataloaders")
            dataset_class = getattr(module, dataset_name)
            dataset = dataset_class()
        else:
            raise Exception('Dataset not found')
        
        print('Loading Splits...')
        splits = split_data(dataset)[0]
        print(f'Splits Loaded.')

        result_path = join(model_path, 'Explanations')

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