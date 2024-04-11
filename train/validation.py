import torch
from torch_geometric.loader import DataLoader
from data.split_generator import split_data
from train.standard_trainer import train_architecture
import shutil
import yaml
import numpy as np
import time
import importlib
import operator
import pandas as pd
import os
import copy
from tensorboardX import SummaryWriter

def assessModel(model, device, test_loader, task_info, logger):
    loss_function = torch.nn.CrossEntropyLoss()

    metric_obj = {}
    metrics = {'test_Loss': 0}

    for metric in task_info['metrics']:
        tasktype = task_info['task'][0]
        num_classes = task_info['num_classes'][0]
        metric_name = metric.split('.')[-1]
        metric_module = '.'.join(metric.split('.')[:-1])
        module = importlib.import_module(metric_module)
        metric_class = getattr(module, metric_name)
        metric_obj[metric_name] = metric_class(task = tasktype, num_classes=num_classes).to(device)
        metrics[f'test_{metric_name}'] = 0

    test_loss = 0
    for i, data in enumerate(test_loader):
        x, edge_index, graph_index, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
        out = model(x, edge_index, graph_index)
        loss = loss_function(out, targets)
        test_loss += loss.item()
        for metric_name in metric_obj:
            preds = torch.nn.functional.softmax(out, -1)
            if tasktype == 'binary':
                preds = preds[:,-1]
            metric_obj[metric_name].update(preds, targets)


    metrics[f'test_Loss'] = test_loss/len(test_loader)
    for metric_name in metric_obj:
        metrics[f'test_{metric_name}'] = metric_obj[metric_name].compute().item()

    main_tag = f'BESTModel_TEST'
    logger.add_scalars(main_tag, metrics, 0)

    return metrics

       
def train_holdout(dataset, hypers_a, task_info, device, result_path, splits):
    train_loader = DataLoader(dataset[splits['train']], batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset[splits['val']], batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset[splits['test']], batch_size=128, shuffle=True)

    logger = SummaryWriter(f'{result_path}/logs')
    if not os.path.isdir(f'{result_path}/logs'): 
        os.makedirs(f'{result_path}/logs')
    input_channels = dataset[0].x.size()[1]
    output_channels = len(torch.unique(dataset.y))

    start = time.time()
    model, model_metrics = train_architecture(train_loader, val_loader, hypers_a, input_channels, output_channels, task_info, device, logger, result_path)
    test_metrics = assessModel(model, device, test_loader, task_info, logger)
    end = time.time()
    elapsed = end-start
    with open(f'{result_path}/time.yaml', "w") as file:
        yaml.dump({"elapsed_time": elapsed}, file, default_flow_style=False, Dumper=yaml.SafeDumper)

    return model, model_metrics, test_metrics


def train_crossvalidation_val(dataset, hypers_a, task_info, device, result_path, k_fold = 5):
    splits = split_data(dataset, mode = 'cross_validation_val', k = k_fold)

    selection_metric = task_info['model_selection_metrics'][0].split('.')[-1]
    selection_obj = task_info['target'][0]
    best_on_val = {'best_fold_index': None,
                   'num_folders': k_fold,
                   'crossVal': 'val'}

    if selection_obj == 'Minimize':
        compare = operator.lt
        best_on_val[f'best_metric_val_{selection_metric}'] = np.inf
        best_on_val[f'best_metric_train_{selection_metric}'] = np.inf
    else:
        compare = operator.gt
        best_on_val[f'best_metric_val_{selection_metric}'] = -np.inf
        best_on_val[f'best_metric_train_{selection_metric}'] = -np.inf

    metrics = {'val_Loss': [],
               'train_Loss': []}
    for metric in task_info['metrics']:
        metric_name = metric.split('.')[-1]
        metrics[f'val_{metric_name}'] = []
        metrics[f'train_{metric_name}'] = []
    
    best_model = None
    best_test = None
    
    for i, folds in enumerate(splits):
        folder_path = f'{result_path}/fold{i}'
        model, model_metrics, test_metrics = train_holdout(dataset,  hypers_a, task_info, device, folder_path, folds)
        
        best_epoch = model_metrics[f'best_epoch']
        for metric in task_info['metrics']:
            metric_name = metric.split('.')[-1]
            metrics[f'train_{metric_name}'].append(model_metrics[f'train_{metric_name}_seq'][best_epoch])
            metrics[f'val_{metric_name}'].append(model_metrics[f'val_{metric_name}_seq'][best_epoch])
        metrics[f'train_Loss'].append(model_metrics[f'train_Loss_seq'][best_epoch])
        metrics[f'val_Loss'].append(model_metrics[f'val_Loss_seq'][best_epoch])

        epoch = model_metrics['best_epoch']
        current_val_value_sel_metric = float(model_metrics[f"val_{selection_metric}_seq"][epoch])
        best_val_value_sel_metric = best_on_val[f'best_metric_val_{selection_metric}']
        current_train_value_sel_metric = float(model_metrics[f"train_{selection_metric}_seq"][epoch])
        best_train_value_sel_metric = best_on_val[f"best_metric_train_{selection_metric}"]

        if compare(current_val_value_sel_metric, best_val_value_sel_metric) or (current_val_value_sel_metric == best_val_value_sel_metric and compare(current_train_value_sel_metric, best_train_value_sel_metric)):
            best_on_val[f'best_metric_val_{selection_metric}'] = current_val_value_sel_metric
            best_on_val[f"best_metric_train_{selection_metric}"] = current_train_value_sel_metric
            best_on_val['best_fold_index'] = i
            best_model = copy.deepcopy(model)
            best_test = test_metrics
    
    bestmodel_src = f'{result_path}/fold{best_on_val["best_fold_index"]}/BestModel'
    bestmode_dest = f'{result_path}/BestModel'
    destination = shutil.copytree(bestmodel_src, bestmode_dest)

    with open(f'{result_path}/BestModel/statistics.yaml', "w") as file:
        yaml.dump(best_on_val, file, default_flow_style=False, Dumper=yaml.Dumper)

    with open(f'{result_path}/BestModel/TestStatistics.yaml', "w") as file:
        yaml.dump(best_test, file, default_flow_style=False, Dumper=yaml.Dumper)

    result_report = {}
    test_results = pd.DataFrame(metrics)
    result_report['avg_train_Loss'] = float(test_results['train_Loss'].mean())
    result_report['std_train_Loss'] = float(test_results['train_Loss'].std())
    result_report['avg_val_Loss'] = float(test_results['val_Loss'].mean())
    result_report['std_val_Loss'] = float(test_results['val_Loss'].std())

    for metric in task_info['metrics']: 
        metric_name = metric.split('.')[-1]
        result_report[f'avg_train_{metric_name}'] = float(test_results[f'train_{metric_name}'].mean())
        result_report[f'std_train_{metric_name}'] = float(test_results[f'train_{metric_name}'].std())
        result_report[f'avg_val_{metric_name}'] = float(test_results[f'val_{metric_name}'].mean())
        result_report[f'std_val_{metric_name}'] = float(test_results[f'val_{metric_name}'].std())

    test_results.to_csv(f'{result_path}/results_report.csv')
    with open(f'{result_path}/result_report.yaml', "w") as file:
        yaml.dump(result_report, file, default_flow_style=False, Dumper=yaml.SafeDumper)

    return best_model, best_on_val[f'best_metric_val_{selection_metric}'], metrics