import torch
import copy
import numpy as np
from models.standard_conv import STDC
from sklearn.model_selection import ParameterGrid
from train.train_loop import train_model
import os.path 
import yaml
import operator
import random

def train_architecture(train_loader, val_loader, hypers_a, input_channels, output_channels, task_info, device, logger, result_path):

    #load task information for model selection
    selection_obj = task_info['target'][0]
    if selection_obj == 'Minimize':
        compare = operator.lt
    else:
        compare = operator.gt

    selection_metric = task_info['model_selection_metrics'][0].split('.')[-1]

    #target 
    best_model_metrics = {
                "val_Loss_seq" : [np.inf],
                "train_Loss_seq" : [],
                "best_epoch" : 0,
                "best_comb" : None,
                "index" : None,
    }

    #Load metrics
    for metric in task_info['metrics']:
        metric_name = metric.split('.')[-1]
        if metric_name == selection_metric:
            if selection_obj == 'Minimize':
                best_model_metrics[f'val_{metric_name}_seq'] = [np.inf]
                best_model_metrics[f'train_{metric_name}_seq'] = [np.inf]
            else:
                best_model_metrics[f'val_{metric_name}_seq'] = [-np.inf]
                best_model_metrics[f'train_{metric_name}_seq'] = [-np.inf]
        else:
            best_model_metrics[f'train_{metric_name}_seq'] = []
            best_model_metrics[f'val_{metric_name}_seq'] = []

    myhypers = list(ParameterGrid(hypers_a))
    random.shuffle(myhypers)
    for num, combination in enumerate(myhypers):
        print(f'Model, combination: {combination}')

        #Holdout code
        model = STDC(input_channels, combination['dim_embed'], output_channels, combination['num_layers'])
        model.train()
        model.to(device)
        metrics, epoch = train_model(model, train_loader, val_loader, combination, task_info, device)

        #select best model
        current_val_value_sel_metric = metrics[f"val_{selection_metric}_seq"][epoch]
        best_val_value_sel_metric = best_model_metrics[f"val_{selection_metric}_seq"][best_model_metrics["best_epoch"]]
        current_train_value_sel_metric = metrics[f"train_{selection_metric}_seq"][epoch]
        best_train_value_sel_metric = best_model_metrics[f"train_{selection_metric}_seq"][best_model_metrics["best_epoch"]]

        #get best on validation, if equal get the best on train too
        if compare(current_val_value_sel_metric, best_val_value_sel_metric) or (current_val_value_sel_metric == best_val_value_sel_metric and compare(current_train_value_sel_metric, best_train_value_sel_metric)):
            best_model_metrics["val_Loss_seq"] = metrics["val_Loss_seq"]
            best_model_metrics["train_Loss_seq"] = metrics["train_Loss_seq"]
            best_model_metrics["best_epoch"] = epoch
            best_model_metrics["best_comb"] = combination
            best_model_metrics["index"] = num
            for metric in task_info['metrics']:
                metric_name = metric.split('.')[-1]
                best_model_metrics[f'train_{metric_name}_seq'] = metrics[f'train_{metric_name}_seq']
                best_model_metrics[f'val_{metric_name}_seq'] = metrics[f'val_{metric_name}_seq']

            best_model = copy.deepcopy(model)

        #logger section
        for i in range(len(metrics["train_Loss_seq"])):
            main_tag = f'Model{num} lr{combination["lr"]} numL{combination["num_layers"]} dimEmb{combination["dim_embed"]} wDecay{combination["weight_decay"]}'
            single_step = {}
            for key, value in metrics.items():
                single_step[key] = value[i]
            logger.add_scalars(main_tag, single_step, i)

        hparams_log = {}
        for key, value in metrics.items():
                hparams_log[f'hparam/{key}'] = value[epoch]
        logger.add_hparams(combination, single_step)
        
        #save in folder section
        savedir = f'{result_path}/Model{num}'
        os.makedirs(savedir)
        with open(f'{savedir}/hypers.yaml', "w") as file:
            yaml.dump(combination, file, default_flow_style=False, Dumper=yaml.Dumper)
        torch.save(model, f'{savedir}/model.pt')

    #Logger section
    print(f'Best_combination: {best_model_metrics["best_comb"]}')
    print(f'train_{selection_metric}: {best_model_metrics[f"train_{selection_metric}_seq"][best_model_metrics["best_epoch"]]}, val_{selection_metric}: {best_model_metrics[f"val_{selection_metric}_seq"][best_model_metrics["best_epoch"]]}')
    

    best_comb = best_model_metrics["best_comb"]
    main_tag = f'BESTModel{best_model_metrics["index"]} lr{best_comb["lr"]} numL{best_comb["num_layers"]} dimEmb{best_comb["dim_embed"]} wDecay{best_comb["weight_decay"]}'
    for i in range(len(best_model_metrics["train_Loss_seq"])):
        single_step = {}
        for key, value in best_model_metrics.items():
            if not key in ["best_epoch","best_comb","index"]:
                single_step[key] = value[i]
        logger.add_scalars(main_tag, single_step, i)

    #save in folder section
    savedir = f'{result_path}/BestModel'
    os.makedirs(savedir)
    with open(f'{savedir}/hypers.yaml', "w") as file:
        yaml.dump(best_model_metrics["best_comb"], file, default_flow_style=False, Dumper=yaml.Dumper)
    torch.save(best_model, f'{savedir}/model.pt')

    return best_model, best_model_metrics