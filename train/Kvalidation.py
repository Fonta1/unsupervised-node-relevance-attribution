import torch
import numpy as np
from data.split_generator import split_data
import importlib
import operator
from ray import train, tune
import ray
from torch_geometric.loader import DataLoader
import os
import yaml
import wandb
from data.dataloaders import ChemXAI
from sklearn.model_selection import StratifiedShuffleSplit
import copy
import pandas as pd

class KfoldCV:
    def __init__(self, experiment_name, dataset_name, exp_config, device, result_path, kfold=5):

        self.hypers = exp_config['hyper_parameters']
        self.task_info = exp_config['task_properties']
        self.data_path = os.path.abspath(exp_config['dataset_properties']['data_path'])
        self.result_path = result_path
        self.kfold = kfold
        self.dataset_name = dataset_name
        self.device = device
        self.experiment_name = experiment_name

        if not os.path.isdir(result_path): 
            os.makedirs(result_path)

        with open(os.path.join(self.result_path, 'GridSearch.yaml'), "w") as file:
                yaml.dump(exp_config, file, default_flow_style=False, Dumper=yaml.SafeDumper)
        
        self.raw_fit_data = None


    def parallel_fit(self, max_trials, cpu, gpu):
        self.raw_fit_data = None

        self.hypers['folds'] = list(range(0,self.kfold))
        hyper_space = {}
        for key, values in self.hypers.items():
            hyper_space[key] = tune.grid_search(values)

        storage = os.path.join(os.getcwd(), 'ray_storage')
        wandblogs = os.path.join(os.getcwd(), 'wandb')

        ray.init(num_cpus=20, num_gpus=1)
        tuner = tune.Tuner(tune.with_resources(self.train_model_parallel, {"cpu": cpu, "gpu": gpu}),
                           param_space=hyper_space,
                           run_config=train.RunConfig(storage_path=storage, name=self.experiment_name),
                           tune_config=tune.TuneConfig(max_concurrent_trials=max_trials))
        results = tuner.fit()

        df = results.get_dataframe()
        df.to_csv(os.path.join(self.result_path, 'RawResults.csv'))
        ray.shutdown()

        del self.hypers['folds']
        self.raw_fit_data = df
        return df

    def select_assess(self):
        readableDF = self.to_readable_results()
        readableDF.to_csv(os.path.join(self.result_path, 'ReadableResults.csv'))

        hyp = list(self.hypers.keys())
        hypG = [f'config/{elem}' for elem in hyp]
        model_selection_metrics = self.task_info["model_selection_metrics"][0].split('.')[-1]
        rdf = self.raw_fit_data.groupby(hypG, as_index=False).mean(numeric_only=True).sample(frac=1)
        indexBest = rdf[f'val_{model_selection_metrics}'].idxmax()
        bestconfig = list(rdf.loc[indexBest][hypG])
        readableDF.loc[[indexBest]].to_csv(os.path.join(self.result_path, 'BestHypersResults.csv'))

        bestconfig_hypers = dict(zip(hyp, bestconfig))

        for key,value in bestconfig_hypers.items():
            if type(value).__module__ == 'numpy':
                bestconfig_hypers[key] = value.item()
        
        print(bestconfig_hypers)

        metrics, best_model = self._fit_and_validate(bestconfig_hypers)

        torch.save(best_model, os.path.join(self.result_path,'model.pt'))
        bestconfig_hypers.update(metrics)
        retraindf = pd.DataFrame(bestconfig_hypers, index=[0])
        retraindf.to_csv(os.path.join(self.result_path, 'FinalModelMetrics.csv'))

        return metrics, best_model

    def to_readable_results(self):
        metrics = [elem.split('.')[-1] for elem in self.task_info['metrics']]
        columns = ['best_epoch']
        for elem in metrics:
            columns.append(f'train_{elem}')
            columns.append(f'val_{elem}')

        hyp = list(self.hypers.keys())
        hypG = [f'config/{elem}' for elem in hyp]

        df1 = self.raw_fit_data.groupby(hypG).mean(numeric_only=True)[columns]
        df2 = self.raw_fit_data.groupby(hypG).std(numeric_only=True)[columns]
        df3 = df1.join(df2, lsuffix='Mean', rsuffix='Std').reset_index()
        for met in metrics:
            mean = df3[f'train_{met}Mean'].round(3)
            std = df3[f'train_{met}Std'].round(3)
            df3[f'{met}_train'] = mean.astype(str) + ' ± ' +  std.astype(str)

            mean = df3[f'val_{met}Mean'].round(3)
            std = df3[f'val_{met}Std'].round(3)
            df3[f'{met}_val'] = mean.astype(str) + ' ± ' +  std.astype(str)

            df3 = df3.drop([f'train_{met}Mean', f'train_{met}Std', f'val_{met}Mean', f'val_{met}Std'], axis=1)

        mean = df3[f'best_epochMean'].round(3)
        std = df3[f'best_epochStd'].round(3)
        df3[f'best_epoch'] = mean.astype(str) + ' ± ' +  std.astype(str)
        df3 = df3.drop([f'best_epochMean', f'best_epochStd'], axis=1)
        
        return df3
    
    def train_model_parallel(self, config):

        task_info = self.task_info
        device = self.device
        kfold = self.kfold
        dataset_name = self.dataset_name
        data_path = self.data_path
               
        if dataset_name in ['Mutagenicity', 'AlkaneCarbonyl', 'FluorideCarbonyl', 'Benzene']:
            dataset = ChemXAI(root = data_path, name = dataset_name)
        elif dataset_name in ['BA2grid', 'GridHouse', 'Stars', 'HouseColors', 'BA2Motif']:
            module = importlib.import_module("data.dataloaders")
            dataset_class = getattr(module, dataset_name)
            dataset = dataset_class(root = data_path)
        else:
            raise Exception('Wrong Dataset')

        splits = split_data(dataset, mode='cross_val', k=kfold)

        epochs = config['mE']
        lr = config['lr']
        patience = config['pt']
        weight_decay = config['wD']
        fold = config['folds']
        batch_size = config['bS']

        output_channels = task_info['num_classes'][0]
        input_channels = dataset[0].x.size()[1]

        model_module = 'models.standard_conv'
        module = importlib.import_module(model_module)
        model_class = getattr(module, 'STDC')
        model = model_class(input_channels, config['dE'], output_channels, config['nL']).to(device)

        train_loader = DataLoader(dataset[splits[fold]['train']], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[splits[fold]['val']], batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, min_lr=1e-4)
        loss_function = torch.nn.CrossEntropyLoss()

        selection_metric = task_info['earlyStoppingMetric'][0].split('.')[-1]
        selection_obj = task_info['earlyStoppingTarget'][0]
        earlyStoppingMinChange = task_info['earlyStoppingMinChange'][0]
        if selection_obj == 'Minimize':
            compare = operator.lt
            best_metric_selection = np.inf
        else:
            compare = operator.gt
            best_metric_selection = -np.inf
            earlyStoppingMinChange = -earlyStoppingMinChange

        metrics = {'train_Loss': [],
                'val_Loss': []}
        
        metric_obj = {}
        for metric in task_info['metrics']:
            metric_name = metric.split('.')[-1]
            metrics[f'train_{metric_name}'] = []
            metrics[f'val_{metric_name}'] = []
            tasktype = task_info['task'][0]
            num_classes = task_info['num_classes'][0]
            metric_module = '.'.join(metric.split('.')[:-1])
            module = importlib.import_module(metric_module)
            metric_class = getattr(module, metric_name)
            metric_obj[metric_name] = metric_class(task = tasktype, num_classes=num_classes).to(device)

        epoch = 0
        best_epoch = 0
        pat = 0
        log_delay = 1

        while epoch < epochs and pat < patience:

            train_loss = 0
            with torch.enable_grad():
                model.train()
                for i, data in enumerate(train_loader):
                    x, edge_index, graph_index, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
                    optimizer.zero_grad()
                    out = model(x, edge_index, graph_index)
                    loss = loss_function(out, targets)

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    for metric_name in metric_obj:
                        preds = torch.nn.functional.softmax(out, -1)
                        if tasktype == 'binary':
                            preds = preds[:,-1]
                        metric_obj[metric_name].update(preds, targets)
            
            
            metrics['train_Loss'].append(train_loss/len(train_loader))
            for metric_name in metric_obj:
                value = metric_obj[metric_name].compute().item()
                metrics[f'train_{metric_name}'].append(value)
                metric_obj[metric_name].reset()
            scheduler.step(train_loss/len(train_loader))

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    x, edge_index, graph_index, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
                    out = model(x, edge_index, graph_index)
                    loss = loss_function(out, targets)
                    val_loss += loss.item()

                    for metric_name in metric_obj:
                        preds = torch.nn.functional.softmax(out, -1)
                        if tasktype == 'binary':
                            preds = preds[:,-1]
                        metric_obj[metric_name].update(preds, targets)

            metrics['val_Loss'].append(val_loss/len(val_loader))
            for metric_name in metric_obj:
                metrics[f'val_{metric_name}'].append(metric_obj[metric_name].compute().item())
                metric_obj[metric_name].reset()

            pat += 1

            if compare(metrics[f'val_{selection_metric}'][-1]+earlyStoppingMinChange, best_metric_selection):
                best_metric_selection = metrics[f'val_{selection_metric}'][-1]
                best_epoch = epoch
                pat = 0

            epoch += 1

            if pat == 0:
                for k in range(log_delay):
                    single_step = {}
                    k = k-log_delay
                    for key, value in metrics.items():
                        single_step[key] = value[k]
                    train.report(single_step)
                log_delay = 1
            else:
                log_delay +=1 
        
        res = {'best_epoch': best_epoch}
        for key, val in metrics.items():
            res[key] = val[best_epoch]
        return res

    def _fit_and_validate(self, config):
        wandb.init(project=self.experiment_name, name=f'SelectedModel', config=config)
    
        if self.dataset_name in ['Mutagenicity', 'AlkaneCarbonyl', 'FluorideCarbonyl', 'Benzene']:
            dataset = ChemXAI(root = self.data_path, name = self.dataset_name )
        elif self.dataset_name  in ['BA2grid', 'GridHouse', 'Stars', 'HouseColors', 'BA2Motif']:
            module = importlib.import_module("data.dataloaders")
            dataset_class = getattr(module, self.dataset_name )
            dataset = dataset_class(root = self.data_path)
        else:
            raise Exception('Wrong Dataset')

        splits = split_data(dataset, mode='cross_val', k=self.kfold)

        epochs = int(config['mE'])
        lr = config['lr']
        weight_decay = config['wD']
        batch_size = int(config['bS'])
        patience = int(config['pt'])

        output_channels = self.task_info['num_classes'][0]
        input_channels = dataset[0].x.size()[1]

        model_module = 'models.standard_conv'
        module = importlib.import_module(model_module)
        model_class = getattr(module, 'STDC')
        model = model_class(input_channels, int(config['dE']), output_channels, int(config['nL'])).to(self.device)
        
        train_split = np.concatenate((splits[0]['train'], splits[0]['val']))
        no_test_dataset = dataset[train_split]

        indices = np.arange(len(no_test_dataset))
        targets = no_test_dataset.y.long().numpy()
        splitter_train_val = StratifiedShuffleSplit(n_splits=1, test_size=1/(self.kfold*2), random_state=5345)
        splits_t_t = splitter_train_val.split(indices, targets)
        train_idx, val_idx =  next(splits_t_t)
    
        train_loader = DataLoader(no_test_dataset[train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(no_test_dataset[val_idx], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset[splits[0]['test']], batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, min_lr=1e-4)
        loss_function = torch.nn.CrossEntropyLoss()

        selection_metric = self.task_info['earlyStoppingMetric'][0].split('.')[-1]
        selection_obj = self.task_info['earlyStoppingTarget'][0]
        earlyStoppingMinChange = self.task_info['earlyStoppingMinChange'][0]
        if selection_obj == 'Minimize':
            compare = operator.lt
            best_metric_selection = np.inf
        else:
            compare = operator.gt
            best_metric_selection = -np.inf
            earlyStoppingMinChange = -earlyStoppingMinChange

        metrics = {'train_Loss': [],
                'val_Loss': []}
        
        metric_obj = {}
        for metric in self.task_info['metrics']:
            metric_name = metric.split('.')[-1]
            metrics[f'train_{metric_name}'] = []
            metrics[f'val_{metric_name}'] = []
            tasktype = self.task_info['task'][0]
            num_classes = self.task_info['num_classes'][0]
            metric_module = '.'.join(metric.split('.')[:-1])
            module = importlib.import_module(metric_module)
            metric_class = getattr(module, metric_name)
            metric_obj[metric_name] = metric_class(task = tasktype, num_classes=num_classes).to(self.device)

        epoch = 0
        best_epoch = 0
        pat = 0
        log_delay = 1

        while epoch < epochs and pat < patience:

            train_loss = 0
            with torch.enable_grad():
                model.train()
                for i, data in enumerate(train_loader):
                    x, edge_index, graph_index, targets = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.y.to(self.device)
                    optimizer.zero_grad()
                    out = model(x, edge_index, graph_index)
                    loss = loss_function(out, targets)

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    for metric_name in metric_obj:
                        preds = torch.nn.functional.softmax(out, -1)
                        if tasktype == 'binary':
                            preds = preds[:,-1]
                        metric_obj[metric_name].update(preds, targets)
            
            
            metrics['train_Loss'].append(train_loss/len(train_loader))
            for metric_name in metric_obj:
                value = metric_obj[metric_name].compute().item()
                metrics[f'train_{metric_name}'].append(value)
                metric_obj[metric_name].reset()
            scheduler.step(train_loss/len(train_loader))

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    x, edge_index, graph_index, targets = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.y.to(self.device)
                    out = model(x, edge_index, graph_index)
                    loss = loss_function(out, targets)
                    val_loss += loss.item()

                    for metric_name in metric_obj:
                        preds = torch.nn.functional.softmax(out, -1)
                        if tasktype == 'binary':
                            preds = preds[:,-1]
                        metric_obj[metric_name].update(preds, targets)

            metrics['val_Loss'].append(val_loss/len(val_loader))
            for metric_name in metric_obj:
                metrics[f'val_{metric_name}'].append(metric_obj[metric_name].compute().item())
                metric_obj[metric_name].reset()

            pat += 1

            if compare(metrics[f'val_{selection_metric}'][-1]+earlyStoppingMinChange, best_metric_selection):
                best_metric_selection = metrics[f'val_{selection_metric}'][-1]
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                pat = 0

            epoch += 1

            if pat == 0:
                for k in range(log_delay):
                    single_step = {}
                    k = k-log_delay
                    for key, value in metrics.items():
                        single_step[key] = value[k]
                    wandb.log(single_step)
                log_delay = 1
            else:
                log_delay +=1 


        model.eval()
        test_loss = 0
        metricsTest = {}
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, edge_index, graph_index, targets = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device), data.y.to(self.device)
                out = model(x, edge_index, graph_index)
                loss = loss_function(out, targets)
                test_loss += loss.item()

                for metric_name in metric_obj:
                    preds = torch.nn.functional.softmax(out, -1)
                    if tasktype == 'binary':
                        preds = preds[:,-1]
                    metric_obj[metric_name].update(preds, targets)

        metricsTest['test_Loss'] = test_loss/len(test_loader)
        for metric_name in metric_obj:
            metricsTest[f'test_{metric_name}'] = metric_obj[metric_name].compute().item()
            metric_obj[metric_name].reset()
        
        wandb.log(metricsTest)
        wandb.finish()

        for met, value in metrics.items():
            metrics[met] = value[best_epoch]

        metrics.update(metricsTest)
        return metrics, best_model