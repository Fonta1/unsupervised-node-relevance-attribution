import torch
import copy
import numpy as np
import importlib
import operator

def train_model(model, train_loader, validation_loader, combination, task_info, device):

    epochs = combination['max_epochs']
    lr = combination ['lr']
    patience = combination['patience']
    weight_decay = combination['weight_decay']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = .5, min_lr= 1e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    selection_metric = task_info['model_selection_metrics'][0].split('.')[-1]
    selection_obj = task_info['target'][0]
    if selection_obj == 'Minimize':
        compare = operator.lt
        best_metric_selection = np.inf
    else:
        compare = operator.gt
        best_metric_selection = -np.inf

    metrics = {'train_Loss_seq': [],
               'val_Loss_seq': []}
    
    metric_obj = {}
    for metric in task_info['metrics']:
        metric_name = metric.split('.')[-1]
        metrics[f'train_{metric_name}_seq'] = []
        metrics[f'val_{metric_name}_seq'] = []
        tasktype = task_info['task'][0]
        num_classes = task_info['num_classes'][0]
        metric_module = '.'.join(metric.split('.')[:-1])
        module = importlib.import_module(metric_module)
        metric_class = getattr(module, metric_name)
        metric_obj[metric_name] = metric_class(task = tasktype, num_classes=num_classes).to(device)

    epoch = 0
    best_epoch = 0
    pat = 0

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
        
        
        metrics['train_Loss_seq'].append(train_loss/len(train_loader))
        for metric_name in metric_obj:
            metrics[f'train_{metric_name}_seq'].append(metric_obj[metric_name].compute().item())
            metric_obj[metric_name].reset()
        scheduler.step(train_loss/len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                x, edge_index, graph_index, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
                out = model(x, edge_index, graph_index)
                loss = loss_function(out, targets)
                val_loss += loss.item()

                for metric_name in metric_obj:
                    preds = torch.nn.functional.softmax(out, -1)
                    if tasktype == 'binary':
                        preds = preds[:,-1]
                    metric_obj[metric_name].update(preds, targets)

        metrics['val_Loss_seq'].append(val_loss/len(validation_loader))
        for metric_name in metric_obj:
            metrics[f'val_{metric_name}_seq'].append(metric_obj[metric_name].compute().item())
            metric_obj[metric_name].reset()

        pat += 1

        if compare(metrics[f'val_{selection_metric}_seq'][-1], best_metric_selection):
            best_metric_selection = metrics[f'val_{selection_metric}_seq'][-1]
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            pat = 0

        epoch += 1

    model = best_model.eval()
    return metrics, best_epoch