import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
import copy
from torchmetrics import Accuracy, Precision, Recall, F1Score
import XAI.metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def StrategyTop(importances, top_k):
    inds = torch.argsort(importances)[-int(top_k):]
    mask = torch.zeros_like(importances)
    mask[inds] = 1
    return mask.long(), torch.count_nonzero(mask.long()).item()

def StrategyBestK(importances, gt_expl):
    gt = gt_expl[-1].node_imp.detach().cpu()
    topk = torch.count_nonzero(gt).item()
    inds = torch.argsort(importances)[-int(topk):]
    mask = torch.zeros_like(importances)
    mask[inds] = 1
    return mask.long(), torch.count_nonzero(mask.long()).item()

def StrategyKmeans(importances):
    scaled_imp = minmaxScaler(importances.cpu())
    scaled_imp = scaled_imp.numpy().reshape(-1,1)
    kms = KMeans(n_clusters=2, init='random', n_init=100).fit(scaled_imp)
    aa = np.argmax(kms.cluster_centers_.reshape(1,-1)[0])
    lab = list(kms.labels_)
    if aa == 0:
        lab = [1 if k==0 else 0 for k in lab]

    mask = torch.tensor(lab).long()
    return mask, torch.count_nonzero(mask.long()).item(), kms

def StrategyFidelityTh(importances, model, target_class, graph):
    graph_index = torch.zeros(len(graph.x), dtype=torch.int64).cuda()
    prediction = torch.nn.functional.softmax(model(graph.x, graph.edge_index, graph_index), -1)
    prediction = prediction[0][target_class].item()
    thresholds = get_thresholds(importances)

    maxth = 0
    maxsum = -np.inf
    for th in thresholds:  
        graph_exp_mask, graph_g_mask = generate_masked_graphs(graph, importances, th, False)
        graph_index_explmask = torch.zeros(len(graph_exp_mask.x), dtype=torch.int64).cuda()
        graph_index_noxplmask = torch.zeros(len(graph_g_mask.x), dtype=torch.int64).cuda()
        expl_prediciton = torch.nn.functional.softmax(model(graph_exp_mask.x, graph_exp_mask.edge_index, graph_index_explmask), -1)[0][target_class].item()
        noexpl_prediction = torch.nn.functional.softmax(model(graph_g_mask.x, graph_g_mask.edge_index, graph_index_noxplmask), -1)[0][target_class].item()
        Fp = prediction - noexpl_prediction
        Fm = prediction - expl_prediciton
        if maxsum < Fp-Fm:
            maxsum = Fp-Fm
            maxth = th

    mask = torch.zeros_like(importances)
    for node_id, value in enumerate(importances):
        if value >= maxth:
            mask[node_id] = 1

    return mask.long(), torch.count_nonzero(mask.long()).item()

def minmaxScaler(x):
    min = torch.min(x)
    max = torch.max(x)
    range = max-min
    return (x-min)/range

def graph_to_networkx(graph, edge_imp, node_imp, node_embeddings, to_undirected=False, remove_self_loops=False, get_map=False):
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    node_list = sorted(torch.unique(graph.edge_index).tolist())
    map_norm =  {node_list[i]:i for i in range(len(node_list))}
    G.add_nodes_from([map_norm[n] for n in node_list])

    if node_embeddings is not None:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({'x': node_embeddings[map_norm[i], :]})

    for i, (u, v) in enumerate(graph.edge_index.t().tolist()):
        u = map_norm[u]
        v = map_norm[v]

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        if edge_imp is not None:
            G[u][v]['edge_imp'] = edge_imp[i].item()

    if node_imp is not None:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({'node_imp': node_imp[map_norm[i]].item()})

    if get_map:
        return G, map_norm
    return G

def get_thresholds(attributions):
    thresholds = []
    sorted, _ = torch.sort(attributions)
    sorted_list = sorted.tolist()
    thresholds = np.array([th for th in sorted_list])
    thresholds = np.unique(thresholds)
    return list(thresholds)[1:]

def generate_masked_graphs(graph, attributions, threshold, logical):
    node_g = []
    node_exp = []
    graph_exp_mask = copy.deepcopy(graph)
    graph_g_mask = copy.deepcopy(graph)

    attr = attributions.tolist()

    for node_id, value in enumerate(attr):
        if value >= threshold:
            node_exp.append(node_id)
        else:
            node_g.append(node_id)
    
    if logical:
        node_exp = torch.tensor(node_exp)
        node_g = torch.tensor(node_g)
        
        mask_exp = torch.zeros(len(attr))
        mask_exp[node_exp] = 1
        mask_exp = mask_exp.reshape(-1,1)

        mask_g = torch.zeros(len(attr))
        mask_g[node_g] = 1
        mask_g = mask_g.reshape(-1,1)

        graph_exp_mask.x = graph_exp_mask.x * mask_exp.cuda()
        graph_g_mask.x = graph_g_mask.x * mask_g.cuda()
        
    else:
        G_exp = graph_to_networkx(graph, None, attributions, graph_exp_mask.x)
        g_exp = nx.subgraph(G_exp,node_exp)
        graph_exp_mask = from_networkx(g_exp)

        G_g = graph_to_networkx(graph, None, attributions, graph_g_mask.x)
        g = nx.subgraph(G_g,node_g)
        graph_g_mask = from_networkx(g)

    return graph_exp_mask.cuda(), graph_g_mask.cuda()

def compute_Accuracy(gt_exp, hard_masked_node_imp) -> float:
    accuracy = Accuracy(task="binary", num_classes=2).cuda()
    acc_accumulator = []

    for exp in gt_exp:
        acc_accumulator.append(accuracy(hard_masked_node_imp.cuda(), exp.node_imp.cuda()).item())
        accuracy.reset()

    return float(max(acc_accumulator))

def compute_Ksilhouette(generated_exp, mask):
    scaled_imp = minmaxScaler(generated_exp.node_imp.cpu())
    scaled_imp = scaled_imp.numpy().reshape(-1,1)
    score = silhouette_score(scaled_imp, mask.cpu().detach().numpy())
    return float(score)

def compute_Precision(gt_exp, hard_masked_node_imp) -> float:
    accuracy = Precision(task="binary", num_classes=2).cuda()
    acc_accumulator = []

    for exp in gt_exp:
        acc_accumulator.append(accuracy(hard_masked_node_imp.cuda(), exp.node_imp.cuda()).item())
        accuracy.reset()
    return float(max(acc_accumulator))

def compute_Recall(gt_exp, hard_masked_node_imp) -> float:
    accuracy = Recall(task="binary", num_classes=2).cuda()
    acc_accumulator = []

    for exp in gt_exp:
        acc_accumulator.append(accuracy(hard_masked_node_imp.cuda(), exp.node_imp.cuda()).item())
        accuracy.reset()
    return float(max(acc_accumulator))

def compute_F1(gt_exp, hard_masked_node_imp) -> float:
    accuracy = F1Score(task="binary", num_classes=2).cuda()
    acc_accumulator = []

    for exp in gt_exp:
        acc_accumulator.append(accuracy(hard_masked_node_imp.cuda(), exp.node_imp.cuda()).item())
        accuracy.reset()
    return float(max(acc_accumulator))

def compute_F1Fidelity(model, graph, hard_masked_node_imp, target_class, logical):
    graph_index = torch.zeros(len(graph.x), dtype=torch.int64).cuda()
    prediction = torch.nn.functional.softmax(model(graph.x, graph.edge_index, graph_index), -1)
    nclasses = prediction.size()[-1]
    prediction = prediction[0][target_class].item()

    graph_exp_mask, graph_g_mask = generate_masked_graphs(graph, hard_masked_node_imp, 0.5, logical)
    graph_index_explmask = torch.zeros(len(graph_exp_mask.x), dtype=torch.int64).cuda()
    graph_index_noxplmask = torch.zeros(len(graph_g_mask.x), dtype=torch.int64).cuda()
    expl_prediciton = torch.nn.functional.softmax(model(graph_exp_mask.x, graph_exp_mask.edge_index, graph_index_explmask), -1)[0][target_class].item()
    noexpl_prediction = torch.nn.functional.softmax(model(graph_g_mask.x, graph_g_mask.edge_index, graph_index_noxplmask), -1)[0][target_class].item()
    Fp = prediction - noexpl_prediction
    Fm = prediction - expl_prediciton

    Fm_norm = (nclasses/(2*nclasses-1))*(1-Fm)

    try:
        F1f = 2*(((1-Fm)*Fp)/(1-Fm+Fp))
    except:
        F1f = 0

    return float(F1f), float(Fm_norm), float(Fp), float(Fm)

def getMetrics(metrics, gt_exp, gen_exp, example_graph, model, filterStrategy, target_class, logical, last, filter_kwargs={}, forward_kwargs = {}):
    result = {}

    if last == True:
        gt_exp = [gt_exp[-1]]

    if filterStrategy.__name__ == 'StrategyFidelityTh':
        filter_kwargs['graph'] = example_graph
        filter_kwargs['model'] = model
        filter_kwargs['target_class'] = target_class
    
    if filterStrategy.__name__ == 'StrategyBestK':
        filter_kwargs['gt_expl'] = gt_exp
    
    if filterStrategy.__name__ == 'StrategyKmeans':
        hard_masked_node_imp, k, kms = filterStrategy(gen_exp.node_imp, **filter_kwargs)
    else:
        hard_masked_node_imp, k = filterStrategy(gen_exp.node_imp, **filter_kwargs)

    hard_masked_node_imp.cuda()

    for met in metrics:
        if met == 'Accuracy':
            f = getattr(XAI.metrics, f"compute_{met}")
            result[met] = f(gt_exp, hard_masked_node_imp)
        elif met == 'F1Fidelity':
            f = getattr(XAI.metrics, f"compute_{met}")
            F1f, Fm_norm, Fp, Fm = f(model, example_graph, hard_masked_node_imp, target_class, logical)
            result[f'FidelityP'] = Fp
            result[f'FidelityM'] = Fm
            result[f'F1Fidelity'] = F1f
            result[f'FidelityM_norm'] = Fm_norm
        elif met == 'Ksilhouette':
            f = getattr(XAI.metrics, f"compute_{met}")
            result[met] = f(gen_exp, hard_masked_node_imp)
        elif met == 'Precision':
            f = getattr(XAI.metrics, f"compute_{met}")
            result[met] = f(gt_exp, hard_masked_node_imp)
        elif met == 'Recall':
            f = getattr(XAI.metrics, f"compute_{met}")
            result[met] = f(gt_exp, hard_masked_node_imp)
        elif met == 'F1':
            f = getattr(XAI.metrics, f"compute_{met}")
            result[met] = f(gt_exp, hard_masked_node_imp)
        elif met == 'K':
            result[met] = k
        else:
            continue

    return result