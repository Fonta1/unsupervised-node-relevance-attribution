import torch

def node_importance(model, example):
    model.eval()
    model.cuda()
    graph_index = torch.zeros(len(example.x), dtype=torch.int64)
    linw = model.out.weight.t()
    embedding_matrix = model.embed_node_level(example.x.cuda(), example.edge_index.cuda(), graph_index.cuda())

    importance = embedding_matrix @ linw
    return importance.detach()