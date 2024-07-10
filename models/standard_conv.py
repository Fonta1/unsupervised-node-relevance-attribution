import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn.conv import GINConv
from torch.nn import ModuleList

class STDC(MessagePassing):
    def __init__(self, in_channel, emb_channel, out_channel, n_layers):
        super().__init__()
        self.layers = ModuleList()
        self.in_channels = in_channel
        self.emb_channels = emb_channel
        self.out_channels = out_channel
        self.n_layers = n_layers
        self.activation = ReLU()

        if n_layers == 1:
            self.lastEmbeddingLayer = GINConv(Linear(in_channel, emb_channel, True))
        else:
            for i in range(n_layers-1): 
                if i == 0:
                    self.layers.append(GINConv(Linear(in_channel, emb_channel, True)))
                else:
                    self.layers.append(GINConv(Linear(emb_channel, emb_channel, True)))
            self.lastEmbeddingLayer = GINConv(Linear(emb_channel, emb_channel, True))

        self.out = Linear(n_layers*emb_channel, out_channel, True)

    def embed_node_level(self, x, edge_index, graph_index):
        for i, lay in enumerate(self.layers):
            if i == 0:
                x = torch.cat((x, self.activation(lay(x, edge_index))),-1)
            else:
                x = torch.cat((x, self.activation(lay(x[:, -self.emb_channels:], edge_index))),-1)
        x = torch.cat((x, self.activation(self.lastEmbeddingLayer(x[:,-self.emb_channels:], edge_index))),-1)
        return x[:,self.in_channels:]

    def embed_graph_level(self, x, edge_index, graph_index):
        layer_wise_concat = self.embed_node_level(x, edge_index, graph_index)
        graph_emebdding = global_add_pool(layer_wise_concat, graph_index)  
        return graph_emebdding

    def forward(self, x, edge_index, graph_index):
        graph_emebdding = self.embed_graph_level(x, edge_index, graph_index)
        y = self.out(graph_emebdding)
        return y