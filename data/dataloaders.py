import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle as pkl
from igraph import Graph
import importlib
import pickle
from graphxai.utils import Explanation
import numpy as np

class ChemXAI(InMemoryDataset):
    def __init__(self, root = f'./data/datasets/', transform=None, pre_transform=None, pre_filter=None, name='Mutagenicity'):
        root = f'{root}/{name}'
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.name == 'Mutagenicity':
            return ['Mutagenicity_A.txt', 'Mutagenicity_edge_labels.txt', 
                    'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 
                    'Mutagenicity_label_readme.txt', 'Mutagenicity_node_labels.txt']
        else:
            return [f'{self.name}.npz']

    @property
    def processed_file_names(self):
        return [f'data.pt', f'explain.pkl']

    def process(self):
        module = importlib.import_module("graphxai.datasets")
        dataset_class = getattr(module, self.name)
        if self.name == 'Mutagenicity':
            dataset = dataset_class(seed = 5843997, root = self.root.split('Mutagenicity', 2)[0])
        else:
            dataset = dataset_class(seed = 5843997, data_path = self.raw_paths[0])
            print(self.raw_paths[0])
        
        data_list, explanations = dataset.get_data_list(range(len(dataset)))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pickle.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pickle.load(f)
        


class BA2grid(InMemoryDataset):
    def __init__(self, root = './data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'BA2grid'
        root = f'{root}/{self.name}'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BA-2grid.pkl']

    @property
    def processed_file_names(self):
        return ['BA-2grid.pt', 'explain.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        explanations = []

        for adj, lab, target in zip(adjs, feas, labels):
            ig = Graph.Adjacency(adj, mode = 'undirected')
            edjI = []
            for edge in ig.es:
                e = list(edge.tuple)
                edjI.append(e)
                edjI.append(list(reversed(e)))

            graph = Data(x = torch.FloatTensor(lab), edge_index= torch.LongTensor(edjI).t().contiguous(), y = torch.LongTensor([target]))
            data_list.append(graph)

            if target == 1:
                z = torch.zeros(len(adj))
                z[-9:] = 1
                explanations.append([Explanation(graph=graph, node_imp=z)])
            else:
                z = torch.zeros(len(adj))
                explanations.append([Explanation(graph=graph, node_imp=z)]) 

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)



class GridHouse(InMemoryDataset):
    def __init__(self, root = './data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'GridHouse'
        root = f'{root}/{self.name}'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BA-2grid-house.pkl']

    @property
    def processed_file_names(self):
        return ['BA-2grid-house.pt', 'explain.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        explanations = []

        for adj, lab, target in zip(adjs, feas, labels):
            ig = Graph.Adjacency(adj, mode = 'undirected')
            edjI = []
            for edge in ig.es:
                e = list(edge.tuple)
                edjI.append(e)
                edjI.append(list(reversed(e)))

            graph = Data(x = torch.FloatTensor(lab), edge_index= torch.LongTensor(edjI).t().contiguous(), y = torch.LongTensor([target]))
            data_list.append(graph)

            if target == 1:
                z = torch.zeros(len(adj))
                z[-14:] = 1
                explanations.append([Explanation(graph=graph, node_imp=z)])
            else:
                z = torch.zeros(len(adj))
                mh = self._my_house()
                mg = self._my_grid()

                sb1 = ig.induced_subgraph(ig.vs[-5:])
                sb2 = ig.induced_subgraph(ig.vs[-9:])

                if sb1.isomorphic(mh):
                    z[-5:] = 1
                    explanations.append([Explanation(graph=graph, node_imp=z)]) 
                else:
                    z[-9:] = 1
                    explanations.append([Explanation(graph=graph, node_imp=z)]) 

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)
        
    def _my_house(self):
        house = Graph()
        house.add_vertices(5)
        house.add_edges([(0,1), (1,2), (2,3), (0,3), (3,4), (2,4)])
        return(house)

    def _my_grid(self):
        grid = Graph()
        grid.add_vertices(9)
        grid.add_edges([(0,1), (1,2), (0,3), (1,4), (2,5), (3,4), (4,5), (3,6), (4,7), (5,8), (6,7), (7,8)])
        return grid


class HouseColors(InMemoryDataset):
    def __init__(self, root = './data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'HouseColors'
        root = f'{root}/{self.name}'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BA-houses_color.pkl']

    @property
    def processed_file_names(self):
        return ['BA-houses_color.pt', 'explain.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            (adjs, feas, labels, Gt) = pkl.load(fin)

        data_list = []
        explanations = []

        for adj, lab, target, gt in zip(adjs, feas, labels, Gt):
            ig = Graph.Adjacency(adj, mode = 'undirected')
            edjI = []
            for edge in ig.es:
                e = list(edge.tuple)
                edjI.append(e)
                edjI.append(list(reversed(e)))

            graph = Data(x = torch.FloatTensor(lab), edge_index= torch.LongTensor(edjI).t().contiguous(), y = torch.LongTensor([target]))
            data_list.append(graph)

            z = torch.zeros(len(adj))
            z[gt[0][0]:gt[0][-1]+1] = 1
            explanations.append([Explanation(graph=graph, node_imp=z)])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)


class BA2Motif(InMemoryDataset):
    def __init__(self, root = './data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'BA2Motif'
        root = f'{root}/{self.name}'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BA-2motif.pkl']

    @property
    def processed_file_names(self):
        return ['BA2Motif.pt', 'explain.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        explanations = []

        for adj, lab, target in zip(adjs, feas, labels):
            ig = Graph.Adjacency(adj, mode = 'undirected')
            edjI = []
            for edge in ig.es:
                e = list(edge.tuple)
                edjI.append(e)
                edjI.append(list(reversed(e)))

            target = np.argmax(target, axis=0)

            if not ig.is_connected():
                continue

            graph = Data(x = torch.FloatTensor(lab), edge_index= torch.LongTensor(edjI).t().contiguous(), y = torch.LongTensor([target]))
            data_list.append(graph)

            sb1 = ig.induced_subgraph(ig.vs[-5:])
            if target == 0:
                mg = self._my_circle()
                if not sb1.isomorphic(mg):
                    raise Exception
                z = torch.zeros(len(adj))
                z[-5:] = 1
                explanations.append([Explanation(graph=graph, node_imp=z)])
            else:
                mh = self._my_house()
                if not sb1.isomorphic(mh):
                    raise Exception
                z = torch.zeros(len(adj))
                z[-5:] = 1
                explanations.append([Explanation(graph=graph, node_imp=z)])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)
        
    def _my_house(self):
        house = Graph()
        house.add_vertices(5)
        house.add_edges([(0,1), (1,2), (2,3), (0,3), (3,4), (2,4)])
        return(house)

    def _my_circle(self):
        circle = Graph()
        circle.add_vertices(5)
        circle.add_edges([(0,1), (1,2), (2,3), (3,4), (4,0)])
        return circle