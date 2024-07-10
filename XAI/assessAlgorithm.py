from graphxai.utils import Explanation
from torch_geometric.explain import Explainer, CaptumExplainer,  GNNExplainer, DummyExplainer
from statistics import mean, pstdev
import torch
import pandas as pd
import pickle
import yaml
import time
import os
from XAI.CAM import node_importance
from XAI.metrics import getMetrics

class Explain():
    def __init__(self, model, dataset, splits, result_path, last):

        self.model = model.cuda()
        self.splits = splits
        self.dataset = dataset
        self.result_path = result_path
        self.last = last

        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

    def getExplainer(self, algorithm):
        explainer = None
        if algorithm in ['GNNExplainer']:

            explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200, lr=0.01),
            explanation_type='phenomenon',
            node_mask_type='object',
            edge_mask_type=None,
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        elif algorithm in ['IntegratedGradients']:
            node_mask_type = 'attributes'
            edge_mask_type = 'object'

            alg = CaptumExplainer(algorithm)

            explainer = Explainer(
                model=self.model,
                algorithm=alg,
                explanation_type='phenomenon',
                node_mask_type=node_mask_type,
                edge_mask_type=edge_mask_type,
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw', 
                ),
            )
        elif algorithm in ['RandomExplainer']:
            
            node_mask_type = 'object'
            edge_mask_type = 'object'

            explainer = Explainer(
                model=self.model,
                algorithm=DummyExplainer(),
                explanation_type='phenomenon',
                node_mask_type=node_mask_type,
                edge_mask_type=edge_mask_type,
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )
        elif algorithm in ['CAM']:
            return explainer
        else:
            raise Exception('XAI algorithm doesn\'t exist')
        
        return explainer

    def getExplanationGraph(self, algorithm, explainer, index, cl):
        example_graph = self.dataset[index].cuda()
        graph_index = torch.zeros(len(example_graph.x), dtype=torch.int64).cuda()
        exp_target = torch.zeros(self.dataset.num_classes).long().cuda()
        exp_target[cl] = 1

        if algorithm in ['RandomExplainer']:
            explanation = explainer(example_graph.x, example_graph.edge_index, target=exp_target, graph_index = graph_index)
            gen_exp = Explanation(node_imp=explanation.node_mask.flatten())
            gen_exp.set_whole_graph(example_graph)
        elif algorithm in ['IntegratedGradients']:
            explanation = explainer(example_graph.x, example_graph.edge_index, target=torch.tensor([[cl]]), graph_index = graph_index)
            node_mask = self.feature_to_node_mask(explanation.node_mask)
            gen_exp = Explanation(node_imp=node_mask)
            gen_exp.set_whole_graph(example_graph)
        elif algorithm in ['GNNExplainer']:
            explanation = explainer(example_graph.x, example_graph.edge_index, target=exp_target, graph_index = graph_index)
            gen_exp = Explanation(node_imp=explanation.node_mask.flatten())
            gen_exp.set_whole_graph(example_graph)
        elif algorithm in ['CAM']:
            relevance = node_importance(self.model, example_graph)
            gen_exp = Explanation(node_imp=relevance[:, example_graph.y.item()])
            gen_exp.set_whole_graph(example_graph)
        return gen_exp


    def assessAlgorithm(self, metrics, algorithm, filterStrategy = None, filter_kwargs = {}, classes=[0,1]):
        
        global_accumulator = None
        global_interesting_examples = None
        global_summary = {}
        alg_path = f'{self.result_path}/{algorithm}/'

        if 'F1fidelityMovingTh' in metrics and ('Sufficiency' not in metrics):
            metrics.extend(['Sufficiency', 'NormalSufficiency', 'Comprehensiveness'])
        if 'F1Fidelity' in metrics and ('FidelityP' not in metrics):
            metrics.extend(['FidelityP', 'FidelityM', 'FidelityM_norm'])

        for cl in classes:   
            rspec_path = f'{self.result_path}/{algorithm}/{cl}/{filterStrategy.__name__}'
            if filterStrategy.__name__ == 'StrategyTopK' or filterStrategy.__name__ == 'StrategyTop':
                top = filter_kwargs['top_k']
                rspec_path = f'{self.result_path}/{algorithm}/{cl}/{filterStrategy.__name__}{top}'
            if not os.path.isdir(rspec_path):
                os.makedirs(rspec_path)

            report = {'sample_index': []}
            for met in metrics:
                report[f'Accumulator{met}'] = []

            explanations = {}
            if os.path.isfile(f'{self.result_path}/{algorithm}/{cl}/Explanation.pkl'):
                with open(f'{self.result_path}/{algorithm}/{cl}/Explanation.pkl', "rb") as f:
                    explanations = pickle.load(f)
           
            tclass = []
            pclass = []
            execTime = []

            number_of_samples = len(self.dataset[self.splits['test']])
            interesting_examples = []

            for i, index in enumerate(self.splits['test']):
                example_graph = self.dataset[index].cuda()
                graph_index = torch.zeros(len(example_graph.x), dtype=torch.int64).cuda()
                graph_expl = self.dataset.explanations[index]

                pred_class = torch.argmax(self.model(example_graph.x.cuda(), example_graph.edge_index.cuda(), graph_index.cuda()),-1)

                if example_graph.y.item() != int(cl) or pred_class != int(cl):
                    continue
                
                if index in list(explanations.keys()):
                    gen_exp = explanations[index]
                    elapsed = 0
                else:
                    start = time.time()
                    explainer = self.getExplainer(algorithm)
                    try:
                        gen_exp = self.getExplanationGraph(algorithm, explainer, index, cl)
                    except Exception as e:
                        print(e)
                        print(f'Error at index: {index}')
                        continue
                    end = time.time()
                    elapsed = end-start
                    explanations[index] = gen_exp

                try:
                    computed_metrics = getMetrics(metrics, graph_expl, gen_exp, example_graph, self.model, filterStrategy, cl, False, self.last, filter_kwargs, forward_kwargs = {'graph_index':graph_index})
                except Exception as e:
                        print(e)
                        print(f'Error metrics at index: {index}')
                        continue
                
                tclass.append(example_graph.y.item())
                pclass.append(pred_class.item())
                execTime.append(elapsed)
                report['sample_index'].append(index)
                for met in metrics:
                    report[f'Accumulator{met}'].append(computed_metrics[met])

                if computed_metrics['F1'] > 0.8:
                    interesting_examples.append(index)


            report['TargetClass'] = tclass
            report['PredictedClass'] = pclass
            report['ElapsedTime'] = execTime

            for key, elem in report.items():
                print(len(elem), key)

            results = pd.DataFrame(report)
            results.to_csv(f'{rspec_path}/Expl_results_list_Lst{self.last}.csv')

            summary = {'interesting_examples': str(interesting_examples),
                    'number_of_examples' : number_of_samples,
                    'AverageTime': mean(execTime)}
            
            for met in metrics:
                    summary[f'Avg{met}'] = mean(report[f'Accumulator{met}'])
                    summary[f'STD{met}'] = pstdev(report[f'Accumulator{met}'])

            with open(f'{rspec_path}/Summary.yaml', "w") as file:
                yaml.dump(summary, file, default_flow_style=False, Dumper=yaml.SafeDumper)

            if not os.path.isfile(f'{self.result_path}/{algorithm}/{cl}/Explanation.pkl'):
                with open(f'{self.result_path}/{algorithm}/{cl}/Explanation.pkl', "wb") as f:
                    pickle.dump(explanations, f)

            if global_accumulator == None:
                global_accumulator = report
            else:
                for key in report.keys():
                    global_accumulator[key].extend(report[key])

            if global_interesting_examples == None:
                global_interesting_examples = interesting_examples
            else:
                global_interesting_examples.extend(interesting_examples)

        for met in metrics:
            global_summary[f'Avg{met}'] = mean(global_accumulator[f'Accumulator{met}'])
            global_summary[f'STD{met}'] = pstdev(global_accumulator[f'Accumulator{met}'])

        global_summary['AvgTime'] =  mean(global_accumulator[f'ElapsedTime'])
        global_summary['STDTime'] =  pstdev(global_accumulator[f'ElapsedTime'])


        path = f'{alg_path}/Global_Summary_{str(classes)}_{filterStrategy.__name__}Lst{self.last}.yaml'
        if filterStrategy.__name__ == 'StrategyTopK' or filterStrategy.__name__ == 'StrategyTop':
            top = filter_kwargs['top_k']
            path = f'{alg_path}/Global_Summary_{str(classes)}_{filterStrategy.__name__}{top}Lst{self.last}.yaml'
        with open(path, "w") as file:
            yaml.dump(global_summary, file, default_flow_style=False, Dumper=yaml.SafeDumper)

        return global_summary, global_interesting_examples
        

    def assessAllAlgorithms(self, metrics, filterStrategy = None, filter_kwargs = {},  classes=[0,1]):
        
        if 'F1fidelityMovingTh' in metrics:
            metrics.extend(['Sufficiency', 'NormalSufficiency', 'Comprehensiveness'])
        if 'F1Fidelity' in metrics:
            metrics.extend(['FidelityP', 'FidelityM', 'FidelityM_norm'])

        report = {'AlgorithmName': []}
        for elem in metrics:
            report[f'Avg{elem}'] = []
            report[f'STD{elem}'] = []
        report[f'AvgTime'] = []
        report[f'STDTime'] = []

        interesting_examples = {}
        for algorithm in ['RandomExplainer', 'CAM', 'IntegratedGradients', 'GNNExplainer']:
            print(algorithm)
            results, examples = self.assessAlgorithm(metrics, algorithm, filterStrategy, filter_kwargs, classes)
            interesting_examples[algorithm] = examples 

            report['AlgorithmName'].append(algorithm)
            for elem in metrics:
                report[f'Avg{elem}'].append(results[f'Avg{elem}'])
                report[f'STD{elem}'].append(results[f'STD{elem}'])
            report[f'AvgTime'].append(results[f'AvgTime'])
            report[f'STDTime'].append(results[f'STDTime'])

        reportPath = f'{self.result_path}/SummaryAlgorithms/{classes}/'
        if not os.path.isdir(reportPath):
            os.makedirs(reportPath)

        res = pd.DataFrame(report)
        rdf = res.round(decimals=3)

        metrics.extend(['Time'])
        for m in metrics:
            avg = 'Avg'+m
            std = 'STD'+m
            rdf[m] = rdf[avg].astype(str) +" ± "+ rdf[std].astype(str) 
            rdf = rdf.drop(avg, axis=1)
            rdf = rdf.drop(std, axis=1)

        path1 = f'{reportPath}/Expl_results_{filterStrategy.__name__}Lst{self.last}.csv'
        path2 = f'{reportPath}/Examples_report_{filterStrategy.__name__}Lst{self.last}.yaml'
        if filterStrategy.__name__ == 'StrategyTopK' or filterStrategy.__name__ == 'StrategyTop':
            top = filter_kwargs['top_k']
            path1 = f'{reportPath}/Expl_results_{filterStrategy.__name__}{top}Lst{self.last}.csv'
            path2 = f'{reportPath}/Examples_report_{filterStrategy.__name__}{top}Lst{self.last}.yaml'

        rdf.to_csv(path1)
        with open(path2, "w") as file:
            yaml.dump(str(interesting_examples), file, default_flow_style=False, Dumper=yaml.SafeDumper)
    
        return report, interesting_examples
    
    def edge_to_node_mask(self, graph, edge_mask):
        node_mask = torch.zeros(len(graph.x))
        for elem in range(len(graph.x)):
            indices = torch.unique((graph.edge_index == elem).nonzero(as_tuple = True)[1])
            importance = torch.sum(edge_mask[indices])/len(indices)
            node_mask[elem] = importance
        return node_mask
    
    def feature_to_node_mask(self, feature_mask):
        return torch.sum(feature_mask, dim=1)