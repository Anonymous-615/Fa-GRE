import torch
from torch import Tensor
from torch_geometric.datasets import PPI, Planetoid

from models.models import GAT, GCN, GraphSAGE





def load_data(name, device):
    if name == 'PPI':
        dataset = PPI(root='./data/PPI')
        data = dataset[18].to(device)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[:int(0.8 * data.num_nodes)] = True  # 将前80%的节点标记为True，表示在训练集中
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[:121] = True
        data.train_mask = train_mask
        data.test_mask = test_mask
    else:
        dataset = Planetoid(root='./data/' + name + '/', name=name)
        data = dataset[0].to(device)
    num_node_features = dataset.num_node_features


    return [data, num_node_features, dataset.num_classes,]


def load_model(name, datas, device):
    num_node_features, num_classes = datas[1], datas[2]
    if name == 'GCN':
        return GCN(num_node_features, num_classes).to(device)

    if name == 'GAT':
        return GAT(num_node_features, num_classes, alpha=0.2, nheads=8).to(device)
    if name == 'GraphSAGE':
        return GraphSAGE(num_node_features, num_classes).to(device)
