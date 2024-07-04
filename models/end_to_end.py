import time
import torch
from loader import load_data, load_model


def train(model, data, device, edge_index):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    for epoch in range(100):
        out = model(data, edge_index)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))


def test(model, data, adj):
    model.eval()
    for epoch in range(10):
        _, pred = model(data, adj).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
    # print('GCN Accuracy: {:.4f}'.format(acc))


def main():
    datanames = ['Cora', 'CiteSeer', 'PubMed', 'PPI']
    modelnames = ['GCN', 'GAT', 'GraphSAGE']
    device = torch.device('cuda')
    print("dataset:", datanames[0])
    datas = load_data(datanames[0], device)



    # redun_free_edge_index = redun_eliminate_qit(data).to(device)

    # a,b,utt=qit_match(torch.load('./data/cora/cora_matrix.pt').float())
    # redun_free_adj=Tensor.to_sparse(utt).to(device)
    # torch.save(redun_free_adj,'temp_qit_cora_adj.pt')
    # redun_free_adj=torch.load('temp_qit_cora_adj.pt').to(device)
    # redun_free_adj2=Tensor.to_sparse(torch.load('temp_qit_cora_adj.pt')).to(device)
    redun_free_adj = torch.load('./data/cora/cora_matrix.pt').float().to(device)
    # redun_free_edge_index=torch.ones_like(data.edge_index,dtype=torch.long).to(device)
    model = load_model(modelnames[0], datas, device)

    start = time.time()
    train(model, datas, device, redun_free_adj)
    a = time.time() - start

    start = time.time()
    test(model, datas, redun_free_adj)
    b = time.time() - start
    print("train time,inference time:%.3f/%.3f" % (a, b))


if __name__ == '__main__':
    main()
