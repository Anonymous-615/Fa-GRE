import pickle


def get_second_order_neighbors_and_count(edge_list):
    prunning = [[] * i for i in range(len(edge_list))]
    for i in range(len(edge_list)):
        second_order_neighbors = []
        for neighbor in edge_list[i]:
            neighbors_neighbor = edge_list[neighbor]
            for second_neighbor in neighbors_neighbor:
                second_order_neighbors.append(second_neighbor)
        a = set(second_order_neighbors)
        # a.discard(i)
        prunning[i] = list(a)

    return prunning


def dump_large_file(data, chunk_size, dump_file):
    with open(dump_file, 'ab') as f:
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            pickle.dump(chunk, f)

def load_large_file(load_file):
    with open(load_file, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break




name = ['cora',
        'flickr',
        'ogbn-arxiv',
        'yelp',
        'reddit',
        'amazon',
        'ogbn-products'
        ]
for i in range(5,7):
    # with open('./data/%s_edge_index' % name[i], 'rb') as file:
    #     edge_index = pickle.load(file)

    # dump_large_file(data, 1000, 'amazon_edge_index.pkl')
    edge_index=[]
    # 逐块加载
    for chunk in load_large_file('./data/%s_edge_index.pkl' % name[i]):
        for j in range(len(chunk)):
            edge_index.append(chunk[j])
    total_count = get_second_order_neighbors_and_count(edge_index)
    # with open('./data/%s_prunning_edge_index' % name[i], 'wb') as file:
    #     pickle.dump(total_count, file)
    dump_large_file(total_count, 1000, './data/%s_prunning_edge_index.pkl' % name[i])
    print(name[i], sum(len(sublist) for sublist in total_count))
