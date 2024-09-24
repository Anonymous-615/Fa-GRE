import math
import pickle
from collections import Counter

import torch


def count_and_sort_elements(lst):
    # Step 1: Count the occurrences of each element
    element_counts = Counter(lst)

    # Step 2: Sort the elements by count (from high to low)
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)

    return sorted_elements


def calculate_neighbor_intersections(edge_list):
    n = len(edge_list)
    total_hit_times = 0
    # Step 1: Convert each neighbor list to a set for easy intersection calculation
    neighbor_sets = [set(neighbors) for neighbors in edge_list]

    # Step 2: Initialize a list to store the intersections for each node
    # intersections = [[] for _ in range(n)]
    # Step 3: Calculate intersections
    for i in range(n):
        if i != 0 and i % 100 == 0:
            print(i, total_hit_times, total_hit_times / i / len(edge_list))
        for j in range(n):
            if i != j:
                intersection = neighbor_sets[i].intersection(neighbor_sets[j])
                hit_time = group(intersection, group_list)
                total_hit_times += hit_time
    return total_hit_times


def group_nodes_by_degree(edge_list):
    n = len(edge_list)
    sqrt_n = int(math.sqrt(n))

    # Step 1: Calculate the degree of each node
    degrees = [(i, len(neighbors)) for i, neighbors in enumerate(edge_list)]

    # Step 2: Sort nodes by degree (from small to large)
    degrees_sorted = sorted(degrees, key=lambda x: x[1])

    # Step 3: Group nodes into sqrt(n) groups
    group_list = [0] * n
    group_size = sqrt_n
    group_id = 0

    for idx, (node, degree) in enumerate(degrees_sorted):
        if idx % group_size == 0 and group_id < sqrt_n:
            group_id += 1
        group_list[node] = group_id

    return group_list


def group(inter_list, group_list):
    rearranged_list = [group_list[i] for i in inter_list]
    return len(set(rearranged_list))


name = ['cora',
        'flickr',
        'ogbn-arxiv',
        'yelp',
        'reddit'
        ]
i = 0
with open('./data/%s_edge_index' % name[i], 'rb') as file:
    edge_index = pickle.load(file)
with open('./data/%s_prunning_neighbors_index' % name[i], 'rb') as file:
    filtered_neighbors_list = pickle.load(file)

group_list = group_nodes_by_degree(edge_index)

index_mat = torch.zeros([len(edge_index), math.ceil((math.sqrt(len(edge_index))))], dtype=torch.bool)
for i in range(len(filtered_neighbors_list)):
    zero_index = torch.zeros([index_mat.shape[1]], dtype=torch.bool)
    for j in range(len(filtered_neighbors_list[i])):
        zero_index[group_list[filtered_neighbors_list[i][j]]] = 1
    index_mat[i] = zero_index


# print(group_list[1701],group_list[1866],group_list[926],group_list[2582],group_list[1166],group_list[1862])
# print(torch.where(index_mat[0]))
def calculate_mat_intersections(index_mat, filtered_neighbors_list, group_list):
    total_hit_times = 0
    for i in range(index_mat.shape[0]):
        if i != 0 and i % 100 == 0:
            print(i, total_hit_times)
        for j in range(len(filtered_neighbors_list[i])):
            intersection = torch.where(index_mat[i] & index_mat[filtered_neighbors_list[i][j]])[0]
            selected_elements = torch.tensor(group_list)[intersection]
            unique_count = torch.unique(selected_elements).numel()
            hit_time = unique_count
            total_hit_times += hit_time
    return total_hit_times



print("剪枝后二级索引乘命中次数为",calculate_mat_intersections(index_mat,filtered_neighbors_list,group_list))