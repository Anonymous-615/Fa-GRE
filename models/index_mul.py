import math
import pickle
from collections import Counter


def count_and_sort_elements(lst):
    # Step 1: Count the occurrences of each element
    element_counts = Counter(lst)

    # Step 2: Sort the elements by count (from high to low)
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)

    return sorted_elements

def calculate_neighbor_intersections(edge_list):
    n = len(edge_list)
    total_hit_times=0
    # Step 1: Convert each neighbor list to a set for easy intersection calculation
    neighbor_sets = [set(neighbors) for neighbors in edge_list]

    # Step 2: Initialize a list to store the intersections for each node
    #intersections = [[] for _ in range(n)]
    # Step 3: Calculate intersections
    for i in range(n):
        if i!=0 and i%100==0:
            print(i,total_hit_times,total_hit_times/i/len(edge_list))
        for j in range(n):
            if i != j:
                intersection = neighbor_sets[i].intersection(neighbor_sets[j])
                hit_time=group(intersection,group_list)
                total_hit_times+=hit_time
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

def group(inter_list,group_list):
    rearranged_list = [group_list[i] for i in inter_list]
    return len(set(rearranged_list))


with open ('./data/reddit_edge_index','rb') as file:
    edge_list = pickle.load(file)
group_list = group_nodes_by_degree(edge_list)


print(calculate_neighbor_intersections(edge_list))