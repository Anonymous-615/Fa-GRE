import pickle


def count_and_sum_second_order_neighbors(edge_list):
    total_count = 0

    # Iterate over each node
    for i in range(len(edge_list)):
        # Get the neighbors of node i
        neighbors_i = edge_list[i]

        # Set to store unique second-order neighbors
        second_order_neighbors = set()

        # Find the second-order neighbors
        for neighbor in neighbors_i:
            if neighbor < len(edge_list):
                # Get the neighbors of this neighbor
                neighbors_neighbor = edge_list[neighbor]
                for second_neighbor in neighbors_neighbor:
                    # Add the second-order neighbor if it is greater than i
                    if second_neighbor > i:
                        second_order_neighbors.add(second_neighbor)
        # Count the number of valid second-order neighbors
        count = len(second_order_neighbors)
        total_count += count
    return total_count


with open('./data/amazon_edge_index', 'rb') as file:
    edge_index = pickle.load(file)

total_count = count_and_sum_second_order_neighbors(edge_index)
print(total_count)
