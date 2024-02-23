#########################################
#           Abul Kalam Azad             #
#           CWID: 12210325              #
#########################################

# Reading Given Graph File, forming Full Adjacency Matrix, drawing the graph, Apply diffetent TSP algorithms, and finally draw the Tour graph.
# Uncomment the respective Function call in the main function using which we want the solution

# Import Packages
import networkx as nx
import matplotlib.pyplot as plt
from sys import maxsize
from itertools import permutations
import datetime
import math
import random
from random import shuffle, randint
import re
import os

# Generate complete Adjacency Matrix from given lower triangular matrix
def complete_adj_matrix(lines, num_nodes):
    adj_matrix = []
    for line in lines:
        distances = line.split()    # Split each line to a list of distances
        distances = [int(distance) for distance in distances]       # Convert string elements to integer
        distances += [0] * (num_nodes - len(distances))         # Fill up the missing values with zeroes
        adj_matrix.append(distances)        # Append row to the adjacency matrix

    # Copy lower triangle distances to upper triangle by maintaining symmetry
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if(i != j):
                if(i != num_nodes - 1) or (j != 0):
                    adj_matrix[i][j] = adj_matrix[j][i]

        # Optional: To print complete adjacency matrix, uncomment below lines
        '''
        for j in range(len(adj_matrix)):
            print(adj_matrix[i][j], end=" ")    # Print element with a space
        print()
        '''
    return adj_matrix      

# Optional: Draw the complete graph and save the graph
def draw_graph(adj_matrix):
    graph = nx.Graph()
    
    # Add nodes with numeric labels from 0 to (num_nodes - 1)
    num_nodes = len(adj_matrix)
    for i in range(num_nodes):
        graph.add_node(i)
    
    # Add edges using the adjacency matrix
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            graph.add_edge(i, j, weight=adj_matrix[i][j])
    
    # Compute the layout for the graph
    nodes_position = nx.spring_layout(graph)
    
    # Draw the complete graph
    nx.draw(graph, nodes_position, with_labels=True, node_size=200, node_color='blue', font_size='7', font_color='black')
    nx.draw_networkx_edge_labels(graph, nodes_position, edge_labels={(i, j): adj_matrix[i][j] for i, j in graph.edges()}, font_size='7', font_color='red')
    
    # Save the figure
    plt.savefig(f"{len(adj_matrix)}_nodes_graph.png")
    plt.show()
    
    return nodes_position

# Implementation of Travelling Salseman Problem using Naive/Brute Force Approach- Reference: https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/?ref=lbp
def tsp_brute_force(start_node, num_nodes, adj_matrix):
    tour = []
    min_path_distance = maxsize

    # Store all nodes except the start_node
    nodes = []
    for i in range(num_nodes):
        if (i != start_node):
            nodes.append(i)

    # Minimum distance Hamiltonian Cycle
    all_permutation = permutations(nodes)
    for permutation in all_permutation:
        # Compute current path distance
        current_path_distance = 0
        s = start_node
        current_tour = [start_node]
        for next_node in permutation:
            current_path_distance += adj_matrix[s][next_node]
            s = next_node
            current_tour.append(next_node)
        current_path_distance += adj_matrix[s][start_node]
        # Update Minimum distance and Tour
        if current_path_distance < min_path_distance:
            min_path_distance = current_path_distance
            current_tour.append(start_node)
            tour = current_tour
    
    return min_path_distance, tour, "\n\nBrute Force Solution for {} nodes: ".format(num_nodes)

# Implementation of Travelling Salseman Problem using Nearest Neighbor Algorithm
def tsp_nn(start_node, num_nodes, adj_matrix):
    nodes = [0] * num_nodes
    tour = []
    total_distance = 0
    current_node = start_node
    tour.append(current_node)
    nodes[current_node] = 1

    for i in range(num_nodes-1):
        nearest_distance = math.inf
        for next_node in range(num_nodes):
            if(nodes[next_node]==0):
                distance = adj_matrix[current_node][next_node]
                if(distance < nearest_distance):
                    nearest_distance = distance
                    nearest_neighbor = next_node

        current_node = nearest_neighbor
        tour.append(current_node)
        nodes[current_node] = 1
        total_distance += nearest_distance

    tour.append(start_node)
    total_distance += adj_matrix[current_node][start_node]
    
    return total_distance, tour, "\n\nNearest Neighbor Solution for {} nodes: ".format(num_nodes)

# Implementation of Travelling Salseman Problem using Genetic Algorithm. Reference: https://www.geeksforgeeks.org/traveling-salesman-problem-using-genetic-algorithm/
def tsp_genetic_alg(start_node, num_nodes, adj_matrix):
    class individual:
        def __init__(self) -> None:
            self.gnome = []
            self.fitness = 0

        def __lt__(self, other):
            return self.fitness < other.fitness

        def __gt__(self, other):
            return self.fitness > other.fitness

    def mutatedGene(gnome):
        r = randint(1, num_nodes-1)
        r1 = randint(1, num_nodes-1)
        if r1 != r:
            gnome[r], gnome[r1] = gnome[r1], gnome[r]
        return gnome

    def create_gnome():
        gnome = list(range(num_nodes))
        gnome.remove(start_node)  # Remove the starting node from the list
        shuffle(gnome)   # Shuffle the remaining nodes
        return [start_node] + gnome + [start_node]

    def cal_fitness(adj_matrix, gnome):
        f = 0
        for i in range(len(gnome) - 1):
            f += adj_matrix[gnome[i]][gnome[i + 1]]
        return f
    
    gen = 1
    
    if(num_nodes <=100):
        POP_SIZE = 200  # Define the population size
        gen_thres = 200
    elif(num_nodes > 100 and num_nodes <= 1000):
        POP_SIZE = 500  # Define the population size
        gen_thres = 500
    else:
        POP_SIZE = 300  # Define the population size
        gen_thres = 300

    population = []
    temp = individual()

    for i in range(POP_SIZE):
        temp.gnome = create_gnome()
        temp.fitness = cal_fitness(adj_matrix, temp.gnome)
        population.append(temp)

    '''
    print("\nInitial population: \n[GNOME]     FITNESS VALUE\n")
    for i in range(POP_SIZE):
        print(f"{population[i].gnome}    {population[i].fitness}")  # Modified print statement
    print()
    '''
    temperature = 10000

    while temperature > 1000 and gen <= gen_thres:
        population.sort()
        #print("\nCurrent temp: ", temperature)
        new_population = []

        for i in range(POP_SIZE):
            p1 = population[i]

            while True:
                new_g = mutatedGene(p1.gnome[:])
                new_gnome = individual()
                new_gnome.gnome = new_g
                new_gnome.fitness = cal_fitness(adj_matrix, new_gnome.gnome)

                if new_gnome.fitness <= population[i].fitness:
                    new_population.append(new_gnome)
                    break
                else:
                    prob = pow(2.7, -1 * ((new_gnome.fitness - population[i].fitness) / temperature))
                    if prob > 0.5:
                        new_population.append(new_gnome)
                        break

        temperature = (90 * temperature) / 100
        population = new_population
        '''
        print("\nGeneration", gen)
        print("[GNOME]     FITNESS VALUE")
        for i in range(POP_SIZE):
            print(f"GNOME: {population[i].gnome}  \n Fitness value:  {population[i].fitness}\n")  # Modified print statement
        '''
        gen += 1

    # Get the best tour and minimum distance
    best_individual = min(population)
    min_distance = best_individual.fitness
    tour = best_individual.gnome

    return min_distance, tour, "\n\nGenetic Algorithm Solution for {} nodes: ".format(num_nodes)

# Implementation of Travelling Salseman Problem using Proposed 2-step Nearest Neighbor Algorithm
def tsp_proposed_alg(start_node, num_nodes, adj_matrix):
    current_node = start_node
    nodes_global = [0] * num_nodes
    nodes_global[current_node] = 1
    total_tour = []
    total_tour.append(current_node)
    total_distance = 0
    num_neighbors=2     # Number of neighbors considering
    subtour_length=2    # Number of edge considering
    subtours = [[] for _ in range(num_neighbors ** subtour_length)]
    distances = [[] for _ in range(num_neighbors ** subtour_length)]

    def find_nearest_neighbors(adj_matrix, current_node, nodes_global, num_neighbors, subtour_length):
    #print("nodes_global: ", nodes_global, "\n")
        no_node = 1
        nodes_pre_global = [[] for _ in range(num_neighbors)]
        nearest_neighbors = []
        nearest_distances = []
        for i in range(num_neighbors):
                nodes_pre_global[i] = nodes_global[:]
        for i in range(num_neighbors):  
            nodes_local = []
            nodes_local = nodes_global
            nearest_distance = math.inf
            nearest_neighbor = None
            for next_node in range(num_nodes):
                if nodes_local[next_node] == 0 and next_node != current_node:
                    distance = adj_matrix[current_node][next_node]
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_neighbor = next_node
    
            if nearest_neighbor is not None:
                nearest_neighbors.append(nearest_neighbor)
                nearest_distances.append(nearest_distance)
                nodes_local[nearest_neighbor] = 1

            else:
                if no_node==1:
                    nearest_neighbors.append(-1)
                    nearest_distances.append(math.inf)
                    no_node=0
            if nearest_neighbors[i] != -1:
                nodes_pre_global[i][nearest_neighbor] = 1

        return nearest_neighbors, nearest_distances, nodes_pre_global
    
    while(len(total_tour)<(num_nodes)):

        nearest_neighbors, nearest_distances, nodes_pre_global = find_nearest_neighbors(adj_matrix, current_node, nodes_global, num_neighbors, subtour_length)
        for i in range(num_neighbors):
            current_node = nearest_neighbors[i]
            # Calculate the start and end indices for the current section
            start_index = i * (num_neighbors ** (subtour_length - 1))
            end_index = (i + 1) * (num_neighbors ** (subtour_length - 1))

            # Assign each entry in the current section to a list with a single node
            for j in range(start_index, end_index):
                subtours[j] = [nearest_neighbors[i]]
            distances[start_index:end_index] = [nearest_distances[i]] * (end_index - start_index)   

        if (len(total_tour)<(num_nodes-2)):
            temp1= nearest_neighbors
            temp2 = nodes_pre_global
            for i in range(len(nearest_neighbors)):
                #print("\n For Nearest-", temp1[i],": ")
                #print("=================================")
                current_node = temp1[i]
                nodes_global = temp2[i]
                nearest_neighbors, nearest_distances, nodes_pre_global = find_nearest_neighbors(adj_matrix, current_node, nodes_global, num_neighbors, subtour_length)

                if(i==0):
                    for j in range(num_neighbors):
                    # Calculate the start and end indices for the current section
                        start_index = j * (num_neighbors-1)
                        end_index = start_index + (num_neighbors-1)
                        for k in range(start_index, end_index):
                            subtours[k].extend([nearest_neighbors[j]])
                            distances[k] += nearest_distances[j]

                if(i==1):
                    for j in range(num_neighbors):
                        # Calculate the start and end indices for the current section
                        start_index = (j * (num_neighbors-1)) + 2
                        end_index = start_index + (num_neighbors-1)

                        # Assign each entry in the current section to a list with a single node
                        for k in range(start_index, end_index):
                            subtours[k].extend([nearest_neighbors[j]])
                            distances[k] += nearest_distances[j]
        min_distance, subtour = min(zip(distances, subtours))

        total_tour.extend(subtour)
        total_distance += min_distance
        current_node = total_tour[-1]
        nodes_global = [0] * num_nodes
        for node in total_tour:
            nodes_global[node] = 1
        subtours = [[] for _ in range(len(subtours))]
        distances = [[] for _ in range(len(distances))]
    total_distance += adj_matrix[total_tour[-1]][start_node]
    total_tour.append(start_node)

    return total_distance, total_tour, "\n\nProposed 2-Step Nearest Neighbor Solution for {} nodes: ".format(num_nodes)

# Optional: Draw and save the graph with only Tour edges
def draw_graph_with_tour(adj_matrix, tour, nodes_position):
    # Create a directed graph for the tour sequence
    G_with_tour = nx.DiGraph()

    # Add nodes with numerical labels
    for node in range(len(adj_matrix)):
        G_with_tour.add_node(node)

    # Add directed edges for the tour sequence
    for i in range(len(tour) - 1):
        G_with_tour.add_edge(tour[i], tour[i + 1], weight=adj_matrix[tour[i]][tour[i + 1]])

    # Draw the tour graph using positions from the complete graph
    nx.draw(G_with_tour, nodes_position, with_labels=True, node_size=200, node_color='blue', font_size='7', font_color='black', arrows=True, width=2.0)
    nx.draw_networkx_edge_labels(G_with_tour, nodes_position, edge_labels={(i, j): adj_matrix[i][j] for i, j in G_with_tour.edges()}, font_size='7', font_color='red')
    
    # Save the figure
    plt.savefig(f"{len(adj_matrix)}_nodes_tour_graph.png")
    
    plt.title("Graph Visualization with Tour Sequence and Start_node Label")
    plt.show()

# Driver Function
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    filename = "Size100.graph"
    with open(filename, "r") as f:       # Input file has to be in the same directory
        lines = f.readlines()
    num_nodes = len(lines)
    numeric_part = re.search(r'\d+', filename).group(0)
    numeric_value = int(numeric_part)

    adj_matrix = complete_adj_matrix(lines, num_nodes) # Complete adjacency matrix function
    # Draw the complete graph and get the node positions
    # Optional: Comment if not necessary
    nodes_position = draw_graph(adj_matrix)

    cost = math.inf
    min_random_number = -1
    min_tour = []    
    # Loop to try multiple times
    num_of_random_tries = 1
    #start_node = 0
    for i in range(num_of_random_tries):  # Loop runs 10 times
        random_number = random.randint(0, numeric_value-1)  # Generating random number from 0 to numeric_value
        #start_node = random_number
        start_node = 0
        
        # Uncomment a function call below using which we want the solution
        ####################################################################

        #min_distance, tour, line = tsp_brute_force(start_node, num_nodes, adj_matrix)    # Uncomment for applying Brute Force Algorithm
        #min_distance, tour, line = tsp_nn(start_node, num_nodes, adj_matrix)    # Uncomment for applying NN Algorithm
        #min_distance, tour, line = tsp_genetic_alg(start_node, num_nodes, adj_matrix)   # Uncomment for applying Genetic Algorithm
        min_distance, tour, line = tsp_proposed_alg(start_node, num_nodes, adj_matrix)    # Uncomment for applying Proposed Algorithm
        
        if  min_distance < cost:
            cost = min_distance
            min_random_number = start_node
            min_tour = tour
    print(line)
    print("Solution with Random Start_Node {} and after {} tries:".format(min_random_number, num_of_random_tries))
    print("====================================================================")
    print("Cost:", cost)
    print("Path:", min_tour)
    
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    hours, remainders = divmod(execution_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainders, 60)
    milliseconds = execution_time.microseconds // 1000  # Convert microseconds to milliseconds

    # Print the execution time
    print("Execution Time: {} Hours, {} Minutes, {} Seconds, and {} Milliseconds\n".format(int(hours), int(minutes), int(seconds), milliseconds))   
    
    # Path to the output file
    output_file_path = "output.txt"

    # Check if the file exists before reading
    if os.path.exists(output_file_path):
        # Open the file and read existing contents
        with open(output_file_path, 'r') as file:
            existing_contents = file.read()
    else:
        existing_contents = ""

    # Open the file in append mode and write the lines
    with open(output_file_path, 'a') as file:
        # If there's existing content, add two empty lines
        if existing_contents.strip():  # Check if existing content is not empty
            file.write("\n\n")
        # Write the lines
        file.write(line)
        file.write("\nSolution with Random Start_Node {} and after {} tries:\n".format(min_random_number, num_of_random_tries))
        file.write("=========================================================================\n")
        file.write("Cost: {}\n".format(cost))
        file.write("Path: {}\n".format(min_tour))
        file.write("Execution Time: {} Hours, {} Minutes, {} Seconds, and {} Milliseconds\n".format(int(hours), int(minutes), int(seconds), milliseconds))
        # To draw graph using the selected edges in the tour only. To use it, we must use nodes_position = draw_graph(adj_matrix) to get the nodes_position
        # Optional: Comment if not necessary
        draw_graph_with_tour(adj_matrix, min_tour, nodes_position)
