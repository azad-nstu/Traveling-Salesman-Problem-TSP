# Travelling-Salesman-Problem-TSP-
Traveling Salesman Problem (TSP) is a well-known NP-hard problem that aims to find the optimal tour among different cities to visit each city exactly once and return to the original city. This directory includes all the graph data used for this experiment.

The repository includes the main program "tsp_sol_final-12210325.py", output file "output.txt", Excel file "Result.xlsx" to organize the output, distance_figure.py, and time_figure.py to draw and store the result analysis graph, saved figures, etc.

tsp_sol_final-12210325.py 
	- is the main programming file where 4 algorithms (Brute force, nearest neighbors, Genetic algorithm, and Proposed 2-step nearest neighbor) were implemented in 4 functions.
	- The .graph file has to be in the same directory to run this and we can choose a different graph by selecting the file name in line 363. 
	- To draw a graph, we can use the function "nodes_position = draw_graph(adj_matrix)" in line 373. If not necessary we can comment on it (especially for large graphs)
	- We will uncomment 1 function (between lines: 389-392) at a time to run 1 of 4 algorithms.
		- To run the algorithm once, we will set the start node in line- 384.
		- To run the algorithm multiple times and get the best result, in line- 379, we can set the "num_of_random_tries" value accordingly, uncomment the line start_node = random_number (line-383), and make comment start_node = 0 (line- 384)
	- To draw the graph for final tour, "draw_graph_with_tour(adj_matrix, min_tour, nodes_position)" function (in line-438) used. 
		- To use this function, we must use "nodes_position = draw_graph(adj_matrix)" in line-373 function because it takes nodes_position from that function.

output.txt
	- Output of the algorithms (cost/distance, tour/path, execution time) stored in this file.
	
Result.xlsx
	- Organize the results from the output.txt file.
	
distance_figure.py
	- Used to draw and save the figure of Distances Vs. Number of Cities comparison among 4 algorithms. 
	
time_figure.py
	- Used to draw and save the figure of Execution Time Vs. Number of Cities comparison among 4 algorithms. 
		
