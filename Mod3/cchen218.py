import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from copy import deepcopy

def color_map( planar_map, colors, trace=False):
	"""
	This function takes the planar_map and tries to assign colors to it.

	planar_map: Dict with keys "nodes", "edges", and "coordinates". "nodes" is a List of node names, "edges"
	is a List of Tuples. Each tuple is a pair of indices into "nodes" that describes an edge between those
	nodes. "coorinates" are x,y coordinates for drawing.

	colors: a List of color names such as ["yellow", "blue", "green"] or ["orange", "red", "yellow", "green"]
	these should be color names recognized by Matplotlib.

	If a coloring cannot be found, the function returns None. Otherwise, it returns an ordered list of Tuples,
	(node name, color name), with the same order as "nodes".
	"""
	### YOUR SOLUTION HERE ###
	# add helper functions as needed for "Clean Code"
	### YOUR SOLUTION HERE ### 
	sol = allSolutions(planar_map, colors, 'LCV', trace)

	if sol:
		return pretty_color_assignments(planar_map, sol) #[(n, "red") for n in planar_map["nodes"]]
	return sol


connecticut = {"nodes": ["Fairfield", "Litchfield", "New Haven", "Hartford", "Middlesex", "Tolland", "New London", "Windham"],
			   "edges": [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (3,6), (4,6), (5,6), (5,7), (6,7)],
			   "coordinates": [( 46, 52), ( 65,142), (104, 77), (123,142), (147, 85), (162,140), (197, 94), (217,146)]}

europe = {
	"nodes":  ["Iceland", "Ireland", "United Kingdom", "Portugal", "Spain",
				 "France", "Belgium", "Netherlands", "Luxembourg", "Germany",
				 "Denmark", "Norway", "Sweden", "Finland", "Estonia",
				 "Latvia", "Lithuania", "Poland", "Czech Republic", "Austria",
				 "Liechtenstein", "Switzerland", "Italy", "Malta", "Greece",
				 "Albania", "Macedonia", "Kosovo", "Montenegro", "Bosnia Herzegovina",
				 "Serbia", "Croatia", "Slovenia", "Hungary", "Slovakia",
				 "Belarus", "Ukraine", "Moldova", "Romania", "Bulgaria",
				 "Cyprus", "Turkey", "Georgia", "Armenia", "Azerbaijan",
				 "Russia" ], 
	"edges": [(0,1), (0,2), (1,2), (2,5), (2,6), (2,7), (2,11), (3,4),
				 (4,5), (4,22), (5,6), (5,8), (5,9), (5,21), (5,22),(6,7),
				 (6,8), (6,9), (7,9), (8,9), (9,10), (9,12), (9,17), (9,18),
				 (9,19), (9,21), (10,11), (10,12), (10,17), (11,12), (11,13), (11,45), 
				 (12,13), (12,14), (12,15), (12,17), (13,14), (13,45), (14,15),
				 (14,45), (15,16), (15,35), (15,45), (16,17), (16,35), (17,18),
				 (17,34), (17,35), (17,36), (18,19), (18,34), (19,20), (19,21), 
				 (19,22), (19,32), (19,33), (19,34), (20,21), (21,22), (22,23),
				 (22,24), (22,25), (22,28), (22,29), (22,31), (22,32), (24,25),
				 (24,26), (24,39), (24,40), (24,41), (25,26), (25,27), (25,28),
				 (26,27), (26,30), (26,39), (27,28), (27,30), (28,29), (28,30),
				 (29,30), (29,31), (30,31), (30,33), (30,38), (30,39), (31,32),
				 (31,33), (32,33), (33,34), (33,36), (33,38), (34,36), (35,36),
				 (35,45), (36,37), (36,38), (36,45), (37,38), (38,39), (39,41),
				 (40,41), (41,42), (41,43), (41,44), (42,43), (42,44), (42,45),
				 (43,44), (44,45)],
	"coordinates": [( 18,147), ( 48, 83), ( 64, 90), ( 47, 28), ( 63, 34),
				   ( 78, 55), ( 82, 74), ( 84, 80), ( 82, 69), (100, 78),
				   ( 94, 97), (110,162), (116,144), (143,149), (140,111),
				   (137,102), (136, 95), (122, 78), (110, 67), (112, 60),
				   ( 98, 59), ( 93, 55), (102, 35), (108, 14), (130, 22),
				   (125, 32), (128, 37), (127, 40), (122, 42), (118, 47),
				   (127, 48), (116, 53), (111, 54), (122, 57), (124, 65),
				   (146, 87), (158, 65), (148, 57), (138, 54), (137, 41),
				   (160, 13), (168, 29), (189, 39), (194, 32), (202, 33),
				   (191,118)]}


COLOR = 1

def test_coloring(planar_map, coloring):
	edges = planar_map["edges"]
	nodes = planar_map[ "nodes"]

	for start, end in edges:
		try:
			assert coloring[ start][COLOR] != coloring[ end][COLOR]
		except AssertionError:
			print("%s and %s are adjacent but have the same color." % (nodes[ start], nodes[ end]))


def assign_and_test_coloring(name, planar_map, colors, trace=False):
	print(f"Trying to assign {len(colors)} colors to {name}")
	coloring = color_map(planar_map, colors, trace=trace)
	if coloring:
		print(f"{len(colors)} colors assigned to {name}.")
		test_coloring(planar_map, coloring)
	else:
		print(f"{name} cannot be colored with {len(colors)} colors.")

def as_dictionary(a_list):
	dct = {}
	for i, e in enumerate(a_list):
		dct[i] = e
	return dct


def draw_map(name, planar_map, size, color_assignments=None): 

	G = nx.Graph()

	labels = as_dictionary(planar_map[ "nodes"])
	pos = as_dictionary(planar_map["coordinates"])

	# create a List of Nodes as indices to match the "edges" entry.
	nodes = [n for n in range(0, len(planar_map[ "nodes"]))]

	if color_assignments:
		colors = [c for n, c in color_assignments]
	else:
		colors = ['red' for c in range(0,len(planar_map[ "nodes"]))]

	G.add_nodes_from( nodes)
	G.add_edges_from( planar_map[ "edges"])

	plt.figure( figsize=size, dpi=600)
	nx.draw( G, node_color = colors, with_labels = True, labels = labels, pos = pos) 
	plt.savefig(name + ".png")

'''
Checks that neighboring nodes have different colors or have not had colors assigned yet
'''
def is_safe(vertex, graph, colors, c):
	# Check that neighbors do not have the same color

	for i in range(len(graph)):
		if graph[vertex][i] and c == colors[i]:
			return False
	return True

'''
Recursive backtracking only algorithm.
Returns boolean when solution found/not found
'''
def backtrack(graph, colors, vertex_colors, current_vertex, trace):
	# Check if last vertex was assigned a color
	if len(graph) == current_vertex:
		return True

	# Trying different colors for vertex current_vertex
	for c in colors:
		# Check if safe color assignment
		if is_safe(current_vertex, graph, vertex_colors, c):
			# Assign color c to current_vertex
			vertex_colors[current_vertex] = c
			# Recurse to assign colors to the remaining vertices
			if backtrack(graph, colors, vertex_colors, current_vertex+1, trace):
				return True
			# Backtrack in case of no possible solution
			vertex_colors[current_vertex] = 0

'''
Recursive backtracking + FC map coloring
Returns boolean when solution found/not found
'''
def forward_check(graph, all_colors, available_colors, vertex_colors, current_vertex, trace):
	# Check if all vertices are assigned a color
	if len(graph) == current_vertex :
		return True

	# Forward checking. Looks for empty possible value sets
	for i in available_colors.values():
		if not i:
			print("No avaialable colors for FC")
			return False

	# Checking all potential colors
	for c in available_colors[current_vertex]:
		# Check if assignment of color c to current_vertex is possible
		if is_safe(current_vertex, graph, vertex_colors, c):
			# Assign color c to current_vertex
			vertex_colors[current_vertex] = c
			# Safely assign color c to current_vertex
			next_available_colors = deepcopy(available_colors)

			# Updating potential color values dict
			neighbors = findNeighbors(graph, current_vertex)
			for i in neighbors:
				if c in next_available_colors[i]:
					next_available_colors[i].remove(c)

			# Recursively assign colors to the rest of the vertices
			if forward_check(graph, all_colors, next_available_colors, vertex_colors, current_vertex+1, trace): return True
			# Solution not found at present state. Remove color assignment and backtrack
			vertex_colors[current_vertex] = 0

'''
Recursive backtracking + FC + Minimum Remaining Value map coloring
Returns boolean when solution found/not found
'''
def minimum_remaining_values(graph, all_colors, available_colors, vertex_colors, current_vertex, trace):
	# Check if all vertices are assigned a color
	if not available_colors :
		return True

	# Forward checking. Looks for empty possible value sets
	for i in available_colors.values():
		if not i:
			print("No avaialable colors for FC")
			return False

	# Checking all potential colors
	for c in available_colors[current_vertex]:
		# Check if assignment of color c to current_vertex is possible
		if is_safe(current_vertex, graph, vertex_colors, c):
			# Safely assign color c to current_vertex
			vertex_colors[current_vertex] = c
			# Remove color from neighbors for forward checking
			next_available_colors = deepcopy(available_colors)

			# Updating potential color values dict
			neighbors = findNeighbors(graph, current_vertex)
			for i in neighbors:
				if i in next_available_colors:
					if c in next_available_colors[i]:
						next_available_colors[i].remove(c)
			next_available_colors.pop(current_vertex)
			

			# Recursively assign colors to the rest of the vertices
			# Assess current_vertex with MRV
			current_vertex = find_minimum_index(next_available_colors, all_colors)

			if minimum_remaining_values(graph, all_colors, next_available_colors, vertex_colors, current_vertex, trace): return True
			# Solution not found at present state. Remove color assignment and backtrack
			vertex_colors[current_vertex] = 0

'''
Recursive backtracking method applying BT + FC, Minimum remaining values, and least constraining value heuristics
'''
def least_constraining_value(graph, all_colors, available_colors, vertex_colors, current_vertex, trace):
	# Check if all vertices are assigned a color
	if not available_colors :
		if trace: print("All vertices assigned color. Assignment Complete")
		return True

	# Forward checking. Looks for empty possible value sets
	for key, value in available_colors.items():
		if not value:
			if trace: print("No avaialable colors found for vertex " + str(key) + " (FC).")
			return False

	if trace: print("\nCurrent vertex: " + str(current_vertex ))


	# Sorting order of color assignment based on least constraining value

	color_key = find_least_constraining_value(all_colors, available_colors, current_vertex, findNeighbors(graph, current_vertex))
	color_list = sorted(available_colors[current_vertex], key = lambda x: color_key.index(x))
	if trace: 
		print("Evaluating Least Constraining Value for " + str(current_vertex))
		print("Value assignment order is: " + str(color_list))

	# Checking all potential colors
	for c in color_list:
		# Check if assignment of color c to current_vertex is possible
		if is_safe(current_vertex, graph, vertex_colors, c):
			# Safely assign color c to current_vertex
			vertex_colors[current_vertex] = c
			# Remove color from neighbors for forward checking
			next_available_colors = deepcopy(available_colors)

			neighbors = findNeighbors(graph, current_vertex)

			# Updating potential color values dict
			for i in neighbors:
				if i in next_available_colors:
					if c in next_available_colors[i]:
						next_available_colors[i].remove(c)
			next_available_colors.pop(current_vertex)

			if trace: print("The color " + c + " was removed from neighbors of vertex " + str(current_vertex))

			# Recursively assign colors to the rest of the vertices
			# Assess current_vertex with MRV
			current_vertex = find_minimum_index(next_available_colors, all_colors)
			if trace: 
				if current_vertex != -1:
					print("The vertex with minimum remaining values is " + str(current_vertex))
					print("Available values: " + str(next_available_colors[current_vertex]))
			
			if least_constraining_value(graph, all_colors, next_available_colors, vertex_colors, current_vertex, trace): return True

			# Solution not found at present state. Remove color assignment and backtrack
			if trace: print("Solution not found for given vertex and color combination: (" + str(current_vertex) + "," + c +  "). Backtracking...")
			vertex_colors[current_vertex] = 0


'''
Returns vertex with minimum remaining values
Loops through all remaining vertices and updates min_set based on length of available values
'''
def find_minimum_index(available_colors, all_colors):
	min_set = len(all_colors) + 1
	res = -1

	for i in available_colors:
		if len(available_colors[i]) < min_set:
			min_set = len(available_colors[i])
			res = i
			
	return res

'''
Returns colors array sorted in order of least constraining value
Sorted Colors Array used to implement a custom sort
'''
def find_least_constraining_value(all_colors, available_colors, current_vertex, neighbors):
	count = {}
	for color in all_colors:
		count[color] = 0

	for neighbor in neighbors:
		if neighbor in available_colors:
			for color in available_colors[neighbor]:
				count[color] += 1
	return sorted(count, key=count.get)

'''
Returns all neighbors, visited and unvisited, of the current vertex
'''
def findNeighbors(graph, current_vertex):
	res = []
	for i in range(len(graph)):
		if graph[current_vertex][i]:
			res.append(i)
	return res



'''
Driver code that lets you choose which search type ('BT', 'FC', 'MRV', or 'LCV') you want to use
Each successive search type builds on the previous ones
Graph represented as adjacency matrix
'''
def allSolutions(planar_map, colors, search_type, trace):

	number_vertices = len(planar_map["nodes"])
	vertex_colors = [0] * number_vertices

	G = nx.Graph()

	labels = as_dictionary(planar_map[ "nodes"])
	pos = as_dictionary(planar_map["coordinates"])

	# create a List of Nodes as indices to match the "edges" entry.
	nodes = [current_vertex for current_vertex in range(0, len(planar_map[ "nodes"]))]

	G.add_nodes_from( nodes)
	G.add_edges_from( planar_map[ "edges"])

	graph = nx.to_numpy_matrix(G).astype(int).tolist()



	# dictionary of {vertex:available colors}
	available_colors = {}
	for current_vertex in range(len(graph)):
		available_colors[current_vertex] = deepcopy(colors)

	# Determins search algorithm to run
	if search_type == 'BT':
		if backtrack(graph, colors, vertex_colors, 0, trace):
			return(vertex_colors)
		else:
			print("No solutions")
			return None
	elif search_type == 'MRV':
		if minimum_remaining_values(graph, colors, available_colors, vertex_colors, 0, trace):
			return(vertex_colors)
		else:
			print("No solutions")
			return None
	elif search_type == 'FC':
		if forward_check(graph, colors, available_colors, vertex_colors, 0, trace):
			return(vertex_colors)
		else:
			print("No solutions")
			return None
	elif search_type == 'LCV':
		if trace: print("CSP with Backtracking, Forward Checking, Minimum Remaining Values, and Least Constraining Value")
		if least_constraining_value(graph, colors, available_colors, vertex_colors, 0, trace):
			return(vertex_colors)
		else:
			if trace: print("No solutions")
			return None
	else:
		print("invalid search type")

def pretty_color_assignments(planar_map, assigned_colors):
	res = []

	for i in range(len(planar_map["nodes"])):
		res.append((planar_map["nodes"][i], assigned_colors[i]))
	print(res)
	return res

if __name__ == "__main__":
	debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

	## edit these to indicate what you implemented.
	
	print("Backtracking...", "yes")
	print("Forward Checking...", "yes")
	print("Minimum Remaining Values...", "yes")
	print("Degree Heuristic...", "no")
	print("Least Constraining Values...", "yes")
	print("")

	three_colors = ["red", "blue", "green"]
	four_colors = ["red", "blue", "green", "yellow"]

	# Easy Map
	assign_and_test_coloring("Connecticut", connecticut, four_colors)
	assign_and_test_coloring("Connecticut", connecticut, three_colors)
	# Difficult Map
	assign_and_test_coloring("Europe", europe, four_colors, trace=debug)
	#draw_map("connecticut", connecticut, (5, 4), color_assignment)
	assign_and_test_coloring("Europe", europe, three_colors, trace=debug)
