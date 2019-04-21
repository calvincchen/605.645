from copy import deepcopy
import random

full_world = [
	['.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], 
	['.', '.', '.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', 'x', 'x', 'x', 'x', 'x', 'x', 'x', '.', '.'], 
	['.', '.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '#', '#', '#', 'x', 'x', '#', '#'], 
	['.', '.', '.', '.', '#', 'x', 'x', 'x', '*', '*', '*', '*', '~', '~', '*', '*', '*', '*', '*', '.', '.', '#', '#', 'x', 'x', '#', '.'], 
	['.', '.', '.', '#', '#', 'x', 'x', '*', '*', '.', '.', '~', '~', '~', '~', '*', '*', '*', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.'], 
	['.', '#', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '~', '~', '~', '~', '~', '.', '.', '.', '.', '.', '#', 'x', '#', '.', '.'], 
	['.', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '#', 'x', 'x', 'x', '~', '~', '~', '.', '.', '.', '.', '.', '#', '.', '.', '.'], 
	['.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#', 'x', 'x', 'x', '~', '~', '~', '.', '.', '#', '#', '#', '.', '.'], 
	['.', '.', '.', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', '.', '~', '~', '.', '.', '#', '#', '#', '.', '.', '.'], 
	['.', '.', '.', '~', '~', '~', '.', '.', '#', '#', '#', 'x', 'x', 'x', 'x', '.', '.', '.', '~', '.', '#', '#', '#', '.', '.', '.', '.'], 
	['.', '.', '~', '~', '~', '~', '~', '.', '#', '#', 'x', 'x', 'x', '#', '.', '.', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.', '.', '.'], 
	['.', '~', '~', '~', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.', '.', '.', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.', '.'], 
	['~', '~', '~', '~', '~', '.', '.', '#', '#', 'x', 'x', '#', '.', '~', '~', '~', '~', '.', '.', '.', '#', 'x', '#', '.', '.', '.', '.'], 
	['.', '~', '~', '~', '~', '.', '.', '#', '*', '*', '#', '.', '.', '.', '.', '~', '~', '~', '~', '.', '.', '#', '.', '.', '.', '.', '.'], 
	['.', '.', '.', '.', 'x', '.', '.', '*', '*', '*', '*', '#', '#', '#', '#', '.', '~', '~', '~', '.', '.', '#', 'x', '#', '.', '.', '.'], 
	['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '#', '#', '.', '~', '.', '#', 'x', 'x', '#', '.', '.', '.'], 
	['.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '.', '.', 'x', 'x', 'x', '.', '.', '.', '.', '.'], 
	['.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', 'x', '.', '.', '.', '.', '.', '.', '.'], 
	['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', '.', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.'], 
	['.', '.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '~', '~', '~', '~'], 
	['.', '.', '#', '#', '#', '#', 'x', 'x', '*', '*', '*', '*', '*', '.', 'x', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'], 
	['.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', '*', '*', 'x', 'x', '.', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'], 
	['.', '.', '.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', 'x', '.', '.', '.', '.', '#', '#', '.', '.', '~', '~', '~', '~', '~', '~'], 
	['.', '#', '#', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', '#', '#', '.', '~', '~', '~', '~', '~'], 
	['#', 'x', '#', '#', '#', '#', '.', '.', '.', '.', '.', 'x', 'x', 'x', '#', '#', 'x', 'x', '.', 'x', 'x', '#', '#', '~', '~', '~', '~'], 
	['#', 'x', 'x', 'x', '#', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', 'x', 'x', '#', '#', '#', '#', 'x', 'x', 'x', '~', '~', '~', '~'], 
	['#', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '#', '#', '#', '.', '.', '.']]

test_world = [
	['.', '*', '*', '*', '*', '*', '*'],
	['.', '*', '*', '*', '*', '*', '*'],
	['.', '*', '*', '*', '*', '*', '*'],
	['.', '.', '.', '.', '.', '.', '.'],
	['*', '*', '*', '*', '*', '*', '.'],
	['*', '*', '*', '*', '*', '*', '.'],
	['*', '*', '*', '*', '*', '*', '.'],
]







cardinal_moves = [(0,-1), (1,0), (0,1), (-1,0)]

costs = { '.': -1, '*': -3, '#': -5, '~': -7}


'''
Main function that runs the Q learning algorithm by evaluating episodes and returning max scores.
Currently a greedy algorithm.
'''
def q_master(world, costs, goal, reward, actions, gamma, alpha):
	visited = {}
	# visited should take in the tuple: count
	epsilon = 1
	delta_epsilon = -.00001

	prev_Q = [[-1000 for _ in range(len(world[0]))] for _ in range(len(world))]
	#current_Q = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	prev_policy = [['?' for _ in range(len(world[0]))] for _ in range(len(world))]

	Q_right = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_left = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_up = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_down = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]

	# initialize every index into visited?
	for i in range(len(world)):
		for j in range(len(world[0])):
			visited[(i, j)] = 0
	visited[(goal[0], goal[1])] = 10000

	# very naive and greedy implementation
	# run algorithm once from every index
	count = 0
	while(min(visited.values())) < 10000 and epsilon > 0:
		count +=1
		print(count)
		index = find_min_visited(visited)
		#print(index)
		current_Q, current_policy, visited = q_episode(world, costs, goal, reward, actions, gamma, alpha, index[0], index[1], Q_right, Q_left, Q_up, Q_down, visited, epsilon)
		# update and comparison function
		prev_Q, prev_policy = max_Q_and_policy(prev_Q, prev_policy, current_Q, current_policy)
		print_matrix(prev_Q)
		print_small_matrix(prev_policy)
		epsilon += delta_epsilon
			# check conversion?


	prev_policy[goal[0]][goal[1]] = 'G'
	for i in range(len(world)):
		for j in range(len(world[0])):
			if world[i][j] == 'x':
				prev_policy[i][j] = 'x'
	print_small_matrix(prev_policy)
	return prev_policy


'''
Return an index with the minimum visits
'''
def find_min_visited(visited):
	positions = [] # output variable
	min_value = float("inf")
	for k, v in visited.items():
		if v == min_value:
			positions.append(k)
		if v < min_value:
			min_value = v
			positions = [] # output variable
			positions.append(k)

	return random.choice(positions)

'''
Comparator function that determines which scores and policies to retain between 
the previous Q score and the current Q score. Currently greedy.
'''
def max_Q_and_policy(prev_Q, prev_policy, current_Q, current_policy):
	temp_Q = [[0 for _ in range(len(prev_Q[0]))] for _ in range(len(prev_Q))]
	temp_policy = [['?' for _ in range(len(prev_Q[0]))] for _ in range(len(prev_Q))]
	for i in range(len(prev_Q)):
		for j in range(len(prev_Q[0])):
			temp_Q[i][j] = max(prev_Q[i][j], current_Q[i][j])
			if temp_Q[i][j] == prev_Q[i][j]:
				temp_policy[i][j] = prev_policy[i][j]
			elif temp_Q[i][j] == current_Q[i][j]:
				temp_policy[i][j] = current_policy[i][j]
	return temp_Q, temp_policy



# Returns the Q table?
'''
Evaluates one episode of the iterative Q algorithm, updating the 4 directional Q table in place.
Uses random movement to traverse the world until goal state reached, using epsilon greedy strategy

'''
def q_episode(world, costs, goal, reward, actions, gamma, alpha, x, y, Q_right, Q_left, Q_up, Q_down, visited, epsilon):
	move_dict = {
	(0,-1): 'left', 
	(1,0): 'down', 
	(0,1): 'right', 
	(-1,0): 'up'
	}

	arrow_dict = {
		(0,-1): '<', 
		(1,0): 'v', 
		(0,1): '>', 
		(-1,0): '^'
		}

	R = [['?' for _ in range(len(world[0]))] for _ in range(len(world))]
	
	#Q = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]

	
	all_Q = [Q_right, Q_left, Q_up, Q_down]
	max_Q = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]

	for q in all_Q:
		q[goal[0]][goal[1]] = reward



	while x != goal[0] or y != goal[1]:
		visited[(x, y)] += 1
		#exploration
		if random.random() <= epsilon:
			intended_move = random_move(actions)
		#exploitation
		else:
			max_score = max(Q_right[x][y], Q_down[x][y], Q_up[x][y], Q_left[x][y])
			if Q_right[x][y] == max_score:
				intended_move = (0, 1)
			elif Q_down[x][y] == max_score:
				intended_move = (1, 0)
			elif Q_up[x][y] == max_score:
				intended_move = (-1, 0)	
			elif Q_left[x][y] == max_score:
				intended_move = (0, -1)
		print(intended_move)
		true_move = simulator(intended_move, actions, probability = 0.7)


		if x + true_move[0] < len(world) and x + true_move[0] >= 0 and y + true_move[1] < len(world[0]) and y + true_move[1] >= 0:
			if world[x + true_move[0]][y + true_move[1]] in costs:

				Q_right, Q_left, Q_up, Q_down = q_score(world, costs, gamma, alpha, x, y, intended_move, true_move, Q_right, Q_left, Q_up, Q_down)
				x = x + true_move[0]
				y = y + true_move[1]
				#print(str(x) + ', ' + str(y))


		#do mountain check here

		#R[x][y] = arrow_dict
	'''
	if x == goal[0] and y == goal[1]:
		# get argmax of all q and return
	else:
		move = random_move(actions)
	'''
	#print('right')
	#print_small_matrix(Q_right)


	#print('left')
	#print_small_matrix(Q_left)

	#print('up')
	#print_small_matrix(Q_up)

	#print('down')
	#print_small_matrix(Q_down)

	# should return max Q + policy?

	for i in range(len(world)):
		for j in range(len(world[0])):
			max_Q[i][j] = max(Q_right[i][j], Q_left[i][j], Q_down[i][j], Q_up[i][j])
			if max_Q[i][j] == Q_right[i][j]:
				R[i][j] = '>'
			elif max_Q[i][j] == Q_left[i][j]:
				R[i][j] = '<'
			elif max_Q[i][j] == Q_up[i][j]:
				R[i][j] = '^'
			elif max_Q[i][j] == Q_down[i][j]:
				R[i][j] = 'v'

	return max_Q, R, visited

# Not maintaining Q's
def q_episode_original_copy(world, costs, goal, reward, actions, gamma, alpha, x, y):
	move_dict = {
	(0,-1): 'left', 
	(1,0): 'down', 
	(0,1): 'right', 
	(-1,0): 'up'
	}

	arrow_dict = {
		(0,-1): '<', 
		(1,0): 'v', 
		(0,1): '>', 
		(-1,0): '^'
		}

	R = [['?' for _ in range(len(world[0]))] for _ in range(len(world))]
	
	#Q = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]

	Q_right = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_left = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_up = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_down = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	all_Q = [Q_right, Q_left, Q_up, Q_down]
	max_Q = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]

	for q in all_Q:
		q[goal[0]][goal[1]] = reward



	while x != goal[0] or y != goal[1]:
		intended_move = random_move(actions)
		true_move = simulator(intended_move, actions, probability = 0.7)

		print(intended_move)
		print(true_move)
		if x + true_move[0] < len(world) and x + true_move[0] >= 0 and y + true_move[1] < len(world[0]) and y + true_move[1] >= 0:
			if world[x + true_move[0]][y + true_move[1]] in costs:

				Q_right, Q_left, Q_up, Q_down = q_score(world, costs, gamma, alpha, x, y, intended_move, true_move, Q_right, Q_left, Q_up, Q_down)
				x = x + true_move[0]
				y = y + true_move[1]
				print(str(x) + ', ' + str(y))

		#do mountain check here

		#R[x][y] = arrow_dict
	'''
	if x == goal[0] and y == goal[1]:
		# get argmax of all q and return
	else:
		move = random_move(actions)
	'''
	print('right')
	print_matrix(Q_right)


	print('left')
	print_matrix(Q_left)

	print('up')
	print_matrix(Q_up)

	print('down')
	print_matrix(Q_down)

	# should return max Q + policy?

	for i in range(len(world)):
		for j in range(len(world[0])):
			max_Q[i][j] = max(Q_right[i][j], Q_left[i][j], Q_down[i][j], Q_up[i][j])
			if max_Q[i][j] == Q_right[i][j]:
				R[i][j] = '>'
			elif max_Q[i][j] == Q_down[i][j]:
				R[i][j] = 'v'
			elif max_Q[i][j] == Q_left[i][j]:
				R[i][j] = '<'
			elif max_Q[i][j] == Q_up[i][j]:
				R[i][j] = '^'
			

	return max_Q, R

'''
Scoring function for the Q matrices
'''
def q_score(world, costs, gamma, alpha, x, y, intended_move, true_move, Q_right, Q_left, Q_up, Q_down):
	graph_dict = {
	(0,-1): Q_left, 
	(1,0): Q_down, 
	(0,1): Q_right, 
	(-1,0): Q_up
	}

	arrow_dict = {
		(0,-1): '<', 
		(1,0): 'v', 
		(0,1): '>', 
		(-1,0): '^'
		}


	new_x = x + true_move[0]
	new_y = y + true_move[1]



	graph_dict[intended_move][x][y] = int((1-alpha) * graph_dict[intended_move][x][y] + alpha*(costs[world[new_x][new_y]] + gamma * graph_dict[true_move][new_x][new_y]))
	
	#print(intended_move)
	#print()
	#print_small_matrix(graph_dict[intended_move])
	return Q_right, Q_left, Q_up, Q_down
	

def test_q_score():
	world = full_world
	costs = { '.': - 1, '*': - 3, '#': - 5, '~': - 7}
	gamma = .9
	alpha = .75
	x = len(world) - 1
	y = len(world[0]) - 2
	reward = 100000
	goal = (len(world) - 1, len(world[0]) - 1)
	actions = cardinal_moves
	intended_move = (0,1)
	true_move = (0,1)

	Q_right = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_left = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_up = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	Q_down = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	all_Q = [Q_right, Q_left, Q_up, Q_down]

	for q in all_Q:
		q[goal[0]][goal[1]] = reward

	#print(Q_right[goal[0]][goal[1]])

	#q_score(world, costs, gamma, alpha, x, y, intended_move, true_move, Q_right, Q_left, Q_up, Q_down)
	
	#q_episode(world, costs, goal, reward, actions, gamma, alpha, x, y)
	q_master(world, costs, goal, reward, actions, gamma, alpha)

	return

'''
Makes a random move
'''
def random_move(actions):
	return random.choice(actions)

def q_learning(world, costs, goal, reward, actions, gamma, alpha):
	pass


def pretty_print_policy(rows, cols, policy):
	print('\n'.join([''.join(['{:}'.format(item) for item in row]) for row in policy]))
	pass

# takes in world, v[s], move direction, probability matrix? and costs array
# save probability chance of moving in main direction
# returns complete map Q[s, direction]
def s_calc(world, Vs, probability, costs, moveset, gamma, epsilon, solution, corner):
	# may need to deepcopy

	Q = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]

	for i in range(len(world)):
		for j in range(len(world[0])):
			if world[i][j] in costs:
				qd1 = compute_score(world, Vs, (0, 1), probability, costs, deepcopy(moveset), i, j, gamma)
				qd2 = compute_score(world, Vs, (0, -1), probability, costs, deepcopy(moveset), i, j, gamma)
				qd3 = compute_score(world, Vs, (1, 0), probability, costs, deepcopy(moveset), i, j, gamma)
				qd4 = compute_score(world, Vs, (-1, 0), probability, costs, deepcopy(moveset), i, j, gamma)


				# need to keep track of which direction leads to which score
				# also need to keep track of previous state so that we have a stopping point

				Q[i][j] = int(max(qd1, qd2, qd3, qd4))#, Vs[i][j]))
				if Q[i][j] == 0:
					solution[i][j] = '?'
				elif Q[i][j] == int(qd1):
					solution[i][j] = '>'
				elif Q[i][j] == int(qd2):
					solution[i][j] = '<'
				elif Q[i][j] == int(qd3):
					solution[i][j] = 'v'
				elif Q[i][j] == int(qd4):
					solution[i][j] = '^'
			else:
				solution[i][j] = 'X'
	# maintaining the corner
	Q[len(Q) - 1][len(Q[0]) - 1] = corner


	complete = True

	for i in range(len(Q)):
		for j in range(len(Q[0])):
			if abs(Q[i][j] - Vs[i][j]) > epsilon:
				complete = False
				break

	if complete:
		print_small_matrix(solution)
			# how to get argmax
		return solution
	else:
		#print_matrix(Q)
		#print()
		new_Vs = deepcopy(Q)
		return s_calc(world, new_Vs, probability, costs, moveset, gamma, epsilon, solution, corner)

'''
Simulator function for the agent.
'''
def simulator(intended_direction, move_set, probability):
	rand = random.uniform(0, 1)
	remaining_moves = deepcopy(move_set)
	remaining_moves.remove(intended_direction)
	if rand <= probability:
		return intended_direction
	else:
		return random.choice(remaining_moves)

'''
Gives mountains a very negative score so that path is never chosen in policy
'''
def update_mountains(world, Vs):
	for i in range(len(world)):
		for j in range(len(world[0])):
			if world[i][j] not in costs:
				Vs[i][j] = -10000

	return Vs  

'''
Helper function for value iteration implementation
Computes the score at a specific index x, y
'''
def compute_score(world, Vs, direction, probability, costs, moveset, x, y, gamma):
	main_move = (direction[0] + x, direction[1] + y)
	moveset.remove(direction)
	remainder_moves = [(m[0] + x, m[1] + y) for m in moveset]

	score = 0

	if main_move[0] >= 0 and main_move[1] >= 0 and main_move[0] < len(Vs) and main_move[1] < len(Vs[0]):
		new_terrain = world[main_move[0]][main_move[1]]
		if new_terrain in costs:
			
			score += costs[new_terrain]
			score += probability * Vs[main_move[0]][main_move[1]] * gamma


	for i in remainder_moves:
		if i[0] >= 0 and i[1] >= 0 and i[0] < len(Vs) and i[1] < len(Vs[0]):
			new_terrain = world[i[0]][i[1]]
			if new_terrain in costs:
				score += (1 - probability) / len(remainder_moves) * Vs[i[0]][i[1]] * gamma
				score += costs[new_terrain]

	return score

def print_matrix(world):
	print('\n'.join([''.join(['{:8}'.format(item) for item in row]) for row in world]))

def print_small_matrix(world):
	print('\n'.join([''.join(['{:2}'.format(item) for item in row]) for row in world]))

if __name__ == "__main__":
	'''
	goal = (5, 6)
	gamma = 0.0  # FILL ME IN
	alpha = 0.0  # FILL ME IN
	reward = 0.0  # FILL ME IN
	test_policy = q_learning(test_world, costs, goal, reward, cardinal_moves, gamma, alpha)
	rows = 0  # FILL ME IN
	cols = 0  # FILL ME IN
	pretty_print_policy(rows, cols, test_policy)
	print()

	goal = (26, 26)
	gamma = 0.0  # FILL ME IN
	alpha = 0.0  # FILL ME IN
	reward = 0.0  # FILL ME IN
	full_policy = q_learning(full_world, costs, goal, reward, cardinal_moves, gamma, alpha)
	rows = 0  # FILL ME IN
	cols = 0  # FILL ME IN
	pretty_print_policy(rows, cols, full_policy)
	print()
	'''

	test_q_score()

	# Value Iteration + Stochastic test code
	'''world = full_world

	sol = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	prev_Vs = [[-1 for _ in range(len(world[0]))] for _ in range(len(world))]
	Vs = [[0 for _ in range(len(world[0]))] for _ in range(len(world))]
	goal = 100000000
	Vs[len(Vs) - 1][len(Vs[0]) - 1] = goal
	#a = compute_score(test_world, Vs, (0, -1), .7, costs, [(1,0), (0,1), (-1,0)], 0, 2, .9)
	#print(a)
	b = s_calc(world, Vs, .7, costs, cardinal_moves, .9, 1, sol, goal)
	'''




	#print()
	#print_matrix(b)
	#c= s_calc(test_world, b, .7, costs, cardinal_moves, .9)
	#print(c)
