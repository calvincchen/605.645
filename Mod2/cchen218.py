import random
import push
import numpy


'''
Generates a random move tuple
'''
def randomMove(N, player):
	directions = ['T', 'B', 'R', 'L']
	return (player, random.choice(directions), random.randint(1, N))


'''
Checks the state of the game and returns True if the game is over
'''
def checkGameOver(board):
	straights = push.straights(board) 
	if straights['X'] != straights['O']:
		return True
	return False


'''
Helper function to identify the opponent
'''
def getOpponent(player):
	opponent = 'X' if player == 'O' else 'O'
	return opponent


'''
Returns list of all possible moves for the current board
'''
def getMoves(n):
	res = []
	directions = ['T', 'B', 'R', 'L']
	for i in range(n, 0, -1):
		for d in directions:
			res.append((d, i))
	# shuffles move list to semi-randomly break ties in scoring
	random.shuffle(res)
	return res


'''
Heuristic used to evaluate board state.
Gives 10^(number of player pieces in winnable configuration - rows, columns, main diagonals)
Subtracts 15^(number of opponent pieces in winnable configuration - rows, columns, main diagonals)
More priority given to opponent pieces and opponent win because the opponent will go next.

'''
def heuristic(board, player):

	opponent = getOpponent(player)

	n = len(board)
	h = 0

	for row in board:
		if row.count(player) == 4:
			h += 100000000
		if row.count(opponent) == 4:
			h -= 1000000000
		h += 10**row.count(player)
		h -= 15**row.count(opponent)

	for column in numpy.transpose(board).tolist():
		if column.count(player) == 4:
			h += 100000000
		if column.count(player) == 4:
			h -= 1000000000
		h += 10**column.count(player)
		h -= 15**column.count(opponent)


	# looking only at main diagonals
	diag1 = numpy.diagonal(board).tolist()
	diag2 = numpy.diagonal(numpy.fliplr(board)).tolist()

	h += 10**diag1.count(player)
	h -= 15**diag1.count(opponent)

	h += 10**diag2.count(player)
	h -= 15**diag2.count(opponent)

	if diag1.count(player) == 4:
		h += 100000000
	if diag2.count(player) == 4:
		h += 100000000

	if diag1.count(opponent) == 4:
		h -= 1000000000
	if diag2.count(opponent) == 4:
		h -= 1000000000

	return h


'''
Returns board determined to be optimal using the mini-max approach
'''
def minimax2(prev_boards, board, player, ply):
	moves = getMoves(len(board))
	opponent = getOpponent(player)
	boards = []
	scores = []

	# evaluates minimax score for all children boards and selects highest score
	for move in moves:
		new_board = push.move(board, (player, move[0], move[1]))
		boards.append(new_board)
		history = prev_boards.copy()
		history.append(new_board)

		scores.append((minimax2_scores(history, new_board, opponent, 1, ply, 'min')))

	return find_max_board(boards, scores, prev_boards)


'''
Recursive function that returns the max heuristic score at the current board, depth, and ply
'''
def minimax2_scores(prev_boards, board, player, depth, ply, state):
	opponent = getOpponent(player)
	moves = getMoves(len(board))

	# base cases
	if depth == ply:
		return heuristic(board, player)

	if checkGameOver(board):
		return heuristic(board, player)

	scores = []
	for move in moves:
			new_board = push.move(board, (player, move[0], move[1]))
			if new_board not in prev_boards:
				history = prev_boards.copy()
				history.append(new_board)
				if state == 'max':
					scores.append((minimax2_scores(history, new_board, opponent, depth + 1, ply, 'min')))
				else:
					scores.append((minimax2_scores(history, new_board, opponent, depth + 1, ply, 'max')))
	if state == 'max':
		return max(scores)
	else:
		return min(scores)


'''
Helper function to determine which board has the highest score and does not conflict with a previous board configuration
'''
def find_max_board(boards, scores, prev_boards):
	max_score = -float("Inf")
	max_board = []

	for i in range(len(scores)):
		if scores[i] > max_score and boards[i] not in prev_boards:
			max_score = scores[i]
			max_board = boards[i]
	return max_board


'''
Returns board determined to be optimal using the alpha-beta approach
'''
def alphabeta2(prev_boards, board, player, ply, alpha, beta):
	moves = getMoves(len(board))
	opponent = getOpponent(player)
	boards = []
	scores = []

	# uses alpha beta pruning to evaluate max scores of all children
	for move in moves:
		new_board = push.move(board, (player, move[0], move[1]))
		boards.append(new_board)
		history = prev_boards.copy()
		history.append(new_board)
		scores.append((alphabeta2_scores(history, new_board, opponent, 1, ply, 'min', alpha, beta)))

	return find_max_board(boards, scores, prev_boards)


'''
Helper function to determine which board has the highest score and does not conflict with a previous board configuration
'''
def alphabeta2_scores(prev_boards, board, player, depth, ply, state, alpha, beta):
	opponent = getOpponent(player)
	moves = getMoves(len(board))

	#base cases
	if depth == ply:
		return heuristic(board, player)
	if checkGameOver(board):
		return heuristic(board, player)

	scores = []
	for move in moves:
			new_board = push.move(board, (player, move[0], move[1]))
			if new_board not in prev_boards:
				history = prev_boards.copy()
				history.append(new_board)
			
				if state == 'max':
					scores.append((alphabeta2_scores(history,new_board, opponent, depth + 1, ply, 'min', alpha, beta)))
					if max(alpha, max(scores)) >= beta:
						break
				else:
					scores.append((alphabeta2_scores(history, new_board, opponent, depth + 1, ply, 'max', alpha, beta)))
					if alpha >= min(beta, min(scores)):
						break
	if state == 'max':
		return max(scores)
	else:
		return min(scores)


"""
Implements one round of minimax vs random strategy, with random as "player"
Returns ending board layout
"""
def minimax2_versus_random(player, ply):
	current = push.create()
	prev_boards = []

	while not checkGameOver(current):
		prev_boards.append(current)

		if player == "X":
			move_tuple  = randomMove(len(current), 'X')
			current = push.move(current, move_tuple)
			player = "O"
		else:
			current = minimax2(prev_boards, current, player, ply)
			player = "X"

	final = push.straights(current)
	return current


"""
Implements one round of minimax vs alphabeta strategy, with minimax as "player"
Returns ending board layout
"""
def ab_versus_minimax(player, plym, plya):
	current = push.create()
	prev_boards = []


	while not checkGameOver(current):
		prev_boards.append(current)

		if player == "X":
			current = minimax2(prev_boards, current, player, plym)
			player = "O"
		else:
			current = alphabeta2(prev_boards, current, player, plya, float('Inf'), -float('Inf'))
			player = "X"
	final = push.straights(current)

	return current


'''
Implements the minimax versus random matchup 5x and records stats
Random will be the input "player"
'''
def minimax_versus_random(player, plym):
	mwins = 0
	rwins = 0

	print("Minimax Player is searching " + str(plym) +  " ply. ")

	for _ in range(5):
		final_board = minimax2_versus_random(player, plym)
		score = push.straights(final_board)
		if score['X'] > score['O']:
			rwins += 1
		else:
			mwins += 1

	print('')
	print('\nFinal Game: \n')
	printpretty(final_board)
	print('\nMinimax won ' + str(mwins) + ' games.')
	print('Random won ' + str(rwins) + ' games.')

	return


'''
Implements the minimax versus alphabeta matchup 5x and records stats
Minimax will be the input "player"
'''
def minimax_versus_alphabeta(player, plym, plya):
	mwins = 0
	awins = 0

	print("Minimax Player is searching " + str(plym) +  " ply. ")
	print("Alph Beta Player is searching " + str(plya) +  " ply. ")

	for _ in range(5):
		final_board = ab_versus_minimax(player, plym, plya)
		score = push.straights(final_board)
		if score['X'] > score['O']:
			mwins += 1
		else:
			awins += 1

	print('\nFinal Game: \n')
	printpretty(final_board)
	print('\nMinimax won ' + str(mwins) + ' games.')
	print('Alpha Beta won ' + str(awins) + ' games.')

	return


'''
Prints the board as specified in instructions
'''
def printpretty(board):
	print('\n'.join([''.join(['{:}'.format(item) for item in row]) 
      for row in board]))

if __name__ == '__main__':

	print("Random v. Minimax")
	minimax_versus_random('X', 2)
	print("\nMinimax v. Alpha Beta")
	minimax_versus_alphabeta('X', 3, 10)