from copy import deepcopy
from collections import Counter
from typing import List, Tuple, Any, Dict

# This module implements the basic functionality for a nxn game of Push.
# The style is functional as much as Python allows and uses basic patterns
# common in Elixir:
#   1. there is no internal state.
#   2. data structures and actions are supplied and a modified copy is returned.
#
# This file only implements the *board* of Push. A  *game* of Push requires 
# keeping track of who's turn it is, checking for winners, and supplying a "brain"
# for the opponent.
#
# Additionally, you must check for cycles (repeating board states).
#
# Supplying the "brain" is the point of Module 2.

def create(n: int=4) -> List[List[str]]:
    """
    create a new board of nxn squares as  list[list[str]]
    """
    return [['.']*n for _ in range(0, n)]


def equal(b1: List[List[str]], b2: List[List[str]]) -> bool:
    """
     Checks to see if two boards are equal. It assumes the boards
     are themselves valid and from the same game.
     """
    n = len(b1)
    for i in range(0, n):
        for j in range(0, n):
            if b1[i][j] != b2[i][j]:
                return False
    return True


# the Python convention for "private" is to preface a function with 
# a single underscore. This is not enforced. The semantics here are 
# "this is not a function meant to be used outside this module, if you
# choose to ignore this warning, it may break later."
def _transpose(lol: List[List[Any]]) -> List[List[Any]]:
    return list(map(list, zip(*lol)))


def _push_left(board, player, index):
    row = board[index-1]
    new_row = [player] + row[:-1]
    board[index-1] = new_row
    return board


def _push_right(board, player, index):
    row = board[index-1]
    new_row = row[1:] + [player]
    board[index-1] = new_row
    return board


def move(board: List[List[str]], move: Tuple[str, str, int]) -> List[List[str]]:
    """
    Given a board and a move (player, edge, index), return a new board. A move is:

    player = "X" or "O"
    edge = "T"(op), "B"(ottom), "L"(eft), "R"(ight)
    index = (1,..., n)

    The board that is returned is a deepcopy of the supplied board.
    """
    player, edge, index = move
    if player not in ['X', 'O']:
        raise ValueError(f"Unknown player: {player}")
    if edge not in ['T', 'B', 'L', 'R']:
        raise ValueError(f"Unknown edge: {edge}")
    n = len(board)
    if index < 1 or n < index:
        raise ValueError(f"Unknown index: {index}")

    new_board = deepcopy(board)
    if edge in ('T', 'B'):
        new_board = _transpose(new_board)
    if edge in ('T', 'L'): # top and left are the same
        new_board = _push_left(new_board, player, index)
    else:  # right and bottom are the same
        new_board = _push_right(new_board, player, index)
    if edge in ('T', 'B'):
        new_board = _transpose(new_board)
    return new_board


def _sequence_winner(n, sequence):
    counts = Counter(sequence)
    if 'O' in counts and counts['O'] == n:
        return 'O'
    if 'X' in counts and counts['X'] == n:
        return 'X'
    return None

def straights(board: List[List[str]]) -> Dict[str, int]:
    """
    Count the number of straights on the board for each player. A 
    straight is a sequence of n pieces.

    Returns a Dict of values with the following format:

    {'O': 2, 'X': 3, None: 5}

    The winner is the player, 'O' or 'X', with the most straights (> 0).
    """
    board = deepcopy(board)
    n = len(board)
    straights = {"O": 0, "X": 0, None: 0}
    # check for horizontals
    for row in board:
        winner = _sequence_winner(n, row)
        straights[winner] += 1
    #print(straights)
    # check main diagonal
    diagonal = []
    for i in range(0, n):
        diagonal.append(board[i][i])
    winner = _sequence_winner(n, diagonal)
    straights[winner] += 1
    #print(straights)
    # check for verticals:
    board = _transpose(board)
    for column in board:
        winner = _sequence_winner(n, column)
        straights[winner] += 1
    #print(straights)
    # off diagonal is the main diagonal of the transposed board.
    diagonal = []
    for i in range(0, n):
        diagonal.append(board[i][n-i-1])
    winner = _sequence_winner(n, diagonal)
    straights[winner] += 1
    #print(straights)
    return straights