###
# Calvin Chen
###

from heapq import heappop, heappush
import numpy
import math

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

costs = { '.': 1, '*': 3, '#': 5, '~': 7}


'''
    Helper function that returns list of all valid directional moves from the current position
    Note: Does not evaluate for presence of mountains
'''
def findNextMoves(world, x, y, moves):
    res = []
    length_x = len(world)
    length_y = len(world[0])


    for m in moves:
        new_x = x + m[0]
        new_y = y + m[1]
        # checks if move is within boundary of the world
        if new_x >= 0 and new_y >= 0 and new_x < length_x and new_y < length_y:
            res.append(m)
    return res


'''
    Heuristic function that evaluates diagonal distance from current location to the goal
    Used as a tiebreaker between paths of equal cost.
'''
def heuristic(pos, goal):
    return math.hypot(goal[1] - pos[1], goal[0] - pos[0])


'''
    Main function implementing A* search
    Essentailly a modified BFS using a priority queue, where priority is lowest cost + heuristic (distance to goal)
'''
def a_star_search( world, start, goal, costs, moves, heuristic): 
    # Use a BFS with a priority queue
    # Elements in queue are a tuple with (heuristic + cost, current cost, location). heapq evaluates on the heuristic+cost portion
    # result is when start == goal. Possible error with multiple paths to goal?
    # visited contains {(x, y): ([path], cost)}

    #Accounting for switched x and y
    world = numpy.transpose(world)

    frontier = []
    heappush(frontier, (heuristic(start, goal), 0, start))
    visited = {}
    visited[start] = ([], 0)
    
    while frontier:
        current = heappop(frontier)
        current_x = current[2][0]
        current_y = current[2][1]
        # list of available next moves
        next_moves = findNextMoves(world, current_x, current_y, moves)

        for m in next_moves:
            new_x = current_x + m[0]
            new_y = current_y + m[1]
            loc = (new_x, new_y)

            # checks to see if index has been visited before or if shorter path found
            if loc not in visited or visited[loc][1] > (current[1] + costs[world[new_x][new_y]]): 
                # checks for mountains
                if world[new_x][new_y] not in costs:
                    break
                else:
                    cost = current[1] + costs[world[new_x][new_y]]
                    heappush(frontier, (heuristic(loc, goal) + cost, cost, loc))
                    # storing (path, cost)
                    visited[loc] = (visited[current[2]][0] + [m], current[1] + costs[world[new_x][new_y]])
            if loc == goal:
                #may have issue here with alternative path
                return visited[loc][0]
    return []


'''
    Helper function for returning symbol of movement from input of cardinal direction tuple
'''
def direction(d):
    if d == (1, 0):
        return '>'
    if d == (0, 1):
        return 'v'
    if d == (-1, 0):
        return '<'
    if d == (0, -1): 
        return '^'
    return "Error"


'''
    Prints the path that the robot takes as described in assignment
    Note: transposes world twice to account for x,y switch
    Loops through the solution elements and assigns directions
'''
def pretty_print_solution(world, solution, start):
    world = numpy.transpose(world)
    x = start[0]
    y = start[1]

    for d in solution:
        world[x][y] = direction(d)
        x += d[0]
        y += d[1]

    world[x][y] = 'G'

    world = numpy.transpose(world)
    for row in world:
        print(" ".join(map(str,row)))
    return 



if __name__ == "__main__":
    print("A* solution for test world")
    test_path = a_star_search(test_world, (0, 0), (6, 6), costs, cardinal_moves, heuristic)
    print(test_path)
    pretty_print_solution( test_world, test_path, (0, 0))

    print("A* solution for full world")
    full_path = a_star_search(full_world, (0, 0), (26, 26), costs, cardinal_moves, heuristic)
    print(full_path)
    pretty_print_solution(full_world, full_path, (0, 0))

'''s1 = a_star_search(test_world, (0,0), (6,6), costs, cardinal_moves, h1)
print(s1)
pretty_print_solution(test_world, s1, (0,0))
s2 = a_star_search(full_world, (0,0), (26, 26), costs, cardinal_moves, h1)
print(s2)
pretty_print_solution(full_world, s2, (0,0))
#print(a_star_search(full_world, (0,0), (26, 26), costs, cardinal_moves, h1))
'''