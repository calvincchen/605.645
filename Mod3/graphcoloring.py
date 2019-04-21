from collections import defaultdict
colors = ['1', '2', '3', '4']
#colors = ['Red', 'Blue', 'Green', ]

states = ["Fairfield", "Litchfield", "New Haven", "Hartford", "Middlesex", "Tolland", "New London", "Windham"]
#states = ['Andhra', 'Karnataka', 'TamilNadu', 'Kerala']

edges = [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (3,6), (4,6), (5,6), (5,7), (6,7)]
neighbors = {}
#neighbors['Andhra'] = ['Karnataka', 'TamilNadu']
#neighbors['Karnataka'] = ['Andhra', 'TamilNadu', 'Kerala']
#neighbors['TamilNadu'] = ['Andhra', 'Karnataka', 'Kerala']
#neighbors['Kerala'] = ['Karnataka', 'TamilNadu']
#print(neighbors)

colors_of_states = {}

def promising(state, color):
    print(state + ' ' + color)
    for neighbor in neighbors.get(state): 
        color_of_neighbor = colors_of_states.get(neighbor)
        if color_of_neighbor == color:
            return False

    return True

def get_color_for_state(state):
    for color in colors:
        if promising(state, color):
            return color

def main():
    for state in states:
        colors_of_states[state] = get_color_for_state(state)

    print(colors_of_states)

def makeNeighbors(states, edges):
    neighbors = defaultdict(list)
    for e in edges:
        neighbors[states[e[0]]].append(states[e[1]])
        neighbors[states[e[1]]].append(states[e[0]])
    return neighbors

neighbors = makeNeighbors(states, edges)
print(neighbors)
main()


'''
connecticut = { "nodes": ["Fairfield", "Litchfield", "New Haven", "Hartford", "Middlese
                        "edges": [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (3,6)
                        "coordinates": [( 46, 52), ( 65,142), (104, 77), (123,142), (147, 85),
        print(connecticut)
'''