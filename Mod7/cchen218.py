import sys
from unification import parse, unify
from copy import deepcopy

def forward_planner( start_state, goal, actions, debug=False):
    '''
    Calls outer layer of DFS to generate path from start_state to goal
    :param start_state: starting state
    :param goal: goal state
    :param actions: possible actions for agent and all pre/post conditions
    :param debug: if True, print intermediate states and actions
    :return: final plan
    '''
    ### YOUR SOLUTION HERE ###
    # Implement a Forward Planner, not STRIPS or GraphPlan
    ### YOUR SOLUTION HERE ### 

    res = dfs2(start_state, goal, actions, debug, [], [])

    return res


def dfs_unification(state, conditions, index, temp_vars, final):
    '''
    Inner layer of DFS to explore all possible actions from current state
    :param state: current state
    :param conditions: preconditions for action we are trying
    :param index: tracks which precondition we need to evaluate
    :param temp_vars: intermediate dictionary containing variable:value mapping from unification
    :param final: list of dicts containing all successful mappings of variable:value
    :return: list of dicts containing all successful mappings of variable:value
    '''

    # base case. All conditions met
    if len(conditions) == index:
        if temp_vars not in final:
            final.append(temp_vars)
        return

    # evaluate all possible unifications given current condition (singular)
    for s in state:
        if unify(s, conditions[index]):

            # vairable not assigned yet, or variable assigned same value
            if all(k not in temp_vars or temp_vars[k] == v for k, v in unify(s, conditions[index]).items()):
                new_temp_vars = deepcopy(temp_vars)
                for k, v in unify(s, conditions[index]).items():
                    new_temp_vars[k] = v
                dfs_unification(state, conditions, index + 1, new_temp_vars, final)

    return final


def check_visited(state, visited):
    '''
    Helper function to evaluate if the current state exists in a list of visited states
    :param state: state to be evaluated
    :param visited: list of previously visited states
    :return: boolean
    '''
    for v in visited:
        if sorted(v) == sorted(state):
            return True
    return False

def dfs2(current_state, goal, actions, debug, plan, visited):
    '''
    Outer layer of dfs to search for appropriate plan. Uses backtracking
    :param current_state: current state
    :param goal: goal state
    :param actions: possible actions for agent and all pre/post conditions
    :param debug: if True, print intermediate states and actions
    :param plan: current plan/path
    :param visited: list of all visited states regardless of path
    :return: final plan    
    '''
    if debug:
        print("Current State")
        print(current_state)
        print("Current Plan")
        print(plan)
        if check_visited(current_state, visited):
            print("Plan Failed")
        print()

    if check_goal(current_state, goal):
        return plan

    if not check_visited(current_state, visited):
        visited.append(current_state)
        frontier = []
        for action in actions:
            # gets list of possible variable assignments for given action
            list_var_assignments = dfs_unification(current_state, actions[action]['conditions'], 0, {}, [])

            if list_var_assignments:
                for variables in list_var_assignments:
                    # adds to frontier
                    frontier.append((action, variables)) 

        for f in frontier:
            new_state = add_and_remove(current_state, actions, f[0], f[1])
            move = merge_action_var(actions[f[0]]['action'], f[1])
            res = dfs2(new_state, goal, actions, debug, plan + [move], visited)
            if res:
                # plan found
                return res

    return []

def merge_action_var(action, variables):
    '''
    Helper function that applies the variable values to the action statement
    :param action: String representing generic action statement w/ variables
    :param variables: dictionary of assigned variable values
    :return: String representing specific action statement
    '''
    for v in variables:
        action = action.replace(v, variables[v])
    return action

def check_goal(current, goal):
    '''
    Helper function that checks if the goal has been met
    :param current: current state
    :param goal: goal state
    :return: boolean
    '''
    return set(current).issubset(set(goal))

def add_and_remove(state, actions, move, variables):
    '''
    Function that carries out the postconditions of an action
    :param state: current state
    :param actions: actions dictionary
    :param move: specific action that was taken
    :param variables: dictionary of assigned variables and their values
    :return: the reordered state after all postconditions have been applied
    '''
    new_state = deepcopy(state)
    for d in actions[move]['delete']:
        del_with_var = d
        for k in variables:
            del_with_var = del_with_var.replace(k, variables[k])
        new_state.remove(del_with_var)

    for a in actions[move]['add']:
        add_with_var = a
        for k in variables:
            add_with_var = add_with_var.replace(k, variables[k])
        new_state.append(add_with_var)

    return reorder(new_state)

def reorder(state):
    '''
    Helper function to reorder a state in the order suggested by program instructions
    :param state: state to be reordered
    :return: sorted state
    '''
    keyorder = ['(item', '(place', '(agent', '(at']
    keyorder = dict(zip(keyorder, range(len(keyorder))))
    return sorted(state, key=lambda x: keyorder[x.split()[0]])


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    start_state = [
        "(item Drill)",
        "(place Home)",
        "(place Store)",
        "(agent Me)",
        "(at Me Home)",
        "(at Drill Store)"
    ]

    goal = [
        "(item Drill)",
        "(place Home)",
        "(place Store)",
        "(agent Me)",
        "(at Me Home)",
        "(at Drill Me)"
    ]

    actions = {
        "drive": {
            "action": "(drive ?agent ?from ?to)",
            "conditions": [
                "(agent ?agent)",
                "(place ?from)",
                "(place ?to)",
                "(at ?agent ?from)"
            ],
            "add": [
                "(at ?agent ?to)"
            ],
            "delete": [
                "(at ?agent ?from)"
            ]
        },
        "buy": {
            "action": "(buy ?purchaser ?seller ?item)",
            "conditions": [
                "(item ?item)",
                "(place ?seller)",
                "(agent ?purchaser)",
                "(at ?item ?seller)",
                "(at ?purchaser ?seller)"
            ],
            "add": [
                "(at ?item ?purchaser)"
            ],
            "delete": [
                "(at ?item ?seller)"
            ]
        }
    }


    plan = forward_planner( start_state, goal, actions, debug=debug)
    print(plan)



