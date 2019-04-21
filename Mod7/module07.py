import sys
from unification import parse, unify
from copy import deepcopy

def forward_planner( start_state, goal, actions, debug=False):
    ### YOUR SOLUTION HERE ###
    # Implement a Forward Planner, not STRIPS or GraphPlan
    ### YOUR SOLUTION HERE ### 
    plan = []
    first_action = "drive"
    visited = []
    res = dfs2(start_state, goal, actions, debug, plan, visited)

    return res
    '''
    parsed = []
    for i in start_state:
        parsed.append(parse(i))

    conditions = []
    for i in actions['drive']['conditions']:
        conditions.append(parse(i))
    print(conditions)
    print(unify_all(start_state, actions, 'drive', debug)) 
    print(unify_all(start_state, actions, 'buy', debug))
    '''
    return

def dfs_unification(state, conditions, index, temp_vars, final, debug = False):
    # final is a list of dictionaries where dictionaries contain variable assignments
    # temp vars is a dictionary of variable:value
    if len(conditions) == index:
        if temp_vars not in final:
            final.append(temp_vars)
            #print('Final')
            #print(temp_vars)
            #print()
        return
    #print(temp_vars)
    for s in state:
        if unify(s, conditions[index]):
            '''
            for k, v in unify(s, conditions[index]).items():
                if k not in temp_vars or temp_vars[k] == v:
                    # variable has not been assigned yet, or variable has been assigned the same value
                    new_temp_vars = deepcopy(temp_vars)
                    new_temp_vars[k] = v
                    dfs_unification(state, conditions, index + 1, new_temp_vars, final, debug)
            '''
            if all(k not in temp_vars or temp_vars[k] == v for k, v in unify(s, conditions[index]).items()):
                new_temp_vars = deepcopy(temp_vars)
                for k, v in unify(s, conditions[index]).items():
                    new_temp_vars[k] = v
                dfs_unification(state, conditions, index + 1, new_temp_vars, final, debug)

    return final



#issue here. need to check all possible values?
def unify_all(state, actions, move, debug=False):
    #returns true if action conditions are possible, false otherwise
    #check by seeing if unification is possible?
    found = {}
    for precondition in actions[move]['conditions']:
        temp = False
        for s in state:
            #print(unify(s, precondition))
            if unify(s, precondition):
                for k, v in unify(s, precondition).items():
                    if k not in found:
                        found[k] = v
                    else:
                        if v != found[k]:
                            if debug: print("Unification failure on precondition" + str(precondition))
                            return False, None
                temp = True

                break
        if not temp:
            if debug: print("Unification failure on precondition" + str(precondition))
            return False, None
    #print(found)
    # temporary hack to fix drive
    if '?from' in found:
        if found['?to'] == found['?from']:
            if found['?from'] == "Home":
                found['?to'] = 'Store'
            else:
                found['?to'] = 'Home'
    return True, found


def dfs(current_state, goal, actions, debug, plan, visited):
    if check_goal(current_state, goal):
        print("Final plan")
        print(plan)
        return plan

    print("Visited")
    print(visited)
    if current_state not in visited:
        visited.append(current_state)
        frontier = []
        for action in actions:
            valid_move, variables = unify_all(current_state, actions, action, debug)
            print(variables)
            if valid_move:
                frontier.append((action, variables))
        print("Frontier")
        print(frontier)
        for f in frontier:
            new_state = add_and_remove(current_state, actions, f[0], f[1])
            print(new_state)
            return dfs(new_state, goal, actions, debug, plan + [f[0]], visited)
            # need some way of removing visited if wrong path? or doesn't matter

    # need to keep track of frontier(state), plan to reach frontier (actions), visited
    return

def check_visited(state, visited):
    for v in visited:
        if sorted(v) == sorted(state):
            return True
    return False

def dfs2(current_state, goal, actions, debug, plan, visited):
    if check_goal(current_state, goal):
        print("Final plan")
        print(plan)
        return plan

    #print("Visited")
    #print(visited)
    #print('current state')
    #print(check_visited(current_state, visited))
    if not check_visited(current_state, visited):
        visited.append(current_state)
        frontier = []
        for action in actions:

            list_var_assignments = dfs_unification(current_state, actions[action]['conditions'], 0, {}, [], debug)
            #valid_move, variables = unify_all(current_state, actions, action, debug)
            #print(variables)
            if list_var_assignments:
                for variables in list_var_assignments:
                    frontier.append((action, variables)) 
        #print("Frontier")
        #print(frontier)
        for f in frontier:
            new_state = add_and_remove(current_state, actions, f[0], f[1])
            #print(new_state)
            move = merge_action_var(actions[f[0]]['action'], f[1])
            res = dfs2(new_state, goal, actions, debug, plan + [move], visited)
            if res:
                return res
            # need some way of removing visited if wrong path? or doesn't matter
    # need to keep track of frontier(state), plan to reach frontier (actions), visited
    return []

def merge_action_var(action, variables):
    for v in variables:
        action = action.replace(v, variables[v])
    return action

def check_goal(current, goal):
    return set(current).issubset(set(goal))

def add_and_remove(state, actions, move, variables):
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
    keyorder = ['(item', '(place', '(agent', '(at']
    keyorder = dict(zip(keyorder, range(len(keyorder))))
    return sorted(state, key=lambda x: keyorder[x.split()[0]])

def remove(state, removals):
    pass

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

    state2 = ['(item Drill)', '(place Home)', '(place Store)', '(agent Me)', '(at Drill Store)', '(at Me Store)']

    plan = forward_planner( state2, goal, actions, debug=debug)

    #print(dfs_unification(start_state, actions['buy']['conditions'], 0, {}, []))
    '''unordered_state = [
        "(item Drill)",
        "(place Store)",
        "(agent Me)",
        "(at Me Home)",
        "(place Home)",
        "(item Me)",
        "(at Drill Store)"
    ]
    print(reorder(unordered_state))

    '''
    print(plan)



