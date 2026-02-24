import numpy as np 

#Nearest Neighbor
def nn(x):
    #Forming sorted reference dict (only holds closest indices, x must be reffed for values)
    nodes = {}
    for idx, row in enumerate(x):
        nodes[idx] = [idy for idy in np.argsort(row) if x[idx][idy] > 0 and idx != idy]

    #Loop to repeatedly select closest 
    starting_idx = np.random.choice(list(nodes.keys()))
    curr = starting_idx
    cost = 0
    visited = set()
    path = []
    
    while True:
        visited.add(curr)
        path.append(int(curr))

        nextn = None 
        for idx in nodes[curr]:
            if idx not in visited:
                nextn = idx
                break

        if nextn == None:
            break

        cost += x[curr][nextn]
        curr = nextn

    #End -> Checking if we can complete + correct length
    if len(visited) < len(x) - 1 or x[curr][starting_idx] == 0:
        return "Some error message"
    else:
        cost += x[curr][starting_idx]
        path += [int(starting_idx)]
        return path, float(cost)



#Two-opt for any path
def two_opt(x, path, cost):
    min_path, min_cost = path, cost

    #Function to try all swap pairs in a path (start + end = starts of pairs)
    def test_swaps(start, end):
        improved = False 

        nonlocal min_cost
        nonlocal min_path

        for i in range(start, end-2):
            for j in range(i + 2, end):
                curr_cost = x[min_path[i]][min_path[i+1]] + x[min_path[j]][min_path[j+1]]
                swap_cost = x[min_path[i]][min_path[j]] + x[min_path[i+1]][min_path[j+1]]
                if swap_cost < curr_cost:
                    new_path = min_path[:i+1] + min_path[i+1:j+1][::-1] + min_path[j+1:]
                    new_cost = min_cost - curr_cost + swap_cost
                    if new_cost < min_cost:
                        min_path, min_cost = new_path, new_cost
                        improved = True
       
        return improved


    while test_swaps(0, len(path) - 2):
        continue

    return min_path, float(min_cost)

#Short wrapper for nearest-neighbor two-opt
def nn_two_opt(x):
    p, c = nn(x)
    return two_opt(x, p, c)



#Repeated Random Nearest Neighbor
def rrnn(x, k, n_repeats, two_opt_func = None):
    #Forming sorted reference dict (only holds closest indices, x must be reffed for values)
    nodes = {}
    for idx, row in enumerate(x):
        nodes[idx] = [idy for idy in np.argsort(row) if x[idx][idy] > 0 and idx != idy]

    best_path, best_cost = [], float("inf")

    for _ in range(n_repeats):
        #Loop to repeatedly select closest 
        starting_idx = np.random.choice(list(nodes.keys()))
        curr = starting_idx
        cost = 0
        visited = set()
        path = []
        
        while True:
            visited.add(curr)
            path.append(int(curr))

            nextn = []
            for idx in nodes[curr]:
                if len(nextn) == k:
                    break
                if idx not in visited:
                    nextn.append(idx)

            if nextn == []:
                break

            nextn = np.random.choice(nextn)
            cost += x[curr][nextn]
            curr = nextn

        #End -> Checking if we can complete + correct length
        if len(visited) < len(x) - 1 or x[curr][starting_idx] == 0:
            pass #Should never happen
        else:
            cost += x[curr][starting_idx]
            path += [int(starting_idx)]
            if two_opt_func:
                path, cost = two_opt_func(x, path, cost)
            if cost < best_cost:
                best_path, best_cost = path, float(cost)

    return best_path, best_cost
