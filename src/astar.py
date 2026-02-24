import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from src.utils import PriorityQueue

def astar(x):
    
    #h(x) (double check this later)
    def h_x(n, visited):
        indices = [idx for idx in range(len(x)) if idx not in visited and idx != n]
        h = minimum_spanning_tree(x[indices][:, indices]).sum()
        h += np.sort([i for i in x[n][indices] if i != 0])[:2].sum()
        return h


    #Init q with start + h(x)
    #Tuples are (nidx, path, visited, pcost, f(x))
    q = PriorityQueue(order = 'min', f = lambda x : x[-1]) 
    start = np.random.choice([i for i in range(len(x))])
    q.append(( int(start), [int(start)], {int(start)}, 0, h_x(start, set()) ))

    #For tracking n expanded
    n_expanded = 0
     
    #Recurs until we find the best path
    while q.__len__() > 0:

        (nidx, path, visited, cost, _) = q.pop()
        #print(f"Expanding path of length {len(path)}")
        neighbors = [idx for idx in range(len(x)) if idx not in visited and x[nidx][idx] > 0]

        if len(neighbors) == 0:
            if len(path) == len(x) and x[nidx][start] > 0:
                return path + [int(start)], float(cost + x[nidx][start]), n_expanded
        
        else:
            for a, nextn in enumerate(neighbors):

                #a = 0 means we expanded at laest one neighbor
                if a == 0:
                    n_expanded += 1

                f_x = cost + x[nidx][nextn] + h_x(nextn, visited)
                v = visited.copy(); v.add(nextn)
                q.append(
                    ( int(nextn), path + [int(nextn)], v, cost + x[nidx][nextn], f_x )
                )

    print("Found no path")
    return [], float('inf'), n_expanded #Should never reach