import heapq 
import numpy as np 

#Imported from AIMA repo -> https://github.com/aimacode/aima-python/blob/master/utils.py#L722

class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)



#Random permutation
def randomp(x):
    while True:
        path = [i for i in range(len(x))]
        np.random.shuffle(path)
        path.append(path[0])
        cost = 0

        cont = True

        for idx in range(0, len(path) - 1):
            edge = x[path[idx]][path[idx+1]]
            if edge == 0:
                cont = False
                break
            else:
                cost += edge

        if cont:
            return path, float(cost)
        


#Helper to find n random valid swaps (assuming x wraps around itself) or returns None is none are available
def find_valid_swap(x, path, cost, swaps):

    tested = set()
    rdepth = 0 #Max 100 as guard, should never happen and will throw errors in method if it does as a form of alert

    while rdepth < 100:
        new_cost = cost
        
        rdepth += 1
        

        if swaps != None and len(swaps) == 0:
            break
 

        if swaps is not None:
            lswap = list(swaps)
            np.random.shuffle(lswap)
            (i, j) = lswap[0]
            swaps.remove((i, j))

        else:
            i = np.random.randint(0, len(x) - 1)
            j = np.random.randint(i, len(x))


        if (i, j) in tested:
            continue

        idx_to_sub = { (i, i+1), (i, i-1), (j, j+1), (j, j-1) }
        idx_to_add = { (i, j+1), (i, j-1), (j, i-1), (j, i+1) }
        if i == 0:
            idx_to_sub.remove((i, i-1))
            idx_to_add.remove((j, i-1))

        tobreak = False
        for a, b in idx_to_add:
            if x[path[a]][path[b]] == 0:
                tobreak = True
                break
            new_cost += x[path[a]][path[b]]

        if tobreak:
            continue
        
        for a, b in idx_to_sub:
            new_cost -= x[path[a]][path[b]]
        
        newpth = path.copy()
        tmp = newpth[i]; newpth[i] = newpth[j]; newpth[j] = tmp
        newpth[-1] = newpth[0] #In case swap since j != end

        return i, j, newpth, new_cost
    
    return None

    



#MPX crossover for genetic impl
def mpx_crossover(x, p1, p2):
    l = len(p1) // 2
    if len(p1) > 10:
        if len(p1) // 2 > 10:
            l = np.random.randint(10, len(p1) // 2)
        else:
            l = 10

    while True:
        startidx = np.random.randint(0, len(p1) - l)
        taken = set(p1[startidx : startidx + l])

        path = p1[startidx : startidx + l]
        cost = 0
        cont = True
        for idx in range(0, len(path) - 1):
            if x[path[idx]][path[idx+1]] == 0:
                cont = False; break
            cost += x[path[idx]][path[idx+1]]
        
        if cont:
            for el in p2:
                if el not in taken:
                    if x[el][path[-1]] == 0:
                        cont = False; break
                    cost += x[el][path[-1]]
                    path.append(el)
                    taken.add(el)

            if x[path[-1]][path[0]] > 0:
                cost += x[path[-1]][path[0]]
                path.append(path[0])
                return path, float(cost)
