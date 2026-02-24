import numpy as np 
from src.utils import mpx_crossover, randomp, find_valid_swap


#Hill climbing
def hill_climbing(x, num_restarts, find_valid_swap = find_valid_swap, n_neighbors = 10):
    min_path, min_cost = [], float('inf')

    ppg, cpg = [], []

    for restartidx in range(num_restarts):
        local_path, local_cost = randomp(x)
        base_path = local_path.copy()

        run = True
        while run:

            improved = False

            #Attempt on n neighbors
            neighbors_left = n_neighbors
            allswaps = { (0, 0) }
            for i in range(0, len(x) - 1):
                for j in range(1, len(x)):
                    allswaps.add((i, j))

            while neighbors_left > 0:
                #Generate random + check valid + recalc cost
                i, j = -1, -1
                p, c = [], float('inf')
                while True:
                    swap_result = find_valid_swap(x, local_path, local_cost, allswaps)
                    if swap_result == None:
                        break
                    i, j, p, c = swap_result

                #Indicates no swaps left 
                if (i, j) == (-1, -1):
                    run = False
                    break

                #Update
                if c < local_cost:
                    local_path, local_cost = p, c
                    improved = True
            
                if tuple(sorted((i, j))) in allswaps:
                    allswaps.remove(tuple(sorted((i, j))))
                neighbors_left -= 1

            #If no progress has been made
            if not improved:
                break


        #Using the path we found
        if local_cost < min_cost:
            min_path, min_cost = local_path, local_cost

        ppg.append((restartidx, min_path))
        cpg.append((restartidx, min_cost))

    return min_path, float(min_cost), cpg



#Simulated Annealing
def simulated_annealing(x, max_iterations, alpha, initial_temperature, find_valid_swap = find_valid_swap, n_neighbors = 10):
    min_path, min_cost = [], float('inf')

    #Only running on one state, but attempting to climb a hill
    local_path, local_cost = randomp(x)
    tmp = initial_temperature

    cpg = []
    ppg = []

    for iteridx in range(max_iterations):


        #Finding the best neighbor
        neighbors_left = n_neighbors
        allswaps = { (0, 0) }
        allswaps = { (0, 0) }
        for i in range(0, len(x) - 1):
            for j in range(1, len(x)):
                allswaps.add((i, j))

        best_path, best_cost = local_path.copy(), local_cost

        neighbors_left = n_neighbors
        while neighbors_left > 0:

            #Generate random + check valid + recalc cost
            i, j = -1, -1
            p, c = [], float('inf')
            while len(allswaps) > 0:
                swapres = find_valid_swap(x, local_path, local_cost, allswaps)
                if swapres == None:
                    break
                i, j, p, c = swapres

            if (i, j) == (-1, -1):
                break

            #Accepting this solution if formula sat
            #Ignoring warnings for prod (solve err in nb)
            with np.errstate(over = "ignore"):
                prob = np.exp( (c - best_cost) / tmp )
            if prob > 1 or np.random.choice([True, False], p=[prob, 1-prob]):
                best_path, best_cost = p, c
                tmp *= alpha
            
            if tuple(sorted((i, j))) in allswaps:
                allswaps.remove(tuple(sorted((i, j))))

            neighbors_left -= 1

        #Update if we made progress
        if best_cost <= local_cost:
            local_path, local_cost = best_path, best_cost

        ppg.append((iteridx, best_path))
        cpg.append((iteridx, best_cost))


    return local_path, float(local_cost), cpg




#Genetic impl
def genetic_algo(x, mutation_chance, population_size, num_generations, 
                 crossover=mpx_crossover, find_valid_swap=find_valid_swap, elim_pc = 0.1): 

    paths = sorted([randomp(x) for _ in range(population_size)], key = lambda p : -p[-1])
    cpg = []
    ppg = []

    for genidx in range(num_generations):

        children = []

        while len(paths) > 0:
            if len(paths) == 1:
                p, c = paths.pop()
                if np.random.choice([True, False], p=[mutation_chance, 1-mutation_chance]):
                    _, _, p, c = find_valid_swap(x, p, c, None)
                children.append((p, c))
            else:
                idx = np.random.randint(0, len(paths))
                idy = np.random.randint(0, len(paths))
                while idy == idx:
                    idy = np.random.randint(0, len(paths))

                p1, _ = paths[idx]
                p2, _ = paths[idy]
                crossp, crossc = crossover(x, p1, p2)

                if np.random.choice([True, False], p=[mutation_chance, 1-mutation_chance]):
                    _, _, crossp, crossc = find_valid_swap(x, crossp, crossc, None)

                children.append((crossp, crossc))
                paths.pop(idx)
                if idy < idx:
                    paths.pop(idy)
                else:
                    paths.pop(idy - 1)

        children = sorted(children, key = lambda p : -p[-1])[int(len(children) * elim_pc) : ]
        paths = children
   
        cpg.append((genidx, paths[-1][-1]))
        ppg.append((genidx, paths[-1][0]))
  
    p, c = paths[-1]
    return p, float(c), cpg