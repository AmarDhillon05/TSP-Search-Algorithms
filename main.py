import numpy as np
import os 
import sys
import time 

#Original package-level imports
from src.nn import nn, nn_two_opt, rrnn
from src.astar import astar
from src.grad import simulated_annealing, hill_climbing, genetic_algo

helpstring = '''
Entrypoint for cmd-level algo running
Usage:

python -m main get_random_path ---> outputs a random matrix path to use 
function-specific syntaxes (args marked with * are optional):

python -m main nn [path]

python -m main nn2opt [path]

python -m main rrnn [path] k=[int] n_repeats=[int] *two_opt_func=[int-bool (0/1)]

python -m main astar [path]

python -m main hill_climbing [path] num_restarts=[int] *n_neighbors=[int]

python -m main simulated_annealing [path] max_iterations=[int] alpha=[int] 
			      initial_temperature=[int] *n_neighbors=[int]

python -m main genetic_algo [path] mutation_chance=[float] population_size=[int] 
                                           num_generations=[int] *elim_pc=[float]

'''

methods = {
    "nn" : nn, "nn2opt" : nn_two_opt, "rrnn" : rrnn, 
    "astar" : astar,
    "hill_climbing" : hill_climbing, "simulated_annealing" : simulated_annealing, "genetic_algo" : genetic_algo
}

try:
    if sys.argv[1] == "get_random_path":
        print( "./matrix/" + np.random.choice(os.listdir("./matrix")) )

    else:
        try:
            func = methods[sys.argv[1]]
            mat = np.loadtxt(sys.argv[2])
            args = {}
            for arg in sys.argv[3:]:
                k, v = arg.split("=")
                args[k] = float(v)
                if '.' not in v:
                    args[k] = int (v)

            print(f"Executing {sys.argv[1]} with args {args}")
            res = func(mat, **args)
            print(f"Path: {res[0]}, Cost: {res[1]}")

        except Exception as e:
            print(e)
            print("Function execution error")
            print(helpstring)

except Exception as e:
    print(helpstring)

print("-" * 50)
time.sleep(0.5) #For gradual printing