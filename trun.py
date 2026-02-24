import numpy as np
from src.trunner import test_simple, test_astar, test_grad
from src.nn import nn, two_opt, nn_two_opt, rrnn
from src.astar import astar
from src.grad import simulated_annealing, hill_climbing, genetic_algo
from src.utils import mpx_crossover, randomp, find_valid_swap

#Test/plot runner/gen
#Toggle T/F to run/notrun

if False:
    test_simple([
        ("NN", nn, {}), 
        ("NN2OPT", nn_two_opt, {}), 
        ("RRNN1", rrnn, {"k" : 5, "n_repeats" : 10, "two_opt_func" : two_opt}), 
        ("RRNN2", rrnn, {"k" : 20, "n_repeats" : 15}),
    ], n_mat = 10)


if False:
    test_astar([
        ("NN", nn, {}), 
        ("NN2OPT", nn_two_opt, {}), 
        ("RRNN1", rrnn, {"k" : 5, "n_repeats" : 10, "two_opt_func" : two_opt}), 
    ], n_mat = 10, astar = astar)


if True:
    test_grad([
        ("HC", hill_climbing, (["num_restarts", "n_neighbors"], [[5, 10], [1, 5, 10]]), ["num_restarts", "n_neighbors"]),
        ("SA", simulated_annealing, (["max_iterations", "alpha", "initial_temperature"], [[5, 10], [0.1, 0.3, 0.7], [10, 100]]), ["initial_temperature", "alpha"]), 
        ("GA", genetic_algo, (['mutation_chance', "population_size", "num_generations", "elim_pc"], [[0.1, 0.7], [10, 100], [5, 10], [0.1, 0.7]]), ["mutation_chance", "elim_pc"])
    ], n_mat=2)