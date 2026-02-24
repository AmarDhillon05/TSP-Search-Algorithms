import os
import numpy as np
import time 
import itertools
import pickle
import matplotlib.pyplot as plt

plt.style.use("bmh")
SEED = 42 #Only for path shuffling, to store data
np.random.seed(SEED)

#Simple test client for cpu, start, cost
def test_simple(methods, n_mat = 10, path = "./data/nn.pkl"):


    #dict = { (name, func, args) }

    sizes = ['5', '10', '15', '20', '25', '30']
    paths = []
    allpaths = [p for p in os.listdir("./matrix") if p != "extra_credit.txt" and "random" in p]
    for size in sizes:
        paths += list(np.random.choice([p for p in allpaths if p.startswith(size) and not p.startswith(size + '0')], size = n_mat))
  
    np.random.shuffle(paths)
    paths = [os.path.join("./matrix", p) for p in paths]
    paths = sorted(paths, key = lambda x : x[0]) #Sorted by size
    matrices = [np.loadtxt(p) for p in paths]

    plt.figure(figsize = [20, 10])

    #Plotting cpu time, time, cost
    cost, paths, cpu_time, raw_time = {}, {}, {}, {}
    cities = []

    for idx, (name, func, args) in enumerate(methods):
        cost[name] = []
        paths[name] = []
        cpu_time[name] = []
        raw_time[name] = []

        cpu_start = time.process_time_ns()
        raw_start = time.time_ns()

        print(f"-" * 50)
        print(f"Testing {name}")

        for i, mat in enumerate(matrices):
            print(f"Entry {i}/{len(matrices)}")
            if idx == 0:
                cities.append(len(mat))

            cpu_start = time.process_time_ns()
            raw_start = time.time_ns()
            p, c = func(mat, **args)
            cpu_end = time.process_time_ns()
            raw_end = time.time_ns()

            cost[name].append(c)
            paths[name].append(p)
            cpu_time[name].append(cpu_end - cpu_start)
            raw_time[name].append(raw_end - raw_start)


    #Actual plotting
    figs = {
        "cost" : cost, "Process Time" : cpu_time, "Raw Time" : raw_time, "sizes" : cities
    }

    #Saving to file
    with open(path, "wb") as f:  
        pickle.dump(figs, f)

     
###################################################################################################################################

import os
import time
import pickle
import numpy as np

def test_astar(methods, astar, n_mat=None,
                       path=["./data/nnastar.pkl", "./data/astarlone.pkl"],
                       sleep_time=1):

    sizes = ['5', '10', '15', '20', '25', '30']
    allpaths = [p for p in os.listdir("./matrix")
                if p != "extra_credit.txt" and "random" in p]

    selected = []

    for size in sizes:
        n_per = n_mat
        if int(size) > 20:
            n_per = 2
        candidates = [p for p in allpaths
                      if p.startswith(size) and not p.startswith(size + '0')]
        selected += list(np.random.choice(candidates, size=n_per))

    selected = [os.path.join("./matrix", p) for p in selected]
    selected = sorted(selected, key=lambda x: x[0])

    # Initialize storage files if they don't exist
    if not os.path.exists(path[0]):
        with open(path[0], "wb") as f:
            pickle.dump({"cost": {}, "Process Time": {},
                         "Raw Time": {}, "sizes": []}, f)

    if not os.path.exists(path[1]):
        with open(path[1], "wb") as f:
            pickle.dump(
                {"acpu": [], "araw": [], "acost": [],
                 "apath": [], "uniq_cities": {}},
                f
            )

    for idx, pth in enumerate(selected):

        print("-" * 50)
        print(f"Processing {idx+1}/{len(selected)} -> {pth}")

        mat = np.loadtxt(pth)
        size = len(mat)

        # -----------------------------
        # Load current saved state
        # -----------------------------
        with open(path[0], "rb") as f:
            figs = pickle.load(f)

        with open(path[1], "rb") as f:
            astar_data = pickle.load(f)

        # -----------------------------
        # Run other methods
        # -----------------------------
        for name, func, args in methods:

            if name not in figs["cost"]:
                figs["cost"][name] = []
                figs["Process Time"][name] = []
                figs["Raw Time"][name] = []

            print(f"Testing {name}")

            cpu_start = time.process_time_ns()
            raw_start = time.time_ns()

            p, c = func(mat, **args)

            cpu_end = time.process_time_ns()
            raw_end = time.time_ns()

            figs["cost"][name].append(c)
            figs["Process Time"][name].append(cpu_end - cpu_start)
            figs["Raw Time"][name].append(raw_end - raw_start)

        figs["sizes"].append(size)

        # -----------------------------
        # Run A*
        # -----------------------------
        print("Testing A*")

        cpu_start = time.process_time_ns()
        raw_start = time.time_ns()

        p, c, e = astar(mat)

        cpu_end = time.process_time_ns()
        raw_end = time.time_ns()

        astar_data["acost"].append(c)
        astar_data["apath"].append(p)
        astar_data["acpu"].append(cpu_end - cpu_start)
        astar_data["araw"].append(raw_end - raw_start)

        if size not in astar_data["uniq_cities"]:
            astar_data["uniq_cities"][size] = []
        astar_data["uniq_cities"][size].append(e)

        # -----------------------------
        # Save immediately
        # -----------------------------
        with open(path[0], "wb") as f:
            pickle.dump(figs, f)

        with open(path[1], "wb") as f:
            pickle.dump(astar_data, f)

        print("Saved progress.")

        # -----------------------------
        # Take a break
        # -----------------------------
        time.sleep(sleep_time)
    
    


#################################################################################################################################

#Generation-based algos
def test_grad(methods, n_mat = None, path = "./data/grad.pkl"):


    #dict = { (name, func, args) }

    sizes = ['5', '10', '15', '20', '25', '30']
    paths = []
    allpaths = [p for p in os.listdir("./matrix") if p != "extra_credit.txt" and "random" in p]
    for size in sizes:
        paths += list(np.random.choice([p for p in allpaths if p.startswith(size) and not p.startswith(size + '0')], size = n_mat))
    if n_mat == None:
        n_mat = len(paths)

    np.random.shuffle(paths)
    rpaths = [os.path.join("./matrix", p) for p in paths]
    paths = sorted(rpaths, key = lambda x : x[0]) #Sorted by size
    matrices = [np.loadtxt(p) for p in paths]
    print([len(m) for m in matrices])

    plt.figure(figsize = [30, 30])

    #Plotting cpu time, time, cost
    #Now formatted as rargs : [names], [poss]
    cost, paths, cpu_time, raw_time = {}, {}, {}, {}
    args_global = {}
    iterd = {}
    cities = {}

    for idx, (name, func, args_raw, _) in enumerate(methods):
        cost[name] = []
        paths[name] = []
        cpu_time[name] = []
        raw_time[name] = []
        args_global[name] = [] #For each run
         #Now its for each run

        print(f"-" * 50)
        print(f"Testing {name}")

        argsets = list(itertools.product(*args_raw[1]))
        aidx = 0
        for argset in argsets:
            args = {}
            aidx += 1

            for argname, argval in zip(args_raw[0], argset):
                if args.get(argname) is None:
                    args[argname] = argval

            print(f"Argset {aidx}/{len(argsets)}: {args}")

            for i, mat in enumerate(matrices):
                print(f"Working on {i}/{len(matrices)} ({rpaths[i]})")
                if cities.get(name) is None:
                    cities[name] = []
                cities[name].append(len(mat))

                cpu_start = time.process_time_ns()
                raw_start = time.time_ns()
                p, c, iterdata = func(mat, **args)
                cpu_end = time.process_time_ns()
                raw_end = time.time_ns()

                cost[name].append(c)
                paths[name].append(p)
                cpu_time[name].append(cpu_end - cpu_start)
                raw_time[name].append(raw_end - raw_start)
                
                args_global[name].append(args)

                if iterd.get(name) == None:
                    iterd[name] = []
                iterd[name].append(iterdata)

            
    #Saving 
    figs = {
        "argset" : args_global, "cost"  : cost, "Process Time" : cpu_time, "Raw Time" : raw_time, "sizes" : cities, 
        "iterdata" : iterd
    }
    with open(path, "wb") as f:  
        pickle.dump(figs, f)