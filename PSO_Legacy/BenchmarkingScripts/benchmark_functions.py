import numpy as np
import random

import numpy as np

class BenchmarkFunctions:

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Sphere Function
    class F0:
        def __init__(sphere):
            sphere.name = 'Sphere'
            sphere.x = None
            sphere.dimension = 30
            sphere.bounds = [[-100, 100] for _ in range(sphere.dimension)]
            sphere.maximise = False
            sphere.optimum = 0
            sphere.goal = 0
            sphere.large_value = 1e10

        def function(sphere, x):
            # Check if any dimension is out of bounds
            for i in range(sphere.dimension):
                lower_bound, upper_bound = sphere.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if sphere.maximise:
                        return -sphere.large_value
                    # If minimising, return a large positive number
                    else:
                        return sphere.large_value
            return sum([xi ** 2 for xi in x])

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Rastrigin Function
    class F1:
        def __init__(rastrigin):
            rastrigin.name = 'Rastrigin'
            rastrigin.x = None
            rastrigin.dimension = 30
            rastrigin.bounds = [[-5.12, 5.12] for _ in range(rastrigin.dimension)]
            rastrigin.maximise = False
            rastrigin.optimum = 0
            rastrigin.goal = 0
            rastrigin.large_value = 1e10

        def function(rastrigin, x):
            # Check if any dimension is out of bounds
            for i in range(rastrigin.dimension):
                lower_bound, upper_bound = rastrigin.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if rastrigin.maximise:
                        return -rastrigin.large_value
                    # If minimising, return a large positive number
                    else:
                        return rastrigin.large_value

            return 10 * len(x) + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Rosenbrock Function
    class F2:
        def __init__(rosenbrock):
            rosenbrock.name = 'Rosenbrock'
            rosenbrock.x = None
            rosenbrock.dimension = 30
            rosenbrock.bounds = [[-30, 30] for _ in range(rosenbrock.dimension)]
            rosenbrock.maximise = False
            rosenbrock.optimum = 0
            rosenbrock.goal = 0 
            rosenbrock.large_value = 1e10

        def function(rosenbrock, x):
            # Check if any dimension is out of bounds
            for i in range(rosenbrock.dimension):
                lower_bound, upper_bound = rosenbrock.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if rosenbrock.maximise:
                        return -rosenbrock.large_value
                    # If minimising, return a large positive number
                    else:
                        return rosenbrock.large_value
            return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)])
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Griewank Function
    class F3:
        def __init__(griewank):
            griewank.name = 'Griewank'
            griewank.x = None
            griewank.dimension = 30
            griewank.bounds = [[-600, 600] for _ in range(griewank.dimension)]
            griewank.maximise = False
            griewank.optimum = 0.05 
            griewank.goal = 0
            griewank.large_value = 1e10 

        def function(griewank, x):
            # Check if any dimension is out of bounds
            for i in range(griewank.dimension):
                lower_bound, upper_bound = griewank.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if griewank.maximise:
                        return -griewank.large_value
                    # If minimising, return a large positive number
                    else:
                        return griewank.large_value
            total1 = 0
            total2 = 1
            for i in range(len(x)):
                total1 += x[i] ** 2
                total2 *= np.cos(x[i] / np.sqrt(i + 1))
            return (1 / 4000) * total1 - total2 + 1
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Ackley Function
    class F4:
        def __init__(ackley):
            ackley.name = 'Ackley'
            ackley.x = None
            ackley.dimension = 30
            ackley.bounds = [[-32.768, 32.768] for _ in range(ackley.dimension)]
            ackley.maximise = False
            ackley.optimum = 0
            ackley.goal = 0
            ackley.large_value = 1e10

        def function(ackley, x):
            # Check if any dimension is out of bounds
            for i in range(ackley.dimension):
                lower_bound, upper_bound = ackley.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if ackley.maximise:
                        return -ackley.large_value
                    # If minimising, return a large positive number
                    else:
                        return ackley.large_value
            first_sum = sum([xi ** 2 for xi in x])
            second_sum = sum([np.cos(2 * np.pi * xi) for xi in x])
            n = len(x)
            return -20 * np.exp(-0.2 * np.sqrt(first_sum / n)) - np.exp(second_sum / n) + 20 + np.e

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Schwefel Function
    class F5:
        def __init__(schwefel):
            schwefel.name = 'Schwefel'
            schwefel.x = None
            schwefel.dimension = 30
            schwefel.bounds = [[-500, 500] for _ in range(schwefel.dimension)]
            schwefel.maximise = False
            schwefel.optimum = 0
            schwefel.goal = 0
            schwefel.large_value = float('inf')

        def function(schwefel, x):
            # Check if any dimension is out of bounds
            for i in range(schwefel.dimension):
                lower_bound, upper_bound = schwefel.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if schwefel.maximise:
                        return -schwefel.large_value
                    # If minimising, return a large positive number
                    else:
                        return schwefel.large_value
            return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Levy Function
    class F6:
        def __init__(levy):
            levy.name = 'Levy'
            levy.x = None
            levy.dimension = 30
            levy.bounds = [[-10, 10] for _ in range(levy.dimension)]
            levy.maximise = False
            levy.optimum = 0
            levy.goal = 0
            levy.large_value = 1e10

        def function(levy, x):
            # Check if any dimension is out of bounds
            for i in range(levy.dimension):
                lower_bound, upper_bound = levy.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if levy.maximise:
                        return -levy.large_value
                    # If minimising, return a large positive number
                    else:
                        return levy.large_value
            w = [1 + (xi - 1) / 4 for xi in x]
            term1 = np.sin(np.pi * w[0]) ** 2
            term2 = sum([(wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2) for wi in w[:-1]])
            term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
            return term1 + term2 + term3

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Michalewicz Function
    class F7:
        def __init__(michalewicz):
            michalewicz.name = 'Michalewicz'
            michalewicz.x = None
            michalewicz.dimension = 30
            michalewicz.bounds = [[0, np.pi] for _ in range(michalewicz.dimension)]
            michalewicz.maximise = False
            michalewicz.optimum = 0
            michalewicz.goal = 0
            michalewicz.large_value = 1e10

        def function(michalewicz, x, m=10):
            # Check if any dimension is out of bounds
            for i in range(michalewicz.dimension):
                lower_bound, upper_bound = michalewicz.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if michalewicz.maximise:
                        return -michalewicz.large_value
                    # If minimising, return a large positive number
                    else:
                        return michalewicz.large_value
            return -sum([np.sin(xi) * (np.sin(i * xi ** 2 / np.pi)) ** (2 * m) for i, xi in enumerate(x, 1)])
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Beale Function
    class F8:
        def __init__(beale):
            beale.name = 'Beale'
            beale.x = None
            beale.dimension = 2
            beale.bounds = [[-4.5, 4.5], [-4.5, 4.5]]
            beale.maximise = False
            beale.optimum = 0
            beale.goal = 0
            beale.large_value = 1e10

        def function(beale, x):
            # Check if any dimension is out of bounds
            for i in range(beale.dimension):
                lower_bound, upper_bound = beale.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if beale.maximise:
                        return -beale.large_value
                    # If minimising, return a large positive number
                    else:
                        return beale.large_value
            x1, x2 = x[0], x[1]
            return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Easom Function
    class F9:
        def __init__(easom):
            easom.name = 'Easom'
            easom.x = None
            easom.dimension = 2
            easom.bounds = [[-100, 100], [-100, 100]]
            easom.maximise = False
            easom.optimum = -1
            easom.goal = 0
            easom.large_value = 1e10

        def function(easom, x):
            # Check if any dimension is out of bounds
            for i in range(easom.dimension):
                lower_bound, upper_bound = easom.bounds[i]
                if x[i] < lower_bound or x[i] > upper_bound:
                    # If out of bounds and maximising, return a large negative number
                    if easom.maximise:
                        return -easom.large_value
                    # If minimising, return a large positive number
                    else:
                        return easom.large_value
            x1, x2 = x[0], x[1]
            return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

