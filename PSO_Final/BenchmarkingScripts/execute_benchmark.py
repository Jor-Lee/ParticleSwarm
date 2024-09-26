import os
import sys
sys.path.append('/work4/clf/Josh/Bayesian/OptimisingHyperParameters')
import write_executable as write_executable
import subprocess
import datetime as datetime
import benchmark_functions as benchmark_functions

output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs/')

function_names = [f'F{i}' for i in range(10)]

# This is the thing you need to change to make the simulation test different things e.g. species weights or hyper_parameters
weights = [[], []]

def RunSimulations(directory, num_scripts):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for function in function_names:
        fn_directory = os.path.join(directory, str(function))
        fn = getattr(benchmark_functions.BenchmarkFunctions(), f'F{i}')
        os.makedirs(fn_directory)
        for i in range(num_scripts): 
            sim_directory = os.path.join(fn_directory, str(f'simulation_{i}'))
            os.makedirs(sim_directory)
            output_path = os.path.join(sim_directory, f'executable.py')
            pickle_path = os.path.join(sim_directory, f'pso.pkl')
            log_file = os.path.join(sim_directory, f'log.log')

            # Example parameter values; you can adjust these as needed
            write_executable.create_executable_script(
                output_path,
                pickle_path,
                log_file=log_file,
                number_of_particles=20*fn.dimension,  # Example: increment number of particles
                use_boxes_cutoff=0,           # Example: increment cutoff
                box_reduction_factor=0.9,  # Example: decrement box reduction factor
                species_weights=[0.2, 0.2, 0.2, 0.2, 0.2],  # Example: adjust species weights
                use_adaptive_boxes=False,
                use_species=True,
                use_explosive_global_parameter=False,
                function_name=function,
            ) 
            subprocess.run(['python', output_path])

RunSimulations(output_directory, 1)

