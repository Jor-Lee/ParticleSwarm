import os
import np

def create_executable_script(
        output_script_path,
        pickle_path,
        log_file,
        number_of_particles=100,
        use_boxes_cutoff=10,
        box_reduction_factor=0.9,
        species_weights=[0.3, 0.3, 0.2, 0.2],
        use_adaptive_boxes=True,
        use_species=True,
        use_explosive_global_parameter=False,
        function_name='Griewank', 
        vthresh=1/(np.e**3),
        vboost=2*(np.e**2)
    ):
    code_template = f""" 
import numpy as np
import sys
sys.path.append('/work4/clf/Josh/Bayesian/OptimisingHyperParameters')
import pso as pso
import benchmark_functions as benchmark_functions
import os

function = benchmark_functions.BenchmarkFunctions().{function_name}()

dimension = function.dimension
number_of_particles = {number_of_particles}
bounds = function.bounds
maximise = function.maximise

def GenerateY(X):
    Y = []
    for x in X:
        Y.append(function.function(x))
    return Y

use_adaptive_hyper_parameters = False

def GlobalWeightFunction():
    return

def LocalWeightFunction():
    return

def InertialWeightFunction():
    return

use_adaptive_boxes = {use_adaptive_boxes}
use_species = {use_species}
species_weights = {species_weights}

use_boxes_cutoff = {use_boxes_cutoff}
boxes_cutoff = 0
box_reduction_factor = {box_reduction_factor}

def BoxWidthFunction(iteration):
    if iteration < boxes_cutoff:
        return (box_reduction_factor**(iteration))*(bounds[0][1] - bounds[0][0])
    else:
        use_adaptive_boxes = False
        return 0.0001

def SampleSizeFunction(iteration):
    if iteration < use_boxes_cutoff:
        if 10 - iteration >= 1:
            return 10 - iteration
    else:
        return 1

use_explosive_global_parameter = {use_explosive_global_parameter}

pso = pso.Pso(dimension, number_of_particles, bounds, GenerateY, 
                maximise, use_adaptive_hyper_parameters, GlobalWeightFunction,
                LocalWeightFunction, InertialWeightFunction, use_adaptive_boxes,
                use_species, species_weights, BoxWidthFunction, SampleSizeFunction,
                use_explosive_global_parameter, log_file="{log_file}")

while pso.current_iteration < 250:

    if pso.current_iteration == 0:
        pso.InitialiseSwarm()
        pso.vthresh = {vthresh}
        pso.vboost = {vboost}
    else:
        pso.UpdatePosition()
    pso.GenerateAllBoxes()
    pso.PopulateAllBoxes()
    pso.GetY()
    pso.UpdateLocalMaxima()
    if pso.use_species == True:
        pso.UpdateSpecies() 
    pso.UpdateGlobalMaxima()
    pso.RankParticles()
    pso.WriteSamplePointsHistory()

    if pso.use_species == True:
        pso.FindLowDensityRegions()
        pso.AssignAdventureLeads()
        pso.PredictOptimum()
    pso.UpdateVelocity()
    pso.current_iteration += 1
    with open("{pickle_path}", 'wb') as file:
        pickle.dump(pso, file)
    """

    with open(output_script_path, 'w') as script_file:
        script_file.write(code_template)
    print(f"Executable script written to: {output_script_path}")


def CreateAnalysisScript(output_script_path,
        pickle_path,
        log_file,
        number_of_particles=100,
        use_boxes_cutoff=10,
        box_reduction_factor=0.9,
        species_weights=[0.3, 0.3, 0.2, 0.2],
        use_adaptive_boxes=True,
        use_species=True,
        use_explosive_global_parameter=False,
        function_name='Griewank', 
    ):
    code_template = f""" 
import numpy as np
import sys
sys.path.append('/work4/clf/Josh/Bayesian/OptimisingHyperParameters')
import pso as pso
import benchmark_functions as benchmark_functions
import os

function = benchmark_functions.BenchmarkFunctions().{function_name}()

dimension = function.dimension
number_of_particles = {number_of_particles}
bounds = function.bounds
maximise = function.maximise

def GenerateY(X):
    Y = []
    for x in X:
        Y.append(function.function(x))
    return Y

use_adaptive_hyper_parameters = False

def GlobalWeightFunction():
    return

def LocalWeightFunction():
    return

def InertialWeightFunction():
    return

use_adaptive_boxes = {use_adaptive_boxes}
use_species = {use_species}
species_weights = {species_weights}

use_boxes_cutoff = {use_boxes_cutoff}
boxes_cutoff = 0
box_reduction_factor = {box_reduction_factor}

def BoxWidthFunction(iteration):
    if iteration < boxes_cutoff:
        return (box_reduction_factor**(iteration))*(bounds[0][1] - bounds[0][0])
    else:
        use_adaptive_boxes = False
        return 0.0001

def SampleSizeFunction(iteration):
    if iteration < use_boxes_cutoff:
        if 10 - iteration >= 1:
            return 10 - iteration
    else:
        return 1

use_explosive_global_parameter = {use_explosive_global_parameter}

with open({pickle_path}, 'rb') as file:
    my_pso = pickle.load(file)


"""