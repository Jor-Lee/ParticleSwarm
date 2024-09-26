import numpy as np
import PSO_species.pso as pso
import PSO_species.species_benchmarking.benchmark_functions as benchmark_functions
import csv
import os

csv_path = os.path.join(os.parth.dirname(os.path.abspath(__file__)), 'output.csv')

function = benchmark_functions.my_function()

dimension = function.dimension
number_of_particles = 100
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

use_adaptive_boxes = True
use_species = True
species_weights = [0.4, 0.3, 0.2, 0.1]

use_boxes_cutoff = 10 
box_reduction_factor = 0.9

def BoxWidthFunction(iteration):
    if iteration < use_boxes_cutoff:
        return (0.9**(iteration))*(bounds[0][1] - bounds[0][0])
    else:
        use_adaptive_boxes = False
        return 0.0001

def SampleSizeFunction(iteration):
    if iteration < use_boxes_cutoff:
        if use_boxes_cutoff - iteration >= 1:
            return use_boxes_cutoff - iteration
    else:
        return 1

use_explosive_global_parameter = True

pso = pso.Pso(dimension, number_of_particles, bounds, GenerateY,
                maximise, use_adaptive_hyper_parameters, GlobalWeightFunction,
                LocalWeightFunction, InertialWeightFunction, use_adaptive_boxes,
                use_species, species_weights, BoxWidthFunction, SampleSizeFunction,
                use_explosive_global_parameter)

with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    while pso.swarm_converged == False:

        if pso.current_iteration == 0:
            pso.InitialiseSwarm()
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

        particle_positions = []
        particle_velocitites = []
        for particle in pso.swarm.particles:
            particle_position_row = [1, pso.current_iteration, particle.species] + [particle.position[i] for i in range(len(particle.position))] + [0] + [particle.velocity[i] for i in range(len(particle.velocity))]
            writer.writerow(particle_position_row)
        for particle in pso.swarm.particles:
            for s, sample_point in enumerate(particle.sample_points):
                sample_point_row = [2, pso.current_iteration, None] + [sample_point[i] for i in range(len(sample_point))] + [particle.sample_points_results[s]] + [0] * len(sample_point)
                writer.writerow(sample_point_row)
        global_max_row = [3, pso.current_iteration, None] + [pso.swarm.global_max_loc[i] for i in range(len(global_max_loc))] + [pso.swarm.global_max] + [0] * len(pso.swarm.global_max_loc)
        writer.writerow(global_max_row)

        if pso.use_species == True:
            pso.FindLowDensityRegions()
            pso.AssignAdventureLeads()
            pso.PredictOptimum()
        pso.UpdateVelocity()
        pso.current_iteration += 1
        pso.DetermineConvergence()
