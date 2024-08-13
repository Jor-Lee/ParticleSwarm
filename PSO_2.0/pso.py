import os
import subprocess
import time
import scipy as sp
from scipy.stats import qmc
import random
import numpy as np
import sys

class Pso:

    def  __init__  (self, dimension, number_of_particles, bounds, use_adaptive_hyper_parameters, use_adaptive_boxes, use_species, species_weights, GenerateY, maximise):

        self.dimension = dimension                                                  # Dimension of the system to be optimised
        self.number_of_particles = number_of_particles                              # Number of swarm particles (not including points tested in boxes)
        self.bounds = bounds                                                        # Bounds of the optimisation e.g. [[0,1], [2, 10]] for a 2D optimisation
        self.use_adaptive_hyper_parameters = use_adaptive_hyper_parameters          # Flag to turn on adaptive hyperparameter functions
        self.use_adaptive_boxes = use_adaptive_boxes                                # Flag to turn on adaptive box sizes
        self.GenerateY = GenerateY                                                  # Black-box function to be optimised
        self.maximise = maximise                                                    # If maximise == True => maximise, else minimise
        self.species_weights = species_weights                                      # [0.3, 0.2, 0.4, 0.1] %ages of each species

        self.initial_velocity = (bounds[0][1]-bounds[0][1])/self.number_of_particles # effectively move the boxes on the first iteration
        self.current_iteration = 0
        self.swarm = None
        

    class Particle:

        def __init__ (self, position, velocity, local_max, local_max_loc, global_max, global_max_loc, species):
            
            self.position = position                                                # position of the particle
            self.velocity = velocity                                                # velocity of the particle
            self.local_max = local_max                                              # the best value of GenerateY found by a given particle
            self.local_max_loc = local_max_loc                                      # the location that gave this best value
            self.global_max = global_max                                            # best value of GenerateY from the particles that it communicates with 
            self.global_max_loc                                                     # location of this best value
            self.species = species                                                  # species of particle - affects hyperparameters and social behaviour

    class Species:
        class Reckless:
            def __init(reckless):
                reckless.id = 0
                reckless.hyper_parameter_set = None
        class Cautious:
            def __init__(cautious):
                cautious.id = 1
                cautious.hyper_parameter_set = None
        class Adventurous:
            def __init__(adventurous):
                adventurous.id = 2
                adventurous.hyper_parameter_set = None
        class Intuitive:
            def __init__(intuitive):
                intuitive.id = 3
                intuitive.hyper_parameter_set = None

    class Swarm:

        def __init__ (swarm, particles, communication_matrix):
            swarm.particles = particles                                             # holds an array of type: particle
            swarm.communication_matrix = communication_matrix                       # number_of_particles X number of particles matrix holding the communication values between particles

    def InitialiseSwarm(self):

        # Create the array of evenly sampledswarm locations 
        sampler = qmc.Sobol(self.dimension)                                                         # Establish the sampler for given dimesnion
        sample = sampler.random(self.number_of_particles)                                           # Generate required number of points (number_of_particles)
        swarm_locs = qmc.scale(sample, np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1])    # Scale the sample to the given bounds

        # Create the set of random swarm velocities
        swarm_velocities = np.random.uniform(-self.initial_velocity, self.initial_velocity, size=(self.number_of_particles, self.dimension))

        species = [0, 1, 2, 3]  # Array of species id's

        particles = []
        for p in range(number_of_particles):
            particles.append(self.Particle(swarm_locs[p], swarm_velocities[p], 0, None, 0, None, random.choices(species, species_weights)[0]))

        self.swarm = self.Swarm(particles, None)

    # def GetY_Values (self):


        
        
