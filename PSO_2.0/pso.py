import os
import subprocess
import time
import scipy as sp
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors

class Pso:

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def  __init__  (self, dimension, number_of_particles, bounds, use_adaptive_hyper_parameters, use_adaptive_boxes, use_species, species_weights, GenerateY, maximise):

        self.dimension = dimension                                                  # Dimension of the system to be optimised
        self.number_of_particles = number_of_particles                              # Number of swarm particles (not including points tested in boxes)
        self.bounds = bounds                                                        # Bounds of the optimisation e.g. [[0,1], [2, 10]] for a 2D optimisation
        self.use_adaptive_hyper_parameters = use_adaptive_hyper_parameters          # Flag to turn on adaptive hyperparameter functions
        self.use_adaptive_boxes = use_adaptive_boxes                                # Flag to turn on adaptive box sizes
        self.GenerateY = GenerateY                                                  # Black-box function to be optimised
        self.maximise = maximise                                                    # If maximise == True => maximise, else minimise
        self.species_weights = species_weights                                      # [0.3, 0.2, 0.4, 0.1] %ages of each species

        self.initial_velocity = 20*(bounds[0][0]-bounds[0][1])/self.number_of_particles # effectively move the boxes on the first iteration
        self.current_iteration = 0
        self.swarm = None

        self.GlobalWeightFunction = None               # Function of iteration number
        self.LocalWeightFunction = None                # Function of iteration number
        self.InertialWeightFunction = None             # Function of iteration number

        self.BoxWidthFunction = None                    # Function of iteration number
        self.SampleSizeFunction = None                 # Function of iteration number

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    class Particle:
        def __init__ (self, position, velocity, local_max, local_max_loc, global_max, global_max_loc, species):
            self.position = position                                                # position of the particle
            self.velocity = velocity                                                # velocity of the particle
            self.local_max = local_max                                              # the best value of GenerateY found by a given particle
            self.local_max_loc = local_max_loc                                      # the location that gave this best value
            self.global_max = global_max                                            # best value of GenerateY from the particles that it communicates with 
            self.global_max_loc = global_max_loc                                    # location of this best value
            self.species = species                                                  # species of particle - affects hyperparameters and social behaviour
            self.box = None
            self.sample_points = None
            self.sample_points_results = None
            self.adventure_lead = None

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    class Box:
        def __init__ (self, bounds):
            self.bounds = bounds

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    class Species:
        # hyper_parameter_set = [Inertia, Local, Global, Adventure, Intuitive]
        class Reckless:
            def __init__(reckless):
                reckless.id = 0
                reckless.hyper_parameter_set = [0.8, 0.4, 0.7, 0, 0]
                
        class Cautious:
            def __init__(cautious):
                cautious.id = 1
                cautious.hyper_parameter_set = [0.8, 0.7, 0.1, 0, 0]

        class Adventurous:
            def __init__(adventurous):
                adventurous.id = 2
                adventurous.hyper_parameter_set = [0.8, 0.1, 0.1, 0.8, 0]

        class Intuitive:
            def __init__(intuitive):
                intuitive.id = 3
                intuitive.hyper_parameter_set = [0.8, 0.3, 0, 0, 0.7]
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    class Swarm:
        def __init__ (swarm, particles, communication_matrix, pso):
            swarm.particles = particles                                             # holds an array of type: particle                                            
            swarm.communication_matrix = communication_matrix                      # number_of_particles X number of particles matrix holding the communication values between particles
            swarm.global_max = None
            swarm.global_max_loc = None
            swarm.local_maxima = np.empty(pso.number_of_particles) 
            swarm.local_maxima_locs = np.empty((pso.number_of_particles, pso.dimension))

            swarm.position_history = None
            swarm.y_value_history = None

            swarm.adventure_leads = np.empty((int(pso.number_of_particles*pso.species_weights[2]), pso.dimension))

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def InitialiseSwarm(self):

        # Create the array of evenly sampledswarm locations 
        sampler = qmc.Sobol(self.dimension)                                                         # Establish the sampler for given dimesnion
        sample = sampler.random(self.number_of_particles)                                           # Generate required number of points (number_of_particles)
        swarm_locs = qmc.scale(sample, np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1])    # Scale the sample to the given bounds

        # Create the set of random swarm velocities
        swarm_velocities = np.random.uniform(-self.initial_velocity, self.initial_velocity, size=(self.number_of_particles, self.dimension))

        species_ids = [0, 1, 2, 3]  # Array of species id's

        particles = []
        for p in range(self.number_of_particles):
            particles.append(self.Particle(swarm_locs[p], swarm_velocities[p], None, None, None, None, random.choices(species_ids, self.species_weights)[0]))

        self.swarm = self.Swarm(np.array(particles), None, self) 

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
        
    def GenerateBox(self, particle):
        half_box_width = self.BoxWidthFunction(self.current_iteration)/2
        box_bounds = []
        for i, x in enumerate(particle.position):
            if x - half_box_width < self.bounds[i][0]:
                lower_bound = self.bounds[i][0]
            else:
                lower_bound = x - half_box_width

            if x + half_box_width > self.bounds[i][1]:
                upper_bound = self.bounds[i][1]
            else:
                upper_bound = x + half_box_width
            box_bounds.append([lower_bound, upper_bound])

        particle.box = self.Box(box_bounds)   
        
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def GenerateAllBoxes(self):
        for particle in self.swarm.particles:
            self.GenerateBox(particle)
       
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def PopulateBox(self, particle):
        if self.use_adaptive_boxes == True:
            box_bounds = particle.box.bounds
            box_points = np.random.rand(self.SampleSizeFunction(self.current_iteration), self.dimension)
            for d in range(self.dimension):
                lower_bound = max(box_bounds[d][0], self.bounds[d][0]) 
                upper_bound = min(box_bounds[d][1], self.bounds[d][1])
                box_points[:, d] = box_points[:, d] * (upper_bound - lower_bound) + lower_bound
            particle.sample_points = box_points
        else:
            particle.sample_points = particle.position

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def PopulateAllBoxes(self):
        for particle in self.swarm.particles:
            self.PopulateBox(particle)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def GetY(self):
        # Create a list of all the points to be sampled in this iteration
        points_to_function = []
        for particle in self.swarm.particles:
            for sample_point in particle.sample_points:
                points_to_function.append(sample_point)

        # Generate the Y values from the black box function
        y_values = self.GenerateY(points_to_function)
        y_values = [y_values[i:i+10] for i in range(0, len(y_values), 10)]
 
        # Redistribute the Y values to the relevant particle
        for p, particle in enumerate(self.swarm.particles):
            particle.sample_points_results = y_values[p]
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def UpdateLocalMaxima(self):
        # Updates each particles Local Maximum
        if self.maximise == True:
            for particle in self.swarm.particles:
                if particle.local_max == None or np.max(particle.sample_points_results) > particle.local_max:
                    particle.local_max = np.max(particle.sample_points_results)
                    particle.local_max_loc = particle.sample_points[np.argmax(particle.sample_points_results)]
        else:
            for particle in self.swarm.particles:
                if particle.local_max == None or np.min(particle.sample_points_results) < particle.local_max:
                    particle.local_max = np.min(particle.sample_points_results)
                    particle.local_max_loc = particle.sample_points[np.argmin(particle.sample_points_results)]

        for p, particle in enumerate(self.swarm.particles):
            self.swarm.local_maxima[p] = particle.local_max
            self.swarm.local_maxima_locs[p, :] = particle.local_max_loc


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def RankParticles(self):
        sorted_indices = np.argsort(self.swarm.local_maxima) # Sorted low -> high
        self.swarm.particles = self.swarm.particles[sorted_indices]
        self.swarm.local_maxima = self.swarm.local_maxima[sorted_indices]
        self.swarm.local_maxima_locs = self.swarm.local_maxima_locs[sorted_indices]

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def UpdateGlobalMaxima(self):
        if self.maximise == True:
            for particle in self.swarm.particles:
                if self.swarm.global_max == None or particle.local_max > self.swarm.global_max:
                    self.swarm.global_max = particle.local_max
                    self.swarm.global_max_loc = particle.local_max_loc
        else:
            for particle in self.swarm.particles:
                if self.swarm.global_max == None or particle.local_max < self.swarm.global_max:
                    self.swarm.global_max = particle.local_max
                    self.swarm.global_max_loc = particle.local_max_loc

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def WriteSamplePointsHistory(self):
        if self.current_iteration == 0:
            self.swarm.position_history = np.empty((self.number_of_particles*len(self.swarm.particles[0].sample_points), self.dimension))
            self.swarm.y_value_history = np.empty((self.number_of_particles*len(self.swarm.particles[0].sample_points), 1))
            for p, particle in enumerate(self.swarm.particles):
                for s, sample_point in enumerate(particle.sample_points):
                    self.swarm.position_history[p*len(particle.sample_points) + s] = sample_point
                    self.swarm.y_value_history[p*len(particle.sample_points) + s] = particle.sample_points_results[s]
        else:
            sampled_point_position_history = self.swarm.position_history
            sampled_y_value_history = self.swarm.y_value_history
            all_sampled_points_positions = np.empty((len(sampled_point_position_history) + self.number_of_particles * len(self.swarm.particle[0].sample_points), self.dimension))
            all_sampled_y_values = np.empty((len(sampled_point_position_history) + self.number_of_particles * len(self.swarm.particle[0].sample_points), 1))
            all_sampled_points_positions[:len(sampled_point_position_history)] = sampled_point_position_history 
            all_sampled_y_values[:len(sampled_point_position_history)] = sampled_y_value_history
            for p, particle in enumerate(self.swarm.particles):
                for s, sample_point in enumerate(particle.sample_points):
                    all_sampled_points_positions[len(sampled_point_position_history) + p*len(particle.sample_points) + s, :] = sample_point
                    all_sampled_y_values[len(sampled_point_position_history) + p*len(particle.sample_points) + s, :] = particle.sample_points_result[s]
            self.swarm.position_history = all_sampled_points_positions
            self.swarm.y_value_history = all_sampled_y_values
        
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def FindLowDensityRegions(self):

        Neighbors = NearestNeighbors(n_neighbors=int(self.number_of_particles / 10)).fit(self.swarm.position_history)
        distances, _ = Neighbors.kneighbors(self.swarm.position_history)

        avg_distances = np.mean(distances, axis=1)

        number_of_low_density_points = int(self.number_of_particles * self.species_weights[2])

        low_density_indices = np.argsort(avg_distances)[-number_of_low_density_points:]

        low_density_positions = self.swarm.position_history[low_density_indices]

        self.swarm.adventure_leads = low_density_positions

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def AssignAdventureLeads(self):

        explorer_particle_locations = []
        for particle in self.swarm.particles:
            if particle.species == 2:
                explorer_particle_locations.append(particle.position)

        explorer_particle_locations = np.array(explorer_particle_locations)

        adventure_leads = self.swarm.adventure_leads
        
        distance_matrix = np.linalg.norm(adventure_leads[:, np.newaxis] - explorer_particle_locations, axis=2)

        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        matched_leads = [adventure_leads[j] for j in col_indices]

        index = 0
        for particle in self.swarm.particles:
            if particle.species == 2:
                particle.adventure_lead = matched_leads[index]
                index += 1

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # def GenerateCommunicationMatrix(self):

    #     communication_matrix = np.empty((self.number_of_particles, self.number_of_particles))
        
    #     for particle in self.swarm.particles:
    #         particle_communication_row = np.empty((self.number_of_particles, 1))
    #         if particle.species == 0:
    #             particle_communication_row = [reckless_global_attraction for _ in range(self.number_of_particles)]
    #         elif particle.species == 1:

    #         elif particle.species == 2:

    #         elif particle.species == 3:

    #         else:
    #             # Log an error
    #             return
