import os
import subprocess
import time
import scipy as sp
from scipy.stats import qmc
import random
import numpy as np
import sys

class pso:
    
    def __init__(self, number_of_particles, dimension):
        
        self.number_of_particles = number_of_particles # Number of particles
        self.dimension = dimension                     # Number of Dimensions
        self.bounds = None                             # Bounds: [[0,1], ..., [0,1]]
        
        self.current_iteration = 0                  # Current iteration of the swarm

        self.initial_velocity = 0.05                   # Maximum initial velocity of the swarm
        self.inertia_weight = None                     # Only used if use_adaptive_hyper_parameters = False
        self.global_weight = None                      # Only used if use_adaptive_hyper_parameters = False
        self.local_weight = None                       # Only used if use_adaptive_hyper_parameters = False

        self.use_adaptive_hyper_parameters = None      # Flag to use adaptive HyperParams = True
        self.GlobalWeightFunction = None               # Function of iteration number
        self.LocalWeightFunction = None                # Function of iteration number
        self.InertialWeightFunction = None             # Function of iteration number
        
        self.box_size = None                           # Only used if use_adaptive_boxes = False
        self.boxes = None
        self.sample_points = None
        self.sample_points_Y = None

        self.use_adaptive_boxes = None                 # Flag to turn on adaptive boxes
        self.BoxSizeFunction = None                    # Function of iteration number
        self.SampleSizeFunction = None                 # Function of iteration number

        self.GenerateY = None                          # Black-Box objective function to optimise
        self.swarm_locations = None                    # Holds swarm locations for the current iteration
        self.swarm_velocities = None                   # Holds swarm velocities for the current iteration

        self.global_maximum = None                        # Holds the global maximum (X, Y) of the swarm
        self.global_maximum_loc = None
        self.local_maxima = None                       # Holds an array of each particles Local Maximum
        self.local_maxima_locs = None

        self.run_sample_points = []
        self.run_sample_points_Y = []

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def InitialiseSwarmLocations(self):
        """
        Creates an array of 'evenly spaced' (or as close as it can get to evenly spaced)
        This should be used to generate the swarm initially

        Inputs:
        Bounds              - The region which you want to create the swarm
        No_ptcs             - The number of particles you want to generate
        dim                 - The dimension of the space to be optimised

        Output:
        swarm_locs          - locations of each of the 100 particles within the bounds
        """
        # Establish the sampler for given dimesnion
        sampler = qmc.Sobol(self.dimension)
        # Generate required number of points (number_of_particles)
        sample = sampler.random(self.number_of_particles)
        # Scale the sample to the given bounds
        swarm_locs = qmc.scale(sample, np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1])
        # swarm_locs = [[x0, ..., xN], [x0, ..., xN], ..., [x0, ..., xN]]
        self.swarm_locations = swarm_locs
        return
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def InitialiseSwarmVelocities(self):
        """
        Initialises the swarm velocities with a random direction and a random magnitude (up to intial_velocity)
        inputs: 
        inital_velocity   - the max velocity of a particle initially
        outputs:
        swarm_velocities  - contains the velocities of each particle and their direction
        """
        self.swarm_velocities = np.random.uniform(-self.initial_velocity, self.initial_velocity, size=(self.number_of_particles, self.dimension))
        return
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def CreateBoxes(self):
        if self.use_adaptive_boxes == True:
            """
            This creates the 'windows' surrounding each particle, within which we random sample for local maxima

            Inputs:
            X           - The location of the particle to create the window around
            box_size    - The length scale of the window you want to check

            Output:
            Box         - This contains the Box bounds for the window
            """
            boxes = []
            for p in range(self.number_of_particles):
                
                box = []
                
                try:
                    box_size = self.BoxSizeFunction(self.current_iteration)/2
                except Exception as e:
                    print('ERR: Create Boxes with use_adaptive_boxes == True requires BoxSizeFunction != None')
                    sys.exit(1)
                
                for i, x in enumerate(self.swarm_locations[p]):

                    if x - box_size < self.bounds[i][0]:
                        lower_bound = self.bounds[i][0]
                    else:
                        lower_bound = x - box_size
        
                    if x + box_size > self.bounds[i][1]:
                        upper_bound = self.bounds[i][1]
                    else:
                        upper_bound = x + box_size
                    box.append([lower_bound, upper_bound])
                boxes.append(box)
            self.boxes = boxes
        else:
            print('ERR: CreateBox requires use_adaptive_boxes == True')
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def UpdateIteration(self):
        """
        Simply updates the current iteration number manually 
        (in case you want to skip ahead to reduce/increase hyper parameters for example)
        """
        self.current_iteration += 1
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def PopulateBoxes(self):
        """
        Populates the Boxes that are created by the CreateBoxes function

        Inputs: 
        swarm_locations              
        BoxSizeFunction        
        bounds 
        boxes

        Outputs:
        points          
        """
        sample_points = []
        for p in range(self.number_of_particles):
            box = self.boxes[p]
            box_points = np.random.rand(self.SampleSizeFunction(self.current_iteration), self.dimension)
            for d in range(self.dimension):
                lower_bound = max(box[d][0], self.bounds[d][0]) 
                upper_bound = min(box[d][1], self.bounds[d][1])
                box_points[:, d] = box_points[:, d] * (upper_bound - lower_bound) + lower_bound
            sample_points.append(box_points)
        self.sample_points = sample_points 
        self.run_sample_points.append(sample_points)
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def Get_Y_Values(self):
        self.sample_points_Y = self.GenerateY(self.sample_points)
        # for p, point in enumerate(self.sample_points):
        #     for i in range(point):
        #         if self.bounds[i][0] < point[i] and point[i] < self.bounds[i][1]:
        #             continue
        #         else:
        #             self.sample_points_Y[p] = 0
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def UpdateGlobalMax(self):
        if self.current_iteration == 0 or self.global_maximum < np.max(self.sample_points_Y):
            index = np.argmax(self.sample_points_Y)
            best_particle = index // self.SampleSizeFunction(self.current_iteration)
            best_sample = index % self.SampleSizeFunction(self.current_iteration)
            self.global_maximum = self.sample_points_Y[best_particle][best_sample]
            self.global_maximum_loc = self.sample_points[best_particle][best_sample]
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ # 

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ # 
    def UpdateLocalMaxima(self):
        best_sample_y = []
        best_sample_locs = []
        for p in range(self.number_of_particles):
            if self.current_iteration == 0:
                best_sample_index = np.argmax(self.sample_points_Y[p])
                best_sample_locs.append(self.sample_points[p][best_sample_index])
                best_sample_y.append(self.sample_points_Y[p][best_sample_index])
            elif np.max(self.sample_points_Y[p]) > self.local_maxima[p]:
                best_sample_index = np.argmax(self.sample_points_Y[p])
                best_sample_locs.append(self.sample_points[p][best_sample_index])
                best_sample_y.append(self.sample_points_Y[p][best_sample_index])
            else:
                best_sample_locs.append(self.local_maxima_locs[p])
                best_sample_y.append(self.local_maxima[p])
        self.local_maxima = best_sample_y
        self.local_maxima_locs = best_sample_locs
        print(f'Size of Local_Maxima_Locs: {len(self.local_maxima_locs)} : {len(self.local_maxima_locs[1])}')
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ # 
  
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def UpdateVelocity(self):
        new_swarm_velocities = []
        for p in range(self.number_of_particles):
            inertial_velocity = self.InertialWeightFunction(self.current_iteration) * self.swarm_velocities[p]
            global_velocity = self.GlobalWeightFunction(self.current_iteration) * (np.subtract(self.global_maximum_loc, self.swarm_locations[p]))
            local_velocity =  self.LocalWeightFunction(self.current_iteration) * (np.subtract(self.local_maxima_locs[p], self.swarm_locations[p]))
            new_swarm_velocities.append(np.add(np.add(inertial_velocity, global_velocity), local_velocity))
        self.swarm_velocities = new_swarm_velocities
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    def UpdateSwarmLocations(self):
        new_swarm_locations = []
        for p in range(self.number_of_particles):
            new_swarm_locations.append(np.add(self.swarm_locations[p], self.swarm_velocities[p]))
        self.swarm_locations = new_swarm_locations
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def IterateSwarm(self):
        self.UpdateIteration()
        self.UpdateVelocity()
        self.UpdateSwarmLocations()
        self.CreateBoxes()
        self.PopulateBoxes()
        self.Get_Y_Values()
        self.UpdateGlobalMax()
        self.UpdateLocalMaxima()

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    

    