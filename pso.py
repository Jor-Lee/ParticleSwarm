import os
from scipy.stats import qmc
import numpy as np
import datetime as datetime
from sklearn.neighbors import KDTree
import pandas as pd 



def generate_random_unit_vector(dimension):
    # Generate a random vector
    random_vector = np.random.randn(dimension)
    # Normalize the vector to get a unit vector
    unit_vector = random_vector / np.linalg.norm(random_vector)
    return unit_vector

class PSO:

    def __init__(self, bounds, number_of_particles, hyper_parameter_sets, species_fractions, velocity_threshold, velocity_boost, maximise=True):

        bounds = np.array(bounds)
        hyper_parameter_sets = np.array(hyper_parameter_sets)

        # Checks! 

        # Check bounds dimensions are correct
        if np.shape(bounds)[1] != 2:
            raise ValueError(f'Bounds needs to be an n by 2 dimentional array but is {np.shape(bounds)[0]} by {np.shape(bounds)[1]}.')
    
        # Check the number of particles is a positive integer
        if not isinstance(number_of_particles, int) or number_of_particles < 1:
            raise ValueError(f'Number of particles needs to be an integer and larger than 0. Instead, number of particles was {number_of_particles}')

        # Check the hyper parameter sets match the number of specie fractions
        if len(hyper_parameter_sets) != len(species_fractions):
            raise ValueError(f'Number of hyper parameter sets provided is {len(hyper_parameter_sets)} but the number of species is {len(species_fractions)}.')
        
        # Check that the hyper parameter sets have the correct length (should be 4)
        if np.shape(hyper_parameter_sets)[1] != 4:
            raise ValueError(f'Length of arrays in hyper parameter sets should be 4 but is {np.shape(hyper_parameter_sets)[1]}. These are defined as inertial, local, global, and exploratory hyper-parameters.')

        self.bounds = bounds
        self.dimension = len(bounds)

        self.number_of_particles = number_of_particles
        self.initial_velocity = np.linalg.norm(bounds[:, 1] - bounds[:, 0])

        # Species stuff
        self.hyper_parameter_sets = hyper_parameter_sets
        self.species_fractions = species_fractions

        # Velocity boost stuff
        self.velocity_threshold = velocity_threshold
        self.velocity_boost = velocity_boost

        self.iteration = 0 

    
    class Particle:
        def __init__(particle, species, position, velocity, hyper_parameter_set, velocity_threshold, velocity_boost, ID):
            
            particle.species = species

            particle.position = position
            particle.velocity = velocity 
            particle.hyper_parameter_set = hyper_parameter_set

            particle.position_history = [position]
            particle.velocity_history = [velocity]
            particle.y_value_history = []

            particle.local_max = -np.inf
            particle.local_max_position = None

            particle.local_max_history = []
            particle.local_max_position_history = []

            particle.velocity_threshold = velocity_threshold
            particle.velocity_boost = velocity_boost

            particle.ID = ID

    class Swarm:

        def __init__(swarm, particles):

            swarm.particles = particles

            swarm.global_max = -np.inf
            swarm.global_max_position = None

            swarm.global_max_history = []
            swarm.global_max_position_history = []

            swarm.particle_positions = np.array([particles[i].position for i in range(len(particles))])
            swarm.particle_positions_history = swarm.particle_positions


    def InitialiseSwarm(self):
        sampler = qmc.Sobol(self.dimension) 
        sample = sampler.random(self.number_of_particles)
        initial_locations = qmc.scale(sample, np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1])

        initial_velocities = np.array([generate_random_unit_vector(self.dimension) * self.initial_velocity for _ in range(self.number_of_particles)])

        particles = []
        sampled_species = np.random.choice(len(self.hyper_parameter_sets), size=self.number_of_particles, p=self.species_fractions)
        sampled_hyper_parameter_sets = self.hyper_parameter_sets[sampled_species]
        for p in range(self.number_of_particles):
            particle = self.Particle(sampled_species[p], initial_locations[p], initial_velocities[p], sampled_hyper_parameter_sets[p], self.velocity_threshold, self.velocity_boost, p)
            particles.append(particle)

        self.swarm = self.Swarm(np.array(particles))



    def UpdateMaxima(self):
        global_max = self.swarm.global_max
        global_max_position = self.swarm.global_max_position

        for particle in self.swarm.particles:
            local_max = np.max(particle.y_value_history)
            particle.local_max = local_max

            max_index = np.argmax(particle.y_value_history)

            local_max_position = particle.position_history[max_index]
            particle.local_max_position = local_max_position

            particle.local_max_history.append(local_max)
            particle.local_max_position_history.append(local_max_position)

            if local_max > global_max:
                global_max = local_max
                global_max_position = local_max_position

        self.swarm.global_max = global_max
        self.swarm.global_max_position = global_max_position

        self.swarm.global_max_history.append(self.swarm.global_max)
        self.swarm.global_max_position_history.append(self.swarm.global_max_position)
        

    
    def UpdateVelocity(self, debugging=False):
        low_density_positions = np.empty([0, self.dimension])
        for i, particle in enumerate(self.swarm.particles):
            if debugging:
                print('Particle:', i)
                print('Initial Velocity:', particle.velocity)

            # Inertial term keeping particle on initial trajectory
            inertial_term = particle.hyper_parameter_set[0] * np.array(particle.velocity)
            if debugging:
                print('Inertial Term:', inertial_term)

            # Local term attracting particles to the location of thier individual maximum
            local_term = particle.hyper_parameter_set[1] * np.subtract(np.array(particle.local_max_position), particle.position)
            if debugging:
                print('Local Subtraction:', np.subtract(np.array(particle.local_max_position), particle.position))
                print('Hyper-parameter 1:', particle.hyper_parameter_set[1])
                print('Local Term:', local_term)

            # Gobal term attracting particles to the location of the swarms maximum
            global_term = particle.hyper_parameter_set[2] * np.subtract(np.array(self.swarm.global_max_position), particle.position)
            if debugging:
                print('Global Subtraction:',np.subtract(np.array(self.swarm.global_max_position), particle.position))
                print('Hyper-parameter 1:', particle.hyper_parameter_set[2])
                print('Global Term:', global_term)

            if particle.hyper_parameter_set[3] != 0:
                # Low density term attacting particle to regions of the domain that have been sparsely sampled
                low_density_position = self.IdentifyLowDensityRegion(low_density_positions)
                low_density_positions = np.vstack([low_density_positions, low_density_position])
                low_density_term = particle.hyper_parameter_set[3] * np.subtract(low_density_position, particle.position)

            else:
                low_density_term = 0

            new_velocity = inertial_term + local_term + global_term + low_density_term
            if debugging:
                print('New Velocity:', new_velocity)
                print('\n')


            # Check if velocity needs boosting
            velocity_magnitude_history = np.linalg.norm(particle.velocity_history, axis=1)
            if np.all(velocity_magnitude_history[-10:] < velocity_magnitude_history[0] * particle.velocity_threshold):
                velocity_unit_vector = new_velocity / np.linalg.norm(new_velocity)
                new_velocity = velocity_unit_vector * np.linalg.norm(particle.velocity_history, axis=1)[0] * particle.velocity_boost

                particle.velocity_threshold = particle.velocity_threshold * particle.velocity_boost
                particle.velocity_boost = particle.velocity_boost * particle.velocity_boost


            particle.velocity = new_velocity
            particle.velocity_history.append(new_velocity)


    def UpdatePosition(self, debugging=False):
        for particle in self.swarm.particles:
            if debugging:
                print('\nParticle:', particle.ID)
                print('Current Position:', particle.position)
                print('Velocity:', particle.velocity)

            
            new_position = particle.position + particle.velocity  # Initial position update based on velocity

            for i in range(len(new_position)):  # Iterate over each dimension
                if debugging:
                    print('\nDimenion', i)
                    print('Bounds', np.array(self.bounds)[i])
                    print('Old Position', particle.position[i])
                    print('New Position', new_position[i])

                # Reflect the position and reverse velocity until the particle is inside the bounds
                while new_position[i] < np.array(self.bounds)[i, 0] or new_position[i] > np.array(self.bounds)[i, 1]:
                    if new_position[i] < np.array(self.bounds)[i, 0]:
                        # Reflect below the lower bound
                        new_position[i] = np.array(self.bounds)[i, 0] + (np.array(self.bounds)[i, 0] - new_position[i])
                        particle.velocity[i] = -particle.velocity[i]  # Reverse velocity

                        if debugging:
                            print('Below lower bound. Refelcting velocity and adjusting position')
                            print('New Position:', new_position[i])
                            print('New Velocity:', particle.velocity[i])

                    
                    elif new_position[i] > np.array(self.bounds)[i, 1]:
                        # Reflect above the upper bound
                        new_position[i] = np.array(self.bounds)[i, 1] - (new_position[i] - np.array(self.bounds)[i, 1])
                        particle.velocity[i] = -particle.velocity[i]  # Reverse velocity

                        if debugging:
                            print('Above upper bound. Refelcting velocity and adjusting position')
                            print('New Position:', new_position[i])
                            print('New Velocity:', particle.velocity[i])

            # Update the particle's position
            particle.position = new_position

            # Record the updated position in the particle's position history
            particle.position_history.append(particle.position)

            if debugging:
                print('New Position:', new_position)


        # Update the swarm's positions
        self.swarm.particle_positions = np.array([particle.position for particle in self.swarm.particles])
        self.swarm.particle_positions_history = np.vstack([self.swarm.particle_positions_history, self.swarm.particle_positions])

    
    def UpdateOptimiser(self, y_values):
        for i, particle in enumerate(self.swarm.particles):
            particle.y_value_history.append(y_values[i])

        self.UpdateMaxima()
        self.UpdateVelocity()
        self.UpdatePosition()

        self.iteration += 1



    def GetNextX(self):
        NextX = []
        for particle in self.swarm.particles:
            NextX.append(particle.position)
        return np.array(NextX)        
    

    # def IdentifyLowDensityRegion(self, low_density_positions):
    #     # Takes all previous and current particle positions into account, including other low 
    #     # density positions that have already been identified this iteration.
    #     particle_position_historys = np.vstack([self.swarm.particle_positions_history, low_density_positions])

    #     lower_bounds = np.array([self.bounds[d][0] for d in range(self.dimension)])
    #     upper_bounds = np.array([self.bounds[d][1] for d in range(self.dimension)])

    #     # Generate random sample points (mesh points) within the search space.
    #     mesh_points = np.random.uniform(lower_bounds, upper_bounds, size=(100 * self.dimension, self.dimension))

    #     # all_position_history = 

    #     Neighbors = NearestNeighbors(n_neighbors=1).fit(particle_position_historys)
    #     distances, _ = Neighbors.kneighbors(mesh_points)

    #     # Calculate the average distance from each mesh point to its nearest neighbors.
    #     avg_distances = np.mean(distances, axis=1)

    #     # Identify the mesh points with the highest average distances (indicating low-density regions).
    #     lowest_density_index = np.argmax(avg_distances)

    #     # Extract the positions of the selected low-density regions.
    #     lowest_density_position = mesh_points[lowest_density_index]

    #     return lowest_density_position
    


    def IdentifyLowDensityRegion(self, low_density_positions, radius=0.1):
        # Combine current particle positions with previously identified low-density positions
        particle_positions = np.vstack([self.swarm.particle_positions_history, low_density_positions])

        # Build a KD-Tree for fast neighborhood queries
        kdtree = KDTree(particle_positions)
        
        lower_bounds = np.array([self.bounds[d][0] for d in range(self.dimension)])
        upper_bounds = np.array([self.bounds[d][1] for d in range(self.dimension)])

        # Generate random mesh points within the search space
        mesh_points = np.random.uniform(lower_bounds, upper_bounds, size=(100 * self.dimension, self.dimension))

        # Query the tree for points within the specified radius around each mesh point
        counts = kdtree.query_radius(mesh_points, r=radius, count_only=True)
        
        # Identify the mesh point with the fewest neighbors (indicating low-density regions)
        lowest_density_index = np.argmin(counts)

        # Return the mesh point in the lowest-density region
        lowest_density_position = mesh_points[lowest_density_index]

        return lowest_density_position


    def WriteOutputToCSV(self, csv_path):
        """
        Write the simulation results to a CSV file.

        This method saves the results of the current iteration, including the input 
        parameters (X) and output results (Y), to a CSV file. If the file does not exist, 
        it creates a new one with headers. Otherwise, it appends the new data to the existing file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file where results should be written.
        raw_X : ndarray, shape (n_samples, n_dimensions)
            The input parameters used in the current iteration.
        raw_y : ndarray, shape (n_samples, 1)
            The output results corresponding to the input parameters.
        """
        # Create arrays for iteration numbers and simulation numbers
        iteration_numbers = np.full(self.number_of_particles, self.iteration - 1)
        simulation_numbers = range(0, self.number_of_particles)

        # Create a dictionary to hold the data with column names
        data = {
            'Iteration': np.array(iteration_numbers),
            'Particle': np.array(simulation_numbers)
        }

        data['Result'] = []
        for particle in self.swarm.particles:
            data['Result'].append(particle.y_value_history[-1])

        # Add raw_X values with column names (X0, X1, X2, ...)
        for i in range(self.dimension):
            data[f'X{i}'] = []
            for particle in self.swarm.particles:
                data[f'X{i}'].append(particle.position_history[-2][i])
            data[f'X{i}'] = np.array(data[f'X{i}'])

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data)

        # Check if the CSV file exists, if not, create it and write the headers
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            # Append new data to the existing CSV file
            df.to_csv(csv_path, mode='a', header=False, index=False)


    def ParticleSummary(self):
        for p in self.swarm.particles:
            print('Particle:', p.ID)
            print('Species:', p.species)
            print('Hyper-parameter Set:', p.hyper_parameter_set)
            print('Position and velocity:', p.position, p.velocity)
            print('Position History:', p.position_history)
            print('Velcoity History:', p.velocity_history)
            print('Y Value History', p.y_value_history)
            print('Local Max and History', p.local_max, p.local_max_history)
            print('Local Max Position and History:', p.local_max_position, p.local_max_position_history)
            print('\n')
    
        print('Global Max and History:', self.swarm.global_max, self.swarm.global_max_history)
        print('Global Max Position and History:', self.swarm.global_max_position, self.swarm.global_max_position_history)