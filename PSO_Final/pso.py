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
import logging
import datetime as datetime

class Pso:

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def  __init__  (self, dimension, number_of_particles, bounds, GenerateY, maximise,
                    use_adaptive_hyper_parameters=False, GlobalWeightFunction=None, LocalWeightFunction=None, InertialWeightFunction=None,
                    use_adaptive_boxes=False, use_species=False, species_weights=None, BoxWidthFunction=None, SampleSizeFunction=None,
                    use_explosive_global_parameter=False, log_file=''):

        self.dimension = dimension                              # Dimensionality of the optimization problem (number of variables to optimize).
        self.number_of_particles = number_of_particles          # Number of particles in the swarm (excluding those used for populating the boxes in box sampling).
        self.bounds = bounds                                    # Search space boundaries for the optimization, e.g., [[0,1], [2, 10]] for a 2D problem.

        #### only one of use_adaptive_hyper_parameters and use_species can be True at once!
        self.use_adaptive_hyper_parameters = use_adaptive_hyper_parameters  # Enables adaptive hyperparameters if True.
        self.use_adaptive_boxes = use_adaptive_boxes            # Enables adaptive box sizes for sampling regions if True.
        self.use_species = use_species                          # Enables the use of different species with varying behaviors if True.
        self.species_weights = species_weights                  # Percentage distribution of particles across different species, e.g., [0.3, 0.2, 0.4, 0.1].

        self.GenerateY = GenerateY                              # The objective function (black-box function) to be optimized.
        self.maximise = maximise                                # Determines the optimization goal: True for maximization, False for minimization.

        self.initial_velocity = 20 * (bounds[0][0] - bounds[0][1]) / self.number_of_particles  # Initial velocity for particles to initiate movement in the first iteration.
        self.current_iteration = 0                              # Tracks the number of iterations completed in the optimization process.
        self.swarm = None                                       # Placeholder for the Swarm instance, which contains all particle information.

        # Static hyperparameters used when neither adaptive hyperparameters nor species-specific parameters are in use.
        self.StaticHyperParameters = [0.8, 0.6, 0.5]            # [Inetia, Global, Local]
        
        # Functions that determine the evolution of weights over iterations (typically dependent on the iteration number).
        self.GlobalWeightFunction = GlobalWeightFunction        # Weight function for global best influence (typically decreasing over iteration).
        self.LocalWeightFunction = LocalWeightFunction          # Weight function for local best influence (typically decreasing over iteration).
        self.InertialWeightFunction = InertialWeightFunction    # Weight function for inertia (typically increasing over iteration).

        # Functions governing the size of the sampling box and the number of samples per box, evolving with iterations.
        self.BoxWidthFunction = BoxWidthFunction                # Function that defines how the width of the sampling box changes over iterations.
        self.SampleSizeFunction = SampleSizeFunction            # Function that defines how the number of samples in the box changes over iterations.

        self.mesh_coarseness = 100                              # Controls the granularity of the search grid; higher values provide finer resolution in finding low-density regions.
        self.mesh_bound_reduction_factor = 1                    # Factor to reduce the search space bounds (between 0 and 1) over time.
        self.length_scale = (bounds[0][1] - bounds[0][0]) / self.mesh_coarseness  # Scale length based on the search space and mesh coarseness.

        self.use_explosive_global_parameter = use_explosive_global_parameter  # Determines if an explosive global parameter should be used to escape local optima.

        # Logging setup
        self.log_file = log_file                                # File path for logging optimization progress.
        self.log_level = logging.INFO                           # Logging level (e.g., INFO, DEBUG).
        self.logger = None                                      # Placeholder for a logger instance.

        self.convergence_threshold = 10     # Number of iterations without improvement in the global maximum before considering the swarm converged.
        self.swarm_converged = False        # Flag indicating if the swarm has converged (True) or is still exploring (False).

        self.number_of_evolutions_cautious_to_explore = 0
        self.number_of_evolutions_explore_to_cautious = 0

        self.vthresh = 1/(np.e**3)
        self.vboost = 2*(np.e**2)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    class Particle:
        def __init__ (self, position, velocity, local_max, local_max_loc, global_max, global_max_loc, species):

            self.position = position                # Current position of the particle in the solution space.
            self.velocity = velocity                # Current velocity of the particle, influencing its movement in the solution space.
            self.local_max = local_max              # Best value of the objective function (GenerateY) found by this particle.
            self.local_max_loc = local_max_loc      # Position in the solution space where the particle found its best value.
            self.local_max_sample_index = 0
            self.global_max = global_max            # Best value of the objective function found by the particle's communication group.
            self.global_max_loc = global_max_loc    # Location where the global maximum was found by the group.
            self.species = species                  # Species of the particle, affecting its behavior and hyperparameters.

            # Attributes related to the adaptive box method (if used)
            self.box = None                         # The box or region the particle is currently exploring.
            self.sample_points = None               # Points within the box that the particle is sampling.
            self.sample_points_results = None       # Results of the objective function at the sample points.

            # Attributes related to the particle's exploration behavior
            self.adventure_lead = None              # Position the particle is leading an exploration or adventure towards.
            self.stuck = False                      # Flag indicating if the particle is stuck, i.e., not making progress.
            self.interesting = False                # Flag indicating if the particle has found an interesting or promising region.

            # Historical data for analysis and plotting
            self.local_max_history = []             # History of the best values found by the particle.
            self.local_max_history_loc = []         # History of the locations of those best values.
            self.position_history = []              # History of the particle's positions over time.
            self.sample_points_history = []         # History of the sample points chosen by the particle.
            self.sample_points_results_history = [] # History of the objective function values at the sample points.

            self.velocity_threshold = 0             # Threshold to determine if the particle's velocity is effectively zero.
            self.iterations_since_evolution = 0
            self.local_max_history_since_evolution = []

            self.counter = 0

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # Box is an attribute given to a particle and sets up a boundary in which the particle can subsample
    class Box:
        def __init__ (self, bounds):
            self.bounds = bounds        # bounds of the box (including shift by particle position)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    class Species:
        
        # hyper_parameter_set = [Inertia, Local, Global, Adventure, Intuitive]

        class Reckless:
            """
            The Reckless species operates like a classical swarm particle however
            with a much higher global hyper-parameter and a lower local pull.
            - This particle will converge on a potential solution quickly
            """
            def __init__(reckless):
                reckless.id = 0
                reckless.hyper_parameter_set = [0.8, 0.2, 0.7, 0, 0]
                
        class Cautious:
            """
            The Cautious species operates like a classical swarm particle however
            with a much lower global hyper-parameter and a higher global pull.

            - This particles aim is to get stuck in its own local optimum irrespective
                of how the global is performing.

            - The inclusion of this species slows down convergence and allows us to thoroughly
                test the space
            """
            def __init__(cautious):
                cautious.id = 1
                cautious.hyper_parameter_set = [0.8, 0.4, 0.0, 0, 0]

        class Adventurous:
            """
            The Adventerous species is less a particle but more a scheme to sample the space 
            in regions in which we have previously ignored. We identify low density sample points
            and move the adventurous particles to these locations.
            
            - The advantage of including adventurous particles is the ability to sample untested regions
                continuously and allows for the possibility of correcting a premature convergence

            - Adventerous particles which identifty a 'promising region' are evolved into cautious particles
                which then locally optimise within this promising region in the hopes to update the gmax
            """
            def __init__(adventurous):
                adventurous.id = 2
                adventurous.hyper_parameter_set = [0, 0, 0, 1, 0]

        class Predictive:
            """
            The Predictive species acts as a particle but has an additional hyperparameter pulling it
            towards the weighted average position in the sample space. This offers a route to bypass the slow 
            convergence on global particles in high dimensional space.

            - This class is only valuable in the first few iterations, outside of this the particle
                is pulled off course and is more useful when evolved into a global or an adventurous particle 
            """
            def __init__(predictive):
                predictive.id = 3
                predictive.hyper_parameter_set = [0.8, 0.3, 0, 0, 0.7]

        class Classic:
            def __init__(classic):
                classic.id = 4
                classic.hyper_parameter_set = [0.8, 0.5, 0.6, 0, 0]
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    class Swarm:

        def __init__(swarm, particles, communication_matrix, pso):
            # Initialize the swarm with an array of particles.
            swarm.particles = particles                         # List of Particle objects that make up the swarm.

            # Communication matrix defining how particles influence each other.
            swarm.communication_matrix = communication_matrix   # Square matrix (number_of_particles x number_of_particles) that represents the strength of communication between particles.
            #

            # Attributes to track the best global position and value found by the swarm.
            swarm.global_max = None                             # Best objective function value found by any particle in the swarm.
            swarm.global_max_loc = None                         # Location in the solution space where the global_max was found.
            swarm.global_max_particle_index = None              # Contains the particle number of the gmax
            swarm.global_max_sample_index = None                # Contains the sample point number of the gmax

            # Arrays to store the local maxima and their corresponding locations for each particle.
            swarm.local_maxima = np.empty(pso.number_of_particles)                          # Array to hold the best values found by each particle.
            swarm.local_maxima_locs = np.empty((pso.number_of_particles, pso.dimension))    # Array to hold the locations corresponding to each particle's best value.

            # History of global maxima over iterations for tracking convergence and progress.
            swarm.global_max_history = []                       # List to store the global_max value at each iteration.

            # Historical data for analysis and visualization.
            swarm.position_history = None                       # Array to track the positions of all particles across iterations.
            swarm.y_value_history = None                        # Array to track the objective function values for all particles across iterations.

            # Array to store the positions of the adventure leads (particles exploring new regions) from the adventurous species.
            swarm.adventure_leads = np.empty((int(pso.number_of_particles * pso.species_weights[2]), pso.dimension))  

            # Predicted location of the global maximum and its associated confidence.
            swarm.predicted_max_loc = None                      # Predicted location of the global maximum based on current swarm data.
            swarm.predicted_weight = 0                          # Confidence level or weight associated with the predicted_max_loc.

            swarm.vthresh = pso.vthresh
            swarm.vboost = pso.vboost

            swarm.imporved = True


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def SetupLogging(self):
        """
        Setup for the Logging file.

        Args:
            pso.log_file: Path to the log file

        Outputs:
            None: Adds the logger to pso.logger
        """

        log_file_path = self.log_file
        
        # Create and configure the logger
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)
        
        # Check if logger already has handlers to avoid duplicate logs
        if not logger.hasHandlers():
            # Create file handler to log messages to a file
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(self.log_level)
            
            # Create a console handler (optional, if you want logs printed to console as well)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create a formatter for the logs and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        self.logger = logger
        self.logger.info('Log file successfully setup.')

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def InitialiseSwarm(self):
        """
        InitialiseSwarm sets up the Particle Swarm Optimization (PSO) system. It populates the 
        swarm with particles, assigns random initial velocities, and establishes velocity 
        thresholds for each particle. Additionally, it logs the initial PSO parameters and 
        records the initial particle positions in the swarm's history.
        """

        # Initialize the logging system and set up the .log file for tracking the swarm's progress.
        self.SetupLogging()

        # Log the PSO parameters, providing a record of the configuration used for the swarm.
        self.logger.info("Initializing the SWARM with the following parameters:")
        self.logger.info(f'dimension = {self.dimension}')                       # Log the dimensionality of the problem.
        self.logger.info(f'bounds = {self.bounds}')                             # Log the bounds of the search space.
        self.logger.info(f'number of particles = {self.number_of_particles}')   # Log the number of particles in the swarm.
        
        # Log whether the objective function will be maximized or minimized.
        if self.maximise:
            self.logger.info(f'swarm will MAXIMIZE the function')
        else:
            self.logger.info(f'swarm will MINIMIZE the function')

        # Log additional PSO configuration settings based on user-defined flags.
        if self.use_adaptive_hyper_parameters:
            self.logger.info('swarm is using adaptive hyperparameters')         # Log if adaptive hyperparameters are used.
        if self.use_species:
            self.logger.info('swarm is using multiple species')                 # Log if multiple species with different behaviors are used.
        if self.use_adaptive_boxes:
            self.logger.info('swarm is using adaptive boxes')                   # Log if adaptive box sizes are used for sampling.
        if self.use_explosive_global_parameter:
            self.logger.info('swarm is using the exploding global optimum parameter')  # Log if the explosive global parameter is used.

        # Initialize particle positions using a Sobol sequence for quasi-random sampling.
        sampler = qmc.Sobol(self.dimension)                                     # Create a Sobol sampler for the given dimensionality.
        sample = sampler.random(self.number_of_particles)                       # Generate a sample of points equal to the number of particles.
        swarm_locs = qmc.scale(sample, np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1])  # Scale the sample to fit within the defined bounds.
        self.logger.info('Particle positions initialized')                      # Log that particle positions have been initialized.


        def generate_random_unit_vector(dimension):
            # Generate a random vector
            random_vector = np.random.randn(dimension)
            # Normalize the vector to get a unit vector
            unit_vector = random_vector / np.linalg.norm(random_vector)
            return unit_vector

        # Generate random velocities with the specified magnitude
        swarm_velocities = np.array([generate_random_unit_vector(self.dimension) * self.initial_velocity for _ in range(self.number_of_particles)])
        self.logger.info('Particle velocities initialized')                     # Log that particle velocities have been initialized.

        # Generate random initial velocities for each particle within the defined range.
        # swarm_velocities = np.random.uniform(-self.initial_velocity, self.initial_velocity, size=(self.number_of_particles, self.dimension))
        # self.logger.info('Particle velocities initialized')                     # Log that particle velocities have been initialized.

        species_ids = [0, 1, 2, 3, 4]                  # Define the possible species IDs, corresponding to different particle behaviors.

        # Create and initialize particles with positions, velocities, and species (if applicable).
        particles = []
        if self.use_species:
            for p in range(self.number_of_particles):
                species_id = random.choices(species_ids, self.species_weights)[0]  # Assign species based on specified weights.
                particles.append(self.Particle(swarm_locs[p], swarm_velocities[p], None, None, None, None, species_id))
        else:
            for p in range(self.number_of_particles):
                particles.append(self.Particle(swarm_locs[p], swarm_velocities[p], None, None, None, None, None))  # No species assignment if not using species.

        # Set up initial history and velocity threshold for each particle.
        for particle in particles:
            particle.position_history.append(particle.position)                 # Record the initial position in the particle's history.
            # Set the velocity threshold to determine when to boost the particles velocity
            particle.velocity_threshold = np.linalg.norm(particle.velocity) * self.vthresh

        # Initialize the swarm with the created particles and no initial communication matrix.
        self.swarm = self.Swarm(np.array(particles), None, self) 
        self.logger.info('Swarm object initialized')                            # Log that the swarm object has been successfully initialized.


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
        
    def GenerateBox(self, particle):
        """
        Generates a bounding box around a given particle's current position. The size of the box
        is determined by the BoxWidthFunction, which varies with the current iteration. This box
        defines the region where the particle will search for new potential solutions.

        Args:
            particle: The particle for which the bounding box is generated. The box will be
                        centered around this particle's current position.

        Returns:
            None: This method modifies the particle's `box` attribute of the particle directly.

        Details:
            - The width of the box is computed as the value returned by the BoxWidthFunction
            for the current iteration.
            - The bounding box is adjusted to stay within the global bounds defined for the problem.
            - If the particle's position plus or minus half the box width exceeds the global bounds,
            the box is clipped to fit within the global bounds.
        """

        # Calculate half of the current box width based on the iteration number.
        half_box_width = self.BoxWidthFunction(self.current_iteration) / 2

        # Initialize an empty list to store the bounds of the box for each dimension.
        box_bounds = []

        # Iterate over each dimension to compute the bounds of the box.
        for i, x in enumerate(particle.position):
            # Determine the lower bound of the box for the current dimension.
            if x - half_box_width < self.bounds[i][0]:
                lower_bound = self.bounds[i][0]
            else:
                lower_bound = x - half_box_width

            # Determine the upper bound of the box for the current dimension.
            if x + half_box_width > self.bounds[i][1]:
                upper_bound = self.bounds[i][1]
            else:
                upper_bound = x + half_box_width
            
            # Append the computed bounds for the current dimension to the list.
            box_bounds.append([lower_bound, upper_bound])

        # Create a Box object using the computed bounds and assign it to the particle.
        particle.box = self.Box(box_bounds)
        
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def GenerateAllBoxes(self):
        """
        Generates bounding boxes for all particles in the swarm. If adaptive boxes are enabled,
        each particle's box is created based on its current position and the BoxWidthFunction.

        Returns:
            None: This method modifies each particle's `box` attribute directly.
        """

        # Generate boxes for all particles if adaptive boxes are in use.
        if self.use_adaptive_boxes:
            for particle in self.swarm.particles:
                self.GenerateBox(particle)
        
        # Log the completion of the box generation process.
        self.logger.info('Generated all boxes for particles in the swarm.')
       
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def PopulateBox(self, particle):
        """
        Populates the bounding box of a given particle with sample points. The method generates
        sample points within the particle's box if adaptive boxes are enabled. Otherwise, it
        assigns the particle's current position as its sole sample point.

        Args:
            particle: The particle for which the sample points are generated.

        Returns:
            None: This method directly modifies the `sample_points` attribute of the given particle.

        Details:
            - When adaptive boxes are used, the number of sample points is determined by
            the `SampleSizeFunction` based on the current iteration.
            - The sample points are randomly distributed within the box's dimensions, which
            are constrained by both the box bounds and the global bounds.
        """

        # Populate the particle's box with sample points if adaptive boxes are enabled.
        if self.use_adaptive_boxes:
            # Retrieve the bounds of the particle's box.
            box_bounds = particle.box.bounds
            
            # Generate random sample points within the unit hypercube.
            box_points = np.random.rand(self.SampleSizeFunction(self.current_iteration), self.dimension)
            
            # Scale and shift the points to fit within the box's bounds.
            for d in range(self.dimension):
                lower_bound = max(box_bounds[d][0], self.bounds[d][0]) 
                upper_bound = min(box_bounds[d][1], self.bounds[d][1])
                box_points[:, d] = box_points[:, d] * (upper_bound - lower_bound) + lower_bound
            
            # Assign the generated points to the particle's sample_points attribute.
            particle.sample_points = box_points
        else:
            # If adaptive boxes are not used, only the particle's current position is considered.
            sample_points = [particle.position]
            particle.sample_points = sample_points


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def PopulateAllBoxes(self):
        """
        Populates the bounding boxes of all particles in the swarm with sample points. For each
        particle, this method calls `PopulateBox` to generate and assign sample points within its
        box. It then records these sample points in the particle's history for tracking purposes.

        Returns:
            None: This method directly modifies the `sample_points` and `sample_points_history` 
                attributes of each particle in the swarm.

        Details:
            - The `sample_points_history` attribute maintains a history of all sample points
            generated for each particle, which can be useful for analyzing the particle's search
            behavior over iterations and for visualisation.
        """
        
        # Populate the bounding boxes of all particles with sample points.
        for particle in self.swarm.particles:
            self.PopulateBox(particle)  # Generate and assign sample points for the particle.

            # Record each generated sample point in the particle's history.
            for sample_point in particle.sample_points:
                particle.sample_points_history.append(sample_point)

        # Log the completion of the box population process.
        self.logger.info('Populated all boxes with sample points for particles in the swarm.')


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def GetY(self):
        """
        Computes the values of the black-box function for all sample points of the particles in the swarm.
        The method evaluates the function at each sample point, processes the results, and assigns them to
        the respective particles. It also maintains a history of results for each particle.

        Returns:
            None: This method directly updates the `sample_points_results` and `sample_points_results_history` 
                attributes of each particle in the swarm.

        Details:
            - `points_to_function` is a list of sample points to be evaluated by the black-box function.
            - `y_values` contains the results of the black-box function applied to `points_to_function`.
            - `sample_points_results` is updated with the corresponding function values for each particle.
            - `sample_points_results_history` maintains a history of all computed results for each particle.

        Notes:
            - The reshaping of `y_values` ensures that the results are correctly distributed among particles when
            adaptive boxes are used.
        """

        # Create a list of all sample points to be evaluated in this iteration.
        points_to_function = []
        for particle in self.swarm.particles:
            for sample_point in particle.sample_points:
                if isinstance(sample_point, float):
                    points_to_function.append(particle.sample_points)  # Catches the case where sample_points only contains a single point
                else:
                    points_to_function.append(sample_point)  # Add the sample point as is.

        # Convert the list of sample points to a NumPy array for batch processing.
        points_to_function = np.array(points_to_function)

        # Evaluate the black-box function at all sample points.
        y_values = self.GenerateY(points_to_function)

        # Reshape y_values if adaptive boxes are used, otherwise wrap each value in a list.
        if self.use_adaptive_boxes:
            y_values = [y_values[i:i + self.SampleSizeFunction(self.current_iteration)] 
                        for i in range(0, len(y_values), self.SampleSizeFunction(self.current_iteration))]
        else:
            y_values = [[value] for value in y_values]
 
        # Log the computed function values.
        self.logger.info(f'y_values : {y_values}')

        # Assign the computed function values to each particle and update their history.
        for p, particle in enumerate(self.swarm.particles):
            particle.sample_points_results = y_values[p]  # Assign results to the particle.
            for sample_point_result in particle.sample_points_results:
                particle.sample_points_results_history.append(sample_point_result)  # Update history.

    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def UpdateLocalMaxima(self):
        """
        Updates the local maxima for each particle in the swarm based on their current sample points' results.
        The method compares the newly evaluated results with the particle's previous local maximum to decide
        if an update is needed. It also records the updated local maxima and their locations in history.

        Returns:
            None: This method directly modifies the `local_max`, `local_max_loc`, `local_max_history`, 
                `local_max_history_loc`, `local_maxima`, and `local_maxima_locs` attributes.

        Details:
            - `local_max` represents the highest (or lowest) value of the function that a single particle has found.
            - `local_max_loc` stores the corresponding location of this local maximum.
            - `local_max_history` and `local_max_history_loc` keep track of the particle's local maxima and locations over iterations.
            - `local_maxima` and `local_maxima_locs` are updated arrays that reflect the current local maxima and their locations for all particles.

        Notes:
            - The method handles both maximization and minimization scenarios based on the `maximise` flag.
            - The `local_max` and `local_max_loc` attributes are updated only if the new results are better (for maximization) or worse (for minimization).
        """

        # Update the local maxima for each particle based on current sample points' results.
        if self.maximise:
            for particle in self.swarm.particles:
                if particle.local_max is None or np.max(particle.sample_points_results) > particle.local_max:
                    particle.local_max = np.max(particle.sample_points_results)
                    # Determine the location of the new local maximum
                    if isinstance(particle.sample_points[0], float):
                        particle.local_max_loc = particle.sample_points
                        particle.local_max_sample_index = 0
                    else:
                        particle.local_max_loc = particle.sample_points[np.argmax(particle.sample_points_results)]
                        particle.local_max_sample_index = np.argmax(particle.sample_points_results)
                # Record the updated local maximum and location in history
                particle.local_max_history.append(particle.local_max)
                particle.local_max_history_since_evolution.append(particle.local_max)
                particle.local_max_history_loc.append(particle.local_max_loc)
        else:
            for particle in self.swarm.particles:
                if particle.local_max is None or np.min(particle.sample_points_results) < particle.local_max:
                    particle.local_max = np.min(particle.sample_points_results)
                    # Determine the location of the new local minimum
                    if isinstance(particle.sample_points[0], float):
                        particle.local_max_loc = particle.sample_points
                        particle.local_max_loc = particle.sample_points
                    else:
                        particle.local_max_loc = particle.sample_points[np.argmin(particle.sample_points_results)]
                        particle.local_max_sample_index = np.argmax(particle.sample_points_results)
                # Record the updated local maximum and location in history
                particle.local_max_history.append(particle.local_max)
                particle.local_max_history_since_evolution.append(particle.local_max)
                particle.local_max_history_loc.append(particle.local_max_loc)

        # Update the swarm's local maxima and their locations
        for p, particle in enumerate(self.swarm.particles):
            self.swarm.local_maxima[p] = particle.local_max
            self.swarm.local_maxima_locs[p, :] = particle.local_max_loc

        self.logger.info('Local Maxima updated')


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def RankParticles(self):
        """
        Ranks the particles in the swarm based on their local maxima. 

        This method sorts the particles and their corresponding local maxima and locations in either ascending or descending order, depending on whether the goal is to maximize or minimize the function value.

        Notes:
            - **Maximization**: If `self.maximise` is `True`, `np.argsort` is used on the negative of `local_maxima` to sort in descending order, placing higher local maxima values first.
            - **Minimization**: If `self.maximise` is `False`, `np.argsort` is used directly to sort in ascending order, placing lower local maxima values first.
            - Sorting is essential for organizing particles based on performance.
        """

        if self.maximise:
            # Sort indices for maximization (highest values first)
            sorted_indices = np.argsort(-self.swarm.local_maxima)  # Negative for descending order
        else:
            # Sort indices for minimization (lowest values first)
            sorted_indices = np.argsort(self.swarm.local_maxima)  # Default ascending order

        # Reorder particles, local_maxima, and local_maxima_locs based on sorted_indices.
        self.swarm.particles = self.swarm.particles[sorted_indices]
        self.swarm.local_maxima = self.swarm.local_maxima[sorted_indices]
        self.swarm.local_maxima_locs = self.swarm.local_maxima_locs[sorted_indices]

        self.logger.info('Particles Ranked')


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def UpdateGlobalMaxima(self):
        """
        Updates the global maximum values based on the local maxima of all particles in the swarm.

        This method compares the local maxima of each particle to the current global maximum and updates the global maximum and its location accordingly. The comparison is done based on whether the optimization goal is to maximize or minimize the function.

        Notes:
            - **Initialization**: If `self.swarm.global_max` is `None`, it indicates that no global maximum has been set yet; thus, any particle's local maximum will set the initial global maximum.
            - **Efficiency**: This method ensures that the global maximum is always updated to reflect the best value found by any particle in the swarm, which is critical for guiding the optimization process.
        """
        
        if self.maximise:
            # Maximization: Update global maximum if a particle's local maximum is greater.
            for p, particle in enumerate(self.swarm.particles):
                if self.swarm.global_max is None or particle.local_max > self.swarm.global_max:
                    self.swarm.improved = True
                    self.swarm.global_max = particle.local_max
                    self.swarm.global_max_loc = particle.local_max_loc
                    self.swarm.global_max_particle_index = p
                    self.swarm.global_max_sample_index = particle.local_max_sample_index
        else:
            # Minimization: Update global maximum if a particle's local maximum is smaller.
            for p, particle in enumerate(self.swarm.particles):
                if self.swarm.global_max is None or particle.local_max < self.swarm.global_max:
                    self.swarm.improved = True
                    self.swarm.global_max = particle.local_max
                    self.swarm.global_max_loc = particle.local_max_loc
                    self.swarm.global_max_particle_index = p
                    self.swarm.global_max_sample_index = particle.local_max_sample_index

        
        # Append the current global maximum to the history list.
        self.swarm.global_max_history.append(self.swarm.global_max)

        self.logger.info('Global Maxima Updated')


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def WriteSamplePointsHistory(self):
        """
        Updates the history of sampled points and their corresponding function values.

        This method manages the storage of sample points and their function values over iterations. 
        It maintains a history of all sample points evaluated and their results to facilitate analysis and debugging of the optimization process.

        Details:
            - **`self.swarm.position_history`**: 2D array where each row corresponds to a sampled point's position.
            - **`self.swarm.y_value_history`**: 2D array where each row corresponds to the (black-box) function value of a sampled point.
            - **Handling Float Sample Points**: In cases where `sample_point` is a float (indicating a single-dimensional point), it is stored as-is. This is handled by checking if `sample_point` is an instance of `float`.
            - **Handling Multi-Dimensional Sample Points**: For multi-dimensional sample points, both the point's position and its result are stored in separate rows of the history arrays.
        """
        
        if self.current_iteration == 0:
            # Initialize history arrays for the first iteration.
            self.swarm.position_history = np.empty((self.number_of_particles * len(self.swarm.particles[0].sample_points), self.dimension))
            self.swarm.y_value_history = np.empty((self.number_of_particles * len(self.swarm.particles[0].sample_points), 1))
            
            # Populate the history arrays with the initial sample points and their results.
            for p, particle in enumerate(self.swarm.particles):
                for s, sample_point in enumerate(particle.sample_points):
                    if isinstance(sample_point, float):
                        self.swarm.position_history[p] = particle.sample_points
                        self.swarm.y_value_history[p] = particle.sample_points_results[0]
                    else:
                        self.swarm.position_history[p * len(particle.sample_points) + s] = sample_point
                        self.swarm.y_value_history[p * len(particle.sample_points) + s] = particle.sample_points_results[s]
        else:
            # Expand history arrays to include new sample points from the current iteration.
            sampled_point_position_history = self.swarm.position_history
            sampled_y_value_history = self.swarm.y_value_history
            
            # Create new arrays with increased size to accommodate additional sample points.
            all_sampled_points_positions = np.empty((len(sampled_point_position_history) + self.number_of_particles * len(self.swarm.particles[0].sample_points), self.dimension))
            all_sampled_y_values = np.empty((len(sampled_point_position_history) + self.number_of_particles * len(self.swarm.particles[0].sample_points), 1))
            
            # Copy existing sample points and their values to the new arrays.
            all_sampled_points_positions[:len(sampled_point_position_history)] = sampled_point_position_history
            all_sampled_y_values[:len(sampled_y_value_history)] = sampled_y_value_history
            
            # Append new sample points and their values to the history arrays.
            for p, particle in enumerate(self.swarm.particles):
                for s, sample_point in enumerate(particle.sample_points):
                    if isinstance(sample_point, float):
                        all_sampled_points_positions[len(sampled_point_position_history) + p, :] = particle.sample_points
                        all_sampled_y_values[len(sampled_point_position_history) + p, :] = particle.sample_points_results[0]
                    else:
                        all_sampled_points_positions[len(sampled_point_position_history) + p * len(particle.sample_points) + s, :] = sample_point
                        all_sampled_y_values[len(sampled_point_position_history) + p * len(particle.sample_points) + s, :] = particle.sample_points_results[s]
            
            # Update the history arrays with the expanded data.
            self.swarm.position_history = all_sampled_points_positions
            self.swarm.y_value_history = all_sampled_y_values

        self.logger.info('Sample Point History Written')

        
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def FindLowDensityRegions(self):
        """
        Identifies and selects low-density regions in the search space for exploration.

        This method locates regions in the search space that are less explored by the swarm particles, 
        particularly targeting low-density areas where fewer particles are present. It assigns these 
        low-density regions to a subset of particles designated as 'explorers' to enhance the exploration 
        capabilities of the swarm.

        Process:
            1. **Count Explorers**:
                - Counts the number of particles classified as 'explorers' (species ID = 2).

            2. **Generate Mesh Points**:
                - Generates random sample points across the entire search space, bounded by the problem's 
                defined limits.
                - These points serve as potential candidates for identifying low-density regions.

            3. **Calculate Distance Metrics**:
                - Uses the `NearestNeighbors` class from `sklearn` to compute the distances between each 
                mesh point and the previously sampled particle positions.
                - Determines the average distance from each mesh point to its nearest neighbors among 
                the swarm's position history.

            4. **Select Low-Density Regions**:
                - Sorts the mesh points based on their average distance metrics.
                - Selects the top regions with the highest average distances (i.e., least dense areas) for 
                further exploration.
                - Assigns these selected positions as 'adventure leads' to the explorers in the swarm.

        Details:
            - **`number_of_explorers`**: Number of particles designated to explore low-density regions.
            - **`mesh_points`**: Randomly generated sample points used to probe the search space.
            - **`Neighbors`**: Instance of `NearestNeighbors` used to compute proximity metrics.
            - **`avg_distances`**: Average distance from each mesh point to its nearest neighbors, reflecting density.
            - **`low_density_indices`**: Indices of mesh points with the highest average distances, indicating low density.
            - **`low_density_positions`**: Selected low-density positions assigned to explorer particles.
        """
        
        # Count the number of explorer particles (species ID = 2).
        number_of_explorers = 0
        for particle in self.swarm.particles:
            if particle.species == 2:
                number_of_explorers += 1

        # Define the bounds of the search space.
        lower_bounds = np.array([self.bounds[d][0] for d in range(self.dimension)])
        upper_bounds = np.array([self.bounds[d][1] for d in range(self.dimension)])
        
        # Generate random sample points (mesh points) within the search space.
        mesh_points = np.random.uniform(lower_bounds, upper_bounds, size=(self.number_of_particles * self.dimension, self.dimension))

        # Print position history for debugging purposes.
        print(self.swarm.position_history)

        # Create a NearestNeighbors instance to compute distances between mesh points and particle positions.
        Neighbors = NearestNeighbors(n_neighbors=int(self.number_of_particles / 10)).fit(self.swarm.position_history)
        distances, _ = Neighbors.kneighbors(mesh_points)

        # Calculate the average distance from each mesh point to its nearest neighbors.
        avg_distances = np.mean(distances, axis=1)

        # Identify the mesh points with the highest average distances (indicating low-density regions).
        low_density_indices = np.argsort(avg_distances)[-number_of_explorers:]

        # Extract the positions of the selected low-density regions.
        low_density_positions = mesh_points[low_density_indices]

        # Assign the low-density positions as adventure leads for the explorer particles.
        self.swarm.adventure_leads = low_density_positions

        self.logger.info('Found adventure leads')

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def AssignAdventureLeads(self):
        """
        Assigns adventure leads (low-density regions) to explorer particles in the swarm.

        This method matches explorer particles with predefined low-density regions (adventure leads) to guide 
        their search towards less-explored areas in the search space. It uses the Hungarian algorithm to find the 
        optimal assignment of adventure leads to explorer particles based on the minimum distance.

        Process:
            1. **Identify Explorer Particles**:
                - Collects the positions of particles that are classified as explorers (species ID = 2).

            2. **Check for Available Explorers**:
                - Ensures that there are explorer particles available to be assigned adventure leads.

            3. **Calculate Distance Matrix**:
                - Computes the distance matrix between the adventure leads and the explorer particle positions.
                - Uses Euclidean distance to measure how far each explorer particle is from each adventure lead.

            4. **Perform Optimal Assignment**:
                - Applies the Hungarian algorithm (via `linear_sum_assignment`) to find the optimal assignment of adventure leads to explorer particles.
                - Minimizes the total distance between assigned leads and particles.

            5. **Assign Adventure Leads**:
                - Updates each explorer particle with its assigned adventure lead based on the optimal assignment.
                - Ensures that each explorer is directed towards a unique low-density region.

        Notes:
            - **Euclidean Distance**: Used to measure distances between explorer particles and adventure leads.
            - **Hungarian Algorithm**: Efficiently solves the assignment problem to minimize the total distance.
            - **Edge Cases**: Handles scenarios where there may be no explorer particles or no adventure leads by checking lists before processing.

        """
        
        # Collect the positions of all explorer particles (species ID = 2).
        explorer_particle_locations = []
        for particle in self.swarm.particles:
            if particle.species == 2:
                explorer_particle_locations.append(particle.position)

        # Check if there are explorer particles to assign adventure leads.
        if explorer_particle_locations != []:

            # Convert the list of explorer particle positions to a NumPy array.
            explorer_particle_locations = np.array(explorer_particle_locations)

            # Get the adventure leads that need to be assigned.
            adventure_leads = self.swarm.adventure_leads
            
            # Compute the distance matrix between adventure leads and explorer particle positions.
            distance_matrix = np.linalg.norm(adventure_leads[:, np.newaxis] - explorer_particle_locations, axis=2)

            # Use the Hungarian algorithm to find the optimal assignment of adventure leads to explorer particles.
            row_indices, col_indices = linear_sum_assignment(distance_matrix)

            # Get the adventure leads assigned to each explorer based on the optimal assignment.
            matched_leads = [adventure_leads[j] for j in col_indices]

            # Assign the matched adventure leads to the explorer particles.
            index = 0
            for particle in self.swarm.particles:
                if particle.species == 2:
                    particle.adventure_lead = matched_leads[index]
                    index += 1

        self.logger.info('Adventure Leads Assigned')

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def PredictOptimum(self):
        """
        Predicts the location of the optimum solution based on the weighted average of sample points and results.

        This method estimates the location of the optimum solution by considering the results from all particles in the swarm.
        The prediction is made differently depending on whether the objective is to maximize or minimize the function.

        **Maximization Case:**
        - Initializes `weighted_loc` with the previous predicted location, weighted by `predicted_weight`, if available.
        - Updates `weighted_loc` by adding the weighted sample points' contributions from each particle.
        - The weight of each sample point is its function result, and the `total_weight` accumulates these results.

        **Minimization Case:**
        - Similar to the maximization case, initializes `weighted_loc` with the previous prediction.
        - Updates `weighted_loc` by adding the inverse of the function results (as weights) multiplied by sample points.
        - The `total_weight` accumulates the inverse of these results.

        Notes:
            - **Maximization:** Focuses on higher function results, treating them as higher weights.
            - **Minimization:** Uses the inverse of function results as weights to favor lower values.
            - **Edge Cases:** Handles scenarios where `predicted_max_loc` is `None` by starting with zeros.
        """
        
        if self.maximise == True:
            # If there's a previous prediction, initialize weighted location using that prediction
            if self.swarm.predicted_max_loc is not None:
                weighted_loc = self.swarm.predicted_weight * self.swarm.predicted_max_loc
                total_weight = self.swarm.predicted_weight
            else:
                weighted_loc = np.zeros(self.dimension)
                total_weight = 0
            
            # Update weighted location based on current sample points and results
            for particle in self.swarm.particles:
                for sample_point, result in zip(particle.sample_points, particle.sample_points_results):
                    weighted_loc = np.add(weighted_loc, result * np.array(sample_point))
                    total_weight += result

        else:
            # If there's a previous prediction, initialize weighted location using that prediction
            if self.swarm.predicted_max_loc is not None:
                weighted_loc = self.swarm.predicted_weight * self.swarm.predicted_max_loc
                total_weight = self.swarm.predicted_weight
            else:
                weighted_loc = np.zeros(self.dimension)
                total_weight = 0
            
            # Update weighted location based on current sample points and results
            for particle in self.swarm.particles:
                for sample_point, result in zip(particle.sample_points, particle.sample_points_results):
                    weighted_loc = np.add(weighted_loc, (1 / result) * np.array(sample_point))
                    total_weight += (1 / result)

        # Update predicted weight and location
        self.swarm.predicted_weight = total_weight
        self.swarm.predicted_max_loc = weighted_loc / total_weight


        self.logger.info('Optimum Predicted')

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def UpdateVelocity(self):
        """
        Updates the velocity of each particle in the swarm based on its species or the chosen hyperparameter strategy.

        This method adjusts the velocity of each particle according to its species-specific parameters or adaptive hyperparameters. The approach varies depending on whether species-based or adaptive hyperparameter-based velocity updates are used.

        **Species-Based Velocity Update:**
        - **Reckless (species 0):**
            - Velocity is computed using a combination of the particle's current velocity, its local best position, and the global best position.
            - If the norm of the computed velocity is below the particle's velocity threshold, it is scaled up to maintain exploration.
        - **Cautious (species 1):**
            - Similar to Reckless, but with different hyperparameters reflecting a more cautious approach.
            - Adjusts velocity if the norm is below the threshold.
        - **Adventurous (species 2):**
            - Includes an additional term for the adventure lead position, promoting exploration towards these leads.
        - **Predictive (species 3):**
            - Includes an additional term for the predicted optimum location to guide particles based on predictions.
        - **V8 (species 4):**
            - Similar to Reckless and Cautious but with a distinct set of hyperparameters for this species.

        **Adaptive Hyperparameters Update:**
        - If adaptive hyperparameters are used, the velocity is updated based on the iteration-dependent global, local, and inertial weight functions.
        - These weights adjust how much the particle's velocity is influenced by its current velocity, local best position, and global best position.

        **Static Hyperparameters Update:**
        - If not using adaptive hyperparameters or species, a fixed set of static hyperparameters is used.
        - The velocity update combines the particle's current velocity with the differences between its local best and global best positions.
        """
        
        species = self.Species()
        reckless = species.Reckless()
        cautious = species.Cautious()
        adventurous = species.Adventurous()
        predictive = species.Predictive()
        classic = species.Classic()

        if self.use_species == True:
            for particle in self.swarm.particles:
                if particle.species == 0:
                    # Reckless species: Update velocity with scaling if below threshold
                    velocity = np.add(np.add(reckless.hyper_parameter_set[0]*np.array(particle.velocity),
                                            reckless.hyper_parameter_set[1]*np.subtract(np.array(particle.local_max_loc), np.array(particle.position))),
                                    reckless.hyper_parameter_set[2]*np.subtract(np.array(self.swarm.global_max_loc), np.array(particle.position)))
                    if np.linalg.norm(velocity) >= particle.velocity_threshold:
                        particle.velocity = velocity
                        particle.counter = 0
                    else:
                        particle.counter += 1
                        if particle.counter >= 5:
                            particle.velocity = (velocity / np.linalg.norm(velocity)) * particle.velocity_threshold * self.swarm.vboost
                            particle.velocity_threshold = particle.velocity_threshold * self.swarm.vthresh
                            particle.counter = 0

                elif particle.species == 1:
                    # Cautious species: Update velocity with scaling if below threshold
                    velocity = np.add(np.add(cautious.hyper_parameter_set[0]*np.array(particle.velocity),
                                            cautious.hyper_parameter_set[1]*np.subtract(np.array(particle.local_max_loc), np.array(particle.position))),
                                    cautious.hyper_parameter_set[2]*np.subtract(np.array(self.swarm.global_max_loc), np.array(particle.position)))
                    if np.linalg.norm(velocity) >= particle.velocity_threshold:
                        particle.velocity = velocity
                        particle.counter = 0
                    else:
                        particle.counter += 1
                        if particle.counter >= 5:
                            particle.velocity = (velocity / np.linalg.norm(velocity)) * particle.velocity_threshold * self.swarm.vboost
                            particle.velocity_threshold = particle.velocity_threshold * self.swarm.vthresh
                            particle.counter = 0

                elif particle.species == 2:
                    # Adventurous species: Includes an additional term for adventure lead
                    particle.velocity = np.add(np.add(np.add(adventurous.hyper_parameter_set[0]*np.array(particle.velocity),
                                                            adventurous.hyper_parameter_set[1]*np.subtract(np.array(particle.local_max_loc), np.array(particle.position))),
                                                    adventurous.hyper_parameter_set[2]*np.subtract(np.array(self.swarm.global_max_loc), np.array(particle.position))),
                                            adventurous.hyper_parameter_set[3]*np.subtract(np.array(particle.adventure_lead), np.array(particle.position)))
                    
                elif particle.species == 3:
                    # Predictive species: Includes an additional term for predicted optimum location
                    velocity = np.add(np.add(np.add(predictive.hyper_parameter_set[0]*np.array(particle.velocity),
                                                            predictive.hyper_parameter_set[1]*np.subtract(np.array(particle.local_max_loc), np.array(particle.position))),
                                                    predictive.hyper_parameter_set[2]*np.subtract(np.array(self.swarm.global_max_loc), np.array(particle.position))),
                                            predictive.hyper_parameter_set[4]*np.subtract(np.array(self.swarm.predicted_max_loc), np.array(particle.position)))
                    if np.linalg.norm(velocity) >= particle.velocity_threshold:
                        particle.velocity = velocity
                        particle.counter = 0
                    else:
                        particle.counter += 1
                        if particle.counter >= 5:
                            particle.velocity = (velocity / np.linalg.norm(velocity)) * particle.velocity_threshold * self.swarm.vboost
                            particle.velocity_threshold = particle.velocity_threshold * self.swarm.vthresh
                            particle.counter = 0
                    
                elif particle.species == 4:
                    # V8 species: Uses V8-specific hyperparameters
                    velocity = np.add(np.add(classic.hyper_parameter_set[0]*np.array(particle.velocity),
                                                    classic.hyper_parameter_set[1]*np.subtract(np.array(particle.local_max_loc), np.array(particle.position))),
                                            classic.hyper_parameter_set[2]*np.subtract(np.array(self.swarm.global_max_loc), np.array(particle.position)))
                    if np.linalg.norm(velocity) >= particle.velocity_threshold:
                        particle.velocity = velocity
                        particle.counter = 0
                    else:
                        particle.counter += 1
                        if particle.counter >= 5:
                            particle.velocity = (velocity / np.linalg.norm(velocity)) * particle.velocity_threshold * self.swarm.vboost
                            particle.velocity_threshold = particle.velocity_threshold * self.swarm.vthresh
                            particle.counter = 0
            
        elif self.use_adaptive_hyper_parameters == True:
            # Adaptive hyperparameters: Use functions based on the current iteration
            global_weight = self.GlobalWeightFunction(self.current_iteration)
            local_weight = self.LocalWeightFunction(self.current_iteration)
            inertial_weight = self.InertialWeightFunction(self.current_iteration)

            for particle in self.swarm.particles:
                particle.velocity = np.add(np.add(inertial_weight*np.array(particle.velocity),
                                                local_weight*np.subtract(np.array(particle.local_max_loc), np.array(particle.position))),
                                        global_weight*np.subtract(np.array(self.swarm.global_max_loc), np.array(particle.position)))
            
        else:
            # Static hyperparameters: Fixed parameters for velocity update
            for particle in self.swarm.particles:
                particle.velocity = np.add(np.add(self.static_hyper_parameters[0]*np.array(particle.velocity),
                                                self.static_hyper_parameters[1]*np.subtract(np.array(particle.local_max_loc), np.array(particle.position))),
                                        self.static_hyper_parameters[2]*np.subtract(np.array(self.swarm.global_max_loc), np.array(particle.position)))

        self.logger.info('Velocity Updated')

        for particle in self.swarm.particles:
            particle.iterations_since_evolution += 1

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def UpdatePosition(self):
        """
        Updates the position of each particle in the swarm based on its current velocity.

        This method iterates over all particles in the swarm and updates their positions by adding the current velocity to the existing position. It also keeps a record of each particle's position history.
        """

        for particle in self.swarm.particles:
            # Update the particle's position by adding its velocity to the current position
            particle.position = np.add(particle.position, particle.velocity)
            
            # Record the updated position in the particle's position history
            particle.position_history.append(particle.position)
        
        self.logger.info('Position Updated')


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def EvolveStuckParticle(self, particle):
        """
        Handles particles that are identified as 'stuck' and evolves them to potentially escape local minima or stagnation.

        This method checks if a given particle is flagged as 'stuck' (i.e., it is not making progress or changing significantly). If the particle is indeed stuck, it performs the following actions to help it escape from its current situation:

        **Process:**
            1. **Check Stuck Status:** The method first verifies if the particle is flagged as 'stuck'.
            2. **Update Species:** If the particle is stuck, it is reclassified to a different species (species 1 in this case). This species may have different behavioral parameters that could help the particle explore new areas.
            3. **Reset Stuck Flag:** The 'stuck' flag is reset to `False` to indicate that the particle is no longer considered stuck.
            4. **Convert Position:** Ensures that the particles position is converted to a numpy array, potentially for compatibility with further operations.

        This method helps in maintaining the efficiency of the Particle Swarm Optimization (PSO) algorithm by addressing particles that are not progressing, thereby improving the overall search capability of the swarm.
        """

        if particle.stuck == True:
 
            # Change the particle's species to 1, which may have different behavioral parameters
            particle.species = 2
            self.number_of_evolutions_cautious_to_explore += 1
            
            # Reset the 'stuck' flag to indicate that the particle is no longer considered stuck
            particle.stuck = False
            
            # Ensure the particle's position is in numpy array format
            particle.position = np.array(particle.position)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def EvolveSuccessfulExpedition(self, particle):
        """
        Handles particles identified as 'interesting' due to their successful exploration or finding valuable regions.

        This method is intended for particles that have been flagged as 'interesting'—typically because they have found promising areas or made significant progress. The method adapts these particles to potentially enhance their exploration capability and effectiveness.

        **Process:**
            1. **Check Interesting Status:** The method first checks if the particle is flagged as 'interesting'.
            2. **Update Species:** If the particle is interesting, it is assigned to become Cautious (Locally optimising).
            3. **Reset Position:** Reshapes the particle's position to a 1D array for consistency.
            4. **Update Velocity:** Randomly initializes the particles velocity within specified bounds to encourage exploration and prevent stagnation.
            5. **Reset Interesting Flag:** Resets the 'interesting' flag to `False` to mark the particle as no longer in the 'interesting' state.

        This method is useful for enhancing the performance of the Particle Swarm Optimization (PSO) algorithm by adapting successful particles to continue exploring or exploiting valuable regions more effectively.
        """

        if particle.interesting == True:
            
            # Change the particle's species to 2, which may have different behavioral parameters
            particle.species = 1
            self.number_of_evolutions_explore_to_cautious += 1
            
            particle.iterations_since_evolution = 0
            particle.local_max_history_since_evolution = []

            # Reshape the particle's position to a 1D array for consistency
            particle.position = particle.position.reshape(-1)
            
            # Randomly initialize the particle's velocity within specified bounds
            particle.velocity = np.array(np.random.uniform(-self.initial_velocity, self.initial_velocity, size=(1, self.dimension))).reshape(-1)
            particle.velocity_threshold = np.linalg.norm(particle.velocity) * self.swarm.vthresh
            
        # Reset the 'interesting' flag to indicate that the particle is no longer in the 'interesting' state
        particle.interesting = False


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def DetermineSuccessfulExpedition(self, particle):
        """
        Evaluates whether a particle has had a successful expedition based on its local maxima and updates its 'interesting' status accordingly.

        This method checks if a particle's performance has improved compared to its previous best performance, indicating that the particle has made significant progress. If so, it flags the particle as 'interesting' for further consideration.

        **Process:**
            1. **Iteration Check:** Ensures that the particle's performance is only evaluated after a certain number of iterations (greater than 10 in this case).
            2. **Evaluate Performance:**
                - For maximization tasks: Compares the current local maximum of the particle with its previous best local maximum. If the current local maximum is better, the particle is flagged as 'interesting'.
                - For minimization tasks: Compares the current local maximum with its previous best. If the current local maximum is worse (i.e., lower), the particle is flagged as 'interesting'.
            3. **Exception Handling:** Catches and ignores exceptions that might arise from the comparison operations.

        **Purpose:**
        - The method helps in identifying and tracking particles that have shown significant progress in their exploration, allowing them to be flagged for potential further exploration or adaptation.

        This approach ensures that particles making notable progress are recognized and adapted for more focused exploitation strategies.
        """

        if self.current_iteration > 10:
            try:
                if particle.species == 2:
                    if self.maximise:
                        # Check if the particle has improved its local maximum from the previous best
                        if particle.local_max > particle.local_max_history[-1] or random.random() < 0.5:
                            particle.interesting = True
                    else:
                        # Check if the particle has worsened its local maximum (better for minimization)
                        if particle.local_max < particle.local_max_history[-1] or random.random() < 0.5:
                            particle.interesting = True
            except:
                # Ignore exceptions that occur during the comparison
                pass


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def DetermineParticleStuck(self, particle):
        """
        Determines if a particle is 'stuck' based on its recent performance history.

        This method evaluates whether a particle's local maximum has stagnated, indicating that it may have become stuck in a suboptimal region of the search space. It does this by comparing the current local maximum to its performance history over the last few iterations.

        **Process:**
            1. **Performance Comparison:** Compares the particles current local maximum to its local maximum from three iterations ago.
            2. **Threshold Check:**
                - For maximization tasks: The particle is considered 'stuck' if its current local maximum is not significantly better (within 5% of) than the local maximum from three iterations ago.
                - For minimization tasks: The particle is considered 'stuck' if its current local maximum is not significantly worse (within 5% of) than the local maximum from three iterations ago.
            3. **Exception Handling:** Catches and ignores exceptions that might occur during the comparison.

        **Purpose:**
        - The method helps identify particles that may be stuck in local optima or unproductive regions of the search space. Recognizing such particles allows for targeted interventions or adaptations to help them escape from these suboptimal regions and continue exploring effectively.

        This approach ensures that particles showing signs of stagnation are flagged for potential re-evaluation or modification in their search strategy.
        """

        if particle.iterations_since_evolution > 10:
            try:
                if particle.species == 1 or particle.species == 3:
                    if self.maximise:
                        # Check if the particle's local maximum has not significantly improved
                        if particle.local_max <= particle.local_max_history_since_evolution[-10]:
                            particle.stuck = True
                    else:
                        # Check if the particle's local maximum has not significantly worsened (better for minimization)
                        if particle.local_max >= particle.local_max_history_since_evolution[-10]:
                            particle.stuck = True
            except:
                # Ignore exceptions that occur during the comparison
                pass


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def CheckGlobalTooDense(self):

        if self.current_iteration > 5:
            gmax_loc = np.array(self.swarm.global_max_loc)

            reckless_particles = []
            for particle in self.swarm.particles:
                if particle.species == 0:
                    reckless_particles.append(particle)

            if reckless_particles != []:
                total_distance = 0
                for reckless_particle in reckless_particles:
                    total_distance += np.linalg.norm(np.subtract(gmax_loc, np.array(reckless_particle.position)))

                average_distance = total_distance / len(reckless_particles)
                print(average_distance)

                if average_distance < np.abs(0.01*(self.bounds[0][1] - self.bounds[0][0])):
                    print('Vroom!')
                    for reckless_particle in reckless_particles:
                        reckless_particle.species = 4
                        reckless_particle.velocity = np.array(np.random.uniform(np.abs(0.08*(self.bounds[0][1] - self.bounds[0][0])), np.abs(0.08*(self.bounds[0][1] - self.bounds[0][0])), size=(1, self.dimension))).reshape(-1)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def CheckGlobalTooSparse(self):

        if self.current_iteration > 5:
            gmax_loc = self.swarm.global_max_loc

            v8_particles = []
            for particle in self.swarm.particles:
                if particle.species == 4:
                    v8_particles.append(particle)
            
            if v8_particles != []:

                total_distance = 0
                for v8_particle in v8_particles:
                    total_distance += np.linalg.norm(gmax_loc - v8_particle.position)

                average_distance = total_distance / len(v8_particles)

                if average_distance >= np.abs(0.2*(self.bounds[0][1] - self.bounds[0][0])):
                    print('Reign it in!')
                    for v8_particle in v8_particles:
                        v8_particle.species = 0

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
 
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def UpdateSpecies(self):
        """
        Updates the species and state of each particle in the swarm based on their performance and current status.

        **Process:**
            1. **Particle Status Updates:**
                - **Determine if Stuck:** Uses `DetermineParticleStuck` to check if a particle is stuck in a local optimum.
                - **Evaluate Expedition Success:** Uses `DetermineSuccessfulExpedition` to identify if a particle's exploration has been successful and interesting.
                - **Evolve Stuck Particles:** Updates particles identified as stuck to potentially escape their current local optima using `EvolveStuckParticle`.
                - **Evolve Successful Expeditions:** Updates particles identified as having had a successful expedition to potentially enhance their exploration capabilities using `EvolveSuccessfulExpedition`.

        **Purpose:**
        - **Particle Adaptation:** Ensures that particles that are either stuck or have had successful explorations are adapted accordingly to improve the swarm's overall performance.

        This method ensures that each particle is appropriately updated based on its current state, and that additional global exploration checks are performed if needed, to maintain the efficiency and effectiveness of the swarm's search process.
        """

        # Update particle status and species
        for particle in self.swarm.particles:
            self.DetermineParticleStuck(particle)
            self.DetermineSuccessfulExpedition(particle)
            self.EvolveStuckParticle(particle)
            self.EvolveSuccessfulExpedition(particle)

        # Perform additional global exploration checks if enabled
        if self.use_explosive_global_parameter:
            self.CheckGlobalTooDense()
            self.CheckGlobalTooSparse()


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def ExplodeGlobal(self):

        self.CheckGlobalTooDense()
        self.CheckGlobalTooSparse()

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def Iterate(self, iterations):
        """
        Executes the iterative process for the specified number of iterations to evolve the swarm and optimize the search.

        **Process:**
            1. **Initialization (for the first iteration):**
                - **Initialise Swarm:** Calls `self.InitialiseSwarm()` to set up the swarm with particles, velocities, and initial positions during the first iteration.

            2. **Iteration Process:**
                - **Update Position:** Calls `self.UpdatePosition()` to update the positions of all particles based on their current velocities.
                
                - **Generate Boxes:** Attempts to call `self.GenerateAllBoxes()` to create and assign adaptive boxes for particles. Logs a critical error if this fails due to issues with `BoxWidthFunction`.
                
                - **Populate Boxes:** Attempts to call `self.PopulateAllBoxes()` to populate the boxes with sample points. Logs a critical error if this fails due to issues with `SampleSizeFunction`.
                
                - **Evaluate Objective Function:** Calls `self.GetY()` to compute the objective function values (`Y`) for the sample points.
                
                - **Update Local Maxima:** Calls `self.UpdateLocalMaxima()` to update the local maxima for each particle based on the computed `Y` values.
                
                - **Species-Specific Updates (if enabled):**
                    - **Update Species:** If `self.use_species` is `True`, calls `self.UpdateSpecies()` to update particles' species and handle specific behaviors for different species.
                
                - **Update Global Maxima:** Calls `self.UpdateGlobalMaxima()` to update the global maximum value and its location.
                
                - **Rank Particles:** Calls `self.RankParticles()` to rank particles based on their local maxima and update their order.
                
                - **Record Sample Points History:** Calls `self.WriteSamplePointsHistory()` to record the positions and `Y` values of all sampled points.
                
                - **Species-Specific Operations (if enabled):**
                    - **Find Low-Density Regions:** If `self.use_species` is `True`, calls `self.FindLowDensityRegions()` to identify and manage low-density regions for exploration.
                    - **Assign Adventure Leads:** If `self.use_species` is `True`, calls `self.AssignAdventureLeads()` to assign adventure leads to exploratory particles.
                    - **Predict Optimum:** If `self.use_species` is `True`, calls `self.PredictOptimum()` to predict the location of the optimum based on current particle data.
                
                - **Update Velocities:** Calls `self.UpdateVelocity()` to adjust particle velocities based on their current states and the chosen strategy.

            3. **Increment Iteration Counter:**
                - **Update Iteration:** Increments `self.current_iteration` to move to the next iteration.

        This method orchestrates the main loop of the optimization process, managing all key steps required for evolving the swarm and adapting its behavior based on the current state and parameters.
        """

        for _ in range(iterations):
            if self.current_iteration == 0:
                self.InitialiseSwarm()
            else:
                self.UpdatePosition()

            try:
                self.GenerateAllBoxes()
            except:
                self.logger.critical('Failed to specify BoxWidthFunction!')

            try:
                self.PopulateAllBoxes()
            except: 
                self.logger.critical('Failed to specify SampleSizeFunction')

            self.GetY()
            self.UpdateLocalMaxima()
            if self.use_species:
                self.UpdateSpecies()
            self.UpdateGlobalMaxima()
            self.RankParticles()
            self.WriteSamplePointsHistory()
            if self.use_species:
                self.FindLowDensityRegions()
                self.AssignAdventureLeads()
                self.PredictOptimum()
            self.UpdateVelocity()
            self.current_iteration += 1
        
        self.logger.info(f'Current Iteration : {self.current_iteration}')


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def DetermineConvergence(self):
        """
        Evaluates whether the swarm has converged based on its performance over a defined number of iterations.

        **Process:**
            1. **Check Iteration Threshold:**
                - **Threshold Validation:** Verifies if the current iteration exceeds the predefined `convergence_threshold`. This ensures that the method only evaluates convergence after enough iterations have occurred.

            2. **Evaluate Convergence:**
                - **Compare Global Maxima:** Compares the current global maximum value (`self.swarm.global_max`) with the global maximum value recorded `self.convergence_threshold` iterations ago (from `self.swarm.global_max_history`).
                - **Set Convergence Flag:** If the current global maximum value matches the historical value from the threshold period, it indicates that the swarm's best solution has not changed over the recent iterations, suggesting convergence. Sets `self.swarm_converged` to `True` in this case.

        **Example:**
        - If `self.current_iteration` is 50 and `self.convergence_threshold` is 10, the method will check if the global maximum value at iteration 50 is the same as at iteration 40. If they are the same, the swarm is considered to have converged.
        """
        
        if self.current_iteration > self.convergence_threshold:
            # Check if the global maximum has remained unchanged for a period equal to the convergence threshold
            if self.swarm.global_max == self.swarm.global_max_history[-self.convergence_threshold]:
                self.swarm_converged = True


    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
            
    def IterateUntilConvergence(self):
        """
        Executes the optimization process iteratively until convergence or a maximum iteration limit is reached.

        **Process:**
            1. **Convergence Check:**
                - Continuously runs iterations as long as the swarm has not converged (`self.swarm_converged` is `False`) and the number of iterations is below the defined maximum (500 iterations).

            2. **Iteration Execution:**
                - Calls the `Iterate` method for each iteration. This method performs the necessary updates and calculations for the swarm, advancing the optimization process.

            3. **Convergence Determination:**
                - After each iteration, checks if the swarm has converged by invoking the `DetermineConvergence` method. This method evaluates if the global maximum has stabilized over the recent iterations.

        **Example:**
        - If the swarm converges before reaching 500 iterations, the optimization process will stop as soon as `self.swarm_converged` is set to `True`. If convergence is not achieved, the process will automatically stop after 500 iterations.

        **Notes:**
        - **Iteration Limit:** The iteration limit of 500 is a safeguard to prevent infinite loops in case convergence is not achieved. This limit can be adjusted based on the specific requirements and complexity of the optimization problem.
        """
        
        while self.swarm_converged == False and self.current_iteration < 500:
            # Perform one iteration of the optimization process
            self.Iterate(1)
            # Check if the swarm has converged to a solution
            self.DetermineConvergence()

            
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #