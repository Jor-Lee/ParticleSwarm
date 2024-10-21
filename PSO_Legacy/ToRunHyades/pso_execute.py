import numpy as np
import pso as pso
import pso_helpers as pso_helper
import pso_Analysis as Analysis
import pickle
import time
import os
import subprocess

number_of_particles = 250
dimension = 7
bounds = [[0, 1] for _ in range(dimension)]
maximise = True
  
# Setup the pso and pso_helper objects 
pso_helper = pso_helper.PSO_Helpers(number_of_particles, dimension)

# Establish the directories that will store .cdfs
PSO_directory = '/work4/clf/Josh/Bayesian/hyades_species_pso'
pso_helper.pso_directory = PSO_directory
pso_helper.output_directory = f'{PSO_directory}/PSO4_output'
pso_helper.base_input_deck = f'{PSO_directory}/pso_decks/input_deck.inf'
pso_helper.SetupPSO()
pso_helper.setup_log()
gmax_directory = f'{PSO_directory}/GMax4'

iteration = 0

# Define the experiment parameters
number_of_runs = 1
pso_helper.number_of_runs = 1

# Define the functions that control experiment parameters:

use_adaptive_hyper_parameters = False

def GenerateY (X):

    # ======------- - - - - - ---------======== #

    def GetResult(simulation_CDF_file, verbose=False):
        """ 
        Function to be optimised. Can make use of the BOA_Helpers/Analysis.py file which contains some pre-made alaysis functions.

        Parameters: 
        simulation_CDF_file (string): path to the cdf file
        raw_X (1d array): the input parameters

        Returns:
        [result] (list of floats): a list of Y values to return to BOA
        """

        # If CDF file exists analyse.
        try:
            # Pull important values from the simulation
            gain_data = Analysis.Gain(simulation_CDF_file)
            CR_data = Analysis.ConvergenceRatio(simulation_CDF_file)
            IFAR_data = Analysis.IFAR(simulation_CDF_file)
            velocity_data = Analysis.ImplosionVelocity(simulation_CDF_file)

            gain = gain_data[0]
            CR = np.max(CR_data[0])
            IFAR = IFAR_data[0][IFAR_data[1]]
            velocity = abs(min(velocity_data[0]))
            parametric_limit = Analysis.LaserProfile(simulation_CDF_file)[2]
            

            def Get_multiplier (x, X_cutoff, half=False):
                if half==False:
                    frac_x = (x - X_cutoff)/(0.25*X_cutoff)
                    # print(frac_x)
                    a = 0.9644
                    b = 13
                    return (1/a)*(1 - (1/(1 + np.exp(-(b*frac_x - 3.3)))))
                else:
                    frac_x = (x - X_cutoff)/(0.25*X_cutoff)
                    # print(frac_x)
                    a = 0.9644
                    b = 13
                    return (1/a)*(1 - (1/(1 + np.exp(-(2*b*frac_x - 3.3)))))

            # Reduce effective gain if CR is above 13. Reduces result by 1/e at a CR of 17
            if CR > 13:
                #CR_multiplier = np.exp(- (CR - 13) / (17 - 13))
                CR_multiplier = Get_multiplier(CR, 13)
            else:
                CR_multiplier = 1

            # Reduce effective gain if IFAR is above 30. Reduces result by 1/e at an IFAR of 40
            if IFAR > 30:
                #IFAR_multiplier = np.exp(- (IFAR - 30) / (40 - 30))
                IFAR_multiplier = Get_multiplier(IFAR, 30)
            else:
                IFAR_multiplier = 1

            # Reduce effective gain if IFAR is above 30. Reduces result by 1/e at an IFAR of 500
            if velocity > 400:
                #velocity_multiplier = np.exp(- (velocity - 400) / (500 - 400))
                velocity_multiplier = Get_multiplier(velocity, 400, half=True)
            else:
                velocity_multiplier = 1

            # Reduce effective gain if IFAR is above 1e14. Reduces result by 1/e at an parametric limit values of 2e14
            if parametric_limit > 1e14:
                # parametric_limit_multiplier = np.exp(- (parametric_limit - 1e14) / (2e14 - 1e14))
                parametric_limit_multiplier = Get_multiplier(parametric_limit, 1e14)
            else:
                parametric_limit_multiplier = 1

            # Combine all multipliers into a single value. This is where instability is factored into the loss function.
            result = gain * CR_multiplier * IFAR_multiplier * velocity_multiplier * parametric_limit_multiplier

        except:
            result = 0.0

        return result
        
        # ======------- - - - - - ---------======== #

    print(f'Len(X) = {len(X)}')
    print(np.array(X))
    sample_points = X.reshape(number_of_particles, SampleSizeFunction(iteration), dimension)

    pso_helper.current_run = 0
    pso_helper.logger.info(f'Starting Run 0')
    pso_helper.Make_Swarm_Iteration_Dir()
    pso_helper.Make_Run_Iteration_Dir()
    pso_helper.run_size = pso.SampleSizeFunction(iteration)
    for p in range(number_of_particles):
        pso_helper.WriteSubfile(run_dir=f'{pso_helper.output_directory}/P{p}/SI{iteration}/R{0}', run_size=SampleSizeFunction(iteration), simulation_time='03:30:00')
        pso_helper.CreateSimulationDirectories(sample_points[p], p)

    pso_helper.RunShellScript()

    while not pso_helper.Check_Hyades_Done():
        if pso_helper.Check_PPFs_Made() == True:
            pso_helper.logger.info('All Hyades jobs have been submitted!')
            pso_helper.logger.info('Waiting for the last jobs to finish...')
            # Break the long sleep into shorter intervals
            total_sleep_time = 3.8 * 3600  # 3.2 hours in seconds
            interval = 120  # Interval to check the condition in seconds
            start_time = time.time()

            while time.time() - start_time < total_sleep_time:
                time.sleep(interval)
                if pso_helper.Check_Hyades_Done():
                    break
            pso_helper.Manually_Make_CDFs()
        else:
            time.sleep(120)

    Y = []
    for p in range(number_of_particles):
        for sim in range(SampleSizeFunction(iteration)):
            if sample_points[p, sim, 6] > sample_points[p, sim, 5] and sample_points[p, sim, 5] > sample_points[p, sim, 4] and sample_points[p, sim, 4] > sample_points[p, sim, 3]:
                Sim_CDF_File = f"{pso_helper.output_directory}/P{p}/SI{iteration}/R{0}/S{sim}/input{sim}.cdf"
                Y.append(GetResult(Sim_CDF_File))
            else:
                Y.append(0)
    return Y

use_adaptive_boxes = True

def LocalWeightFunction (iteration):
    return

def GlobalWeightFunction (iteration):
    return 

def InertialWeightFunction (iteration):
    return

use_adaptive_boxes = True
use_species = True
species_weights = [0.1, 0.2, 0.1, 0.1, 0.5]

use_boxes_cutoff = 1
box_reduction_factor = 0.8

def BoxWidthFunction(iteration):
    if iteration < use_boxes_cutoff:
        return (box_reduction_factor**(iteration))*(bounds[0][1] - bounds[0][0])
    else:
        use_adaptive_boxes = False
        return 0.0001

def SampleSizeFunction(iteration):
    if iteration < use_boxes_cutoff:
        return 4
    else:
        return 1

use_explosive_global_parameter = False
    
pso = pso.Pso(dimension, number_of_particles, bounds, GenerateY, 
                maximise, use_adaptive_hyper_parameters, GlobalWeightFunction,
                LocalWeightFunction, InertialWeightFunction, use_adaptive_boxes,
                use_species, species_weights, BoxWidthFunction, SampleSizeFunction,
                use_explosive_global_parameter, log_file="/work4/clf/Josh/Bayesian/hyades_species_pso/log.log")

while pso.current_iteration < 300:

    pso_helper.current_iteration = pso.current_iteration
    pso_helper.run_size = pso.SampleSizeFunction(pso.current_iteration)
    
    if pso.current_iteration == 0:
        pso.vthresh = 0.03
        pso.vboost = 30
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

    if pso.use_species == True:
        pso.FindLowDensityRegions()
        pso.AssignAdventureLeads()
        pso.PredictOptimum()
    pso.UpdateVelocity()

    if pso.swarm.improved == True:
        subprocess.run(['rm', '-r', gmax_directory])
        subprocess.run(['mkdir', gmax_directory])
        subprocess.run(['cp', '-r', f'{pso_helper.output_directory}/P{pso.swarm.global_max_particle_index}/SI{iteration}/R{0}/S{pso.swarm.global_max_sample_index}', gmax_directory])
    pso.swarm.improved = False

    pso_helper.CleanUpDirectories()
    pso.current_iteration += 1
    iteration += 1

    with open('/work4/clf/Josh/Bayesian/hyades_species_pso/pso.pkl', 'wb') as file:
        pickle.dump(pso, file)