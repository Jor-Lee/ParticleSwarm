import numpy as np
import time
import os
import glob
import logging
import subprocess
import csv

class PSO_Helpers:
    
    def __init__ (self, number_of_particles, dimension):

        self.number_of_particles = number_of_particles
        self.dimension = dimension

        self.pso_directory = None
        self.output_directory = None
        self.gmax_directory = None

        self.last_complete_iteration = None
        self.last_complete_run = None
        self.current_iteration = None
        self.current_run = None
        self.run_size = None
        self.number_of_runs = None

        self.log_path = None
        self.logger = None

        self.script_content = None
        self.base_input_deck = None
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def CurrentSwarmIteration(self):
        """
        Function identifies the current swarm iteration number of the experiment. 
        Function only works if the swarm iteration folders are named 'SIX' (with X being an integer).

        Parameters:
        OutputDirectory (string): path to the output directory of the experiment.

        Returns:
        CurrentIterationNumber (int): the next iteration number of the experiment
        """
        IterationDirectories = glob(self.output_directory + "/P1" + "/SI*", recursive=True)
        
        iteration_numbers = []
        for directory_long in IterationDirectories:
            directory_short = directory_long.split('/')[-1]
            for i, character in enumerate(directory_short):
                if character.isdigit():
                    number = int(directory_short[i:])
                    iteration_numbers.append(number)
                    break

        if iteration_numbers: 
            self.last_complete_iteration = max(iteration_numbers)
            self.current_iteration = max(iteration_numbers) + 1
            return max(iteration_numbers) + 1
        else: 
            self.current_iteration = 0
            return 0 

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def CurrentSwarmRun(self):
        """
        Function identifies the current swarm run, number of the experiment. 
        Function only works if the swarm run folders are named 'RX' (with X being an integer).

        Parameters:
        OutputDirectory (string): path to the output directory of the experiment.

        Returns:
        CurrentRunNumber (int): the next iteration number of the experiment
        """
        IterationDirectories = glob(self.output_directory + "/P1" + "/SI1" + "/R*", recursive=True)
        
        IterationNumbers = []
        for directory_long in IterationDirectories:
            directory_short = directory_long.split('/')[-1]
            for i, character in enumerate(directory_short):
                if character.isdigit():
                    number = int(directory_short[i:])
                    IterationNumbers.append(number)
                    break

        if IterationNumbers: 
            self.last_complete_run = max(IterationNumbers)
            self.current_run = max(IterationNumbers) + 1
            return max(IterationNumbers) + 1
        else: 
            self.current_run = 0
            return 0

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def WriteSubfile(self, run_dir, run_size, simulation_time, verbose=False):
   
        subfile_path = run_dir + '/submit.slurm'

        with open(subfile_path, 'w') as f:
            f.write('#!/bin/bash')
            f.write('\n#SBATCH --ntasks=1')
            f.write(f'\n#SBATCH --time={simulation_time}')
            f.write('\n#SBATCH --array=0-%i' %(run_size - 1))
            f.write('\n#SBATCH --wait')
            f.write('\n#SBATCH -o %s/BatchOutput.txt' %run_dir)
            
            f.write('\n\ncd %s' %run_dir)
            
            f.write('\n\nhyades S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt')
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)
            
            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)
            
            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)

            f.write('\n\nhyades %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID} >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)
            f.write('\nppf2ncdf %s/S${SLURM_ARRAY_TASK_ID}/input${SLURM_ARRAY_TASK_ID}.ppf >> S${SLURM_ARRAY_TASK_ID}/output.txt' %run_dir)

        if verbose: print('Written the submission file %s' %subfile_path)

        return
        #return subfile_path

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def setup_log(self):
        """
        Simply sets up the log file
        
        Inputs:
        log_path        - where you want to store the log
        log_name        - the file name you want to give it (including .loc)

        Output:
        logger          - object allows you to write to the log file using logger.info('your messege')
        """
        # Join the log path and log name 
        log_path = os.path.join(self.output_directory, 'PSO_log.log')
        self.log_path = log_path

        # Setup logger and set level to INFO
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Setup Log_handler - set mode to 'w' to write
        log_handler = logging.FileHandler(log_path, mode='w')
        log_handler.setLevel(logging.INFO)
        
        # Define the log format (preamble before your message is displayed)
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(log_format)
        
        # add the handler to the logger object so you can start writing to the log file
        logger.addHandler(log_handler)

        self.logger = logger
        
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def Make_Swarm_Iteration_Dir(self):
        """
        Creates the Swarm iteration directories, each particle has its own set of iteration directories
        
        Inputs:
        output_directory            - Where all the output from the swarm is going
        iteration                   - the current iteration we need to make the directory for
        No_ptcs                     - number of particles in the swarm (number of directories we need to make)

        Output:
                                    - Just creates the directories
        """
        for p in range(self.number_of_particles):
            SI_Dir = f"{self.output_directory}/P{p}/SI{self.current_iteration}"
            os.mkdir(SI_Dir)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def Make_Run_Iteration_Dir(self):
        """
        Creates the RUN directories - this should be 5 run directories (each will eventually hold 10 sims)

        Inputs:
        output_directory            - Where all the output from the swarm is going
        iteration                   - the current iteration we need to make the directory for
        No_ptcs                     - number of particles in the swarm (number of directories we need to make)
        Run                         - the current run of the iteration

        Output:
                                    - Just creates the directories
        """
        for p in range(self.number_of_particles):
            SI_Dir = f"{self.output_directory}/P{p}/SI{self.current_iteration}"
            Run_Dir = f"{SI_Dir}/R{self.current_run}"
            os.mkdir(Run_Dir)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def generate_shell_script(self):
        script_content = """#!/bin/bash\n\n"""

        # Navigate to the parent directory
        script_content += f'cd {self.output_directory}\n\n'

        # Iterate through each P* directory
        for p_dir in sorted(os.listdir(self.output_directory)):
            p_dir_path = os.path.join(self.output_directory, p_dir)
            if not os.path.isdir(p_dir_path):
                continue
            
            si_dir_path = os.path.join(p_dir_path, f'SI{self.current_iteration}')
            r_dir_path = os.path.join(si_dir_path, f'R{self.current_run}')
                    
            # Find the .slurm file in the R* directory
            slurm_files = [f for f in os.listdir(r_dir_path) if f.endswith('.slurm')]
            if slurm_files:
                slurm_file = slurm_files[0]  # Assuming there's exactly one .slurm file per folder
                #script_content += f'echo "Submitting job in folder: {r_dir_path}"\n'
                script_content += f'cd {r_dir_path}\n'
                script_content += f'sleep 2\n'
                script_content += f'sbatch {slurm_file} &\n'
                script_content += f'cd {self.output_directory}\n\n'

        self.script_content = script_content
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def Check_Hyades_Done (self):
        """
        If theres a .cdf file not created yet then the swarm has to wait before carrying on
        """
        number_unfinished = 0
        for p in range(self.number_of_particles):
            for sim in range(self.run_size):
                cdf_path = f'{self.output_directory}/P{p}/SI{self.current_iteration}/R{self.current_run}/S{sim}/input{sim}.cdf'
                if os.path.exists(cdf_path): 
                    continue
                else:
                    # logger.info(f'Couldnt find: {cdf_path}')
                    number_unfinished += 1
                    return False
                
        if number_unfinished != 0:
            self.logger.info(f'There are still {number_unfinished} .cdf files to be made')
            return False
        else:
            self.logger.info('All Hyades Simulations are complete')
            return True

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def Check_PPFs_Made (self):
        """
        If theres a .cdf file not created yet then the swarm has to wait before carrying on
        """
        No_unfinished = 0
        for p in range(self.number_of_particles):
            for sim in range(self.run_size):
                ppf_path = f'{self.output_directory}/P{p}/SI{self.current_iteration}/R{self.current_run}/S{sim}/input{sim}.ppf'
                if os.path.exists(ppf_path): 
                    continue
                else:
                    # logger.info(f'Couldnt find: {cdf_path}')
                    No_unfinished += 1
                
        if No_unfinished != 0:
            self.logger.info(f'There are still {No_unfinished} .ppf files to be made')
            return False
        else:
            self.logger.info('All Hyades Simulations are complete')
            return True
        
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def Manually_Make_CDFs (self):
    
        for p in range(self.number_of_particles):
            for sim in range(self.run_size):
                cdf_path = f'{self.output_directory}/P{p}/SI{self.current_iteration}/R{self.current_run}/S{sim}/input{sim}.cdf'
                # sim_path = f'{output_directory}/P{p}/SI{current_iteration}/R{current_run}/S{sim}'
                if not os.path.exists(cdf_path): 
                    self.logger.info(f'Manually Creating: /P{p}/SI{self.current_iteration}/R{self.current_run}/S{sim}/input{sim}.cdf')
                    subprocess.run(['ppf2ncdf', f'{self.output_directory}/P{p}/SI{self.current_iteration}/R{self.current_run}/S{sim}/input{sim}.ppf'])

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def CleanUpDirectories (self):
        for p in range(self.number_of_particles):
            for r in range(self.number_of_runs):
                for sim in range(self.run_size):
                    ppf_Dir = f'{self.output_directory}/P{p}/SI{self.current_iteration}/R{r}/S{sim}/input{sim}.ppf'
                    cdf_Dir = f'{self.output_directory}/P{p}/SI{self.current_iteration}/R{r}/S{sim}/input{sim}.cdf'
                    subprocess.run(['rm', '-r', f'{ppf_Dir}'])
                    subprocess.run(['rm', '-r', f'{cdf_Dir}'])
        return

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def SetupPSO (self):
        
        # Check the output_directory has been made
        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)

        # Create the Output Directory for the Global Maximum
        self.gmax_directory = f'{self.pso_directory}/GMax'
        if not os.path.isdir(self.gmax_directory):
            os.mkdir(self.gmax_directory)

        # Create the individual swarm directories 
        for i in range(self.number_of_particles):
            if not os.path.isdir(f"{self.output_directory}/P{i}"):
                os.mkdir(f"{self.output_directory}/P{i}")

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def ReplaceLine(self, file_name, line_num, text):
        """
        Function that opens a text file and updates text on a specific line. Will be very helpful for modifying input decks.

        Parameters:
        file_name (string): path to input deck
        line_num (int): the number of the line to overwrite
        text (string): text to put on this line
        """
        # Open and read the file
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # Edit the specified line
        lines[line_num] = text + '\n'  # Ensure text ends with a newline

        # Write the edited file
        with open(file_name, 'w') as file:
            file.writelines(lines)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def WriteInputDeck(self, simulation_input_deck, raw_X):
        """
        Write the input deck for the simulation using the X values. 

        Parameters:
        base_input_deck (string): path to a template input deck
        simulation_input_deck (string): path to write new input deck file

        raw_X (1d array): the input parameters

        Returns:
        [result] (list of floats): a list of Y values to return to BOA
        """

        # Copy base file into simulation directory
        subprocess.run(['cp', self.base_input_deck, simulation_input_deck])

        # Calculate values to change in the input deck

        # Laser timings 
        laser4_start_time = raw_X[2] * 12e-9
        laser3_start_time = raw_X[1] * laser4_start_time 
        laser2_start_time = raw_X[0] * laser3_start_time 
        laser1_start_time = 2.000e-10 
        
        # Laser power
        # Maps X[i] ([0, 1]) onto [2e18, 2e21] on a logarithmic scale.
        laser1_power = 2 * 10**(18 + (raw_X[3] * 3))
        laser2_power = 2 * 10**(18 + (raw_X[4] * 3))
        laser3_power = 2 * 10**(18 + (raw_X[5] * 3))
        laser4_power = 2 * 10**(18 + (raw_X[6] * 3))

        # Forcing Laser Power to take expected pulse shape
        #laser4_power = 2 * 10**(18 + (raw_X[6] * 3))
        #laser3_power = raw_X[5] * laser4_power
        #laser2_power = raw_X[4] * laser3_power
        #laser1_power = raw_X[3] * laser2_power

        self.ReplaceLine(simulation_input_deck, 31, f'tv {laser1_start_time:.3e} {laser1_power:.3e}') 
        self.ReplaceLine(simulation_input_deck, 32, f'tv {laser2_start_time:.3e} {laser1_power:.3e}') 
        self.ReplaceLine(simulation_input_deck, 33, f'tv {(laser2_start_time + 0.2e-9):.3e} {laser2_power:.3e}') 
        self.ReplaceLine(simulation_input_deck, 34, f'tv {laser3_start_time:.3e} {laser2_power:.3e}') 
        self.ReplaceLine(simulation_input_deck, 35, f'tv {(laser3_start_time + 0.2e-9):.3e} {laser3_power:.3e}') 
        self.ReplaceLine(simulation_input_deck, 36, f'tv {laser4_start_time:.3e} {laser3_power:.3e}') 
        self.ReplaceLine(simulation_input_deck, 37, f'tv {(laser4_start_time + 0.2e-9):.3e} {laser4_power:.3e}') 
        
        return
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def CreateSimulationDirectories(self, X, p):
        for simulation_number, x in enumerate(X):
            # Make the directory
            simulation_directory = f"{self.output_directory}/P{p}/SI{self.current_iteration}/R{self.current_run}/S{simulation_number}"
            os.mkdir(simulation_directory)

            # Write the input decks
            simulation_input_deck = simulation_directory + '/input%i.inf' %simulation_number
            self.WriteInputDeck(simulation_input_deck, x)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def RunShellScript(self):
        self.generate_shell_script()
        shell_filename = f'ShellSubmitI{self.current_iteration}R{self.current_run}.sh'
        with open(shell_filename, 'w') as f:
            f.write(self.script_content)

        subprocess.run(['chmod', '+x', shell_filename])
        subprocess.run(['./' + shell_filename])

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def WriteStartOfIterationToCSV (self, HyperParams, box_size, run_size, swarm_locations, swarm_velocities):
        self.logger.info('Writing Swarm Locations to the csv file')
        with open(self.output_directory + '/Data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            hyper_row = [2, self.current_iteration, HyperParams[0], HyperParams[1], HyperParams[2], box_size, run_size, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            writer.writerow(hyper_row)
            for p in range(len(swarm_locations)):
                swarm_loc_row = [0, swarm_locations[p][0], swarm_locations[p][1], swarm_locations[p][2], swarm_locations[p][3],
                                 swarm_locations[p][4], swarm_locations[p][5], swarm_locations[p][6], 0, 
                                 swarm_velocities[p][0], swarm_velocities[p][1], swarm_velocities[p][2], swarm_velocities[p][3],
                                 swarm_velocities[p][4], swarm_velocities[p][5], swarm_velocities[p][6],]
                writer.writerow(swarm_loc_row)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def WriteSampledPointsToCsv (self, sample_points, sample_points_Y):
        self.logger.info('Writing Sampled Points to the csv file')
        with open(self.output_directory + '/Data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for p in range(len(sample_points_Y)):
                for sim in range(len(sample_points_Y[0])):
                    swarm_loc_row = [1, sample_points[p][sim][0], sample_points[p][sim][1], sample_points[p][sim][2], sample_points[p][sim][3],
                                        sample_points[p][sim][4], sample_points[p][sim][5], sample_points[p][sim][6], sample_points_Y[p][sim][0], 
                                        0, 0, 0, 0, 0, 0, 0]
                    writer.writerow(swarm_loc_row)

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def WriteGlobalMaxToCSV (self, run_sample_points, run_sample_points_Y, GMAX, GMAX_LOC):
        self.logger.info('Writing the Global Max to the csv')
        gmax = GMAX
        gmax_loc = GMAX_LOC
        gmax_index = []
        with open(self.output_directory + '/Data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for r in range(len(run_sample_points_Y)):
                for p in range(len(run_sample_points_Y[0])):
                    for sim in range(len(run_sample_points_Y[0][0])):
                        try:
                            self.logger.info(f'In WriteGlobalMaxToCSV gmax = {gmax}')
                            if run_sample_points_Y[r][p][sim][0] > gmax:
                                gmax = run_sample_points_Y[r][p][sim][0]
                                self.logger.info(f'{run_sample_points_Y}')
                                self.logger.info(f'{run_sample_points}')
                                gmax_loc = run_sample_points[r][p][sim]
                                gmax_index = [r, p, sim]
                        except:
                            self.logger.info(f'In WriteGlobalMaxToCSV gmax = {gmax} ??')
                            if run_sample_points_Y[r][p][sim][0] > gmax[0]:
                                gmax = run_sample_points_Y[r][p][sim][0]
                                self.logger.info(f'{run_sample_points_Y}')
                                self.logger.info(f'{run_sample_points}')
                                gmax_loc = run_sample_points[r][p][sim]
                                gmax_index = [r, p, sim]

            swarm_loc_row = [3, gmax_loc[0], gmax_loc[1], gmax_loc[2], gmax_loc[3],
                                gmax_loc[4], gmax_loc[5], gmax_loc[6], gmax, 
                                0, 0, 0, 0, 0, 0, 0]
            writer.writerow(swarm_loc_row)
          
        return [gmax, gmax_loc, gmax_index]
  
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    def SaveGlobalMaxima(self, iteration, gmax_index):
        subprocess.run(['rm', '-r', self.gmax_directory])
        subprocess.run(['mkdir', f'{self.output_directory}/P{gmax_index[1]}/SI{iteration}/R{gmax_index[0]}/S{gmax_index[2]}', f'{self.gmax_directory}'])
        subprocess.run(['cp', '-r', f'{self.output_directory}/P{gmax_index[1]}/SI{iteration}/R{gmax_index[0]}/S{gmax_index[2]}', f'{self.gmax_directory}'])
    
    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ # 