## ECL Discrete Workflow Simulation:

### Version 1.3
*by Ricky Huang*

Required python packages:
numpy, scipy, pandas, matplotlib

### - Tests folder

This folder contains all the testing functions/files for recording and displaying the simulations results.

### - Utils folder

This folder contains all the utility functions for running the simulations and plotting the graphs.

#### Sim1_Util.py
This file contains the basic plotting functions

     Simulation related:
      |- roundhalf:       rounds to integers
      |- poisson_sim:     models Poisson arrival
      |- sort_ptcls:      sorts ptcls based on arrival time
      
     General plotting:
      |- plot_2D:         plots multiple sets of data in one plot
      |- plot_hist:       plots a histogram of a set of data
      |- plot_2Dy2:       plot_2D with two separate y-axes
      
     Single-run plotting:
      |- plot_QueuesTime: queue size v. Time
      |- plot_QueueWait:  queue size & wait time v. Time
      |- plot_Ptclsta:    ptcl status number v. Time

#### Sim13_ecl_Util.py
This file contains the .csv-related functions 

**Note:** all the data .csv files need to be stored locally. You can specify customized path to these .csv files when calling the correspondent functions

     ECL data related:
      |- data_to_csv:     replaces NaN values with -1 in the original .csv data
      |- csv_to_jobs:     reads .csv data and produces a job dict 
                          (also takes a multiplier variable for job operating time)
      |- get_multiplier:  returns a multiplier for the mean job operating time in the ECL data
      |- sample_job:      samples ptcls from a job dict
      |- res_to_csv:      turns batch simulation results into a .csv file

     Data plotting:
      |- plot_scatter:    plots avg_wt v. eps with (different) pause_prob

#### Sim13_Classes.py
This file contains all the essential class objects for the simulation

      |- class customer:  generates a customer to submit ptcls at a fixed rate
      |- class protocol:  generates a ptcl with status, operating time, pause probability, release time
      |- class operator:  generates an operator with status to grab, process, and dispose ptcls
      |- class onthread:  generates an Onthread queue to hold on progress, paused, and waiting ptcls
      |- class backlog:   generates a backlog queue to hold ptcls when Onthread reaches capacity

#### Sim13_Init.py
This file contains the key simulation function for randomly generted jobs

      |- initialize:      generates an empty onthread, a backlog with initial jobs, idle operators, and customers
      |- sim_run:         performs one workflow simulation under a given timespan, this is the key function
      
#### Sim13_ecl_Init.py
This file contains the key simulation function that incorporates ECL .csv data
The only difference from Sim13_Init.py is that the simulated jobs are sampled from the ECL data

      |- initialize:      generates an empty onthread, a backlog with initial jobs, idle operators, and customers
      |- sim_run:         performs one workflow simulation under a given timespan, this is the key function

#### Sim13_Batch.py
This file contains the functions that conduct batch-testing simulations and reports

      Batch-testing related:
      |- test_runs:      running and recording simulations, with the only variable being the operator number
      |- setup:          produces an array of test_param, each test_param is a dictionary specifying the simulation configurations
      |- test_func:      inputs a test_data array by setup function, outputs the test_data array with simulation results
      
      Auxiliary functions:
      |- get_mean:	 	   computes the mean value of specific attributes in test_param
      |- plot_multi_QueuesTime:    plots queue size vs. time plot with multiple queues from different test_param
      |- plot_multi_WaitTime:      plots waiting time vs. time plot with multiple waiting times from different test_param
      |- plot_multi_OutstandTime:  plots outstanding time vs. time plot with multiple outstanding times from different test_param
            
#### Sim13_ecl_Batch.py
This file contains the functions that conduct batch-testing simulations and reports
The only difference from Sim13_Batch.py is that the simulated jobs are sampled from the ECL data

      Batch-testing related:
      |- test_batch_wpic: 	   conducts a systematic batch-test and outputs the .csv data and the graphs
      |- test_runs:      running and recording simulations, with the only variable being the operator number
      |- setup:          produces an array of test_param, each test_param is a dictionary specifying the simulation configurations
      |- test_func:      inputs a test_data array by setup function, outputs the test_data array with simulation results
      |- single_run:	 conducts one simulation with specified parameters and records the results.
      
      Auxiliary functions:
      |- get_mean:	 	   computes the mean value of specific attributes in test_param
      |- plot_multi_QueuesTime:    plots queue size vs. time plot with multiple queues from different test_param
      |- plot_multi_WaitTime:      plots waiting time vs. time plot with multiple waiting times from different test_param
      |- plot_multi_OutstandTime:  plots outstanding time vs. time plot with multiple outstanding times from different test_param
