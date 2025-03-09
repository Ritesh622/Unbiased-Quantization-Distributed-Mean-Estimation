Algorithm Evaluation: Maximum and Average NMSE Calculations
===========================================================

This code simulates the performance of various quantization schemes, focusing on calculating 
the Normalized Mean Squared Error (NMSE) for different numbers of users, vectors, and 
quantization strategies.

Setup
------------------------------------------------------------

Install all dependencies listed in the `requirements.txt` file.

Files and Structure
------------------------------------------------------------

- `All_Schemes.py`: This modular file contains the implementation of all algorithms. 
  Each algorithm is defined as a function within this file.

Distribution-Specific Files
---------------------------

Each distribution has its own dedicated file (e.g., for Normal distribution `Normal_dist.py`, 
for Laplace `Laplace_dist.py`). These files are used to implement various schemes.

Number of Trials (`num_trials`)
-------------------------------

The number of repetitions for averaging/maximizing the results of each experiment. 
The code sets `num_trials = 50`.

Number of Instances (`num_instances`)
------------------------------------

For each number of users, the experiment is repeated 50 times. These instances allow 
for more robust estimates of NMSE across different random initializations.

Vector Dimension (`dim`)
------------------------

The size of each vector generated in the experiment. Here, the vector dimension is set to 2048. 
Each user generates a vector of this size.

Additional Notes
----------------

For the `QUICKFL` algorithm, ensure that the `tables` folder is located in the same 
directory as `All_Schemes.py`.

Running the Scripts
===================
To run the scripts for all distributions, simply execute the bash file `run_simulation.sh`. 
This will generate and save the data as `.pkl` files in the `NMSE_results_data` folder.

Plotting Results
================

To visualize the results stored in the `.pkl` files, use the `Max_Plot_format.py` 
and `Avg_Plot_format.py` scripts. These files can be found in the 
`NMSE_results_data` folder.

Centralized vs. Distributed Data
--------------------------------

The results are organized into subdirectories based on the data distribution method, such as:

- `nmse_avg_data_distribution_name_centralized.pkl`
- `nmse_avg_data_distribution_name_distributed.pkl`

Example
-------

For the Bernoulli distribution, the average data is saved as:

```bash
nmse_avg_Ber_dist.pkl

Running the Plotting Scripts:
--------------------------------------------------
To generate the plots, simply execute the bash file run_plot_simulation.sh. 
This will automatically run both the Max_Plot_format.py and Avg_Plot_format.py scripts, producing the required plots.










