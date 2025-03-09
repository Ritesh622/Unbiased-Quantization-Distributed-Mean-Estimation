Installation
============
Install all dependencies listed in the `requirements.txt` file.

Files and Structure
-------------------
- **All_Schemes.py**: Contains the implementation of all algorithms, each defined as a function.
- **Distribution-Specific Files**: Each quantization scheme has its own file (e.g., `DRIVE.py`, `EDEN.py`).

Additional Notes
----------------
For the `QUICKFL` algorithm, ensure the `tables` folder is located in the same directory as `All_Schemes.py`.

Running the Scripts
===================
To run the scripts for all schemes, execute:
bash run_simulation.sh

This generates and saves the simulation data (`.pkl` files) in the `Simulation_results_dataset` folder.
For example, in the CIFAR10 dataset folder `Simulation_results_CIFAR10`, accuracy data is saved as:
- centralized_accuracy_DRIVE_results.pkl
- distributed_accuracy_DRIVE_results.pkl

NMSE Computation
----------------
Each scheme file (e.g., DRIVE.py, EDEN.py) computes the NMSE across each round and saves multiple `.pkl` files (one per rate) under the `NMSE_Results_datasetname` folder.
For instance, running `EDEN.py` with CIFAR10 will create:
NMSE_Results_CIFAR10 -> EDEN -> rate_1 -> *.pkl
                               -> rate_2 -> *.pkl

Quantization timing data is saved in the `Simulation_timing_datasetname` folder as `SimulationTimes.xlsx`.

Plotting Results
================
To generate accuracy plots, run:
bash run_plot_simulation.sh

This executes:
- Plot_Format_dist.py (for distributed data)
- Plot_format_centralized.py (for centralized data)
- Plot_customized_centralized.py and Plot_customized_dist.py (for customized ranges)

NMSE Results
------------
For NMSE statistics, run:
python NMSE_results.py

This produces the `NMSE_stats_dataset_name.xlsx` file containing average and maximum NMSE values.

