import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# Define label, marker, and line dictionaries
label_dict = {
    'EDEN_Hadamard_1bit': 'EDEN-Hadamard R=1',
    'EDEN_Hadamard_2bit': 'EDEN-Hadamard R=2',
    'DRIVE_Hadamard_1bit': 'DRIVE R=1',
    'DRIVE_Hadamard_2bit': 'DRIVE R=2',
    'QUICFL_1bit': 'QUIC-FL R=1',
    'QUICFL_2bit': 'QUIC-FL R=2',
    'QUICFL_4bit': 'QUIC-FL R=4',
    'Kashin_1bit': 'Kashin R=1',
    'Kashin_2bit': 'Kashin R=2',
    'Type_Unbiased_1bit': 'Type unbiased R=1',
    'Type_Unbiased_2bit': 'Type unbiased R=2',
    'Type_Biased_1bit': 'Type biased R=1',
    'Type_Biased_2bit': 'Type biased R=2',
    'No_Quantize_32bit': 'No Quantization R=32',
    'Scalar_Quantize_1bit': 'Scalar Quantization R=1',
    'Scalar_Quantize_2bit': 'Scalar Quantization R=2',
    'Scalar_Quantize_4bit': 'Scalar Quantization R=4',
}

marker_dict = {
    'EDEN_Hadamard_1bit': 'd',
    'EDEN_Hadamard_2bit': 'd',
    'DRIVE_Hadamard_1bit': 's',
    'DRIVE_Hadamard_2bit': 's',
    'QUICFL_1bit': 'o',
    'QUICFL_2bit': 'o',
    'QUICFL_4bit': 'o',
    'Kashin_1bit': 's',
    'Kashin_2bit': 's',
    'Type_Unbiased_1bit': '*',
    'Type_Unbiased_2bit': '*',
    'Type_Biased_1bit': '+',
    'Type_Biased_2bit': '+',
    'No_Quantize_32bit': '>',
    'Scalar_Quantize_1bit': '^',
    'Scalar_Quantize_2bit': '^',
    'Scalar_Quantize_4bit': '^',
}

line_dict = {
    'EDEN_Hadamard_1bit': '--',
    'EDEN_Hadamard_2bit': '-',
    'DRIVE_Hadamard_1bit': '--',
    'DRIVE_Hadamard_2bit': '-',
    'QUICFL_1bit': '--',
    'QUICFL_2bit': '-',
    'QUICFL_4bit': '--',
    'Kashin_1bit': '--',
    'Kashin_2bit': '-',
    'Type_Unbiased_1bit': '--',
    'Type_Unbiased_2bit': '-',
    'Type_Biased_1bit': '--',
    'Type_Biased_2bit': '-',
    'No_Quantize_32bit': '-',
    'Scalar_Quantize_1bit': '--',
    'Scalar_Quantize_2bit': '-.',
    'Scalar_Quantize_4bit': '-',
}

# Mapping from data algorithm names to dictionary keys
algo_name_mapping = {
    'EDEN_quantize_Hadamard': 'EDEN_Hadamard',
    'DRIVE_quantize_Hadamard': 'DRIVE_Hadamard',
    'QUICFL_quantize': 'QUICFL',
    'Kashin_quantize': 'Kashin',
    'Type_unbiased_quantize': 'Type_Unbiased',
    'Type_biased_quantize': 'Type_Biased',
    'No_quantize': 'No_Quantize',
    'Scalar_quantize': 'Scalar_Quantize',
}

def plot_accuracy_centralized(pkl_file_paths, filename):
    # Check if the './Simulation_Plots/' directory exists, if not create it
    if not os.path.exists('./Simulation_Plots/'):
        os.makedirs('./Simulation_Plots/')

    # Define a set of algorithms to ignore. You can update this as needed.
    ignore_algos = {'Scalar_Quantize_1bit', 'QUICFL_1bit'}

    # Initialize a dictionary to hold all algorithms' data
    all_algo_data = {}

    # Iterate over each .pkl file
    for pkl_file_path in pkl_file_paths:
        # Load the accuracy data from the pickle file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        # Iterate over each algorithm in the data
        for algo_name, data_rates in data.items():
            # Map the algorithm name to match dictionary keys
            mapped_algo_name = algo_name_mapping.get(algo_name, algo_name)

            # Iterate over each data rate (e.g., 1, 2)
            for data_rate, metrics in data_rates.items():
                # Create a unique key by combining algorithm name and data rate
                if isinstance(data_rate, int):
                    data_rate_str = f"{data_rate}bit"
                else:
                    data_rate_str = data_rate

                algo_key = f"{mapped_algo_name}_{data_rate_str}"

                # Check if this algorithm key is in the ignore list and skip it if so
                if algo_key in ignore_algos:
                    print(f"Ignoring {algo_key} as per ignore list")
                    continue

                # Skip if the combined name is not in label_dict
                if algo_key not in label_dict:
                    print(f"Skipping {algo_key} as it's not in label_dict")
                    continue

                # Initialize the list to hold accuracy values for this algorithm and data rate
                if algo_key not in all_algo_data:
                    all_algo_data[algo_key] = {'iterations': None, 'accuracies': []}

                # Collect accuracy data
                acc_centralized = metrics.get('acc_centralized', [])
                if acc_centralized:
                    iterations, accuracies = zip(*acc_centralized)
                    # Assuming iterations are consistent across runs
                    if all_algo_data[algo_key]['iterations'] is None:
                        all_algo_data[algo_key]['iterations'] = iterations
                    elif all_algo_data[algo_key]['iterations'] != iterations:
                        print(f"Warning: Iterations mismatch in {algo_key}")
                    all_algo_data[algo_key]['accuracies'].append(accuracies)

    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy for each algorithm and data rate, with an optional custom iteration range
    for algo_key, data in all_algo_data.items():
        accuracies_runs = np.array(data['accuracies'])
        # Extract the iterations; if it's a list of tuples, take the first element from each tuple
        if isinstance(data['iterations'][0], tuple):
            iterations = np.array([x[0] for x in data['iterations']])
        else:
            iterations = np.array(data['iterations'])
        for run in accuracies_runs:
            run_percent = np.array(run) * 100
            plt.plot(
                iterations,
                run_percent,
                marker=marker_dict[algo_key],
                markersize=7,
                linestyle=line_dict[algo_key],
                linewidth=1.5,
                label=label_dict[algo_key]
            )
    # Add labels, grid, and legend
    plt.xlabel('No of rounds', fontsize=14)
    plt.ylabel('Accuracy(%)', fontsize=14)
    plt.grid(True, which='both', axis='both')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend(loc='lower right')
    # Save the figure as a PDF with the specified filename
    plt.savefig(f'./Simulation_Plots/{filename}.pdf', format='pdf')
    # Display the plot
    plt.show()
  
if __name__ =="__main__": 
    file_path_folder = "folder_path"
    
    pkl_files = [os.path.join(file_path_folder, f) for f in os.listdir(file_path_folder) if f.endswith('.pkl')]
    
    plot_accuracy_centralized(pkl_files, "centralized_CIFAR10")
    
