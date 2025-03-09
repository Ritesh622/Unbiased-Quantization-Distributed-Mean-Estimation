# =============================================================================
# 1. IMPORTS AND INITIAL SETUP
# =============================================================================
import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --------------------- Dictionaries for Labels, Markers, and Line Styles ---------------------
label_dict = {
    'EDEN_Hadamard_blackbox_1bit': 'EDEN-Hadamard R=1',
    'EDEN_Hadamard_blackbox_2bit': 'EDEN-Hadamard R=2',
    'Type_Unbiased_2bit': 'Type Unbiased R=2',
    'DRIVE_Hadamard': 'DRIVE R=1',
    'Scalar_1bit': 'Scalar Quantization R=1',
    'Scalar_2bit': 'Scalar Quantization R=2',
    'Scalar_4bit': 'Scalar Quantization R=4',
    'Kashin_2bit': 'Kashin R=2',
    'Type_Biased_1bit': 'Type Biased R=1',
    'Type_Biased_2bit': 'Type Biased R=2',
    'No_Quantize': 'No Quantization',
    'EDEN_Hadamard_1bit': 'EDEN-Hadamard R=1',
    'Type_Unbiased_1bit': 'Type Unbiased R=1',
    'QUICFL_2bit': 'QUIC-FL R=2',
    'QUICFL_1bit': 'QUIC-FL R=1',
    'Kashin_1bit': 'Kashin R=1',
    'EDEN_Hadamard_2bit': 'EDEN-Hadamard R=2'
}

marker_dict = {
    'EDEN_Hadamard_blackbox_1bit': 'd',
    'EDEN_Hadamard_blackbox_2bit': 'd',
    'Type_Unbiased_2bit': '*',
    'Type_Unbiased_1bit': '*',
    'DRIVE_Hadamard': 'd',
    'Scalar_1bit': '^',
    'Scalar_2bit': 'v',
    'Scalar_4bit': '^',
    'Kashin_2bit': 's',
    'Kashin_1bit': 's',
    'Type_Biased_1bit': '+',
    'Type_Biased_2bit': '+',
    'QUICFL_2bit': 'o',
    'QUICFL_1bit': 'o',
    'No_Quantize': '>',
}

line_dict = {
    'EDEN_Hadamard_blackbox_1bit': '--',
    'EDEN_Hadamard_blackbox_2bit': '-',
    'Type_Unbiased_2bit': '-',
    'Type_Unbiased_1bit': '--',
    'DRIVE_Hadamard': '--',
    'Scalar_1bit': '--',
    'Scalar_2bit': '-',
    'Scalar_4bit': '-',
    'Kashin_2bit': '-',
    'Kashin_1bit': '--',
    'Type_Biased_1bit': '--',
    'Type_Biased_2bit': '-',
    'QUICFL_2bit': '-',
    'QUICFL_1bit': '--',
    'No_Quantize': '-',
}

# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================
def plot_nmse(folder_path, output_filename):
    parent_dir = os.getcwd()
    plots_dir = os.path.join(parent_dir, "Plots_Distribution_NMSE")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load NMSE data from the pickle file
    with open(folder_path, 'rb') as f:
        nmse_dict = pickle.load(f)
    
    # Helper: strip prefixes/suffixes so keys match our dictionaries
    def strip_key(key):
        return key.replace("NMSE_", "").replace("_avg", "").replace("_max", "")
    
    plt.figure(figsize=(10, 6))
    
    # Plot each algorithm's NMSE values
    for algorithm, nmse_values in nmse_dict.items():
        algo_name = strip_key(algorithm)
        if algo_name in label_dict:
            nmse_values = np.array(nmse_values)
            # x-axis: one point per user, with integer ticks
            #x_values = np.arange(1, 5*len(nmse_values)-1, 5)
            x_values = np.linspace(1, 5 * (len(nmse_values) - 1), len(nmse_values), dtype=int)
            nmse_values = np.clip(nmse_values, a_min=1e-10, a_max=None)
            plt.semilogy(
                x_values, nmse_values,
                marker=marker_dict.get(algo_name, 'o'),
                markersize=7,
                linestyle=line_dict.get(algo_name, '-'),
                linewidth=1.5,
                label=label_dict[algo_name]
            )
        else:
            print(f"Warning: '{algo_name}' not found in label_dict.")
    
    plt.xlabel("Number of users", fontsize=14)
    plt.ylabel("Max NMSE", fontsize=14)
    plt.xticks(x_values)  # Only integer x-ticks
    plt.grid(True, which="both")
    plt.legend(loc='upper right')
    
    # Save the plot (override if exists)
    output_path = os.path.join(plots_dir, f"{output_filename}.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()

def plot_all_distributions(parent_folder, file_type):
    parent_dir = os.getcwd()
    plots_dir = os.path.join(parent_dir, "Plots_Distribution_NMSE")
    os.makedirs(plots_dir, exist_ok=True)
    pattern = os.path.join(parent_folder, f"nmse_{file_type}_data_*.pkl")
    pkl_files = glob.glob(pattern)
    
    if not pkl_files:
        print("No NMSE pickle files found with pattern:", pattern)
        return
    
    for pkl_file in pkl_files:
        base = os.path.basename(pkl_file)
        prefix = f"nmse_{file_type}_data_"
        distribution = base[len(prefix):].replace(".pkl", "")
        output_filename = f"NMSE_{distribution}_{file_type}"
        print(f"Plotting {base} ...")
        plot_nmse(pkl_file, output_filename)

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    parent_folder = "path of parent folder"
    file_type = "max"  # Set to "max" for reading  max .pkl files and then NMSE plotting
    
    plot_all_distributions(parent_folder, file_type)
