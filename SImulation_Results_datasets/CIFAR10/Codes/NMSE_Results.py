import os
import pickle
import numpy as np
import pandas as pd
import torch  

def data_format(data):
    if np.isnan(data):
        return "nan"
    
    if 0.01 <= data < 1e4:
        truncated = int(data * 1000) / 1000  # truncate without rounding
        return f"{truncated:.3f}"
    
    if 0.001 <= data < 0.01:
        sig_digits = 2
    else:
        sig_digits = 5
    
    # Convert to high-precision scientific notation.
    s = f"{data:.12e}" 
    mantissa, exp = s.split("e")
    
    # removing trailing zeros and the dot.
    mantissa = mantissa.rstrip("0").rstrip(".")
    digits = mantissa.replace(".", "")
    
    # Truncating digits 
    if len(digits) > sig_digits:
        truncated_digits = digits[:sig_digits]
    else:
        truncated_digits = digits
    
    # Reinsert a decimal point after the first digit (if more than one digit exists).
    if len(truncated_digits) > 1:
        new_mantissa = truncated_digits[0] + "." + truncated_digits[1:]
    else:
        new_mantissa = truncated_digits
    
    exp_int = int(exp)  # convert exponent to integer to remove any extra zeros
    return f"{new_mantissa}e{exp_int}"   

def compute_nmse_stats_auto(parent_folder: str, sampled_clients_per_round: int = 5):
    results = []  # List to collect result dictionaries

    # Loop over each scheme subfolder.
    for scheme in os.listdir(parent_folder):
        scheme_path = os.path.join(parent_folder, scheme)
        if not os.path.isdir(scheme_path):
            continue
        print(f"Scheme: {scheme}")     
        
        # Loop over each rate folder within the scheme folder.
        for rate in os.listdir(scheme_path):
            rate_path = os.path.join(scheme_path, rate)
            if not os.path.isdir(rate_path):
                continue
            print(f"  Rate folder: {rate}")           
            
            # List and sort all NMSE_info_*.pkl files.
            files = sorted([f for f in os.listdir(rate_path)
                            if f.startswith("NMSE_info_") and f.endswith(".pkl")])
            total_files = len(files)          
            # Discard the initialization file (NMSE_info_1.pkl).
            effective_total = total_files - 1
            expected_rounds = effective_total // sampled_clients_per_round
            if effective_total % sampled_clients_per_round != 0:
                print(f"Warning: (total files - 1) = {effective_total} is not exactly divisible by {sampled_clients_per_round}.")
                print(f"Processing {expected_rounds} complete rounds only.")
                # Optionally trim files list.
                files = files[:1 + expected_rounds * sampled_clients_per_round]
            
            print(f"Total effective files (discarding initialization): {effective_total}")
            print(f"Expected rounds: {expected_rounds}")
            print(f"Clients per round: {sampled_clients_per_round}")
            
            nmse_rounds = []
            # Process each round (0-indexed). File numbering starts at 1, so we skip file 1.
            for r in range(expected_rounds):
                NMSE_numerator = None
                NMSE_denominator = None
                # Process each client file in the current round.
                for j in range(sampled_clients_per_round):
                    # File index: skip the first file (initialization), so add 2.
                    file_index = r * sampled_clients_per_round + j + 2
                    file_name = f"NMSE_info_{file_index}.pkl"
                    file_path = os.path.join(rate_path, file_name)
                    if not os.path.exists(file_path):
                        print(f"Warning: {file_name} not found; skipping.")
                        continue
                    with open(file_path, "rb") as f:
                        # Each file is  stored as  [quant_error_vector, gradient_norm].
                        data = pickle.load(f)
                    if NMSE_numerator is None:
                        NMSE_numerator = data[0]
                        NMSE_denominator = data[1] ** 2
                    else:
                        NMSE_numerator += data[0]
                        NMSE_denominator += data[1] ** 2
                # Compute NMSE for the round if data was collected.
                if NMSE_numerator is None or NMSE_denominator is None:
                    continue
                avg_error = NMSE_numerator / sampled_clients_per_round
                avg_grad_norm_sq = NMSE_denominator / sampled_clients_per_round

                # If these are torch tensors on GPU, move them to CPU and convert to numpy.
                if hasattr(avg_error, 'cpu'):
                    avg_error = avg_error.cpu().numpy()
                if hasattr(avg_grad_norm_sq, 'cpu'):
                    avg_grad_norm_sq = avg_grad_norm_sq.cpu().numpy()

                nmse_value = np.nan if avg_grad_norm_sq == 0 else np.linalg.norm(avg_error)**2 / avg_grad_norm_sq
                nmse_rounds.append(nmse_value)
            
            if nmse_rounds:
                nmse_arr = np.array(nmse_rounds)
                max_nmse = np.nanmax(nmse_arr)
                avg_nmse = np.nanmean(nmse_arr)
                print(f"Computed over {len(nmse_arr)} rounds:")
                print(f"Max NMSE: {max_nmse}")
                print(f"Avg NMSE: {avg_nmse}")
            else:
                max_nmse = np.nan
                avg_nmse = np.nan
                print("No complete rounds processed.")
            
            results.append({
                "Scheme": scheme,
                "Rate Folder": rate,
                "Total Files": effective_total,
                "No of Rounds": expected_rounds,
                "Clients Per Round": sampled_clients_per_round,
                "Max NMSE": data_format(max_nmse),
                "Avg NMSE": data_format(avg_nmse)
            })
    
    df = pd.DataFrame(results)
    excel_filename = "NMSE_stats_CIFAR10.xlsx"
    df.to_excel(excel_filename, index=False, engine="openpyxl")
    print(f"NMSE statistics saved in: {excel_filename}")

if __name__ == "__main__":
    parent_folder_path = "/home/ritesh/Ritesh/WorkWithNithish/AISTATS2025/CIFAR10/Codes/NMSE_Results_CIFAR10"
    compute_nmse_stats_auto(parent_folder_path, sampled_clients_per_round=5)
