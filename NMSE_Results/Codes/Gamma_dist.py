# =============================================================================
# 1. IMPORTS AND DEVICE SETUP
# =============================================================================
from All_Schemes import *
import numpy as np
import torch
import pickle
import os
import pathlib
import sys
import time, pickle

#============= Reproducibility Setup =============
np.random.seed(42)
torch.manual_seed(42)
#================================================

#============= Set Device =======================
# Use GPU if available; otherwise, use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
#=======================================================
# Set up parent directory for NMSE results
parent_dir = os.getcwd()
files_dir = os.path.join(parent_dir, 'NMSE_results_data')
os.makedirs(files_dir, exist_ok=True)

# Set up current path for imports
current_path = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(current_path) + "/../")
#=======================================================

#============= Define Parameters =============
# Define simulation parameters.
max_vectors   = 100   # maximum number of users
num_trials    = 50
dim           = 2048  # reduced dimension
num_instances = 50

num_users_list = np.arange(1, max_vectors + 1, 5)
print(num_users_list)
#======================
#============= Initialize NMSE Result Tensors =============
# Create result tensors on the specified device for each quantization scheme
NMSE_DRIVE_Hadamard_list    = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_EDEN_Hadamard_1bit_list  = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_EDEN_Hadamard_2bit_list  = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_DRIVE_Unstructured_list = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_EDEN_Unstructured_1bit_list = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_EDEN_Unstructured_2bit_list = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_QUICFL_1bit_list       = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_QUICFL_2bit_list       = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Type_Unbiased_1bit_list  = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Type_Biased_1bit_list    = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Type_Unbiased_2bit_list  = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Type_Biased_2bit_list    = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Kashin_1bit_list       = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Kashin_2bit_list       = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Scalar_1bit_list       = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Scalar_2bit_list       = torch.zeros([len(num_users_list), num_instances], device=device)
NMSE_Scalar_4bit_list       = torch.zeros([len(num_users_list), num_instances], device=device)
#======================

#============= Initialize Helper Variables =============
# Variables for computing the empirical mean and storing vector norms
emp_mean = torch.zeros(dim, device=device)
vec_norm_squared_list = []
vec_list = []
#======================

#============= Main Simulation Loop =============
# Loop over different numbers of users and simulation instances

for n_index in range(len(num_users_list)):
    num_users = num_users_list[n_index]
    start_time = time.time()
    
    for instanceno in range(num_instances):
        i = num_users
        vec_norm_squared_list.clear()
        vec_list.clear()
        
        #----- Generate Gamma Distributed Vectors -----
        # Generate 'i' random vectors drawn from Gamma(shape=2, scale=2)
        for _ in range(i):
            vec = np.random.gamma(shape=2, scale=2, size=dim)
            vec_norm_squared_list.append(np.linalg.norm(vec) ** 2)
            # Use torch.as_tensor to minimize overhead
            vec_list.append(torch.as_tensor(vec, device=device, dtype=torch.float32))
        #------------------------------------------------
        
        vec_norm_squared = sum(vec_norm_squared_list)
        emp_mean = torch.stack(vec_list).sum(dim=0) / i
        
        #----- Initialize NMSE Accumulators -----
        NMSE_DRIVE_Hadamard_temp   = 0
        NMSE_EDEN_Hadamard_1bit_temp = 0
        NMSE_EDEN_Hadamard_2bit_temp = 0
        NMSE_Type_Biased_1bit_temp   = 0
        NMSE_Type_Unbiased_1bit_temp = 0
        NMSE_Type_Biased_2bit_temp   = 0
        NMSE_Type_Unbiased_2bit_temp = 0
        NMSE_QUICFL_1bit_temp        = 0
        NMSE_QUICFL_2bit_temp        = 0
        NMSE_Kashin_1bit_temp        = 0
        NMSE_Kashin_2bit_temp        = 0
        NMSE_Scalar_1bit_temp        = 0
        NMSE_Scalar_2bit_temp        = 0
        NMSE_Scalar_4bit_temp        = 0
        #-----------------------------------------
        
        #----- Initialize Estimator Tensors -----
        emp_mean_est_DRIVE_Hadamard    = torch.zeros(dim, device=device)
        emp_mean_est_EDEN_Hadamard_1bit  = torch.zeros(dim, device=device)
        emp_mean_est_EDEN_Hadamard_2bit  = torch.zeros(dim, device=device)
        emp_mean_Type_Unbiased_1bit    = torch.zeros(dim, device=device)
        emp_mean_Type_Biased_1bit      = torch.zeros(dim, device=device)
        emp_mean_Type_Unbiased_2bit    = torch.zeros(dim, device=device)
        emp_mean_Type_Biased_2bit      = torch.zeros(dim, device=device)
        emp_mean_QUICFL_1bit           = torch.zeros(dim, device=device)
        emp_mean_QUICFL_2bit           = torch.zeros(dim, device=device)
        emp_mean_Kashin_1bit           = torch.zeros(dim, device=device)
        emp_mean_Kashin_2bit           = torch.zeros(dim, device=device)
        emp_mean_Scalar_1bit           = torch.zeros(dim, device=device)
        emp_mean_Scalar_2bit           = torch.zeros(dim, device=device)
        emp_mean_Scalar_4bit           = torch.zeros(dim, device=device)
        #-----------------------------------------
        
        #----- Compute Quantized Estimates -----
        # For each vector, compute the quantized output from each scheme and update the running average
        for v in vec_list:
            emp_mean_est_DRIVE_Hadamard   += torch.as_tensor(DRIVE_quantize_Hadamard(v, 1), device=device) / i
            emp_mean_est_EDEN_Hadamard_1bit += torch.as_tensor(EDEN_quantize_Hadamard(v, 1), device=device) / i
            emp_mean_est_EDEN_Hadamard_2bit += torch.as_tensor(EDEN_quantize_Hadamard(v, 2), device=device) / i
            emp_mean_Type_Unbiased_1bit   += torch.as_tensor(Type_unbiased_quantize(v, 1), device=device) / i
            emp_mean_Type_Unbiased_2bit   += torch.as_tensor(Type_unbiased_quantize(v, 2), device=device) / i
            emp_mean_Type_Biased_1bit     += torch.as_tensor(Type_biased_quantize(v, 1), device=device) / i
            emp_mean_Type_Biased_2bit     += torch.as_tensor(Type_biased_quantize(v, 2), device=device) / i
            emp_mean_QUICFL_1bit          += torch.as_tensor(QUICFL_quantize(v, 1), device=device) / i
            emp_mean_QUICFL_2bit          += torch.as_tensor(QUICFL_quantize(v, 2), device=device) / i
            emp_mean_Kashin_1bit          += torch.as_tensor(Kashin_quantize(v, 1), device=device) / i
            emp_mean_Kashin_2bit          += torch.as_tensor(Kashin_quantize(v, 2), device=device) / i
            emp_mean_Scalar_1bit          += torch.as_tensor(Scalar_quantize(v, 1), device=device) / i
            emp_mean_Scalar_2bit          += torch.as_tensor(Scalar_quantize(v, 2), device=device) / i
            emp_mean_Scalar_4bit          += torch.as_tensor(Scalar_quantize(v, 4), device=device) / i
        #-----------------------------------------
        
        #----- Accumulate NMSE for Each Scheme -----
        NMSE_DRIVE_Hadamard_temp   += torch.norm(emp_mean_est_DRIVE_Hadamard - emp_mean).pow(2)   / (num_trials * vec_norm_squared * i)
        NMSE_EDEN_Hadamard_1bit_temp += torch.norm(emp_mean_est_EDEN_Hadamard_1bit - emp_mean).pow(2) / (num_trials * vec_norm_squared * i)
        NMSE_EDEN_Hadamard_2bit_temp += torch.norm(emp_mean_est_EDEN_Hadamard_2bit - emp_mean).pow(2) / (num_trials * vec_norm_squared * i)
        NMSE_Type_Biased_1bit_temp   += torch.norm(emp_mean_Type_Biased_1bit - emp_mean).pow(2)   / (num_trials * vec_norm_squared * i)
        NMSE_Type_Unbiased_1bit_temp += torch.norm(emp_mean_Type_Unbiased_1bit - emp_mean).pow(2) / (num_trials * vec_norm_squared * i)
        NMSE_Type_Biased_2bit_temp   += torch.norm(emp_mean_Type_Biased_2bit - emp_mean).pow(2)   / (num_trials * vec_norm_squared * i)
        NMSE_Type_Unbiased_2bit_temp += torch.norm(emp_mean_Type_Unbiased_2bit - emp_mean).pow(2) / (num_trials * vec_norm_squared * i)
        NMSE_QUICFL_1bit_temp        += torch.norm(emp_mean_QUICFL_1bit - emp_mean).pow(2)        / (num_trials * vec_norm_squared * i)
        NMSE_QUICFL_2bit_temp        += torch.norm(emp_mean_QUICFL_2bit - emp_mean).pow(2)        / (num_trials * vec_norm_squared * i)
        NMSE_Kashin_1bit_temp        += torch.norm(emp_mean_Kashin_1bit - emp_mean).pow(2)        / (num_trials * vec_norm_squared * i)
        NMSE_Kashin_2bit_temp        += torch.norm(emp_mean_Kashin_2bit - emp_mean).pow(2)        / (num_trials * vec_norm_squared * i)
        NMSE_Scalar_1bit_temp        += torch.norm(emp_mean_Scalar_1bit - emp_mean).pow(2)        / (num_trials * vec_norm_squared * i)
        NMSE_Scalar_2bit_temp        += torch.norm(emp_mean_Scalar_2bit - emp_mean).pow(2)        / (num_trials * vec_norm_squared * i)
        NMSE_Scalar_4bit_temp        += torch.norm(emp_mean_Scalar_4bit - emp_mean).pow(2)        / (num_trials * vec_norm_squared * i)
        #-----------------------------------------
        
        #----- Store Instance Results -----
        NMSE_DRIVE_Hadamard_list[n_index, instanceno]   = NMSE_DRIVE_Hadamard_temp
        NMSE_EDEN_Hadamard_1bit_list[n_index, instanceno] = NMSE_EDEN_Hadamard_1bit_temp
        NMSE_EDEN_Hadamard_2bit_list[n_index, instanceno] = NMSE_EDEN_Hadamard_2bit_temp
        NMSE_Type_Biased_1bit_list[n_index, instanceno]   = NMSE_Type_Biased_1bit_temp
        NMSE_Type_Unbiased_1bit_list[n_index, instanceno] = NMSE_Type_Unbiased_1bit_temp
        NMSE_Type_Biased_2bit_list[n_index, instanceno]   = NMSE_Type_Biased_2bit_temp
        NMSE_Type_Unbiased_2bit_list[n_index, instanceno] = NMSE_Type_Unbiased_2bit_temp
        NMSE_QUICFL_1bit_list[n_index, instanceno]        = NMSE_QUICFL_1bit_temp
        NMSE_QUICFL_2bit_list[n_index, instanceno]        = NMSE_QUICFL_2bit_temp
        NMSE_Kashin_1bit_list[n_index, instanceno]        = NMSE_Kashin_1bit_temp
        NMSE_Kashin_2bit_list[n_index, instanceno]        = NMSE_Kashin_2bit_temp
        NMSE_Scalar_1bit_list[n_index, instanceno]        = NMSE_Scalar_1bit_temp
        NMSE_Scalar_2bit_list[n_index, instanceno]        = NMSE_Scalar_2bit_temp
        NMSE_Scalar_4bit_list[n_index, instanceno]        = NMSE_Scalar_4bit_temp
        #-----------------------------------------
        
    print("***********************************************")
    print(f"Simulation for num_users={num_users} done...")
    print("***********************************************")
    end_time = time.time()
    print(f"Time taken for num_users={num_users}: {(end_time - start_time)/60:.2f} mins")
#======================

#============= Compute NMSE Statistics =============
# Compute maximum and average NMSE for each scheme across all instances
NMSE_DRIVE_Hadamard_max    = torch.max(NMSE_DRIVE_Hadamard_list, dim=1).values
NMSE_EDEN_Hadamard_1bit_max  = torch.max(NMSE_EDEN_Hadamard_1bit_list, dim=1).values
NMSE_EDEN_Hadamard_2bit_max  = torch.max(NMSE_EDEN_Hadamard_2bit_list, dim=1).values
NMSE_Type_Biased_1bit_max    = torch.max(NMSE_Type_Biased_1bit_list, dim=1).values
NMSE_Type_Unbiased_1bit_max  = torch.max(NMSE_Type_Unbiased_1bit_list, dim=1).values
NMSE_Type_Biased_2bit_max    = torch.max(NMSE_Type_Biased_2bit_list, dim=1).values
NMSE_Type_Unbiased_2bit_max  = torch.max(NMSE_Type_Unbiased_2bit_list, dim=1).values
NMSE_QUICFL_1bit_max         = torch.max(NMSE_QUICFL_1bit_list, dim=1).values
NMSE_QUICFL_2bit_max         = torch.max(NMSE_QUICFL_2bit_list, dim=1).values
NMSE_Kashin_1bit_max         = torch.max(NMSE_Kashin_1bit_list, dim=1).values
NMSE_Kashin_2bit_max         = torch.max(NMSE_Kashin_2bit_list, dim=1).values
NMSE_Scalar_1bit_max         = torch.max(NMSE_Scalar_1bit_list, dim=1).values
NMSE_Scalar_2bit_max         = torch.max(NMSE_Scalar_2bit_list, dim=1).values
NMSE_Scalar_4bit_max         = torch.max(NMSE_Scalar_4bit_list, dim=1).values

NMSE_DRIVE_Hadamard_avg    = torch.mean(NMSE_DRIVE_Hadamard_list, dim=1)
NMSE_EDEN_Hadamard_1bit_avg  = torch.mean(NMSE_EDEN_Hadamard_1bit_list, dim=1)
NMSE_EDEN_Hadamard_2bit_avg  = torch.mean(NMSE_EDEN_Hadamard_2bit_list, dim=1)
NMSE_Type_Biased_1bit_avg    = torch.mean(NMSE_Type_Biased_1bit_list, dim=1)
NMSE_Type_Unbiased_1bit_avg  = torch.mean(NMSE_Type_Unbiased_1bit_list, dim=1)
NMSE_Type_Biased_2bit_avg    = torch.mean(NMSE_Type_Biased_2bit_list, dim=1)
NMSE_Type_Unbiased_2bit_avg  = torch.mean(NMSE_Type_Unbiased_2bit_list, dim=1)
NMSE_QUICFL_1bit_avg         = torch.mean(NMSE_QUICFL_1bit_list, dim=1)
NMSE_QUICFL_2bit_avg         = torch.mean(NMSE_QUICFL_2bit_list, dim=1)
NMSE_Kashin_1bit_avg         = torch.mean(NMSE_Kashin_1bit_list, dim=1)
NMSE_Kashin_2bit_avg         = torch.mean(NMSE_Kashin_2bit_list, dim=1)
NMSE_Scalar_1bit_avg         = torch.mean(NMSE_Scalar_1bit_list, dim=1)
NMSE_Scalar_2bit_avg         = torch.mean(NMSE_Scalar_2bit_list, dim=1)
NMSE_Scalar_4bit_avg         = torch.mean(NMSE_Scalar_4bit_list, dim=1)
#======================

#============= Save NMSE Data =============
# Create dictionaries for average and maximum NMSE values and save to pickle files
nmse_avg_data = {
    "NMSE_DRIVE_Hadamard_avg":    NMSE_DRIVE_Hadamard_avg.cpu().numpy(),
    "NMSE_EDEN_Hadamard_1bit_avg":  NMSE_EDEN_Hadamard_1bit_avg.cpu().numpy(),
    "NMSE_EDEN_Hadamard_2bit_avg":  NMSE_EDEN_Hadamard_2bit_avg.cpu().numpy(),
    "NMSE_Type_Biased_1bit_avg":    NMSE_Type_Biased_1bit_avg.cpu().numpy(),
    "NMSE_Type_Unbiased_1bit_avg":  NMSE_Type_Unbiased_1bit_avg.cpu().numpy(),
    "NMSE_Type_Biased_2bit_avg":    NMSE_Type_Biased_2bit_avg.cpu().numpy(),
    "NMSE_Type_Unbiased_2bit_avg":  NMSE_Type_Unbiased_2bit_avg.cpu().numpy(),
    "NMSE_QUICFL_1bit_avg":         NMSE_QUICFL_1bit_avg.cpu().numpy(),
    "NMSE_QUICFL_2bit_avg":         NMSE_QUICFL_2bit_avg.cpu().numpy(),
    "NMSE_Kashin_1bit_avg":         NMSE_Kashin_1bit_avg.cpu().numpy(),
    "NMSE_Kashin_2bit_avg":         NMSE_Kashin_2bit_avg.cpu().numpy(),
    "NMSE_Scalar_1bit_avg":         NMSE_Scalar_1bit_avg.cpu().numpy(),
    "NMSE_Scalar_2bit_avg":         NMSE_Scalar_2bit_avg.cpu().numpy(),
    "NMSE_Scalar_4bit_avg":         NMSE_Scalar_4bit_avg.cpu().numpy()
}

nmse_max_data = {
    "NMSE_DRIVE_Hadamard_max":    NMSE_DRIVE_Hadamard_max.cpu().numpy(),
    "NMSE_EDEN_Hadamard_1bit_max":  NMSE_EDEN_Hadamard_1bit_max.cpu().numpy(),
    "NMSE_EDEN_Hadamard_2bit_max":  NMSE_EDEN_Hadamard_2bit_max.cpu().numpy(),
    "NMSE_Type_Biased_1bit_max":    NMSE_Type_Biased_1bit_max.cpu().numpy(),
    "NMSE_Type_Unbiased_1bit_max":  NMSE_Type_Unbiased_1bit_max.cpu().numpy(),
    "NMSE_Type_Biased_2bit_max":    NMSE_Type_Biased_2bit_max.cpu().numpy(),
    "NMSE_Type_Unbiased_2bit_max":  NMSE_Type_Unbiased_2bit_max.cpu().numpy(),
    "NMSE_QUICFL_1bit_max":         NMSE_QUICFL_1bit_max.cpu().numpy(),
    "NMSE_QUICFL_2bit_max":         NMSE_QUICFL_2bit_max.cpu().numpy(),
    "NMSE_Kashin_1bit_max":         NMSE_Kashin_1bit_max.cpu().numpy(),
    "NMSE_Kashin_2bit_max":         NMSE_Kashin_2bit_max.cpu().numpy(),
    "NMSE_Scalar_1bit_max":         NMSE_Scalar_1bit_max.cpu().numpy(),
    "NMSE_Scalar_2bit_max":         NMSE_Scalar_2bit_max.cpu().numpy(),
    "NMSE_Scalar_4bit_max":         NMSE_Scalar_4bit_max.cpu().numpy()
}


#============= Save NMSE Data =============
# Dump the average and maximum NMSE data pickle files into the designated folder
with open(os.path.join(files_dir, 'nmse_avg_data_gamma_dist.pkl'), 'wb') as avg_file:
    pickle.dump(nmse_avg_data, avg_file)

with open(os.path.join(files_dir, 'nmse_max_data_gamma_dist.pkl'), 'wb') as max_file:
    pickle.dump(nmse_max_data, max_file)
#======================
