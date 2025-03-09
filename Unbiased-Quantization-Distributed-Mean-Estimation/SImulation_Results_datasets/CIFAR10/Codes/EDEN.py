# =============================================================================
# 1. IMPORTS AND DEVICE SETUP
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import CIFAR10
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar, Metrics, Context
import pickle
import os
import pandas as pd
import time
import shutil
import openpyxl  # For Excel writing in pandas
from colorama import Fore, Style
from All_Schemes import *  
import global_variables

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ===============file to save time data=============
totaltime_file = 'totaltime_file.npy'
# =============================================================================
#  CNN MODEL DEFINITION
# =============================================================================
class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        # Define a simple CNN architecture for CIFAR-10 classification
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the forward pass through the network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =============================================================================
# DATASET PREPARATION
# =============================================================================
def get_CIFAR10(data_path: str = "./data"):
    tr = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = CIFAR10(data_path, train=True, download=True, transform=tr)
    testset = CIFAR10(data_path, train=False, download=True, transform=tr)
    return trainset, testset

def train(model, trainloader, optimizer, epochs, device):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

def test(model, testloader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_CIFAR10()
    trainset.targets = torch.tensor(trainset.targets)
    testset.targets = torch.tensor(testset.targets)

    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2024))

    trainloaders, valloaders = [], []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2024)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        )

    testloader = DataLoader(testset, batch_size=128, pin_memory=True)
    return trainloaders, valloaders, testloader

NUM_CLIENTS = 100
trainloaders, valloaders, testloader = prepare_dataset(num_partitions=NUM_CLIENTS, batch_size=64)

# =============================================================================
# 3) FLOWER CLIENT DEFINITION
# =============================================================================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, quant_func, bits_per_dimension) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes=10).to(device)
        self.device = device
        self.quantization_func = quant_func
        self.bits_per_dimension = bits_per_dimension
        self.global_model_params_np_array = np.zeros(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )
    def set_parameters(self, parameters):
        concat_array = np.concatenate([p.reshape(-1) for p in parameters])
        self.global_model_params_np_array = concat_array.copy()
        with torch.no_grad():
            for param, p in zip(self.model.parameters(), parameters):
                param.copy_(torch.tensor(p, dtype=param.dtype, device=self.device))

    def get_parameters(self, config: Dict[str, Scalar]):
        param_arrays = [p.detach().cpu().numpy() for p in self.model.parameters()]
        array_shapes = [arr.shape for arr in param_arrays]
        array_sizes = [arr.size for arr in param_arrays]
        concat_array = np.concatenate([arr.reshape(-1) for arr in param_arrays])
        gradient = concat_array - self.global_model_params_np_array
        # ===================time calculation======================
        t_start = time.time() 
        # Apply quantization: convert gradient to a PyTorch tensor if necessary
        gradient_tensor = torch.from_numpy(gradient).float().to(device)
        quantized_gradient_out = self.quantization_func(gradient_tensor, self.bits_per_dimension)
        t_end = time.time()
        t_quantize = t_end-t_start
        time_taken = np.load(totaltime_file)
        time_taken += t_quantize
        np.save(totaltime_file,time_taken)
        #=======================================================================
        if isinstance(quantized_gradient_out, np.ndarray):
                quantized_gradient_tensor = torch.from_numpy(quantized_gradient_out).to(device)
        else:
            quantized_gradient_tensor = quantized_gradient_out
        quantization_error_vec = quantized_gradient_tensor - gradient_tensor
        gradient_norm = torch.norm(gradient_tensor).item()
        NMSE_information = [quantization_error_vec, gradient_norm]
        #===================================== NMSE results ======================
        nmse_dir = "NMSE_Results_CIFAR10"
        os.makedirs(nmse_dir, exist_ok=True)
        quantization_func_name = self.quantization_func.__name__
        scheme_dir = os.path.join(nmse_dir, quantization_func_name)
        os.makedirs(scheme_dir, exist_ok=True)
        rate_dir = os.path.join(scheme_dir, f"rate_{self.bits_per_dimension}")
        os.makedirs(rate_dir, exist_ok=True)
        # Get the list of existing NMSE files
        existing_files = sorted( [f for f in os.listdir(rate_dir) if f.startswith("NMSE_info_") and f.endswith(".pkl")])
        # Ensure `existing_indices` is always initialized
        existing_indices = sorted(int(f.split("_")[-1].split(".")[0]) for f in existing_files) if existing_files else []

        # Determine the next available index
        next_index = 1
        if existing_indices:
            next_index = max(existing_indices)+1
        else:
            next_index = 1
        file_path = os.path.join(rate_dir, f"NMSE_info_{next_index}.pkl")

        # Save NMSE information (override existing if index exists)
        with open(file_path, "wb") as f:
            pickle.dump(NMSE_information, f)

        print(f"NMSE information saved at: {file_path}")
        #===========================================================================       
        # Ensure quantized_gradient is a NumPy array
        if isinstance(quantized_gradient_out, torch.Tensor):
            quantized_gradient_out = quantized_gradient_out.cpu().numpy()
        elif isinstance(quantized_gradient_out, np.ndarray):
            pass
        else:
            raise TypeError("Quantization function must return a PyTorch tensor or a NumPy array.")
        quantized_model_params = quantized_gradient_out + self.global_model_params_np_array        
        quantized_list_arrays = []
        offset = 0
        for shape, size in zip(array_shapes, array_sizes):
            chunk = quantized_model_params[offset:offset + size]
            quantized_list_arrays.append(chunk.reshape(shape))
            offset += size
        return quantized_list_arrays
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr, epochs = config["lr"], config["epochs"]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        train(self.model, self.trainloader, optimizer, epochs=epochs, device=self.device)
        return self.get_parameters({}), len(self.trainloader), {}
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def get_evaluate_fn(testloader):
    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]):
        model = Net(num_classes=10).to(device)
        with torch.no_grad():
            for param, new_param in zip(model.parameters(), parameters):
                param.copy_(torch.tensor(new_param, dtype=torch.float32, device=device))
        loss, accuracy = test(model, testloader, device)
        return float(loss), {"accuracy": accuracy}
    return evaluate_fn

def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {"epochs": 1, "lr": 0.01}

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.05,
    fraction_evaluate=0.05,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=int(NUM_CLIENTS * 0.75),
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(testloader),
)

def generate_client_fn(trainloaders, valloaders, quantization_func, bits_per_dimension):
    def client_fn(cid: str):
        cid_int = int(cid)
        return FlowerClient(
            trainloader=trainloaders[cid_int],
            valloader=valloaders[cid_int],
            quant_func=quantization_func,
            bits_per_dimension=bits_per_dimension,
        )
    return client_fn

# =============================================================================
# 5) DIRECTORY SETUP FOR RESULTS & TIMING
# =============================================================================
dir_ = os.getcwd()
files_dir = os.path.join(dir_, "Simulation_results_CIFAR10")
os.makedirs(files_dir, exist_ok=True)
files_dir_time = os.path.join(dir_, "Simulation_timing_CIFAR10")
os.makedirs(files_dir_time, exist_ok=True)
parentFilename = os.path.join(files_dir_time, "SimulationTimes.xlsx")

def log_simulation_time(scheme_name: str, bit_depth: int, elapsed_minutes: float):
    try:
        os.makedirs(files_dir_time, exist_ok=True)
        if os.path.isfile(parentFilename):
            df_existing = pd.read_excel(parentFilename, engine="openpyxl")
        else:
            df_existing = pd.DataFrame(columns=["Scheme", "BitDepth", "TimeInMinutes"])
        mask = ((df_existing["Scheme"] == scheme_name) & (df_existing["BitDepth"] == bit_depth))
        if mask.any():
            df_existing.loc[mask, "TimeInMinutes"] = elapsed_minutes
        else:
            new_row = pd.DataFrame([{"Scheme": scheme_name, "BitDepth": bit_depth, "TimeInMinutes": elapsed_minutes}])
            df_existing = pd.concat([df_existing, new_row], ignore_index=True)
        df_existing.to_excel(parentFilename, index=False, engine="openpyxl")
        print(f"{Fore.GREEN}Simulation time logged/updated in {parentFilename}.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to log simulation time: {e}{Style.RESET_ALL}")

# =============================================================================
# 6) MAIN SIMULATION LOOP FOR EDEN_quantize_Hadamard SCHEME
# =============================================================================
quantization_schemes = [
    EDEN_quantize_Hadamard,
]
centralized_results = {}
distributed_results = {}

for quantization_func in quantization_schemes:
    quantization_scheme_name = quantization_func.__name__
    scheme_results_centralized = {}
    scheme_results_distributed = {}
    nmse_dir = "NMSE_Results_CIFAR10"
    scheme_dir = os.path.join(nmse_dir, quantization_scheme_name)
    if os.path.exists(scheme_dir):
        shutil.rmtree(scheme_dir)
    os.makedirs(scheme_dir, exist_ok=True)
    if quantization_scheme_name == "DRIVE_quantize_Hadamard":
        current_bit_depths = [1]
    elif quantization_scheme_name == "QUICFL_quantize":
        current_bit_depths = [1, 2, 4]
    elif quantization_scheme_name == "Scalar_quantize":
        current_bit_depths = [1, 2, 4]
    elif quantization_scheme_name == "No_quantize":
        current_bit_depths = [32]
    else:
        current_bit_depths = [1, 2]
    for bits_per_dimension in current_bit_depths:
        time_taken = 0.0
        np.save(totaltime_file, time_taken)
        client_fn_callback = generate_client_fn(trainloaders, valloaders, quantization_func, bits_per_dimension)
        client_resources = {"num_cpus": 1, "num_gpus": 1}
        start_time = time.time()
        history = fl.simulation.start_simulation(
            client_fn=client_fn_callback,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=300),
            strategy=strategy,
            client_resources=client_resources,
        )
        end_time = time.time()
        time_taken = np.load(totaltime_file)
        elapsed_minutes = time_taken / 60.0
        print("="*20)
        print("GLOB_TIME_TAKEN (in minutes) = ", elapsed_minutes)
        print("GLOB_TIME_TAKEN (in seconds) = ", time_taken)
        print("END-START (in minutes) = ", (end_time - start_time) / 60)
        print("="*20)
        log_simulation_time(quantization_scheme_name, bits_per_dimension, elapsed_minutes)
        scheme_results_centralized[bits_per_dimension] = {"acc_centralized": history.metrics_centralized["accuracy"]}
        scheme_results_distributed[bits_per_dimension] = {"acc_distributed": history.metrics_distributed["accuracy"]}
        print(f"{Fore.YELLOW}Finished {quantization_scheme_name} with {bits_per_dimension} bits in {elapsed_minutes:.2f} min.{Style.RESET_ALL}")
    centralized_results[quantization_scheme_name] = scheme_results_centralized
    distributed_results[quantization_scheme_name] = scheme_results_distributed

centralized_filename = "centralized_accuracy_EDEN_results.pkl"
with open(os.path.join(files_dir, centralized_filename), "wb") as f:
    pickle.dump(centralized_results, f)
distributed_filename = "distributed_accuracy_EDEN_results.pkl"
with open(os.path.join(files_dir, distributed_filename), "wb") as f:
    pickle.dump(distributed_results, f)

print(f"{Fore.YELLOW}All quantization scheme accuracy results saved in {centralized_filename}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}All quantization scheme accuracy results saved in {distributed_filename}{Style.RESET_ALL}")
