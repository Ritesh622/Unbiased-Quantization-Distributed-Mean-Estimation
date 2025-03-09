# =============================================================================
# 1. IMPORTS AND DEVICE SETUP
# =============================================================================
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, random_split
import flwr as fl
from flwr.common import NDArrays, Scalar, Metrics
from typing import List, Optional, Tuple, Dict
import openpyxl
from colorama import Fore, Style
from All_Schemes import *

#===============================================================================
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#==============================================================================
# ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# 2) CNN MODEL DEFINITION
# =============================================================================
class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        # Simple CNN architecture for CIFAR-10 classification
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =============================================================================
# 3) DATASET AND HELPERS
# =============================================================================
def get_CIFAR10(data_path: str = "./data"):
    # Load CIFAR-10 dataset and apply normalization
    tr = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = CIFAR10(data_path, train=True, download=True, transform=tr)
    testset = CIFAR10(data_path, train=False, download=True, transform=tr)
    return trainset, testset

def train(model, trainloader, optimizer, epochs, device):
    # Training loop for a single client
    criterion = nn.CrossEntropyLoss()  # Define loss function
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

def test(model, testloader, device):
    # Evaluation loop for a single client
    criterion = nn.CrossEntropyLoss()  # Define loss function
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Forward pass
            total_loss += criterion(outputs, labels).item()  # Compute loss
            _, predicted = torch.max(outputs, 1)  # Get predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    # Partition the dataset for federated learning
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

        # Split each partition into training and validation sets
        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2024)
        )

        # Create DataLoaders for train and validation sets
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        )

    testloader = DataLoader(testset, batch_size=128, pin_memory=True)
    return trainloaders, valloaders, testloader
# =============================================================================
# dEFINING NUM_CLIENTS AND ROUNDS
# =============================================================================
NUM_CLIENTS = 100
n_round = 10
#================================================================================

trainloaders, valloaders, testloader = prepare_dataset(num_partitions=NUM_CLIENTS, batch_size=64)

# =============================================================================
# 4) FLOWER CLIENT WITH NMSE TRACKING (ACCURACY PART UNCHANGED)
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
        self.global_model_params_np_array = np.zeros(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.true_gradient = None
        self.quantized_gradient = None

    def set_parameters(self, parameters):
        # Update the model parameters with those received from the server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # Move each tensor to the device first, then to NumPy
        state_dict = OrderedDict({k: torch.Tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # Flatten model parameters for gradient calculation
        global_model_params_np_arrays = [val.cpu().numpy() for val in state_dict.values()]
        self.global_model_params_np_array = np.concatenate([np.reshape(x, x.size) for x in global_model_params_np_arrays])

    def get_parameters(self, config: Dict[str, Scalar]):
        # Extract model parameters and flatten them
        np_arrays = [val.cpu().numpy() for val in self.model.state_dict().values()]  # Ensure all tensors are moved to CPU
        array_shapes = [np.shape(x) for x in np_arrays]
        array_sizes = np.array([np.size(x) for x in np_arrays])
        concat_array = np.concatenate([np.reshape(x, x.size) for x in np_arrays])

        # Calculate the model update (gradient)
        gradient = concat_array - self.global_model_params_np_array

        # Apply quantization
        gradient_tensor = torch.from_numpy(gradient).float().to('cpu')

        quantized_gradient = self.quantization_func(
            gradient_tensor, 
            self.bits_per_dimension
        )

        # Ensure quantized_gradient is a NumPy array
        if isinstance(quantized_gradient, torch.Tensor):
            quantized_gradient = quantized_gradient.cpu().numpy()
        elif isinstance(quantized_gradient, np.ndarray):
            # Already a NumPy array, no need to convert
            pass
        else:
            raise TypeError("Quantization function must return a PyTorch tensor or a NumPy array.")

        # Make sure global_model_params_np_array is on CPU before operations
        quantized_model_params = quantized_gradient + self.global_model_params_np_array  # NumPy arrays only

        # Reshape quantized updates to original dimensions
        quantized_list_arrays = []
        for i, size in enumerate(array_sizes):
            init_loc = sum(array_sizes[:i])
            reshaped_subarray = quantized_model_params[init_loc:init_loc + size].reshape(array_shapes[i])
            quantized_list_arrays.append(reshaped_subarray)

        # Store for NMSE computation
        self.true_gradient = gradient  # NumPy array
        self.quantized_gradient = quantized_gradient  # NumPy array

        return quantized_list_arrays

    def fit(self, parameters, config):
        # Set model parameters and train on local data
        self.set_parameters(parameters)
        lr, epochs = config["lr"], config["epochs"]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        train(self.model, self.trainloader, optimizer, epochs=epochs, device=self.device)
        updated_params, num_examples, _ = self.get_parameters({})
        # Return gradients in the metrics dict
        return updated_params, len(self.trainloader), {
            "true_gradient": self.true_gradient,
            "quantized_gradient": self.quantized_gradient,
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Evaluate the model on local validation data
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def get_evaluate_fn(testloader):
    # Define global evaluation function for server
    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]):
        model = Net(num_classes=10).to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(model, testloader, device)
        return float(loss), {"accuracy": accuracy}
    return evaluate_fn

def fit_config(server_round: int) -> Dict[str, Scalar]:
    # Define training configuration
    return {"epochs": 1, "lr": 0.1}

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    # Aggregate accuracy using weighted average
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
    # Generate client function for each client
    def client_fn(cid: str):
        cid_int = int(cid)
        return FlowerClient(
            trainloader=trainloaders[cid_int],
            valloader=valloaders[cid_int],
            quant_func=quantization_func,
            bits_per_dimension=bits_per_dimension
        )
    return client_fn

#============================================================================================================





class NMSEFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nmse_history = []  

    def aggregate_fit(self, rnd: int, results, failures: List[BaseException]) -> Optional[fl.common.Parameters]:
        # Perform standard FedAvg aggregation (accuracy part unchanged)
        aggregated_parameters_nmse = super().aggregate_fit(rnd, results, failures)
        # Extract NMSE-related metrics from each client's fit result
        true_gradients_nmse = []
        quantized_gradients_nmse = []
        for client, fit_res in results:
            metrics = fit_res.metrics
            if metrics is not None:
                if "true_gradient" in metrics and "quantized_gradient" in metrics:
                    true_gradients_nmse.append(metrics["true_gradient"])
                    quantized_gradients_nmse.append(metrics["quantized_gradient"])
        if true_gradients_nmse and quantized_gradients_nmse:
            nmse = compute_nmse(true_gradients_nmse, quantized_gradients_nmse)
        else:
            nmse = None
        self.nmse_history.append(nmse)
        return aggregated_parameters_nmse

def compute_nmse(true_gradients_nmse: List[np.ndarray], quantized_gradients_nmse: List[np.ndarray]) -> float:
    """
    Compute NMSE for a single round:
    NMSE = ||mean(quantized_gradients) - mean(true_gradients)||^2 / ||mean(true_gradients)||^2
    A small epsilon (1e-10) is added to the denominator for numerical stability.
    """
    true_mean = np.mean(true_gradients_nmse, axis=0)
    quant_mean = np.mean(quantized_gradients_nmse, axis=0)
    error = quant_mean - true_mean
    nmse = np.linalg.norm(error)**2 / (np.linalg.norm(true_mean)**2 + 1e-10)
    return nmse

strategy_nmse = NMSEFedAvg(
    fraction_fit=1.0,            # Use all clients in each round for NMSE
    fraction_evaluate=0.0,       # No evaluation needed here
    min_fit_clients=10,
    min_available_clients=10,
    on_fit_config_fn=lambda rnd: {"epochs": 1},
)
def generate_client_fn_nmse(trainloaders, valloaders, quantization_func, bits_per_dimension):
    def client_fn_nmse(cid: str):
        cid_int = int(cid)
        return FlowerClient(
            trainloader=trainloaders[cid_int],
            valloader=valloaders[cid_int],
            quant_func=quantization_func,
            bits_per_dimension=bits_per_dimension
        )
    return client_fn_nmse

#==================================================================================================

# -------------------------------------------------------------------------
# LOGGING FUNCTION TO OVERRIDE (Scheme, BitDepth) BUT KEEP OTHER ROWS
# -------------------------------------------------------------------------
base_folder = "Simulation_timing_CIFAR10"
parentFilename = os.path.join(base_folder, "SimulationTimes.xlsx")

def log_simulation_time(scheme_name: str, bit_depth: int, elapsed_minutes: float):
    """
    Read the existing SimulationTimes.xlsx, update or insert
    a single row with (scheme_name, bit_depth, elapsed_minutes).
    Other existing rows remain unchanged.
    """
    try:
        # Create the folder if it doesn't exist
        os.makedirs(base_folder, exist_ok=True)

        # Either load existing data, or create an empty DataFrame
        if os.path.isfile(parentFilename):
            df_existing = pd.read_excel(parentFilename, engine="openpyxl")
        else:
            df_existing = pd.DataFrame(columns=["Scheme", "BitDepth", "TimeInMinutes"])

        # Look for existing row(s) matching scheme_name & bit_depth
        mask = (
            (df_existing["Scheme"] == scheme_name) &
            (df_existing["BitDepth"] == bit_depth)
        )
        # If found, override that row's 'TimeInMinutes'
        if mask.any():
            df_existing.loc[mask, "TimeInMinutes"] = elapsed_minutes
        else:
            # Otherwise, create a new row
            new_row = pd.DataFrame([{
                "Scheme": scheme_name,
                "BitDepth": bit_depth,
                "TimeInMinutes": elapsed_minutes
            }])
            df_existing = pd.concat([df_existing, new_row], ignore_index=True)

        # Save back to Excel
        df_existing.to_excel(parentFilename, index=False, engine="openpyxl")

        print(f"{Fore.GREEN}Simulation time logged/updated in {parentFilename}.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to log simulation time: {e}{Style.RESET_ALL}")

# =============================================================================================================

# Directory for saving results
dir = os.getcwd()
files_dir = os.path.join(dir, 'Simulation_results_CIFAR10')
os.makedirs(files_dir, exist_ok=True)

# List of quantization schemes to test
quantization_schemes = [
    # Type_unbiased_quantize,
    # Type_biased_quantize,
    # DRIVE_quantize_Hadamard,
     EDEN_quantize_Hadamard,
    # QUICKFL_quantize,
    # Scalar_quantize,
    # Kashin_quantize,
    # No_quantize
]

# Initialize dictionary to store all accuracy results
centralized_results = {}
distributed_results = {}

for quantization_func in quantization_schemes:
    quantization_scheme_name = quantization_func.__name__
    scheme_results_centralized = {}
    scheme_results_distributed = {}
    nsme_results_max = {}
    scheme_results_avg = {}

    # Define bit depths based on quantization scheme
    if quantization_scheme_name == "DRIVE_quantize_Hadamard":
        current_bit_depths = [1]
    elif quantization_scheme_name == "Scalar_quantize":
        current_bit_depths = [1, 2, 4]
    elif quantization_scheme_name == "No_quantize":
        current_bit_depths = [32]
    else:
        current_bit_depths = [1, 2]

    for bits_per_dimension in current_bit_depths:
        start_time = time.time()
        client_fn_callback = generate_client_fn(trainloaders, valloaders, quantization_func, bits_per_dimension)
        client_resources = {"num_cpus": 1, "num_gpus": 1}
        history = fl.simulation.start_simulation(
            client_fn=client_fn_callback,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=n_round),
            strategy=strategy,
            client_resources=client_resources,
        )
        history_nmse = fl.simulation.start_simulation(
        client_fn_nmse=client_fn_nmse,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=n_round),
        strategy_nmse=strategy_nmse,
        )
        
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60.0

        # Log or override the row for this (scheme, bit_depth)
        log_simulation_time(quantization_scheme_name, bits_per_dimension, elapsed_minutes)

        # Store only accuracy results for this bit depth
        scheme_results_centralized[bits_per_dimension] = {
            "acc_centralized": history.metrics_centralized["accuracy"]
        }
        scheme_results_distributed[bits_per_dimension] = {
            "acc_distributed": history.metrics_distributed["accuracy"]
        }
        # Store only accuracy results for this bit depth
        scheme_nmse[bits_per_dimension] = strategy_instance.nmse_history

        print(f"{Fore.YELLOW}Finished {quantization_scheme_name} with {bits_per_dimension} bits in {(time.time() - start_time)/60:.2f} min.{Style.RESET_ALL}")

        
    centralized_results[quantization_scheme_name] = scheme_results_centralized
    distributed_results[quantization_scheme_name] = scheme_results_distributed
    nmse_results[quantization_scheme_name] = scheme_nmse

#  Save centralized results to a .pkl file
centralized_filename = "centralized_accuracy_EDEN_results.pkl"
with open(os.path.join(files_dir, centralized_filename), 'wb') as f:
    pickle.dump(centralized_results, f)
    
# Save distributed results to a .pkl file
distributed_filename = "distributed_accuracy_EDEN_results.pkl"
with open(os.path.join(files_dir, distributed_filename), 'wb') as f:
    pickle.dump(distributed_results, f)
nmse_filename = "nmse_results.pkl"

with open(os.path.join(files_dir, nmse_filename), 'wb') as f:
    pickle.dump(nmse_results, f)

print(f"{Fore.GREEN}All quantization scheme accuracy results saved in {centralized_filename}{Style.RESET_ALL}")
print(f"{Fore.GREEN}All quantization scheme accuracy results saved in {distributed_filename}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}All quantization scheme NMSE results saved in {nmse_filename}{Style.RESET_ALL}")
