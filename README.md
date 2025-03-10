# Unbiased Type-based Quantization for Distributed Mean Estimation

This is official code for the paper: “[Unbiased Quantization of the $L_1$ Ball for Communication-Efficient Distributed Mean Estimation](https://openreview.net/forum?id=AdXSZNm3SL)” by Nithish Suresh Babu, Ritesh Kumar, Shashank Vatedka, AISTATS 2025.

In this paper, we propose unbiased and biased quantizers for the $L_1$ ball, and use this for distributed mean estimation (DME) and federated learning. For distributed mean estimation, these schemes achieve order-optimal $O(1/n)$ normalized mean squared error, where $n$ denotes the number of users/clients.

There are two subfolders:
- NMSE_Results: this contains code to simulate DME where each user has an independent random vector with i.i.d. components. Various distributions are used.
- Simulation_Results_Datasets: this contains code to simulate communication-efficient federated learning over the MNIST, CIFAR-10 and Fashion-MNIST datasets.

For comparison, the code also includes implementations of other algorithms taken from [this Github repo](https://github.com/amitport/QUIC-FL-Quick-Unbiased-Compression-for-Federated-Learning).
