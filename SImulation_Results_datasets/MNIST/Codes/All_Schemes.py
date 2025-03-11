""" 
Code implementations for some existing algorithms taken from https://github.com/amitport/QUIC-FL-Quick-Unbiased-Compression-for-Federated-Learning
Copyright (c) 2024, Ran Ben-Basat, Yaniv Ben-Itzhak, Amit Portnoy, and Shay Vargaftik
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import xxhash
import pickle
import os
from scipy.stats import halfnorm
from scipy.stats import norm
import glob
import pathlib
import sys
import torch.nn as nn
import torch.nn.functional as F


# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

current_dir = os.getcwd()
path = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(path) + "/../")






##############################################################################
# 1) Fast Walsh–Hadamard Transform (1D)
##############################################################################
def fast_walsh_hadamard_transform(x, normalize=False):
    """
    In-place Fast Walsh–Hadamard Transform on a 1D torch tensor x.
    Assumes x.size(0) is a power of 2.
    If normalize=True, divides by sqrt(n) at the end.
    """
    n = x.shape[0]
    h = 1
    while h < n:
        # Reshape so that pairs of length h are along the last dimension
        # shape => (n//(2h), h, 2)
        x_reshaped = x.view(-1, h, 2)
        a = x_reshaped[:, :, 0]
        b = x_reshaped[:, :, 1]
        # FWHT step: (a+b, a-b)
        x_reshaped[:, :, 0] = a + b
        x_reshaped[:, :, 1] = a - b
        x = x_reshaped.reshape(n)
        h <<= 1  # multiply h by 2

    if normalize:
        x /= (n ** 0.5)
    return x

##############################################################################
class StochasticQuantizationSender:
    def __init__(self, device=device):
        self.device = device
        self.prng = torch.Generator(device=device)

    def compress(self, data):
        self.prng.manual_seed(xxhash.xxh64(str(data["seed"])).intdigest() % 2 ** 16)

        data['vec'] = data['vec'].to(self.device)

        if data.get('step', 'standard') == "standard":
            step = (data['vec'].max() - data['vec'].min()) / (data['nlevels'] - 1)  # step size
        elif data["step"] == "norm":
            step = 1.4142135623730951 * torch.norm(data['vec']) / (data['nlevels'] - 1)  # step size
        else:
            raise Exception("unknown step size")

        r = (data['vec'] - data['vec'].min()) / step  # number of steps from the min
        r = torch.floor(r) + torch.bernoulli(r - torch.floor(r), generator=self.prng).to(self.device)  # sq

        return {'bins': r, 'min': data['vec'].min(), 'step': step}

##############################################################################

class StochasticQuantizationReceiver:
    def __init__(self, device=device):
        self.device = device

    def decompress(self, data):
        return data['min'] + data['bins'].to(self.device) * data['step']
 
##############################################################################   

class Hadamard:
    def __init__(self, device=device):
        self.device = device
        self.prng = torch.Generator(device=device)

    def hadamard(self, vec):
        vec = vec.to(self.device)
        d = vec.numel()
        if d & (d - 1) != 0:
            raise Exception("input numel must be a power of 2")

        h = 2
        while h <= d:
            hf = h // 2
            vec = vec.view(d // h, h)
            vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
            vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
            h *= 2
        vec /= np.sqrt(d)

        return vec.view(-1)

    def random_diagonal(self, size, seed):
        self.prng.manual_seed(seed)
        # Create a tensor of random 1s and -1s on the GPU
        return 2 * torch.bernoulli(torch.ones(size=(size,), device=self.device) / 2, generator=self.prng).to(self.device) - 1

##############################################################################
class HadamardSender(Hadamard):
    def __init__(self, device=device):
        super().__init__(device=device)

    def randomized_hadamard_transform(self, vec, seed):
        dim = vec.numel()
        vec = vec.to(self.device)

        if not dim & (dim - 1) == 0:
            padded_dim = int(2 ** (np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = vec

            padded_vec = padded_vec * self.random_diagonal(padded_vec.numel(), seed)
            padded_vec = self.hadamard(padded_vec)

            return padded_vec
        else:
            vec = vec * self.random_diagonal(vec.numel(), seed)
            vec = self.hadamard(vec)

            return vec
##############################################################################

class HadamardReceiver(Hadamard):
    def __init__(self, device=device):
        super().__init__(device=device)

    def randomized_inverse_hadamard_transform(self, vec, seed):
        vec = vec.to(self.device)
        vec = self.hadamard(vec)
        vec = vec * self.random_diagonal(vec.numel(), seed)

        return vec

##############################################################################

class HadamardStochasticQuantizationSender(HadamardSender, StochasticQuantizationSender):

    def __init__(self, device=device):
        super(HadamardSender, self).__init__(device=device)
        super(StochasticQuantizationSender, self).__init__(device=device)

    def compress(self, data):
        hvec = self.randomized_hadamard_transform(data['vec'], data['rotation_seed'])
        sq_data = StochasticQuantizationSender.compress(self, {
            'vec': hvec,
            'nlevels': data['nlevels'],
            'seed': data['seed'],
            'step': data.get('step', 'standard')
        })

        return {'data': sq_data, 'seed': data['seed'], 'rotation_seed': data['rotation_seed'], 'dim': data['vec'].numel()}

##############################################################################

class HadamardStochasticQuantizationReceiver(HadamardReceiver, StochasticQuantizationReceiver):

    def __init__(self, device=device):
        super(HadamardReceiver, self).__init__(device=device)
        super(StochasticQuantizationReceiver, self).__init__(device=device)

    def decompress(self, data):
        qhvec = StochasticQuantizationReceiver.decompress(self, data['data'])
        return self.randomized_inverse_hadamard_transform(qhvec, data['rotation_seed'])[:data['dim']]

##############################################################################

class KashinSender(HadamardSender, HadamardReceiver):
    def __init__(self, device, eta, delta, pad_threshold, niters):
        HadamardSender.__init__(self, device=device)
        HadamardReceiver.__init__(self, device=device)
        
        # Kashin-specific parameters
        self.eta = eta
        self.delta = delta
        self.pad_threshold = pad_threshold
        self.niters = niters

    def kashin_padded_dim(self, dim, pad_threshold):
        padded_dim = dim
        if not dim & (dim - 1) == 0:
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            if dim / padded_dim > pad_threshold:
                padded_dim = 2 * padded_dim
        else:
            padded_dim = 2 * dim
        return padded_dim

    def kashin_coefficients(self, data):
        orig_vec = data["vec"].clone().to(self.device)
        dim = orig_vec.numel()
        padded_dim = self.kashin_padded_dim(dim, self.pad_threshold)

        kashin_coefficients = torch.zeros(padded_dim, device=orig_vec.device)
        padded_x = torch.zeros(padded_dim, device=orig_vec.device)
        
        M = torch.norm(orig_vec) / np.sqrt(self.delta * padded_dim)
        for i in range(self.niters):
            padded_x[:] = 0
            padded_x[:dim] = orig_vec
            padded_x = self.randomized_hadamard_transform(padded_x, data['rotation_seed'])

            b = padded_x
            b_hat = torch.clamp(b, min=-M, max=M)
            kashin_coefficients += b_hat

            if i < self.niters - 1:
                b_hat = self.randomized_inverse_hadamard_transform(b_hat, data['rotation_seed'])
                orig_vec -= b_hat[:dim]
                M *= self.eta

            err = (data["vec"] - self.randomized_inverse_hadamard_transform(kashin_coefficients.clone(), data['rotation_seed'])[:dim]).norm(2) / orig_vec.norm(2)
            if err < data.get('err', 1e-6):
                break

        return kashin_coefficients, dim

##############################################################################

#####################################################################################
class KashinStochasticQuantizationSender(KashinSender, StochasticQuantizationSender):
    def __init__(self, device='cpu', eta=0.9, delta=1.0, pad_threshold=0.85, niters=3):
        KashinSender.__init__(self, device=device, eta=eta, delta=delta, pad_threshold=pad_threshold, niters=niters)
        StochasticQuantizationSender.__init__(self, device=device)

    def compress(self, data):
        self.niters = data.get('niters', 2147483647)
        self.err = data.get('err', 0.000001)

        kashin_coefficients, dim = self.kashin_coefficients(data)
        sq_data = StochasticQuantizationSender.compress(self, {
            'vec': kashin_coefficients,
            'nlevels': data['nlevels'],
            'seed': data['seed'],
            'step': data.get('step', 'standard')
        })

        return {'data': sq_data, 'seed': data['seed'], 'rotation_seed': data['rotation_seed'], 'dim': dim}

#####################################################################################
class KashinStochasticQuantizationReceiver(HadamardReceiver, StochasticQuantizationReceiver):
    def __init__(self, device='cpu'):
        HadamardReceiver.__init__(self, device=device)
        StochasticQuantizationReceiver.__init__(self, device=device)

    def decompress(self, data):
        # Decompress using the StochasticQuantizationReceiver's decompress method
        qhvec = StochasticQuantizationReceiver.decompress(self, data['data'])
        
        # Perform the randomized inverse Hadamard transform
        return self.randomized_inverse_hadamard_transform(qhvec, data['rotation_seed'])[:data['dim']]
#####################################################################################

def gen_cscale(centorids):
    res = 0
    prev = centorids[0]
    prev_mid = 0

    for curr in centorids:
        mid = (prev + curr) / 2
        res += prev ** 2 * (halfnorm.cdf(mid) - halfnorm.cdf(prev_mid))
        prev = curr
        prev_mid = mid

    res += curr ** 2 * (1 - halfnorm.cdf(mid))
    return res

#####################################################################################

def centroid_lookup_table(centroids, boundries, delta, device='cpu'):
    table_size = 2 * int(boundries.max() / delta + 2)
    table = torch.arange(-table_size // 2, table_size // 2, device=device) * delta + delta / 2
    bins = torch.bucketize(table, boundries)
    return bins

#####################################################################################

def gen_normal_centoirds_and_boundries(device='cpu', nbits='all'):
    centroids = {
        1: [0.7978845608028654],
        2: [0.4527800398860679, 1.5104176087114887],
        # add other centroids based on nbits
    }

    cscales = {i: gen_cscale(centroids[i]) for i in centroids}
    centroids = {i: torch.Tensor([-j for j in centroids[i][::-1]] + centroids[i]).to(device) for i in centroids}

    def gen_boundries(centorids):
        return [(a + b) / 2 for a, b in zip(centorids[:-1], centorids[1:])]

    boundries = {1: centroids[1][0]**2}
    boundries.update({i: torch.Tensor(gen_boundries(centroids[i])).to(device) for i in centroids})

    if nbits == 'all':
        return centroids, boundries, cscales

    return centroids[nbits], boundries[nbits], cscales[nbits]


class EdenSender(HadamardSender):
    def __init__(self, device=device, delta=None):
        super().__init__(device=device)
        self.centroids, self.boundries, self.cscales = gen_normal_centoirds_and_boundries(device)
        self.delta = delta

        if self.delta is not None:
            self.centroid_lookup_table = {
                b: centroid_lookup_table(self.centroids[b], self.boundries[b], delta)
                for b in self.centroids
            }

    def quantize(self, vec, nbits, cscale):
        vec = vec.to(self.device)
        if self.delta is not None:
            normalized_vec = vec * (vec.numel() ** 0.5) / torch.norm(vec, 2)
            lt_size = len(self.centroid_lookup_table[nbits])
            table_entries = torch.clamp(normalized_vec / self.delta + lt_size / 2, 0, lt_size - 1).long()
            bins = torch.take(self.centroid_lookup_table[nbits], table_entries)
        else:
            bins = torch.bucketize(vec * (vec.numel() ** 0.5) / torch.norm(vec, 2), self.boundries[nbits])

        if cscale:
            scale = torch.norm(vec, 2) / (self.cscales[nbits] * np.sqrt(vec.numel()))
        else:
            scale = torch.norm(vec, 2) ** 2 / torch.dot(torch.take(self.centroids[nbits], bins), vec)

        return bins, scale

    def stochastic_quantize(self, vec, nbits_low, nbits_high, p_high, seed):
        vec = vec.to(self.device)
        bins_low = torch.bucketize(vec * (vec.numel() ** 0.5) / torch.norm(vec, 2), self.boundries[nbits_low])
        bins_high = torch.bucketize(vec * (vec.numel() ** 0.5) / torch.norm(vec, 2), self.boundries[nbits_high])

        centroids_low = torch.take(self.centroids[nbits_low], bins_low)
        centroids_high = torch.take(self.centroids[nbits_high], bins_high)

        self.prng.manual_seed(seed)
        mask_high = torch.bernoulli(torch.ones(size=(vec.numel(),), device=self.device) * p_high, generator=self.prng).bool()

        bins_low.masked_scatter_(mask_high, torch.masked_select(bins_high, mask_high))
        centroids_low.masked_scatter_(mask_high, torch.masked_select(centroids_high, mask_high))

        scale = torch.norm(vec, 2) ** 2 / torch.dot(centroids_low, vec)

        return bins_low, scale

    def compress(self, data):
        data['vec'] = data['vec'].to(self.device)
        dim = data['vec'].numel()

        if not dim & (dim - 1) == 0:
            padded_dim = int(2 ** (np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = data['vec']
            vec = self.randomized_hadamard_transform(padded_vec, data['seed'])
        else:
            vec = self.randomized_hadamard_transform(data['vec'], data['seed'])

        if data['nbits'] == round(data['nbits']) or data['nbits'] < 1:
            bins, scale = self.quantize(vec, int(np.ceil(data['nbits'])), False)
            return {'bins': bins, 'scale': scale, 'seed': data['seed'], 'nbits': data['nbits'], 'dim': dim, 'pdrop': data.get('pdrop', 0)}
        else:
            bits_h = int(np.ceil(data['nbits']))
            bits_l = int(np.floor(data['nbits']))
            p_high = data['nbits'] - bits_l
            bins, scale = self.stochastic_quantize(vec, bits_l, bits_h, p_high, data['seed'] * 7 + 13)
            return {'bins': bins, 'scale': scale, 'seed': data['seed'], 'nbits': (bits_l, bits_h, p_high), 'dim': dim, 'pdrop': data.get('pdrop', 0)}
        
#####################################################################################
class EdenReceiver(HadamardReceiver):
    def __init__(self, device=device):
        super().__init__(device=device)
        self.centroids, _, _ = gen_normal_centoirds_and_boundries(device)

    def decompress(self, data):
        if not isinstance(data['nbits'], tuple):
            vec = torch.take(self.centroids[int(np.ceil(data['nbits']))], data['bins']).to(self.device)
        else:
            self.prng.manual_seed(data['seed'] * 7 + 13)
            mask_high = torch.bernoulli(
                torch.ones(size=(data['bins'].numel(),), device=self.device) * data['nbits'][2],
                generator=self.prng
            ).bool()
            vec = torch.take(self.centroids[int(data['nbits'][1])], data['bins']).to(self.device)
            vec.masked_scatter_(
                ~mask_high,
                torch.take(self.centroids[int(data['nbits'][0])], torch.masked_select(data['bins'], ~mask_high)).to(self.device)
            )

        pdrop = 0
        if not isinstance(data['nbits'], tuple) and data['nbits'] < 1:
            pdrop = 1 - data['nbits']
        if data['pdrop'] > 0:
            pdrop += (1 - pdrop) * data['pdrop']

        if pdrop > 0:
            dim = vec.numel()
            ri = torch.randperm(dim, device=self.device)[:round(dim * pdrop)]
            vec[ri] = 0
            vec /= (1 - pdrop)

        vec = self.randomized_inverse_hadamard_transform(vec, data['seed'])
        return (data['scale'] * vec)[:data['dim']].cpu()

##############################################################################
class QuicFLSender(HadamardSender):

    def __init__(self, device='cpu', bits=[1, 2, 3, 4], sr_bits=[6, 5, 4, 4], prefix=str(path) + '/tables/'):
        super().__init__(device=device)
        self.local_prng = torch.Generator(device=device)

        self.sender_table_X = {}
        self.sender_table_p = {}
        self.data = {}
        self.half_table_size = {}

        for i in range(4):
            fn = prefix + '{}_X_{}_h_256_q_'.format(bits[i], sr_bits[i])
            self.sender_table_X[bits[i]], self.sender_table_p[bits[i]], self.data[bits[i]] = self.sender_table(fn, device)
            self.half_table_size[bits[i]] = ((self.sender_table_X[bits[i]].numel() // self.data[bits[i]]['h_len']) - 1) * self.data[bits[i]]['h_len'] // 2

    ##########################################################################

    def sender_table(self, prefix, device):
        sender_table_X = torch.load(prefix + 'sender_table_X.pt').to(device)
        sender_table_p = torch.load(prefix + 'sender_table_p.pt').to(device)
        data = eval(open(prefix + 'data.txt').read())
        return sender_table_X, sender_table_p, data

    ##########################################################################

    def compress(self, data):
        dim = data['vec'].numel()
        prng_seed = xxhash.xxh64(str(data["seed"])).intdigest() % 2 ** 16
        self.local_prng.manual_seed(prng_seed)

        if not dim & (dim - 1) == 0:
            padded_dim = int(2 ** (np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = data['vec']
            vec = self.randomized_hadamard_transform(padded_vec, data['rotation_seed'])
            h = torch.randint(0, self.data[data['nbits']]['h_len'], (padded_dim,), device=self.device, generator=self.local_prng)
            scale = np.sqrt(padded_dim) / vec.norm(2)
        else:
            vec = self.randomized_hadamard_transform(data['vec'].to(self.device), data['rotation_seed'])
            h = torch.randint(0, self.data[data['nbits']]['h_len'], (dim,), device=self.device, generator=self.local_prng)
            scale = np.sqrt(dim) / vec.norm(2)

        rotated_and_scaled_vec = vec * scale
        #exact = (rotated_and_scaled_vec > self.data[data['nbits']]['T']) + (rotated_and_scaled_vec < -self.data[data['nbits']]['T'])
        
        def Q_inv(p):
         return norm.ppf(1-p)
        p_value = 2**(-8) 
        exact = (rotated_and_scaled_vec > Q_inv(p_value/2)) + (rotated_and_scaled_vec < -Q_inv(p_value/2))
        
        q_rotated_and_scaled_vec = rotated_and_scaled_vec / self.data[data['nbits']]['delta']
        q_rotated_and_scaled_vec[exact] = 0

        p = q_rotated_and_scaled_vec - q_rotated_and_scaled_vec.floor()
        int_q_rotated_and_scaled_vec = q_rotated_and_scaled_vec.floor() + torch.bernoulli(p, generator=self.local_prng)

        X = torch.take(self.sender_table_X[data['nbits']], (int_q_rotated_and_scaled_vec * self.data[data['nbits']]['h_len'] + h + self.half_table_size[data['nbits']]).long())
        p_X = torch.take(self.sender_table_p[data['nbits']], (int_q_rotated_and_scaled_vec * self.data[data['nbits']]['h_len'] + h + self.half_table_size[data['nbits']]).long())

        X += torch.bernoulli(p_X)
        X = X.long()

        return {
            'X': X,
            'exact_values': rotated_and_scaled_vec[exact],
            'exact_indeces': exact,
            'seed': data['seed'],
            'prng_seed': prng_seed,
            'rotation_seed': data['rotation_seed'],
            'dim': dim,
            'scale': scale,
            'nbits': data['nbits'],
            'h_len': self.data[data['nbits']]['h_len']
        }

##############################################################################

class QuicFLReceiver(HadamardReceiver):

    def __init__(self, device='cpu', bits=[1, 2, 3, 4], sr_bits=[6, 5, 4, 4], prefix=str(path) + '/tables/'):
        super().__init__(device=device)
        self.local_prng = torch.Generator(device=device)

        self.recv_table = {}
        for i in range(4):
            fn = prefix + '{}_X_{}_h_256_q_'.format(bits[i], sr_bits[i])
            self.recv_table[bits[i]] = self.receiver_table(fn, device)

    ##########################################################################

    def receiver_table(self, prefix, device):
        recv_table = torch.load(prefix + 'recv_table.pt').to(device)
        return recv_table

    ##########################################################################

    def decompress(self, data):
        self.local_prng.manual_seed(data['prng_seed'])
        h = torch.randint(0, data['h_len'], (data['X'].numel(),), device=self.device, generator=self.local_prng)

        client_rotated_and_scaled_vec = torch.take(self.recv_table[data['nbits']], (data['X'] * data['h_len'] + h))
        client_rotated_and_scaled_vec[data['exact_indeces']] = data['exact_values']
        client_rotated_and_scaled_vec /= data['scale']

        vec = self.randomized_inverse_hadamard_transform(client_rotated_and_scaled_vec, data['rotation_seed'])
        return vec[:data['dim']]

##############################################################################
##############################################################################
Type_quantize_algo_rate_l_dict = {1: 0.21403, 2: 0.63752, 3: 1.41725, 4: 2.91504, 5: 5.87195, 6: 11.76507, 7: 23.54075, 8: 47.0868, 9: 94.17625, 10: 188.35383}

one_sided_quant_vals_dict = {1: [0.7978845608028654],
                    2: [0.4527800398860679, 1.5104176087114887],
                    3: [0.24509416307340598, 0.7560052489539643, 1.3439092613750225, 2.151945669890335],
                    4: [0.12839501671105813, 0.38804823445328507, 0.6567589957631145, 0.9423402689122875,
                        1.2562309480263467, 1.6180460517130526, 2.069016730231837, 2.732588804065177],
                    5: [0.06588962234909321, 0.1980516892038791, 0.3313780514298761, 0.4666991751197207,
                        0.6049331689395434, 0.7471351317890572, 0.89456439585444, 1.0487823813655852, 1.2118032120324,
                        1.3863389353626248, 1.576226389073775, 1.7872312118858462, 2.0287259913633036,
                        2.3177364021261493, 2.69111557955431, 3.260726295605043],
                    6: [0.0334094558802581, 0.1002781217139195, 0.16729660990171974, 0.23456656976873475,
                        0.3021922894403614, 0.37028193328115516, 0.4389488009177737, 0.5083127587538033,
                        0.5785018460645791, 0.6496542452315348, 0.7219204720694183, 0.7954660529025513,
                        0.870474868055092, 0.9471530930156288, 1.0257343133937524, 1.1064859596918581,
                        1.1897175711327463, 1.2757916223519965, 1.3651378971823598, 1.458272959944728,
                        1.5558274659528346, 1.6585847114298427, 1.7675371481292605, 1.8839718992293555,
                        2.009604894545278, 2.146803022259123, 2.2989727412973995, 2.471294740528467, 2.6722617014102585,
                        2.91739146530985, 3.2404166403241677, 3.7440690236964755],
                    7: [0.016828143177728235, 0.05049075396896167, 0.08417241989671888, 0.11788596825032507,
                        0.1516442630131618, 0.18546025708680833, 0.21934708340331643, 0.25331807190834565,
                        0.2873868062260947, 0.32156710392315796, 0.355873075050329, 0.39031926330596733,
                        0.4249205523979007, 0.4596922300454219, 0.49465018161031576, 0.5298108436256188,
                        0.565191195643323, 0.600808970989236, 0.6366826613981411, 0.6728315674936343,
                        0.7092759460939766, 0.746037126679468, 0.7831375375631398, 0.8206007832455021,
                        0.858451939611374, 0.896717615963322, 0.9354260757626341, 0.9746074842160436,
                        1.0142940678300427, 1.054520418037026, 1.0953237719213182, 1.1367442623434032,
                        1.1788252655205043, 1.2216138763870124, 1.26516137869917, 1.309523700469555, 1.3547621051156036,
                        1.4009441065262136, 1.448144252238147, 1.4964451375010575, 1.5459387008934842,
                        1.596727786313424, 1.6489283062238074, 1.7026711624156725, 1.7581051606756466,
                        1.8154009933798645, 1.8747553268072956, 1.9363967204122827, 2.0005932433837565,
                        2.0676621538384503, 2.1379832427349696, 2.212016460501213, 2.2903268704925304,
                        2.3736203164211713, 2.4627959084523208, 2.5590234991374485, 2.663867022558051,
                        2.7794919110540777, 2.909021527386642, 3.0572161028423737, 3.231896182843021,
                        3.4473810105937095, 3.7348571053691555, 4.1895219330235225],
                    8: [0.008445974137017219, 0.025338726226901278, 0.042233889994651476, 0.05913307399220878,
                        0.07603788791797023, 0.09294994306815242, 0.10987089037069565, 0.12680234584461386,
                        0.1437459285205906, 0.16070326074968388, 0.1776760066764216, 0.19466583496246115,
                        0.21167441946986007, 0.22870343946322488, 0.24575458029044564, 0.2628295721769575,
                        0.2799301528634766, 0.29705806782573063, 0.3142150709211129, 0.3314029639954903,
                        0.34862355883476864, 0.3658786774238477, 0.3831701926964899, 0.40049998943716425,
                        0.4178699650069057, 0.4352820704086704, 0.45273827097956804, 0.4702405882876,
                        0.48779106011037887, 0.505391740756901, 0.5230447441905988, 0.5407522460590347,
                        0.558516486141511, 0.5763396823538222, 0.5942241184949506, 0.6121721459546814,
                        0.6301861414640443, 0.6482685527755422, 0.6664219019236218, 0.684648787627676,
                        0.7029517931200633, 0.7213336286470308, 0.7397970881081071, 0.7583450032075904,
                        0.7769802937007926, 0.7957059197645721, 0.8145249861674053, 0.8334407494351099,
                        0.8524564651728141, 0.8715754936480047, 0.8908013031010308, 0.9101374749919184,
                        0.9295877653215154, 0.9491559977740125, 0.9688461234581733, 0.9886622867721733,
                        1.0086087121824747, 1.028689768268861, 1.0489101021225093, 1.0692743940997251,
                        1.0897875553561465, 1.1104547388972044, 1.1312812154370708, 1.1522725891384287,
                        1.173434599389649, 1.1947731980672593, 1.2162947131430126, 1.238005717146854,
                        1.2599130381874064, 1.2820237696510286, 1.304345369166531, 1.3268857708606756,
                        1.349653145284911, 1.3726560932224416, 1.3959037693197867, 1.419405726021264,
                        1.4431719292973744, 1.4672129964566984, 1.4915401336751468, 1.5161650628244996,
                        1.541100284490976, 1.5663591473033147, 1.5919556551358922, 1.6179046397057497,
                        1.6442219553485078, 1.6709244249695359, 1.6980300628044107, 1.7255580190748743,
                        1.7535288357430767, 1.7819645728459763, 1.81088895442524, 1.8403273195729115, 1.870306964218662,
                        1.9008577747790962, 1.9320118435829472, 1.9638039107009146, 1.9962716117712092,
                        2.0294560760505993, 2.0634026367482017, 2.0981611002741527, 2.133785932225919,
                        2.170336784741086, 2.2078803102947337, 2.2464908293749546, 2.286250990303635, 2.327254033532845,
                        2.369604977942217, 2.4134218838650208, 2.458840003415269, 2.506014300608167, 2.5551242195294983,
                        2.6063787537827645, 2.660023038604595, 2.716347847697055, 2.7757011083910723, 2.838504606698991,
                        2.9052776685316117, 2.976670770545963, 3.0535115393558603, 3.136880130166507,
                        3.2282236667414654, 3.3295406612081644, 3.443713971315384, 3.5751595986789093,
                        3.7311414987004117, 3.9249650523739246, 4.185630113705256, 4.601871059539151]}
#################################################################################################################

######################################################### Quantization Functions #########################################################

def Type_unbiased_quantize(input_vector, bits_per_dimension=1): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device=device)

    # Table from original code
    type_quantize_algo_rate_l_dict = {
        0.5: 0.08282,  1: 0.21403,  1.5: 0.39443,  2: 0.63752,
        2.5: 0.96656,  3: 1.41725,  3.5: 2.04187,  4: 2.91504,
        4.5: 4.14217,  5: 5.87195,  5.5: 8.31416,  6: 11.76507,
        6.5: 16.64332, 7: 23.54075, 7.5: 33.29414, 8: 47.0868,
        8.5: 66.59204, 9: 94.17625, 9.5: 133.18596, 10: 188.35383
    }

    d = input_vector.numel()
    m = int(type_quantize_algo_rate_l_dict[bits_per_dimension] * d)
    L1_norm = input_vector.abs().sum()
    v = input_vector / (L1_norm + 1e-12)  # avoid /0 in degenerate case
    p = v.abs()

    # Floor and fractional part
    mp = m * p
    floor_mp = torch.floor(mp)
    frac_mp = mp - floor_mp

    # Single random scalar X
    X = torch.rand(1, device=device)
    c = torch.cat([torch.zeros(1, device=device), frac_mp.cumsum(dim=0)], dim=0)
    diff = torch.floor(c[1:] - X) - torch.floor(c[:-1] - X)
    r = (diff == 1).float()

    # Combine sign, floor_mp, r
    quantized_vector = L1_norm * v.sign() * (floor_mp + r) / m
    return quantized_vector


def Reznik(p, m):
    """
    Vectorized Reznick rounding using partial top-k instead of full sorting.
    """
    k_prime = torch.floor(m * p + 0.5)
    m_prime = k_prime.sum()

    if m_prime == m:
        return k_prime / m

    delta_prime = k_prime - m * p
    Delta = int(m_prime - m)

    if Delta > 0:
        # Adjust the largest Delta elements
        _, idx = torch.topk(delta_prime, Delta)
        k_prime[idx] -= 1
    else:
        # Adjust the smallest -Delta elements
        _, idx = torch.topk(-delta_prime, -Delta)
        k_prime[idx] += 1

    return k_prime / m


def Type_biased_quantize(input_vector, bits_per_dimension=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device=device)
    type_quantize_algo_rate_l_dict = {
        0.5: 0.08282, 1: 0.21403, 1.5: 0.39443, 2: 0.63752, 2.5: 0.96656, 3: 1.41725,
        3.5: 2.04187, 4: 2.91504, 4.5: 4.14217, 5: 5.87195, 5.5: 8.31416, 6: 11.76507,
        6.5: 16.64332, 7: 23.54075, 7.5: 33.29414, 8: 47.0868, 8.5: 66.59204, 9: 94.17625,
        9.5: 133.18596, 10: 188.35383
    }

    quantized_vector = input_vector.clone()
    d = quantized_vector.size(0)
    L1_norm = quantized_vector.abs().sum()
    signs = quantized_vector.sign()
    p = quantized_vector.abs() / (L1_norm + 1e-12)
    m = int(type_quantize_algo_rate_l_dict[bits_per_dimension] * d)
    p_hat = Reznik(p, m)

    return L1_norm * signs * p_hat

def walsh_hadamard_transform(input_vec):
    arr = input_vec.clone().to(input_vec.device)
    n = arr.size(0)
    
    if n < 2:
        return arr

    h = 2
    while h <= n:
        hf = h // 2
        for i in range(0, n, h):
            for j in range(hf):
                u, v = arr[i + j], arr[i + j + hf]
                arr[i + j], arr[i + j + hf] = u + v, u - v
        h *= 2

    return arr

def DRIVE_quantize_Hadamard(input_vector, bits_per_dimension=1):
    """
    Example with chunking preserved to minimize memory usage.
    Replaces the old nested loops with fast_walsh_hadamard_transform calls.
    """
    batch_size = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device=device)
    quantized_vector = input_vector.clone()
    d = quantized_vector.size(0)
    num_batches = int(np.ceil(d / batch_size))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, d)
        vec = quantized_vector[start_idx:end_idx].clone()

        # Pad to next power-of-2 if needed for the hadamard
        batch_len = vec.size(0)
        size_power2 = 1
        while size_power2 < batch_len:
            size_power2 <<= 1

        if size_power2 > batch_len:
            pad_extra = size_power2 - batch_len
            vec = torch.cat([vec, torch.zeros(pad_extra, device=vec.device)])

        # 1) random sign
        D = (torch.rand_like(vec) > 0.5).float() * 2 - 1
        # 2) forward hadamard
        Rx = D * vec
        Rx = fast_walsh_hadamard_transform(Rx, normalize=False)

        # 3) scaling
        S = (vec[:batch_len].norm(2).pow(2)) / (Rx.abs().sum() + 1e-12)
        # 4) sign quant
        Rx_hat = S * Rx.sign()

        # 5) inverse hadamard
        out = fast_walsh_hadamard_transform(Rx_hat, normalize=False)
        out = out * D

        # remove any padding
        quantized_vector[start_idx:end_idx] = out[:batch_len]

    return quantized_vector

# Similar updates are applied to the rest of the functions, ensuring device compatibility and consistency.
def Scalar_quantize(input_vector, bits_per_dimension=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device=device)
    q = input_vector.clone()
    x_min = q.min()
    x_max = q.max()
    denom = x_max - x_min
    if denom == 0:
        # No variation => nothing to quantize, just return original
        return q

    # Normalize to [0,1]
    q = (q - x_min) / denom

    # Number of equally spaced intervals minus 1
    nlevels = 2 ** bits_per_dimension - 1
    if nlevels < 1:
        return input_vector

    # Clamp values to the [0,1] range to avoid floating round-off
    q.clamp_(0, 1)

    # Determine bucket index
    bin_floor = torch.floor(q * nlevels)
    frac = q * nlevels - bin_floor

    # Bernoulli draw for each element to decide whether to "round up"
    rnd = torch.rand_like(frac)
    bin_idx = bin_floor + (rnd < frac).to(bin_floor.dtype)

    # Map bin indices back into [0, 1], then scale to [x_min, x_max]
    q = bin_idx / nlevels
    q = q * (x_max - x_min) + x_min

    return q

def EDEN_quantize_Hadamard(input_vector, bits_per_dimension=1):
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    quantized_vector = input_vector.clone()
    d = quantized_vector.size(0)

    data = {
        'vec': quantized_vector,
        'seed': int(torch.randint(0, 100, (1,), device=quantized_vector.device).item()),
        'nbits': bits_per_dimension,
        'rotation_seed': 123,
        'nlevels': 2 ** bits_per_dimension
    }

    EDEN_sender = EdenSender(device=quantized_vector.device)
    EDEN_receiver = EdenReceiver(device=quantized_vector.device)

    data = EDEN_sender.compress(data)
    client_received_vec = EDEN_receiver.decompress(data)

    return client_received_vec.cpu().numpy()

def QUICFL_quantize(input_vector, bits_per_dimension=1):
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    quantized_vector = input_vector.clone()
    d = quantized_vector.size(0)

    data = {
        'vec': quantized_vector,
        'seed': int(torch.randint(0, 100, (1,), device=quantized_vector.device).item()),
        'nbits': bits_per_dimension,
        'rotation_seed': 123,
        'nlevels': 2 ** bits_per_dimension
    }

    quicfl_sender = QuicFLSender(device=quantized_vector.device)
    quicfl_receiver = QuicFLReceiver(device=quantized_vector.device)

    data = quicfl_sender.compress(data)
    client_received_vec = quicfl_receiver.decompress(data)

    return client_received_vec.cpu().numpy()

def Kashin_quantize(input_vector, bits_per_dimension=1): 
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    quantized_vector = input_vector.clone()
    d = quantized_vector.size(0)

    data = {
        'vec': quantized_vector,
        'seed': int(torch.randint(0, 100, (1,), device=device).item()),
        'nbits': bits_per_dimension,
        'rotation_seed': 123,
        'nlevels': 2 ** bits_per_dimension,
        'niters': 3
    }

    kashin_sender = KashinStochasticQuantizationSender(device=device)
    kashin_receiver = KashinStochasticQuantizationReceiver(device=device)

    data = kashin_sender.compress(data)
    client_received_vec = kashin_receiver.decompress(data)

    return client_received_vec.cpu().numpy()

def No_quantize(input_vector, bits_per_dimension=1):
    # Explicitly convert to torch tensor, regardless of input type
    input_vector = torch.tensor(input_vector, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    return input_vector
#############################################################################################################################
