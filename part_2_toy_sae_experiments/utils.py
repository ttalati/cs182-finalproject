import torch
import numpy as np

def plot_neuron_activations(sae_activations, sae_type, hidden_layer_dim):
    '''
    plots the neuron activation densities/frequencies of our SAE
        sae_activations: iterator through desired number of SAE activations, [1, C, D]
        sae_type: i.e. relu SAE, topK SAE, etc.
    '''
    activation_densities = np.zeroes((hidden_layer_dim))
    for iter, activation in enumerate(sae_activations):
        if sae_type == 'relu':
            activation.squeeze_(0) #shape = [C, D]
            active = (activation > 0).mean(dim=0)
        elif sae_type == 'topk':
            in_top_k = torch.isclose(activation, torch.zeros(activation.shape), rtol=1e4)
            active = in_top_k.sum(0)
        
        activation_densities = activation_densities * (iter - 1 / iter) #to finish!




