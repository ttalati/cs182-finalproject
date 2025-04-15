import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
#import tiktoken
#from torch.nn.utils.rnn import pad_sequence
from topoloss import TopoLoss, LaplacianPyramid
from dataclasses import dataclass
from topoloss.cortical_sheet.output import get_cortical_sheet_linear
#import einops
import transformer_lens
import toponets
from datasets import load_dataset
#import math
import argparse
from networks import Sparse_AutoEnc_ReLU, Sparse_AutoEnc_TopK, Sparse_AutoEnc_NoAct
import wandb
import random


parser = argparse.ArgumentParser(description = "Script for training SAE")
parser.add_argument("--batch_size", type=int, default = 8) #might have to change -> 8/16
parser.add_argument("--learning_rate", type=float, default = 0.0001)
parser.add_argument("--num_epochs", type=int, default = 1)
parser.add_argument("--dataset", type=str, default = "fine_web") #change which dataset to use
parser.add_argument("--l1_coeff", type=float, default = 3e-4) #increasing sparsity regularization
parser.add_argument("--topo_loss_scale", type=float, default = 2.0)
parser.add_argument("--hidden_dim_multiplier", type=int, default = 2)
parser.add_argument("--tied_weights", type=bool, default = False)
parser.add_argument("--sae_type", type=str, default = "relu")
parser.add_argument("--base_model", type=str, default = "1_layer")
parser.add_argument("--save_dir", type=str, default = "./sae_checkpoints")
parser.add_argument("--plot_dir", type=str, default = "./training_plots")
parser.add_argument("--sae_activations_dir", type=str, default = "./sae_activations")

args = parser.parse_args()


if __name__ == "__main__":
    
    wandb.login()

    run = wandb.init(
    project="topo-sae",
    config=vars(args),
    )

    np.random.seed(30)

    if args.base_model == "1_layer":
        if args.sae_type == "relu":
            sae = Sparse_AutoEnc_ReLU(in_feats=512, out_feats=512*args.hidden_dim_multiplier)
        elif args.sae_type == "topk":
            sae = Sparse_AutoEnc_TopK(in_feats=512, out_feats=512*args.hidden_dim_multiplier, tied_weights=args.tied_weights)
        elif args.sae_type == "no_act":
            sae = Sparse_AutoEnc_NoAct(in_feats=512, out_feats=512*args.hidden_dim_multiplier)
        base_model = transformer_lens.HookedTransformer.from_pretrained("gelu-1l")
    elif args.sae_type == "topo_nanogpt":
        #to implement later
        pass
    if args.dataset == "fine_web":
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    if args.dataset == "pile":
        pass
    
    optim = torch.optim.Adam(sae.parameters())
    train_data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False) #TODO: shuffle handled by buffer
    metrics = {
        'total': [],
        'recon': [],
        'sparsity': [],
        'topo': []
    }
    activations_cache = {
        'activations': [],
        'data': []
    }

    plt.switch_backend('Agg')
    fig_total, ax_total = plt.subplots(figsize=(10, 6))
    fig_recon, ax_recon = plt.subplots(figsize=(10, 6))
    fig_sparsity, ax_sparsity = plt.subplots(figsize=(10, 6))
    fig_topo, ax_topo = plt.subplots(figsize=(10, 6))
    plot_path = args.plot_dir + f"/{args.base_model}_{args.sae_type}_lr_{args.learning_rate}_l1_{args.l1_coeff}_topo_{args.topo_loss_scale}"
    chkpt_path = args.save_dir + f"/{args.base_model}_{args.sae_type}_lr_{args.learning_rate}_l1_{args.l1_coeff}_topo_{args.topo_loss_scale}"
    activations_path = args.sae_activations_dir + f"/{args.base_model}_{args.sae_type}_dataset_{args.dataset}_lr_{args.learning_rate}_l1_{args.l1_coeff}_topo_{args.topo_loss_scale}"
    for diff_path in [args.save_dir, args.plot_dir, args.sae_activations_dir, plot_path, chkpt_path]:
        if not os.path.exists(diff_path):
            os.makedirs(diff_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topoloss = TopoLoss(losses=[LaplacianPyramid(layer_name="enc", factor_h = 4, factor_w = 4, scale = args.topo_loss_scale)])
    base_model.to(device)
    sae.to(device)
    for epoch in tqdm(range(args.num_epochs)): #look at gpu memory
        for index, data in tqdm(enumerate(train_data)):
            #tokenize data
            tokens = base_model.to_tokens(data['text'])
            tokens = tokens.to(device)

            #get intermediate activations
            out, cache = base_model.run_with_cache(tokens)
            inp = cache['blocks.0.hook_mlp_out'].clone().detach()
            
            #train sae
            sae.train()
            inp = inp.to(device)
            pred, enc, inp = sae(inp)

            #calculate loss
            optim.zero_grad()
            l_reconstruction = (pred.float() - inp.float()).pow(2).sum(-1).mean(0).mean(0) #divide by batchsize
            l_sparsity = (enc.float().abs().sum())
            l_topo = topoloss.compute(sae)
            l = l_reconstruction + (args.l1_coeff*l_sparsity) + l_topo
            
            #backwards pass
            l.backward()
            optim.step()

            if index % 1000 == 0 and index != 0:
                wandb.log({"loss": l_reconstruction.item(), "reconstruction_loss": l_reconstruction.item(), "sparsity loss": l_sparsity.item(), "topo loss": l_topo.item()})
                print(f'total loss: {l.item()}, reconstruction loss: {l_reconstruction.item()}, sparsity loss: {l_sparsity.item()}, topo loss: {l_topo.item()}')
                
                metrics['total'].append(l.item())
                metrics['recon'].append(l_reconstruction.item())
                metrics['sparsity'].append(l_sparsity.item())
                metrics['topo'].append(l_topo.item())
                
                #put in utils...
                for iter, loss_type in enumerate(['total', 'recon', 'sparsity', 'topo']):
                    fig = locals()[f'fig_{loss_type}']
                    ax = locals()[f'ax_{loss_type}']
                    
                    ax.clear()
                    ax.plot(metrics[loss_type], color=f'C{iter}')
                    ax.set_title(f"{loss_type.capitalize()} Loss (Epoch {epoch}, Batch {index})")
                    ax.set_xlabel("Batch")
                    ax.set_ylabel("Loss")
                    ax.grid(True)
                    
                    path = plot_path + f'/{loss_type}_loss.png'

                    fig.savefig(path)
                
                activation = enc.cpu().detach().clone()
                activations_cache['activations'].append(activation)
                activations_cache['data'].append(data)
                torch.save(activations_cache, activations_path)

            if index % 10_000 == 0 and index != 0:
                path = chkpt_path + f"/iter_{index}"
                state = {'sae': sae.state_dict(), 'optim': optim.state_dict(), 'config':vars(args)}
                torch.save(state, path)
