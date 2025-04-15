import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops
import config



class Sparse_AutoEnc_ReLU(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        torch.manual_seed(30)
        self.enc = nn.Linear(self.in_feats, self.out_feats)
        self.dec = nn.Linear(self.out_feats, self.in_feats)
        
        #normalize decoder weights
        with torch.no_grad():
          self.dec.weight.data = F.normalize(self.dec.weight.data, p=2, dim=1)

    def forward(self, x):        
        # Use the captured activation as input to the autoencoder
        x_inp = x - self.dec.bias
        encoded_x = F.relu(self.enc(x_inp))
        decoded_x = self.dec(encoded_x)
        return decoded_x, encoded_x, x


class Sparse_AutoEnc_TopK(nn.Module):
    def __init__(self, in_feats, out_feats, tied_weights):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.enc = nn.Linear(self.in_feats, self.out_feats)
        if tied_weights:
          dec_weights = self.enc.weight.data.transpose(0, 1)
          self.dec = nn.Linear(self.out_feats, self.in_feats)
          self.dec.weight.data = dec_weights
        else:
          self.dec = nn.Linear(self.out_feats, self.in_feats)
        # self.dec = nn.Linear(self.out_feats, self.in_feats)


    def forward(self, x: torch.Tensor):
      """
      Args:
        x: tensor of shape [Batchsize, In_dim]
      """
      emb = self.enc(x)
      vals, idx = torch.topk(emb, 128, dim=-1)
      mask = torch.zeros_like(emb, dtype=torch.bool)  # Shape: [2, 256, 6144]
      mask.scatter_(dim=-1, index=idx, src=torch.ones_like(mask, dtype=torch.bool))
      encoded_x = emb * mask
      decoded_x = self.dec(encoded_x)
      return decoded_x, encoded_x, x

#removing the relu as a sanity check to verify nothing wrong with code
class Sparse_AutoEnc_NoAct(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        torch.manual_seed(30)
        self.enc = nn.Linear(self.in_feats, self.out_feats)
        self.dec = nn.Linear(self.out_feats, self.in_feats)
        
        #normalize decoder weights
        with torch.no_grad():
          self.dec.weight.data = F.normalize(self.dec.weight.data, p=2, dim=1)

    def forward(self, x):        
        # Use the captured activation as input to the autoencoder
        x_inp = x - self.dec.bias
        encoded_x = self.enc(x_inp) #remove relu as a sanity check
        decoded_x = self.dec(encoded_x)
        return decoded_x, encoded_x, x


# class Sparse_AutoEnc(nn.Module):
#     def __init__(self, cfg, base_model):
#         super().__init__()
#         self.cfg = cfg
#         self.in_feats = cfg.in_feats
#         self.base_model = base_model
        
#         # Register forward hook to capture intermediate output
#         self.intermediate_output = None  # Stores activation from base_model
#         self._register_hook()  # Initialize hook
        
#         # Autoencoder layers
#         self.out_feats = cfg.out_feats
#         self.enc = nn.Linear(self.in_feats, self.out_feats)
#         self.dec = nn.Linear(self.out_feats, self.in_feats)

#     def _forward_hook(self, module, input, output):
#         """Hook function to retain intermediate output."""
#         self.intermediate_output = output.detach()  # Detach to avoid gradients

#     def _register_hook(self):
#         """Attach hook to the target layer in base_model."""
#         target_layer = self.base_model.transformer.h[-1].mlp.c_proj
#         self.hook_handle = target_layer.register_forward_hook(self._forward_hook)

#     def forward(self, x):
#         # Run base_model to trigger the hook
#         _ = self.base_model(x)  # Intermediate output captured in self.intermediate_output
        
#         # Use the captured activation as input to the autoencoder
#         encoded_x = F.relu(self.enc(self.intermediate_output))
#         decoded_x = self.dec(encoded_x)
#         return decoded_x, encoded_x

#     def remove_hook(self):
#         """Optional: Cleanup hook to prevent memory leaks."""
#         self.hook_handle.remove()



class Toy_Enc_Dec(nn.Module):
    def __init__(self, in_dim: int = 784, hidden_layers: list[int] = [128, 64], cfg: config.Base_Enc_Dec_Cfg=None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.flatten = nn.Flatten()
        self.encoder = []
        self.encoder.append(nn.Linear(in_dim, hidden_layers[0]))
        self.encoder.append(nn.ReLU())
        hd = 1
        while hd < len(hidden_layers):
            self.encoder.append(nn.Linear(hidden_layers[hd-1], hidden_layers[hd]))
            self.encoder.append(nn.ReLU())
            hd += 1
        self.encoder = nn.Sequential(*self.encoder)
        hd -= 1
        self.decoder = []
        while hd > 0:
            self.decoder.append(nn.Linear(hidden_layers[hd], hidden_layers[hd-1]))
            self.decoder.append(nn.ReLU())
            hd -= 1
        self.decoder.append(nn.Linear(hidden_layers[0], in_dim))
        self.decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.decoder)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def reconstruct_im(self, x: torch.Tensor) -> None:
        x = x.detach().numpy()
        x = x[0]
        x = einops.rearrange(x, '(h w) -> h w', h=int(self.in_dim**0.5))
        plt.imshow(x, cmap='gray')
        plt.show()


class Topo_Enc_Dec(Toy_Enc_Dec):
    def __init__(self):
        super().__init__()
















