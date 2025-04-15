import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from topoloss import TopoLoss, LaplacianPyramid
from dataclasses import dataclass

class Base_Enc_Dec_Cfg:
    def __init__(self):
        self.batchsize = 128
        self.learning_rate = 0.01
        self.num_epochs = 25
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = optim.Adam
        self.loss = nn.MSELoss()

class Topo_Enc_Dec_Cfg:
    def __init__(self):
        self.batchsize = 128
        self.learning_rate = 0.01
        self.num_epochs = 25
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = optim.Adam
        self.loss = TopoLoss(
        losses=[
        LaplacianPyramid(layer_name="encoder", factor_h = 4, factor_w = 4, scale = 3.0)
    ]
)

class Topo_SAE_Cfg:
    def __init__(self):
        self.batchsize = 128
        self.learning_rate = 0.01
        self.num_epochs = 25
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = optim.Adam
        self.tied_weights = False
        self.l1 = 2 #SPARITY PENALTY
        self.l2 = 4 #WEIGHT PENALTY
        self.loss = nn.MSELoss() #CHANGE TO CUSTOM

class Toy_SAE_Cfg:
    def __init__(self):
        self.batchsize = 128
        self.learning_rate = 0.01
        self.num_epochs = 25
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = optim.Adam
        self.tied_weights = False
        self.l1 = 2 #SPARITY PENALTY
        self.l2 = 4 #WEIGHT PENALTY
        self.loss = nn.MSELoss() #CHANGE TO CUSTOM


@dataclass
#how to train an SAE?
class SAE_Cfg:
    def __init__(self):
        self.batch_size: int = 64
        self.learning_rate: float = 0.01
        self.num_epochs: int = 5
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_feats: int = 768
        self.out_feats: int = 8 * 768
        self.optimizer: optim = optim.Adam
        self.l1_coeff = 0.2 #sparsity penalty
        self.topo_loss = TopoLoss(
        losses=[
        LaplacianPyramid(layer_name="enc", factor_h = 4, factor_w = 4, scale = 3.0)
    ]
)
    


    # wandb_log = False
    # wandb_project = 'nesim-nanogpt'
    # effective_batch_size = 512
    # batch_size = 64
    # block_size = 1024
    # num_tokens_per_step = 524288
    # num_tokens_total = 9949090040
    # gradient_accumulation_steps = 8
    # max_iters = 18976
    # lr_decay_iters = 18976
    # eval_interval = 1000
    # eval_iters = 200
    # log_interval = 10
    # dataset = 'fineweb-edu-10B'
    # init_from = 'scratch'
    # eval_only = False
    # always_save_checkpoint = True
    # weight_decay = 0.1
    # n_layer = 12
    # n_head = 12
    # n_embd = 768
    # dropout = 0.0
    # bias = False
    # learning_rate = 0.0006
    # beta1 = 0.9
    # beta2 = 0.95
    # grad_clip = 1.0
    # decay_lr = True
    # warmup_iters = 2000
    # min_lr = 6e-05
    # backend = 'nccl'
    # device = 'cuda'
    # dtype = 'bfloat16'
    # compile = True
