from typing import Sequence
import numpy as np
import scipy
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F

import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, remove_diag
from calibloss import NodewiseECE
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit_calibration(temp_model, data, train_mask, test_mask, args, patience=100):
    """
    Train calibrator
    """    
    ece_fn = NodewiseECE(test_mask, args.n_bins, 'equal_width', 1)
    vlss_mn = float('Inf')
    # 1. Obtain uncalibrated logits from trained GNN.
    with torch.no_grad():
        logits = temp_model.model(data.x, data.adj_t) # uncalibrated logits
        _, preds = F.softmax(logits, dim=1).max(1)
        labels = data.y
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    
    # 2. Train calibrator.
    for epoch in range(2000):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        temp_model.model.eval() ### Post-hoc calibration set the classifier to the evaluation mode
        assert not temp_model.model.training
        
        logits_scaled, binconf, binacc = temp_model.propagate(logits, labels)

        loss_cls = F.cross_entropy(logits_scaled[train_mask], labels[train_mask])
        
        ### bin-wise reg.
        loss_lcc = torch.norm(binconf - binacc) ** 2
        
        loss = loss_cls + args.lamb * loss_lcc
        loss.backward()
        
        temp_model.optimizer.step()
        
        with torch.no_grad():
            temp_model.eval()
            logits_scaled, _, _ = temp_model.propagate(logits, labels)
            val_ece = ece_fn(F.log_softmax(logits_scaled, dim=1), labels)
            if val_ece <= vlss_mn:
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = val_ece
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
                  
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)


class Simi_Mailbox(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        
        self.temperature = nn.Parameter(torch.ones(args.n_simi_bins))
        
        self.train_mask = data.train_mask.clone()
        self.val_mask = data.val_mask.clone()
        self.test_mask = data.test_mask.clone()
        
        self.centroids = None
        self.bin_assignments = None
        
    def compute_mailbox(self, logits, data):
        num_nodes, num_classes = logits.shape
        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, 1)
        
        tmp_adj = remove_diag(data.adj_t)
        deg = torch_sparse.sum(tmp_adj, 1)
        deg_inv = deg.pow(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        
        row, col, _ = tmp_adj.coo()
        sim = torch.sigmoid(torch.sum(logits[row] * logits[col], 1))
        edge_index = torch.stack((row, col))
        
        tmp_adj = SparseTensor.from_edge_index(edge_index, sim,
                                        sparse_sizes=(num_nodes, num_nodes))
        
        del (edge_index, sim)
        torch.cuda.empty_cache()
        
        avg_sim = torch_sparse.sum(tmp_adj, 1) * deg_inv
        # Perform Min-Max scaling
        min_avg_sim = torch.min(avg_sim)
        max_avg_sim = torch.max(avg_sim)
        avg_sim = (avg_sim - min_avg_sim) / (max_avg_sim - min_avg_sim)

        min_conf = torch.min(confs)
        max_conf = torch.max(confs)
        confs = (confs - min_conf) / (max_conf - min_conf)
        
        mailbox = torch.stack((confs, avg_sim), 1)
        
        kmeans = KMeans(n_clusters=self.args.n_simi_bins, random_state=0, 
                        n_init='auto').fit(mailbox.detach().cpu().numpy())
        
        self.centroids = kmeans.cluster_centers_ # (n_clusters, 2)
        self.bin_assignments = torch.from_numpy(kmeans.labels_).to(device).type(torch.long)
        return mailbox

    def forward(self, logits):
        T = self.temperature[self.bin_assignments]
        T = F.relu(T) + 1e-8
        temperature = T.unsqueeze(1).expand(logits.size(0), logits.size(1))
        
        logits_scaled = logits / temperature
        return logits_scaled
    
    def propagate(self, logits, labels):
        T = self.temperature[self.bin_assignments]
        T = F.relu(T) + 1e-8
        temperature = T.unsqueeze(1).expand(logits.size(0), logits.size(1))
        
        logits_scaled = logits / temperature
        probs_scaled = F.softmax(logits_scaled, dim=1)
        confs_scaled, preds = torch.max(probs_scaled, 1)
        accs = (preds == labels)
        
        binconf = torch.zeros(self.args.n_simi_bins).to(device)
        binacc = torch.zeros(self.args.n_simi_bins).to(device)
        for i in range(self.args.n_simi_bins):
            mask = self.bin_assignments == i
            mask_acc = mask & self.val_mask
            if mask.sum() > 0:
                binconf[i] = confs_scaled[mask].mean()
                if mask_acc.sum() > 0:
                    binacc[i] = accs[mask].float().mean()
        return logits_scaled, binconf, binacc

    def fit(self, data, train_mask, test_mask, lr, wdecay):
        self.to(device)
        self.train_params = [self.temperature]
        self.optimizer = optim.Adam(self.train_params, lr=lr, weight_decay=wdecay)
        
        fit_calibration(self, data, train_mask, test_mask, self.args)
        return self
    

