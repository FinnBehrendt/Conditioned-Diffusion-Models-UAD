import torch
import torch.nn as nn
class L1_AE(torch.nn.Module):
    def __init__(self, cfg) :
        super().__init__()
        self.strat = cfg.lossStrategy

    def forward(self, output_batch, input_batch) :
        if isinstance(output_batch, dict):
            output_batch = output_batch['x_hat']
        else: 
            output_batch = output_batch
        if self.strat == 'sum' :
            L1Loss = nn.L1Loss(reduction = 'sum') 
            L1 = L1Loss(output_batch, input_batch)/input_batch.shape[0]
        elif self.strat == 'mean' :
            L1Loss = nn.L1Loss(reduction = 'mean') 
            L1 = L1Loss(output_batch, input_batch)
        loss = {}
        loss['combined_loss'] = L1
        loss['reg'] = L1 # dummy
        loss['recon_error'] = L1 
        return loss 
