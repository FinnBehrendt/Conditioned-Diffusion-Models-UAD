from src.models.modules.spark.Spark_2D import SparK_2D 
from src.models.losses import L1_AE
import torch
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any
import torchio as tio


class Spark_2D(LightningModule):
    def __init__(self,cfg,prefix=None):
        super().__init__()
        self.cfg = cfg
        # Model 
        self.model = SparK_2D(cfg)
        self.L1 = L1_AE(cfg)

        self.prefix = prefix
        self.save_hyperparameters()

    def forward(self, x):
        active_ex, reco, loss, latent = self.model(x)
        if self.cfg.get('loss_on_mask', False): # loss is calculated only on the masked patches
            loss = loss
        else: 
            loss = self.L1({'x_hat':reco},x)['recon_error'] + self.cfg.get('delta_mask',0) * loss 
        return loss, reco, latent[0].mean([2,3])

    def training_step(self, batch, batch_idx: int):
        # process batch
        input = batch['vol'][tio.DATA].squeeze(-1) # add dimension for channel
        loss, reco, latent = self(input) # loss, reconstruction, latent

        self.log(f'{self.prefix}train/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)

        return {"loss": loss} # , 'latent_space': z}



    def validation_step(self, batch: Any, batch_idx: int):
        input = batch['vol'][tio.DATA].squeeze(-1) # add dimension for channel
        loss, reco, _ = self(input)
        # log val metrics
        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self,'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreReco = []


        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4):
            num_slices = self.cfg.get('num_eval_slices', input.size(4))  # number of center slices to evaluate. If not set, the whole Volume is evaluated
            start_slice = int((input.size(4) - num_slices) / 2)

            input = input[...,start_slice:start_slice+num_slices]
            data_orig = data_orig[...,start_slice:start_slice+num_slices] 
            data_seg = data_seg[...,start_slice:start_slice+num_slices]
            data_mask = data_mask[...,start_slice:start_slice+num_slices]
        
        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)
        
        # reorder depth to batch dimension
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]

        # compute reconstruction
        loss, output_slice, _ = self(input)

        # calculate loss and Anomalyscores
        AnomalyScoreReco.append(loss.item())

        # reassamble the reconstruction volume
        final_volume = output_slice.squeeze().permute(1,2,0) # to HxWxD
    

        # average across slices to get volume-based scores
        AnomalyScoreReco_vol = np.mean(AnomalyScoreReco)


        self.eval_dict['AnomalyScoreRegPerVol'].append(0)


        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(0)
            self.eval_dict['AnomalyScoreCombiPerVol'].append(0)
            self.eval_dict['AnomalyScoreCombPriorPerVol'].append(0)
            self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(0)

        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)
        
        # calculate metrics
        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label) # everything that is independent of the model choice

           
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 


    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.get('weight_decay', 0.05), betas=[0.9,0.95])
    
    def update_prefix(self, prefix):
        self.prefix = prefix 