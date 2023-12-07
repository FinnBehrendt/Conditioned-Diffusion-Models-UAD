
from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet

import torch
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any, List
import torchio as tio
from src.utils.patch_sampling import BoxSampler
from src.utils.generate_noise import gen_noise
class DDPM_2D(LightningModule):
    def __init__(self,cfg,prefix=None):
        super().__init__()
        
        self.cfg = cfg

        # Model 

        model = OpenAI_UNet(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = cfg.get('unet_dim',64),
                            out_channels = 1,
                            num_res_blocks = cfg.get('num_res_blocks',3),
                            attention_resolutions = (int(cfg.imageDim[0])/int(32),int(cfg.imageDim[0])/int(16),int(cfg.imageDim[0])/int(8)),
                            dropout=cfg.get('dropout_unet',0), # default is 0.1
                            channel_mult=cfg.get('dim_mults',[1, 2, 4, 8]),
                            conv_resample=True,
                            dims=2,
                            num_classes=None,
                            use_checkpoint=True,
                            use_fp16=True,
                            num_heads=cfg.get('num_heads',1),
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=False,    
                            transformer_depth=1,              
                            )
        model.convert_to_fp16()
        

        timesteps = cfg.get('timesteps',1000)
        self.test_timesteps = cfg.get('test_timesteps',150) 
        sampling_timesteps = cfg.get('sampling_timesteps',self.test_timesteps)
    
        self.diffusion = GaussianDiffusion(
        model,
        image_size = (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)), # only important when sampling
        timesteps = timesteps,   # number of steps
        sampling_timesteps = sampling_timesteps,
        objective = cfg.get('objective','pred_x0'), # pred_noise or pred_x0
        channels = 1,
        loss_type = cfg.get('loss','l1'),    # L1 or L2
        p2_loss_weight_gamma = cfg.get('p2_gamma',0),
        inpaint = cfg.get('inpaint',False),
        cfg=cfg
        )
        
        self.boxes = BoxSampler(cfg) # initialize box sampler

        self.prefix = prefix
        
        self.save_hyperparameters()


    def forward(self):
        return None


    def training_step(self, batch, batch_idx: int):
        # process batch
        input = batch['vol'][tio.DATA].squeeze(-1)



        # generate bboxes for DDPM 
        if self.cfg.get('grid_boxes',False): # sample boxes from a grid
            bbox = torch.zeros([input.shape[0],4],dtype=int)
            bboxes = self.boxes.sample_grid(input)
            ind = torch.randint(0,bboxes.shape[1],(input.shape[0],))
            for j in range(input.shape[0]):
                bbox[j] = bboxes[j,ind[j]]
            bbox = bbox.unsqueeze(-1)
        else: # sample boxes randomly
            bbox = self.boxes.sample_single_box(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None
        # reconstruct
        loss, reco = self.diffusion(input, box=bbox,noise=noise)

        self.log(f'{self.prefix}train/Loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        input = batch['vol'][tio.DATA].squeeze(-1) 
        # generate bboxes for DDPM 
        if self.cfg.get('grid_boxes',False): # sample boxes from a grid
            bbox = torch.zeros([input.shape[0],4],dtype=int)
            bboxes = self.boxes.sample_grid(input)
            ind = torch.randint(0,bboxes.shape[1],(input.shape[0],))
            for j in range(input.shape[0]):
                bbox[j] = bboxes[j,ind[j]]
            bbox = bbox.unsqueeze(-1)
        else:  # sample boxes randomly
            bbox = self.boxes.sample_single_box(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None

        loss, reco = self.diffusion(input, box=bbox, noise=noise)


        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
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
        age = batch['age']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []

        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4):
            num_slices = self.cfg.get('num_eval_slices', input.size(4))  # number of center slices to evaluate. If not set, the whole Volume is evaluated
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[...,start_slice:start_slice+num_slices]
            data_orig = data_orig[...,start_slice:start_slice+num_slices] 
            data_seg = data_seg[...,start_slice:start_slice+num_slices]
            data_mask = data_mask[...,start_slice:start_slice+num_slices]
            ind_offset = start_slice
        else: 
            ind_offset = 0 

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)

        # reorder depth to batch dimension
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]
        
        latentSpace.append(torch.tensor([0],dtype=float).repeat(input.shape[0])) # dummy latent space 

        # generate bboxes for DDPM 
        bbox = self.boxes.sample_grid(input)
        reco_patched = torch.zeros_like(input)

        
        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None


        for k in range(bbox.shape[1]):
            box = bbox[:,k]
            # reconstruct
            loss_diff, reco = self.diffusion(input,t=self.test_timesteps-1, box=box,noise=noise)

            if reco.shape[1] == 2:
                reco = reco[:,0:1,:,:]
    
            for j in range(reco_patched.shape[0]): 
                if self.cfg.get('overlap',False): # cut out the overlap region
                    grid = self.boxes.sample_grid_cut(input)
                    box_cut = grid[:,k]
                    if self.cfg.get('agg_overlap','cut') == 'cut': # cut out the overlap region
                        reco_patched[j,:,box_cut[j,1]:box_cut[j,3],box_cut[j,0]:box_cut[j,2]] = reco[j,:,box_cut[j,1]:box_cut[j,3],box_cut[j,0]:box_cut[j,2]]
                    elif self.cfg.get('agg_overlap','cut') == 'avg': # average the overlap region
                        reco_patched[j,:,box[j,1]:box[j,3],box[j,0]:box[j,2]] = reco_patched[j,:,box[j,1]:box[j,3],box[j,0]:box[j,2]] + reco[j,:,box[j,1]:box[j,3],box[j,0]:box[j,2]]
                else:
                    reco_patched[j,:,box[j,1]:box[j,3],box[j,0]:box[j,2]] = reco[j,:,box[j,1]:box[j,3],box[j,0]:box[j,2]]


            if self.cfg.get('overlap',False) and self.cfg.get('agg_overlap','cut') == 'avg': # average the intersection of all patches
                mask = torch.zeros_like(reco_patched) 
                # create mask 
                for k in range(bbox.shape[1]):
                    box = bbox[:,k]
                    for l in range(mask.shape[0]):
                        mask[l,:,box[l,1]:box[l,3],box[l,0]:box[l,2]] = mask[l,:,box[l,1]:box[l,3],box[l,0]:box[l,2]] + 1
                # divide by the mask to average the intersection of all patches
                reco_patched = reco_patched/mask

            reco = reco_patched.clone()

        
        
        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(AnomalyScoreComb) # dummy
        AnomalyScoreReco.append(AnomalyScoreComb) # dummy

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD
       

        # average across slices to get volume-based scores
        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace),0))

        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb) 
        AnomalyScoreReg_vol = AnomalyScoreComb_vol # dummy
        AnomalyScoreReco_vol = AnomalyScoreComb_vol # dummy

        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)


        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)


        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)

        # calculate metrics
        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label) 

           
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix):
        self.prefix = prefix 