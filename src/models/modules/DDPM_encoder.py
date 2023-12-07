import timm 
import torch 
import torchvision
from src.models.modules.spark.Spark_2D import SparK_2D_encoder

def get_encoder(cfg):
    """
    Available backbones (some of them): 
    Resnet: 
        resnet18,
        resnet34,
        resnet50, 
        resnet101
    """
    backbone = cfg.get('backbone','resnet50')  
    chans = 1 
    if 'spark' in backbone.lower(): # spark encoder
        encoder = SparK_2D_encoder(cfg) 
    else : # 2D CNN encoder
        encoder = timm.create_model(backbone, pretrained=cfg.pretrained_backbone, in_chans=chans, num_classes = cfg.get('cond_dim',256) )
                               
    out_features = cfg.get('cond_dim',256) # much adaptive..


        
    return encoder, out_features