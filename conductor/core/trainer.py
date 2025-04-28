import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class GRPOOptimizer:
    def __init__(self, model: Any, tokenizer: Any, reward_manager:Any, temp:float =0.1, kl_weight:float=0.1, 
                 learning_rate:float=5e-5, gradient_accum_steps:int =1, max_grad_norm: float=1.0, device:str=None
                 ):
        
        self.model = model
        self.tokenizer=tokenizer
        self.reward_manager = reward_manager
        self.temp = temp
        self.kl_weight= kl_weight
        self.learning_rate= learning_rate
        self.gradient_accum_steps= gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        
        #set device
        self.device = device or ('cuda'if torch.cuda.is_available()else'cpu')
         
        if hasattr(model,'device'):
             self.device = model.device
        
        
        # create optimizer 
        
        self.optimizer= torch.optim.AdamW(
            self.model.parameters(),
            lr= learning_rate,
            weight_decay= 0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )     
            
        """
        Learning Rate
        |
        5e-5|    *        
            |      *     
            |        *  
            |          *
            |           *
            |            *
        5e-6|--------------> Steps

        """ 
        
        # create scheduler
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000, # well be updated in traning
            eta_min=learning_rate/10
        )
        
        logger.info(f'Iniit Grpo optimizer on device {self.device}')
        
    
        
        
        