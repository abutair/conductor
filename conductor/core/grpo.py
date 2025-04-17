import torch
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm
from typing import  Dict, List, Optional, Union, Tuple,Any
import logging


logger = logging.getLogger(__name__)


class GRPOOptimzer:

    def __init__(
            self, 
            model: Any, 
            tokenizer: Any, 
            reward_manager: Any, 
            temperature: float = 1.0, 
            kl_weight: float = 0.1,
            learning_rate: float = 5e-5,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            device: str = None,
        ):
            """
            Initialize the GRPO optimizer.
            
            """
            self.model = model
            self.tokenizer = tokenizer
            self.reward_manager = reward_manager
            self.temperature = temperature
            self.kl_weight = kl_weight
            self.learning_rate = learning_rate
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.max_grad_norm = max_grad_norm
            
            # Set device
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(model, 'device'):
                self.device = model.device
            
            # Create optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Create scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=1000,  
                eta_min=learning_rate / 10
            )
            
            logger.info(f"Initialized GRPO optimizer on device {self.device}")

    def compute_logs(self, prompts:List[str], completions:List[str],batch_size:int = 4)->torch.Tensor:
         all_logs_probs=[]

         for i in range(0,len(prompts),batch_size):
              batch_prompts= prompts[i:i+batch_size]
              batch_comppletions = completions[i:i+batch_size]

              batch_log_probs =[]

              for prompt, completion in zip(batch_prompts, batch_comppletions):
                   prompt_token = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                   full_text = prompt + completion
                   full_tokens = self.tokenizer(full_text,return_tennsors='pt').to(self.device)
            
                   prompt_length= prompt_token['input_ids'].shape[1]

                   with torch.no_grad():
                        outputs = self.model(**full_tokens)
                        logits= outputs.logits

                   completion_logits = logits[:, prompt_length-1:-1,:]
                   completion_ids = full_tokens["input_ids"][:, prompt_length:]

                   # compute log probabilities

                   log_probs = F.log_softmax(completion_logits/self.temperature,dim=-1)

                   token_log_probs = torch.gather(
                       log_probs,
                       dim=2,
                       index=completion_ids.unsqueeze(-1)
                   ).squeeze(-1)

                   seq_log_prob = token_log_probs.sum().item()
                   batch_log_probs.append(seq_log_prob)

              all_logs_probs.extend(batch_log_probs)

              return torch.tensor(all_logs_probs, device=self.device)

                    

    

    



                
        
        
         
        
        
        
        
        
    


        
