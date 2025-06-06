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
        
    
    def  compute_logprobs(self, prompts:List[str], completions:List[str], batch_size)->torch.Tensor:
        all_log_probs = []
        

        for i in range(0,len(prompts),batch_size):
            
            batch_prompts =prompts[i:i+batch_size]
            
            batch_completions = completions[i:i+batch_size]
            
            batch_log_probs= []
            
            for prompt, completion in zip(batch_prompts, batch_completions):
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
                full_text = prompt + completion
            
                full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                
                prompt_length = prompt_tokens["input_ids"].shape[1]
                
                with torch.no_grad():
                    outputs = self.model(** full_tokens)
                    logits = outputs.logits
                
                completion_logits = logits[:, prompt_length-1:-1, :]
                completion_ids = full_tokens["input_ids"][:, prompt_length:]
                log_probs = F.log_softmax(completion_logits / self.temperature, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, 
                    dim=2, 
                    index=completion_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                seq_log_prob = token_log_probs.sum().item()
                batch_log_probs.append(seq_log_prob)
                
            all_log_probs.extend(batch_log_probs)
            
        return torch.tensor(all_log_probs, device=self.device)
    
    
    def _generate_completions(self, prompts:List[str], num_completions:int =4, max_length:int =512)-> Tuple[list[str],list[str]]:
        
        all_flat_prompts= []
        all_completions =[]
        
        for prompt in tqdm(prompts, desc='Generating completions'):
            
            prompt_completions= []
            
            for _ in range(num_completions):
                
                inputs = self.tokenizer(prompt, return_tensors="pt",padding=True, truncatioon=True).to(self.device)
                
                # generate completion
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attentoin_mask = inputs.get("attention_mask",None), 
                        max_length= max_length,
                        temperature= self.temp,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id= self.tokenizer.pad_token_id,
                        num_return_sequences=1
                    )
                    
                
                # get the completion part 
                
                completion_ids = outputs[0][inputs['input_ids'].shape[1]:]
                
                # decode completion
                completion_text = self.tokenizer.decode(
                    completion_ids, 
                    skip_special_tokens=True
                )
                
                prompt_completions.append(completion_text)
                all_flat_prompts.append(prompt)
                all_completions.append(completion_text)
                
        return all_flat_prompts, all_completions
    
     
    def _calculate_grpo_loss(self, log_probs:torch.Tensor, rewards:torch.Tensor)-> torch.Tensor:
        
        if len(rewards)>1 :
            normalized_rewards= (rewards- rewards.mean())/(rewards.std()+1e-8)
            
        else:
            normalized_rewards = rewards 
            
        grpo_loss = -(log_probs* normalized_rewards).mean()
        
        return grpo_loss
               
                
                
    def optimize_step(self, prompts:List[str], references:Optional[List[str]]=None, num_completions:int=4, max_length:int =512)-> Dict[str, Any]:
        
        flat_prompts, completions = self._generate_completions(prompts=prompts, num_completions=num_completions,max_length=max_length)
        
        flat_references= None
        
        if references:
            flat_references=[]
            for i,ref in enumerate(references):
                flat_references([ref]*num_completions)
                
        
                rewards = self.reward_manager.calculate_rewards(
            completions=completions,
            references=flat_references
        )
        rewards_tensor = torch.tensor(rewards, device=self.device)
        
        log_probs = self.compute_logprobs(flat_prompts, completions)
        
        loss = self._calculate_grpo_loss(log_probs, rewards_tensor)
        
        loss.backward()
        
        return {
            "loss": loss.item(),
            "rewards": rewards,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
            "completions": completions,
            "log_probs": log_probs.detach().cpu().numpy().tolist()
        }
              
        
