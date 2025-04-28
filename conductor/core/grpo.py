import torch
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm
from typing import  Dict, List, Optional, Union, Tuple,Any
import logging

from conductor import rewards



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

    def _generate_completions(self, prompts:List[str], num_completions:int=4, max_length:int=512)-> Tuple[list[str],list[str]]:
        
        all_flat_prompts = []
        all_completions = []
        
        for prompt in tqdm(prompts, desc="Generating completions"):
            prompt_completions = []
            
            for _ in range(num_completions):
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        max_length=max_length,
                        temperature=self.temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=1
                    )
                
                completion_ids = outputs[0][inputs["input_ids"].shape[1]:]
                
                completion_text = self.tokenizer.decode(
                    completion_ids, 
                    skip_special_tokens=True
                )
                
                prompt_completions.append(completion_text)
                all_flat_prompts.append(prompt)
                all_completions.append(completion_text)
            
        return all_flat_prompts, all_completions
                            
    def _calculate_grpo_loss(self, log_probs:torch.Tensor, rewards:torch.Tensor)->torch.Tensor:
        
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        grpo_loss = -(normalized_rewards*log_probs).mean() # negative becase we want to max rewards
        
        return grpo_loss
    
    def optmize_step(self, prompts:list[str], references: Optional[List[str]] = None, num_completions: int = 4,max_length: int = 512)->Dict[str, Any]:
        flat_prompts, completions = self._generate_completions(prompts= prompts,num_completions=num_completions, max_length= max_length)
        
        flat_refernces = None
        
        if references:
            flat_refernces=[]
            for i, ref in enumerate(references):
                flat_refernces.extend([ref]* num_completions)
                
                
        rewards = self.reward_manager.calculate_rewards(completions=completions,references=flat_refernces)
        
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
        
    
    def train(self,dataset: Dict[str, List[str]],num_epochs: int = 1,batch_size: int = 4,num_completions: int = 4,
        max_length: int = 512,eval_steps: Optional[int] = None,save_path: Optional[str] = None,) -> Dict[str, List[float]]:
        prompts = dataset['prompts']
        refernces = dataset.get("references")
        total_steps = (len(prompts)// batch_size+1)* num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.learning_rate/10
        )
        
        metrics = {
            'epochs':[],
            'steps':[],
            'loss':[],
            'mean_reward':[],
            'learning_rate':[]
        }

        step=0 
        for epoch in range(num_epochs):
            logger.info(f"epoch{epoch+1}/{num_epochs}")

            #shuffle data 
            indices = np.random.permuation(len(prompts))
            shuffled_prompts = [prompts[i] for i in indices]
            shuffled_refernces = None
            if refernces:
                shuffled_refernces= [refernces[i] for i in indices]

            # traning loop
            epoch_loss =0 
            epoch_rewards = []

            for batch_idx in tqdm(range(0,len(shuffled_prompts),batch_size),
                                  desc=f'traning loop {epoch+1}'):

                batch_prompts = shuffled_prompts[batch_idx:batch_idx+batch_size]
                
                batch_refs = None
                if refernces:
                    batch_refs = shuffled_refernces[batch_idx:batch_idx+batch_size]


                self.optimizer.zero_grad()

                batch_metrics={
                    'loss':0,
                    'rewards':[],
                    'mean_reward':0,
                    'completion':[]
                }
 
                for i in range(len(batch_prompts)):
                    prompt = [batch_prompts[i]]
                    ref = [batch_refs[i]]

                    prompt_metrics=  self.optmize_step(
                        prompts=prompts,
                        refernces=ref,
                        num_completions=num_completions,
                        max_length=max_length
                    )
                    batch_metrics['loss']+=prompt_metrics["loss"]/len(batch_prompts)

                    batch_metrics['rewards'].extend(prompt_metrics['rewards'])
                    batch_metrics['completion'].extend(prompt_metrics['completions']
                                                       )    

                    batch_metrics['mean_reward'] = (
                       sum(batch_metrics['rewards']/len(batch_metrics['rewards']))
                        if batch_metrics['rewards'] else 0 
                    )

                    
    def evelaute(self,eval_dataset: Dict[str, List[str]],num_completions: int = 4,max_length: int = 512)->Dict[str,Any]:
        prompts = eval_dataset["prompts"]
        references= eval_dataset.get('references')
        
        self.model.eval()
        
        flat_prompts, completions = self._generate_completions(
            prompts=prompts,
            num_completions=num_completions,
            max_length=max_length
        )
        
        flat_references = None
        if references:
            flat_references =[]
            
            for i, ref in enumerate(references):
                flat_references.extend([ref]*num_completions)
        
        rewards = self.reward_manager.calculate_rewards(
            completions=completions,
            references=flat_references
        )    
        
        reward_breakdown = self.reward_manager.get_reward_breakdown(
            completions=completions,
            references=flat_references
        )

        
        grouped_completions = []
        grouped_rewards = []
        for i in range(0, len(completions), num_completions):
            grouped_completions.append(completions[i:i+num_completions])
            grouped_rewards.append(rewards[i:i+num_completions])
            
        mean_reward = sum(rewards) / len(rewards) if rewards else 0
        
        self.model.train()
        
        return {
            "mean_reward": mean_reward,
            "rewards": rewards,
            "completions": completions,
            "grouped_completions": grouped_completions,
            "grouped_rewards": grouped_rewards,
            "reward_breakdown": reward_breakdown
        }
    



                
        
        
         
        
        
        
        
        
    


        
