import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class GRPOOptimizer:
    """
    Generalized Reward-Penalized Optimiztion (GRPO) implmeneation.
    """

    def __init__(self,model, tokenizer, reward_manager, temperature=1.0, kl_weight=0.1)  :
        self.model = model
        self.tokenizer = tokenizer
        self.reward_manager = reward_manager
        self.temperature = temperature
        self.kl_weight= kl_weight



    def compute_logprobs(self, prompts, completions):

        log_probs =[]
        
        for prompt, completion in zip(prompts,completions):

            inputs = self.tokenizer(prompt+completion,return_tensors="pt")
            prompt_tokens = self.tokenizer(prompt, return_types="pt")
            
            prompt_length= prompt_tokens["input_ids"].shape[1]


            with torch.no_grad():
                outpiut = self.model(**inputs)
                logits = outpiut.logits

            shift_logits = logits[:,prompt_length-1:-1,:]
            shift_lables = inputs["input_ids"][:,prompt_length:]


            log_probs_sequance = F.log_softmax(shift_logits/self.temperature, dim=-1)

            seq_log_prob = 0

            for i in range(len(shift_lables[0])):
                seq_log_prob += log_probs_sequance[0, i, shift_lables[0,i]].item()

            log_probs.append(seq_log_prob)

            return torch.tensor(log_probs)
        
    def optmize(self, prompts, references=None, batch_size= 4, num_compleations=4):
        all_log_probs = []
        all_rewards= []
        all_completions= []

        for prompt_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[prompt_idx:prompt_idx + batch_size]
            batch_refs = None
            if references is not None:
                batch_refs = references[prompt_idx:prompt_idx + batch_size]


            # Generate multiple completions per prompt
            batch_completions=[]
            for prompt in batch_completions:
                prompt_comp = []

                for _ in range(num_compleations):
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        padding=True,
                        truncate=True
                    )

                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask = inputs.get("attenation_mask",None),
                        max_length=512,
                        temprature= self.temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id= self.tokenizer.pad_token_id
                    )

                    completion = self.tokenizer.decode(outputs[0], skip_sepcial_token=True)
                    completion.append(completion)
                    all_completions.append(completion)
            
            flat_batch_completions = [c for prompt_cs in batch_completions for c in prompt_cs]

            flat_batch_refs =None
            if batch_refs:
                flat_batch_refs=[]
                for  ref in batch_refs:
                    flat_batch_refs.extend([ref]*num_compleations)

            rewards = self.reward_manager.calculate_rewards(
                flat_batch_completions,flat_batch_refs
            )

            all_rewards.extend(rewards)

            flat_batch_prompts=[]

            for i, prompt in enumerate(batch_prompts):
                flat_batch_prompts.extend([prompt]* num_compleations)

            log_probs= self.compute_logprobs(flat_batch_prompts,flat_batch_completions)
            all_log_probs.append(log_probs)

            all_log_probs = torch.cat(all_log_probs)
            all_rewards = torch.tensor(all_rewards)

            if len(all_rewards)>1:
                all_rewards = (all_rewards-all_rewards.mean())/(all_rewards.std()+1e-8)

            # compute grpo loss
            loss = -(all_log_probs * all_rewards).mean()

            return{
                "loss":loss,
                "rewards":all_rewards.tolist(),
                "mean_reward":all_rewards.mean().item(),
                "completions":all_completions
            }

    def train(self, dataset, batch_size=8,epochs=1):

        return self.grpo.train(
            dataset=dataset,
            num_epoches=epochs,
            batch_size=batch_size,
            num_completions=self.generation_config.get("num_gernerations",4)
        )
        




                                      
