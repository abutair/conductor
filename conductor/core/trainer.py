import torch
from tqdm import tqdm

class ConductorTrainer:
    """Main trainer class for reinforcement learning with LLMs."""
    
    def __init__(self, model, tokenizer, reward_manager, generation_config=None):
        """
        Initialize the trainer.
        
        Args:
            model: Pretrained language model
            tokenizer: Tokenizer for the model
            reward_manager: RewardManager instance
            generation_config: Configuration for text generation
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.reward_manager = reward_manager
        self.generation_config = generation_config or {
            "num_generations": 4,
            "max_length": 512,
            "temperature": 1.0,
            "top_p": 0.9
        }
        self.optimizer = None
    
    def generate_completions(self, prompts, num_completions=None):
        """
        Generate multiple completions for each prompt.
        
        Args:
            prompts: List of input prompts
            num_completions: Number of completions to generate per prompt
            
        Returns:
            List of lists of completions
        """
        if num_completions is None:
            num_completions = self.generation_config.get("num_generations", 4)
            
        all_completions = []
        
        for prompt in prompts:
            completions = []
            
            for _ in range(num_completions):
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),  # Add this line
                    max_length=self.generation_config.get("max_length", 512),
                    temperature=self.generation_config.get("temperature", 1.0),
                    top_p=self.generation_config.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id  # Add this line
                )
                
                completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completions.append(completion)
                
            all_completions.append(completions)
            
        return all_completions
    
    def train_step(self, batch):
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary with prompts and references
            
        Returns:
            Dictionary with training metrics
        """
        # Generate completions
        prompts = batch["prompts"]
        references = batch.get("references")
        
        all_completions = self.generate_completions(prompts)
        flat_completions = [c for prompt_completions in all_completions 
                            for c in prompt_completions]
        
        # Calculate rewards
        flat_references = None
        if references:
            flat_references = references * self.generation_config.get("num_generations", 4)
            
        rewards = self.reward_manager.calculate_rewards(
            flat_completions, flat_references
        )
        
        # Compute RL loss
        # This is a simplified version - actual implementation would use GRPO or similar
        loss = self._compute_rl_loss(flat_completions, rewards)
        
        # Update model
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=5e-5
            )
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "rewards": rewards
        }
    
    def _compute_rl_loss(self, completions, rewards):
        """
        Compute reinforcement learning loss.
        This is a placeholder implementation.
        
        Args:
            completions: List of generated completions
            rewards: List of reward values
            
        Returns:
            Tensor containing loss value
        """
        # This is where you'd implement the actual RL algorithm (PPO, GRPO, etc.)
        # For now, we'll use a dummy implementation
        return torch.tensor(1.0, requires_grad=True)
    
    def train(self, dataset, batch_size=8, epochs=1):
        """
        Train the model using reinforcement learning.
        
        Args:
            dataset: Dataset containing prompts and references
            batch_size: Batch size for training
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary with training metrics
        """
        metrics = {
            "epochs": [],
            "loss": [],
            "mean_reward": []
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_metrics = {"loss": 0, "rewards": []}
            
            # Create batches
            batches = []
            for i in range(0, len(dataset), batch_size):
                batch = {
                    "prompts": dataset["prompts"][i:i+batch_size],
                    "references": dataset.get("references", None)
                }
                if batch["references"]:
                    batch["references"] = batch["references"][i:i+batch_size]
                batches.append(batch)
            
            # Process batches
            for batch in tqdm(batches):
                batch_metrics = self.train_step(batch)
                epoch_metrics["loss"] += batch_metrics["loss"]
                epoch_metrics["rewards"].extend(batch_metrics["rewards"])
            
            # Calculate epoch metrics
            epoch_metrics["loss"] /= len(batches)
            epoch_metrics["mean_reward"] = sum(epoch_metrics["rewards"]) / len(epoch_metrics["rewards"])
            
            metrics["epochs"].append(epoch+1)
            metrics["loss"].append(epoch_metrics["loss"])
            metrics["mean_reward"].append(epoch_metrics["mean_reward"])
            
            print(f"Epoch {epoch+1}: Loss = {epoch_metrics['loss']:.4f}, "
                  f"Mean Reward = {epoch_metrics['mean_reward']:.4f}")
            
        return metrics