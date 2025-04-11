class RewardManager:
    """Manages multiple reward functions and aggregates their results."""
    
    def __init__(self, reward_functions):
        """
        Initialize the reward manager.
        
        Args:
            reward_functions: List of BaseReward instances
        """
        self.reward_functions = reward_functions
    
    def calculate_rewards(self, completions, references=None, **kwargs):
        """
        Calculate rewards using all registered reward functions.
        
        Args:
            completions: List of model completions
            references: Optional reference answers or target outputs
            **kwargs: Additional arguments passed to reward functions
            
        Returns:
            Dictionary mapping completion indices to total rewards
        """
        all_rewards = []
        
        # Calculate rewards from each function
        for reward_func in self.reward_functions:
            rewards = reward_func(completions, references, **kwargs)
            weighted_rewards = [r * reward_func.weight for r in rewards]
            all_rewards.append(weighted_rewards)
        
        # Aggregate rewards
        total_rewards = []
        for i in range(len(completions)):
            total = sum(rewards[i] for rewards in all_rewards)
            total_rewards.append(total)
            
        return total_rewards
    
    def get_reward_breakdown(self, completions, references=None, **kwargs):
        """
        Get detailed breakdown of rewards from each function.
        
        Args:
            completions: List of model completions
            references: Optional reference answers
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with detailed reward information
        """
        breakdown = {}
        
        for reward_func in self.reward_functions:
            rewards = reward_func(completions, references, **kwargs)
            breakdown[reward_func.name] = {
                'raw': rewards,
                'weighted': [r * reward_func.weight for r in rewards]
            }
            
        # Add total rewards
        total_rewards = []
        for i in range(len(completions)):
            total = sum(breakdown[func.name]['weighted'][i] 
                        for func in self.reward_functions)
            total_rewards.append(total)
            
        breakdown['total'] = total_rewards
        
        return breakdown