import re
from .base_reward import BaseReward

class FormatReward(BaseReward):
    """Reward based on following a specific format."""
    
    def __init__(self, format_pattern, weight=1.0):
        """
        Initialize format reward.
        
        Args:
            format_pattern: Regex pattern to match desired format
            weight: Weight of this reward
        """
        super().__init__(weight=weight)
        self.pattern = re.compile(format_pattern)
    
    def compute(self, completions, references=None, **kwargs):
        """Check if completions match the desired format."""
        rewards = []
        
        for completion in completions:
            if self.pattern.search(completion):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        return rewards


class LengthReward(BaseReward):
    """Reward based on completion length."""
    
    def __init__(self, min_length=0, max_length=float('inf'), weight=1.0):
        """
        Initialize length reward.
        
        Args:
            min_length: Minimum desired length
            max_length: Maximum desired length
            weight: Weight of this reward
        """
        super().__init__(weight=weight)
        self.min_length = min_length
        self.max_length = max_length
    
    def compute(self, completions, references=None, **kwargs):
        """Check if completions are within desired length range."""
        rewards = []
        
        for completion in completions:
            length = len(completion)
            
            # If within range, give full reward
            if self.min_length <= length <= self.max_length:
                rewards.append(1.0)
            else:
                # Calculate partial reward based on how far off target
                if length < self.min_length:
                    ratio = length / self.min_length
                else:
                    ratio = self.max_length / length
                # Apply penalty but keep some reward
                rewards.append(max(0.0, ratio * 0.8))
                
        return rewards


class AccuracyReward(BaseReward):
    """Reward based on matching reference answers."""
    
    def __init__(self, extract_pattern=None, weight=1.0):
        """
        Initialize accuracy reward.
        
        Args:
            extract_pattern: Optional regex to extract answer from completion
            weight: Weight of this reward
        """
        super().__init__(weight=weight)
        self.extract_pattern = None
        if extract_pattern:
            self.extract_pattern = re.compile(extract_pattern)
    
    def compute(self, completions, references=None, **kwargs):
        """Check if completions match reference answers."""
        if references is None:
            raise ValueError("References required for AccuracyReward")
            
        rewards = []
        
        for completion, reference in zip(completions, references):
            # Extract answer if pattern provided
            if self.extract_pattern:
                match = self.extract_pattern.search(completion)
                if match:
                    extracted = match.group(1).strip()
                    reward = 1.0 if extracted == str(reference).strip() else 0.0
                else:
                    reward = 0.0
            else:
                # Simple string matching as fallback
                reward = 1.0 if completion.strip() == str(reference).strip() else 0.0
                
            rewards.append(reward)
                
        return rewards