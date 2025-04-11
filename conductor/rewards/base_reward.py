class BaseReward:
    """Base class for all reward functions"""

    def __init__(self,weight, name=None):
        self.weight = weight
        self.name = name or self.__class__.__name__


    def compute(self, completions, refernces=None,**kwargs):
        """
        Compute reward for given completions.
        
        Args:
            completions: List of model completions
            references: Optional reference answers or target outputs
            **kwargs: Additional arguments that may be needed
            
        Returns:
            List of float rewards, one per completion
        """

        raise NotImplementedError("Reward functions must be implment compute method")
    

    def __call__(self, completions, references=None, **kwargs):
        """Make the reward funcation callable directly"""
        return self.compute(completions,references,**kwargs)


