# examples/basic_training.py
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conductor.core.trainer import ConductorTrainer
from conductor.core.reward_manager import RewardManager
from conductor.rewards.common_rewards import FormatReward, LengthReward, AccuracyReward
from conductor.utils.visualization import RewardVisualizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"  # Use a small model for demonstration
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define reward functions
reward_manager = RewardManager([
    FormatReward(format_pattern=r"Answer\s*=\s*(\d+)", weight=0.4),
    LengthReward(min_length=50, max_length=200, weight=0.2),
    AccuracyReward(extract_pattern=r"Answer\s*=\s*(\d+)", weight=0.4)
])

# Create trainer
trainer = ConductorTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_manager=reward_manager,
    generation_config={
        "num_generations": 2,  # Reduced for faster testing
        "max_length": 256,
        "temperature": 0.8,
        "top_p": 0.9
    }
)

# Sample dataset
dataset = {
    "prompts": [
        "What is 2+2? Provide a step-by-step explanation.",
        "What is 7*8? Provide a step-by-step explanation."
    ],
    "references": ["4", "56"]
}

# Train model
print("Starting training...")
metrics = trainer.train(
    dataset=dataset,
    batch_size=1,
    epochs=2
)

print("Training complete!")
print(f"Final mean reward: {metrics['mean_reward'][-1]}")

# Visualize results
print("Generating visualizations...")
RewardVisualizer.plot_rewards(metrics)
RewardVisualizer.plot_loss(metrics)

# Generate completions with the trained model
print("\nGenerating completions with trained model:")
final_completions = trainer.generate_completions(dataset["prompts"])

for i, prompt_completions in enumerate(final_completions):
    print(f"\nPrompt: {dataset['prompts'][i]}")
    print(f"Reference: {dataset['references'][i]}")
    print("Generated completions:")
    for j, completion in enumerate(prompt_completions):
        print(f"Completion {j+1}: {completion}")
        
    # Calculate rewards for these completions
    rewards = reward_manager.calculate_rewards(prompt_completions, 
                                              [dataset["references"][i]] * len(prompt_completions))
    
    print(f"Rewards: {rewards}")