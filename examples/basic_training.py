from conductor import ConductorTrainer, RewardManager
from conductor.rewards import FormatReward, LengthReward, AccuracyReward
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2" 
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
        "num_generations": 4,
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
metrics = trainer.train(
    dataset=dataset,
    batch_size=1,
    epochs=2
)

print("Training complete!")
print(f"Final loss: {metrics['loss'][-1]}")
print(f"Final mean reward: {metrics['mean_reward'][-1]}")