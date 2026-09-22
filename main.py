import random
import collections

# Simulate a very simplified LLM's next-token probability output.
# In a real LLM, these probabilities would be dynamically calculated by the model
# based on its training and the input context. Here, we hardcode them for demonstration.
SIMULATED_PROBABILITIES = {
    "The cat sat on the": {
        "mat": 0.6,
        "rug": 0.3,
        "floor": 0.08,
        "dog": 0.02,
    },
    "I like to eat": {
        "pizza": 0.5,
        "apples": 0.3,
        "burgers": 0.15,
        "rocks": 0.05,
    },
    "The capital of France is": {
        "Paris": 0.9,
        "London": 0.07,
        "Berlin": 0.03,
    }
}

def get_next_token_probabilities(prompt_prefix: str) -> dict:
    """
    Simulates an LLM returning a probability distribution for the next token.
    This is the core mathematical output of an LLM for a given input context.
    """
    probs = SIMULATED_PROBABILITIES.get(prompt_prefix, {})
    if not probs:
        return {}
    
    # Normalize probabilities to ensure they sum to 1, handling potential floating point issues
    total_prob = sum(probs.values())
    if total_prob == 0:
        return {}
    
    normalized_probs = {token: prob / total_prob for token, prob in probs.items()}
    return normalized_probs

def greedy_sampling(probabilities: dict) -> str:
    """
    Selects the next token by picking the one with the highest probability.
    This is a deterministic approach, similar to setting 'temperature=0' in LLMs.
    """
    if not probabilities:
        return ""
    return max(probabilities, key=probabilities.get)

def probabilistic_sampling(probabilities: dict) -> str:
    """
    Selects the next token by sampling from the given probability distribution.
    This is a stochastic approach, meaning different runs can yield different results,
    reflecting the LLM's non-deterministic nature when 'temperature > 0'.
    """
    if not probabilities:
        return ""
    tokens = list(probabilities.keys())
    weights = list(probabilities.values())
    
    # random.choices performs sampling with replacement based on weights
    return random.choices(tokens, weights=weights, k=1)[0]

def demonstrate_llm_token_generation(prompt: str, num_samples: int = 10):
    """
    Demonstrates how an LLM generates the next token using both greedy and probabilistic sampling.
    """
    print(f"\n--- Demonstrating for prompt: '{prompt}' ---")
    
    # Step 1: LLM calculates next token probabilities
    # This is the fundamental output of an LLM, not 'written text'.
    next_token_probs = get_next_token_probabilities(prompt)
    
    if not next_token_probs:
        print("No next token probabilities found for this prompt. Skipping demonstration.")
        return

    print("\nSimulated LLM's next token probability distribution:")
    for token, prob in sorted(next_token_probs.items(), key=lambda item: item[1], reverse=True):
        print(f"  '{token}': {prob:.4f}")

    # Step 2: Greedy Sampling (always picks the most probable)
    # This shows how a deterministic output is chosen from the probabilities.
    greedy_choice = greedy_sampling(next_token_probs)
    print(f"\nGreedy sampling (always highest probability): '{greedy_choice}'")

    # Step 3: Probabilistic Sampling (samples based on distribution)
    # This shows how stochastic outputs are chosen, reflecting the LLM's probabilistic nature.
    print(f"\nProbabilistic sampling (repeated {num_samples} times):")
    sampled_choices = []
    for i in range(num_samples):
        choice = probabilistic_sampling(next_token_probs)
        sampled_choices.append(choice)
        print(f"  Sample {i+1}: '{choice}'")
    
    # Show distribution of sampled choices to illustrate how probabilities influence outcomes
    choice_counts = collections.Counter(sampled_choices)
    print("\nDistribution of sampled choices:")
    for choice, count in choice_counts.most_common():
        print(f"  '{choice}': {count} times ({count/num_samples:.1%})")

if __name__ == "__main__":
    print("This example demonstrates how Large Language Models (LLMs) generate text")
    print("by calculating the probabilities of the next 'token' (word/word-piece)")
    print("and then sampling from that probability distribution. LLMs don't 'understand' or 'write'")
    print("in a human sense; they predict probabilities and then select based on them.")

    demonstrate_llm_token_generation("The cat sat on the")
    demonstrate_llm_token_generation("I like to eat")
    demonstrate_llm_token_generation("The capital of France is")
    demonstrate_llm_token_generation("A non-existent prompt") # To show handling of unknown prompts
