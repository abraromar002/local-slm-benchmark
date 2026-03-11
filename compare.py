import ollama
import time
import json

MODELS = ["llama3.2:3b", "phi4-mini", "mistral:7b"]

TEST_PROMPTS = [
    # Logic
    {"category": "logic", "prompt": "If I have 3 apples and you take away 2, how many apples do YOU have?"},
    {"category": "logic", "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?"},
    {"category": "logic", "prompt": "A doctor gives you 3 pills and tells you to take one every 30 minutes. How long until you run out?"},
    {"category": "logic", "prompt": "What comes next: 2, 4, 8, 16, ?"},
    {"category": "logic", "prompt": "Is it possible to end a sentence with the word 'the the'?"},
    # Coding
    {"category": "coding", "prompt": "Write a Python function to check if a string is a palindrome."},
    {"category": "coding", "prompt": "Write a Python function that returns the fibonacci sequence up to n."},
    {"category": "coding", "prompt": "Write a CSS rule to center a div both vertically and horizontally."},
    {"category": "coding", "prompt": "Write a Python one-liner to remove duplicates from a list."},
    {"category": "coding", "prompt": "Write a SQL query to find the second highest salary."},
    # Creative
    {"category": "creative", "prompt": "Write a 2-sentence horror story about a sentient AI."},
    {"category": "creative", "prompt": "Explain cloud computing using a library analogy."},
    {"category": "creative", "prompt": "Write a haiku about machine learning."},
    {"category": "creative", "prompt": "Describe the color blue to someone who has never seen color."},
    {"category": "creative", "prompt": "Write a one-sentence pitch for an app that helps people drink more water."},
]

def run_comparison():
    all_results = []

    for model in MODELS:
        print(f"\nTesting: {model}")
        print("=" * 40)

        model_results = {
            "model": model,
            "prompts": [],
            "summary": {}
        }

        total_latency = 0
        total_words = 0

        for i, item in enumerate(TEST_PROMPTS):
            print(f"  [{i+1}/{len(TEST_PROMPTS)}] {item['category']}: {item['prompt'][:40]}...")

            start = time.time()
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": item["prompt"]}]
            )
            latency = round(time.time() - start, 2)

            content = response["message"]["content"]
            word_count = len(content.split())
            total_latency += latency
            total_words += word_count

            model_results["prompts"].append({
                "category": item["category"],
                "prompt": item["prompt"],
                "response": content,
                "latency": latency,
                "word_count": word_count
            })

            print(f"  Done in {latency}s | {word_count} words")

        # Summary per model
        model_results["summary"] = {
            "avg_latency": round(total_latency / len(TEST_PROMPTS), 2),
            "avg_word_count": round(total_words / len(TEST_PROMPTS), 1),
            "total_time": round(total_latency, 2)
        }

        print(f"\n  Avg latency: {model_results['summary']['avg_latency']}s")
        print(f"  Avg words:   {model_results['summary']['avg_word_count']}")

        all_results.append(model_results)

    with open("results/comparison_data.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 40)
    print("FINAL SUMMARY")
    print("=" * 40)
    for r in all_results:
        s = r["summary"]
        print(f"\n{r['model']}")
        print(f"  Avg Latency:    {s['avg_latency']}s")
        print(f"  Avg Words:      {s['avg_word_count']}")
        print(f"  Total Time:     {s['total_time']}s")

    print("\nSaved to results/comparison_data.json")

if __name__ == "__main__":
    run_comparison()