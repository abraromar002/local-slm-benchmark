import ollama
import json
import time

def test_temperature(model: str, prompt: str, temperatures: list):
    print(f"\nModel: {model}")
    print(f"Prompt: {prompt[:60]}...")
    print("-" * 50)
    
    results = []
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        
        responses = []
        for run in range(3):
            start = time.time()
            
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temp}
            )
            
            latency = round(time.time() - start, 3)
            content = response["message"]["content"]
            responses.append(content)
            print(f"  Run {run+1}: {latency}s | {content[:80]}...")
        
        # Check variance between runs
        unique_responses = len(set(responses))
        variance = "high" if unique_responses == 3 else "medium" if unique_responses == 2 else "low"
        
        results.append({
            "temperature": temp,
            "responses": responses,
            "variance": variance,
            "unique_count": unique_responses
        })
        
        print(f"  Variance: {variance} ({unique_responses}/3 unique responses)")
    
    return results

def main():
    models = ["llama3.2:3b", "phi4-mini", "mistral:7b"]
    
    prompts = [
        "Write a one sentence story about a robot.",
        "What is 2+2? Answer in one word.",
        "Give me one word that describes summer."
    ]
    
    temperatures = [0.0, 0.5, 1.0]
    all_results = {}
    
    for model in models:
        all_results[model] = {}
        for prompt in prompts:
            results = test_temperature(model, prompt, temperatures)
            all_results[model][prompt] = results
    
    with open("results/temperature_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for model in models:
        print(f"\nModel: {model}")
        for prompt, results in all_results[model].items():
            print(f"  Prompt: {prompt[:40]}...")
            for r in results:
                print(f"    Temp {r['temperature']}: variance={r['variance']}")
    
    print("\nResults saved to results/temperature_results.json")

if __name__ == "__main__":
    main()