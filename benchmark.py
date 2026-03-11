import ollama
import time
import psutil
import json
import os
from datetime import datetime

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_model(model_name: str, prompt: str, runs: int = 3):
    print(f"\nBenchmarking: {model_name}")
    print(f"Prompt: {prompt[:50]}...")
    
    results = []
    
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...")
        
        ram_before = get_ram_usage()
        start_time = time.time()
        first_token_time = None
        full_response = ""
        token_count = 0
        
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
            
            content = chunk["message"]["content"]
            full_response += content
            token_count += len(content.split())
        
        end_time = time.time()
        ram_after = get_ram_usage()
        
        total_latency = end_time - start_time
        time_to_first_token = first_token_time - start_time if first_token_time else 0
        tokens_per_second = token_count / total_latency if total_latency > 0 else 0
        
        result = {
            "run": i + 1,
            "time_to_first_token": round(time_to_first_token, 3),
            "total_latency": round(total_latency, 3),
            "tokens_per_second": round(tokens_per_second, 2),
            "token_count": token_count,
            "ram_used_mb": round(ram_after - ram_before, 2),
            "response_preview": full_response[:100]
        }
        
        results.append(result)
        print(f"  Latency: {total_latency:.2f}s | Tokens/sec: {tokens_per_second:.1f} | First token: {time_to_first_token:.3f}s")
    
    avg = {
        "model": model_name,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "runs": results,
        "average": {
            "time_to_first_token": round(sum(r["time_to_first_token"] for r in results) / runs, 3),
            "total_latency": round(sum(r["total_latency"] for r in results) / runs, 3),
            "tokens_per_second": round(sum(r["tokens_per_second"] for r in results) / runs, 2),
            "ram_used_mb": round(sum(r["ram_used_mb"] for r in results) / runs, 2),
        }
    }
    
    return avg

def save_results(results: dict, model_name: str):
    os.makedirs("results", exist_ok=True)
    filename = f"results/{model_name.replace(':', '_')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def main():
    test_prompts = [
        "What is machine learning? Explain in 3 sentences.",
        "Write a Python function to reverse a string.",
        "What are the symptoms of diabetes?"
    ]
    
    model = "llama3.2:3b"
    all_results = []
    
    for prompt in test_prompts:
        result = benchmark_model(model, prompt, runs=3)
        all_results.append(result)
    
    save_results(all_results, model)
    
    print("\nSUMMARY")
    print("=" * 50)
    for r in all_results:
        avg = r["average"]
        print(f"\nPrompt: {r['prompt'][:40]}...")
        print(f"  Avg Latency:        {avg['total_latency']}s")
        print(f"  Avg Tokens/sec:     {avg['tokens_per_second']}")
        print(f"  Avg First Token:    {avg['time_to_first_token']}s")
        print(f"  Avg RAM used:       {avg['ram_used_mb']} MB")

if __name__ == "__main__":
    main()