import ollama
import json
import time
from pydantic import BaseModel, ValidationError
from typing import Optional

# Pydantic schema
class MedicalInfo(BaseModel):
    condition: str
    symptoms: list[str]
    treatments: list[str]
    severity: str
    seek_doctor: bool

class CodeSolution(BaseModel):
    language: str
    function_name: str
    code: str
    explanation: str

class FactAnswer(BaseModel):
    question: str
    answer: str
    confidence: str
    source_type: str

def ask_structured(model: str, prompt: str, schema: dict, retries: int = 2):
    attempt = 0
    
    while attempt <= retries:
        attempt += 1
        print(f"  Attempt {attempt}...")
        
        system_prompt = f"""You must respond ONLY with valid JSON.
No extra text, no markdown, no code blocks.
Follow this exact schema:
{json.dumps(schema, indent=2)}"""

        start = time.time()
        
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0}
        )
        
        latency = round(time.time() - start, 3)
        raw = response["message"]["content"].strip()
        
        # Clean response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        
        try:
            parsed = json.loads(raw)
            print(f"  Success in {latency}s")
            return {"success": True, "data": parsed, "latency": latency, "attempts": attempt}
        except json.JSONDecodeError as e:
            print(f"  JSON error: {e}")
            if attempt > retries:
                return {"success": False, "error": str(e), "raw": raw, "attempts": attempt}

def run_tests(model: str):
    print(f"\nTesting structured output: {model}")
    print("=" * 50)
    
    tests = [
        {
            "name": "Medical Info",
            "prompt": "Give me information about diabetes.",
            "schema": MedicalInfo.model_json_schema()
        },
        {
            "name": "Code Solution",
            "prompt": "Write a Python function to reverse a string.",
            "schema": CodeSolution.model_json_schema()
        },
        {
            "name": "Fact Answer",
            "prompt": "What is the capital of France?",
            "schema": FactAnswer.model_json_schema()
        }
    ]
    
    results = []
    passed = 0
    
    for test in tests:
        print(f"\nTest: {test['name']}")
        result = ask_structured(model, test["prompt"], test["schema"])
        result["test_name"] = test["name"]
        results.append(result)
        
        if result["success"]:
            passed += 1
            print(f"  Result: {json.dumps(result['data'], indent=2)[:200]}")
        else:
            print(f"  Failed after {result['attempts']} attempts")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return results

if __name__ == "__main__":
    models = ["llama3.2:3b", "phi4-mini", "mistral:7b"]
    
    all_results = {}
    for model in models:
        all_results[model] = run_tests(model)
    
    with open("results/structured_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to results/structured_results.json")