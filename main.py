import ollama
import time
import psutil
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Local SLM Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = ["llama3.2:3b", "phi4-mini", "mistral:7b"]

# Request models
class ChatRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7

class BenchmarkRequest(BaseModel):
    model: str
    prompt: str
    runs: int = 3

# Routes
@app.get("/")
def root():
    return {"status": "running", "models": MODELS}

@app.get("/models")
def get_models():
    return {"models": MODELS}

@app.post("/chat")
def chat(req: ChatRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {req.model} not supported")
    
    start = time.time()
    first_token_time = None
    full_response = ""
    token_count = 0

    stream = ollama.chat(
        model=req.model,
        messages=[{"role": "user", "content": req.prompt}],
        options={"temperature": req.temperature},
        stream=True
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        content = chunk["message"]["content"]
        full_response += content
        token_count += len(content.split())

    total_latency = round(time.time() - start, 3)
    ttft = round(first_token_time - start, 3) if first_token_time else 0
    tps = round(token_count / total_latency, 2) if total_latency > 0 else 0

    return {
        "model": req.model,
        "prompt": req.prompt,
        "response": full_response,
        "metrics": {
            "total_latency": total_latency,
            "time_to_first_token": ttft,
            "tokens_per_second": tps,
            "token_count": token_count
        }
    }

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {req.model} not supported")

    results = []

    for i in range(req.runs):
        start = time.time()
        first_token_time = None
        full_response = ""
        token_count = 0

        stream = ollama.chat(
            model=req.model,
            messages=[{"role": "user", "content": req.prompt}],
            stream=True
        )

        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
            content = chunk["message"]["content"]
            full_response += content
            token_count += len(content.split())

        total_latency = round(time.time() - start, 3)
        ttft = round(first_token_time - start, 3) if first_token_time else 0
        tps = round(token_count / total_latency, 2) if total_latency > 0 else 0

        results.append({
            "run": i + 1,
            "total_latency": total_latency,
            "time_to_first_token": ttft,
            "tokens_per_second": tps,
            "token_count": token_count
        })

    avg = {
        "total_latency": round(sum(r["total_latency"] for r in results) / req.runs, 3),
        "time_to_first_token": round(sum(r["time_to_first_token"] for r in results) / req.runs, 3),
        "tokens_per_second": round(sum(r["tokens_per_second"] for r in results) / req.runs, 2),
    }

    return {
        "model": req.model,
        "prompt": req.prompt,
        "runs": results,
        "average": avg
    }

@app.get("/results")
def get_results():
    results = {}
    results_dir = "results"

    if not os.path.exists(results_dir):
        return {"error": "No results found"}

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            with open(f"{results_dir}/{filename}") as f:
                results[filename] = json.load(f)

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)