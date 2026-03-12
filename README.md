# 🧠 Local SLM Benchmark

A systematic benchmarking study comparing **local language models running entirely offline via Ollama**.
The project evaluates inference speed, structured output reliability, and response behavior across standardized prompt categories.

All experiments run **locally with no cloud APIs or external services**.

---

#  Benchmark Dashboard

<img width="1175" height="870" alt="image" src="https://github.com/user-attachments/assets/1eb32a45-4cad-4c34-ab33-cd86432cb8ad" />


<img width="1226" height="460" alt="image" src="https://github.com/user-attachments/assets/b6d8eb5e-b5dd-4d10-8734-f8063db0872c" />



The dashboard visualizes benchmark results including:

* Model latency comparison
* Tokens per second
* Structured output success rate
* Temperature variance across prompts

---

#  Models Tested

| Model       | Size   | Format       |
| ----------- | ------ | ------------ |
| llama3.2:3b | 2.0 GB | Q4 Quantized |
| phi4-mini   | 2.5 GB | Q4 Quantized |
| mistral:7b  | 4.4 GB | Q4 Quantized |

All models are executed locally using **Ollama**.

---

#  Benchmark Phases

## Phase 1 — Inference Benchmark

Measures core inference performance:

* Latency per prompt
* Tokens per second
* Time to first token (TTFT)

---

## Phase 2 — Structured Output Compliance

Tests whether models correctly follow **JSON schema instructions** using **Pydantic validation with retry logic**.

---

## Phase 3 — Model Comparison

Compares the three models using **15 standardized prompts** covering:

* Logic reasoning
* Coding tasks
* Creative responses

---

## Phase 4 — Temperature Variance

Evaluates how model responses change across temperature settings:

* `temperature = 0.0` → deterministic output
* `temperature = 0.5` → moderate creativity
* `temperature = 1.0` → high randomness

---

#  Key Findings

* **llama3.2:3b** → fastest inference performance
* **phi4-mini** → best structured output compliance
* **mistral:7b** → significantly slower in CPU-only environments

---

# 📁 Project Structure

```
local-slm-benchmark/

├── benchmark.py           # Phase 1 — latency and tokens/sec
├── structured.py          # Phase 2 — JSON schema validation
├── temperature_test.py    # Phase 4 — temperature variance
├── compare.py             # Phase 3 — multi-model comparison
├── main.py                # FastAPI backend
├── dashboard.html         # Results dashboard
└── results/
    ├── llama3.2_3b.json
    ├── structured_results.json
    ├── temperature_results.json
    └── comparison_data.json
```

---

#  Setup

### Pull models

```bash
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b
```

### Install dependencies

```bash
pip install fastapi uvicorn ollama pydantic
```

### Run the API

```bash
uvicorn main:app --reload
```

---

# Tech Stack

* **Ollama**
* **FastAPI**
* **Pydantic**
* **Python 3.12**

---

#  Endpoints

| Method | Endpoint     | Description                   |
| ------ | ------------ | ----------------------------- |
| GET    | `/models`    | List available models         |
| POST   | `/chat`      | Send a prompt to a model      |
| POST   | `/benchmark` | Run benchmark on a model      |
| GET    | `/results`   | Fetch saved benchmark results |

---


