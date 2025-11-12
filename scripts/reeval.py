import sys
import json

with open("../mt-rag-benchmark/human/evaluations/RAG.json") as f:
    j = json.load(f)

evals = {
    e["task_id"]: e["model_response"]
    for e in j["evaluations"]
    if e["model_id"] == "llama-3.1-8b-instruct"
}

with open("../mt-rag-benchmark/human/generation_tasks/RAG.jsonl") as f:
    for line in f.readlines():
        t = json.loads(line)
        e = evals[t["task_id"]]
        t["predictions"] = [{"text": e}]
        print(json.dumps(t))
