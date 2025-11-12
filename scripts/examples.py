import sys
import json

with open("../../mt-rag-benchmark/human/evaluations/RAG.json") as f:
    j = json.load(f)

tasks = {
    t["task_id"]: t["annotations"]["rb_llm"]["composite"]["value"]
    for t in j["evaluations"]
    if t["model_id"] == "llama-3.1-8b-instruct"
}

test_tasks = []

with open("../../eval_dir_Meta-Llama-3.1-8B-Instruct_0_1/meta-llama_Meta-Llama-3.1-8B-Instruct_raft_0_cot_1.eval.jsonl") as f:
    for line in f.readlines():
        t = json.loads(line)
        tid = t["task_id"]
        test_tasks.append((tasks[tid], t["metrics"]["RB_llm"][0], tid))

for t in sorted(test_tasks, key=lambda t: t[0]):
    print(t)
