"""
# Evaluation Metrics Summary

This document explains each metric reported in `metrics_summary.csv`.

| Metric | Description | Importance |
| ------- | ------------ | ----------- |
| **Recall** | Proportion of relevant docs retrieved. | Measures retriever coverage. |
| **RB_agg** | Retrieval-based aggregate combining recall@k, precision, etc. | Captures retriever effectiveness. |
| **RB_llm** | Retrieval quality judged by LLM. | Semantic retrieval quality. |
| **RL_F** | ROUGE-L F1 score. | Generation faithfulness. |

See the IBM MTRAG repo for how `RB_agg` is computed.
"""

import json
import pandas as pd
import argparse


def get_metrics_original(input_file,model_name,raft,cot):

    eval_dir = 'metrics/'

    # === STEP 1: Read JSON and flatten metrics ===
    records = []
    with open(input_file, "r") as f:
        for line in f:
            rec = {}
            line = line.strip()
            try:
                data = json.loads(line)
                rec['RL_F'] = data['metrics']['RL_F'][0]
                rec['RB_agg'] = data['metrics']['RB_agg'][0]
                rec['RB_llm'] = data['metrics']['RB_llm'][0]

                # The conditional IDK metrics are the ones reported in the IBM paper.
                # These are the ones we should report for our results
                rec['RL_F_idk'] = data['metrics']['RL_F_idk'][0]
                rec['RB_agg_idk'] = data['metrics']['RB_agg_idk'][0]
                rec['RB_llm_idk'] = data['metrics']['RB_llm_idk'][0]

                rec['Multi-Turn'] = data["Multi-Turn"][0]
                rec["Answerability"] = data["Answerability"][0]
                rec["Question Type"] = data["Question Type"][0]
                if 'govt' in data['Collection']:
                    rec['domain'] = 'govt'
                elif 'cloud' in data['Collection']:
                    rec['domain'] = 'cloud'
                elif 'fiqa' in data['Collection']:
                    rec['domain'] = 'fiqa'
                else:
                    rec['domain'] = 'clapnq'

                records.append(rec)

            except Exception as e:
                print ('The faulty line')
                print (data)
                raise

    # === STEP 2: Convert to DataFrame ===

    df = pd.DataFrame(records)
    df.to_csv(
        eval_dir + 'raw_metrics_' + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.csv',
        index=False)
    print ('Overall aggregate')
    df1 = df[['RL_F_idk', 'RB_agg_idk', 'RB_llm_idk']].mean()
    print(df1)
    df1.to_csv(
        eval_dir + 'overall_aggregate_' + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.csv',
        index=False)

    print ('\n')

    print ('Domain')
    df1 = df.groupby(['domain'],as_index=False)[['RL_F_idk','RB_agg_idk','RB_llm_idk']].mean()
    print (df1)
    df1.to_csv(eval_dir + 'domain_aggregate_' + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.csv',index=False)
    print ('\n')

    print ('Answerability')
    df1 = df.groupby(['Answerability'], as_index=False)[
        ['RL_F_idk', 'RB_agg_idk', 'RB_llm_idk']].mean()
    print (df1)
    df1.to_csv(
        eval_dir + 'answerability_aggregate_' + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.csv',
        index=False)

    print('\n')

    print('Question Type')
    df1 = df.groupby(['Question Type'], as_index=False)[
        ['RL_F_idk', 'RB_agg_idk', 'RB_llm_idk']].mean()
    print(df1)
    df1.to_csv(
        eval_dir + 'qntype_aggregate_' + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.csv',
        index=False)

    print('\n')

    print('Multi-Turn')
    df1 = df.groupby(['Multi-Turn'], as_index=False)[
        ['RL_F_idk', 'RB_agg_idk', 'RB_llm_idk']].mean()
    print(df1)
    df1.to_csv(
        eval_dir + 'mturn_aggregate_' + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.csv',
        index=False)

    print('\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str,
                        help="The path to the jsonl file for the test data, with the metrics")
    parser.add_argument('--raft', type=int, default=0,
                        help="include distractor docs or not, 0, 3, 6")
    parser.add_argument('--cot', type=int, default=1,
                        help="include chain-of-thought answers or not for training; 0 or 1")
    parser.add_argument('--modelname', type=str, default="meta-llama/Meta-Llama-3.1-8B",
                        help="name of the LLM model used to fine-tune")

    args = parser.parse_args()

    input_file = args.input_file
    raft = args.raft
    cot = args.cot
    model_name = args.modelname  # should have a "/" in the modelname for HF models

    if raft not in [0, 3, 6]:
        print('This must be either 0, 3, or 6 distractor documents')
        return

    if cot not in [0, 1]:
        print('This is a binary value')
        return
    
    get_metrics_original(input_file,model_name, raft, cot)


if __name__ == "__main__":
    main()
