import datasets
import pandas as pd
def load_mtrag(split):
    try:
        if split == "train":
            # This training dataset already contains distractor documents and CoT answer format in a column mixed with in-domain documents using the P/1-P% RAFT mix
            df = pd.read_csv("/blue/cis6930/team7/CIS6930_T7_MTRAG/data/task_C_baseline/train_df_distractors_cot.tsv", sep="\t")
            df = pd.DataFrame(df)
            ds = datasets.Dataset.from_pandas(df)
        elif split == "val":
            df = pd.read_csv("/blue/cis6930/team7/CIS6930_T7_MTRAG/data/task_C_baseline/dev_df.tsv", sep="\t")
            df = pd.DataFrame(df)
            ds = datasets.Dataset.from_pandas(df)
        else: # test
            df = pd.read_csv("/blue/cis6930/team7/CIS6930_T7_MTRAG/data/task_C_baseline/test_df.tsv", sep="\t")
            df = pd.DataFrame(df)
            ds = datasets.Dataset.from_pandas(df)


    except ValueError as e:
        if "trust_remote_code" in str(e):
          raise ValueError("Loading Samsung/samsum requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set HF_DATASETS_TRUST_REMOTE_CODE env variable to True.") from e
        else:
          raise e
    return ds




def get_preprocessed_mtrag(tokenizer, split,raft,cot=0):
    dataset = load_mtrag(split)

    #if split == "train":
    if cot == 1:
        prompt = (
            f"Question: {{question}}\n\n context: {{context}}\n\n Instructions: Given the question, context, and answer above, provide a logical reasoning for that answer. Please use the format of: ##Reason: (reason) ##Answer: (answer). \n\n "
        )
    else:
        prompt = (
            f"Question: {{question}}\n\n context: {{context}}\n\n Instructions: Given the question, context, and answer above, provide the answer. Please use the format of: ##Answer: (answer). \n\n "
        )
    #else: # call the normal prompt on dev split; since we only want the comparative cross-entropy distributions on the answer and not the CoT answer
    #    prompt = (
    #        f"Question: {{question}}\n\n context: {{context}}\n\n Instructions: Given the question, context, and answer above, provide the answer. Please use the format of: ##Answer: (answer). \n\n "
    #    )



    def apply_prompt_template(sample,split,raft=0,cot=0):

        if split == 'train':
            if raft == 0 and cot == 0:
                return {
                    "prompt": prompt.format(question=sample["question"],context=sample["context"]),
                    "answer": sample["answer"]
                }
            elif raft==3 and cot == 0:
                return {
                    "prompt": prompt.format(question=sample["question"], context=sample["context_distractors"]),
                    "answer": sample["answer"]
                }
            elif raft==6 and cot == 0:
                return {
                    "prompt": prompt.format(question=sample["question"], context=sample["context_distractors_6"]),
                    "answer": sample["answer"]
                }
            elif raft==0 and cot == 1:
                return {
                    "prompt": prompt.format(question=sample["question"], context=sample["context"]),
                    "answer": sample["cot"]
                }
            elif raft == 3 and cot == 1:
                return {
                    "prompt": prompt.format(question=sample["question"], context=sample["context_distractors"]),
                    "answer": sample["cot"]
                }
            else:
                return {
                    "prompt": prompt.format(question=sample["question"], context=sample["context_distractors_6"]),
                    "answer": sample["cot"]
                }
        else:
            #if cot == 0:
            #    return {
            #        "prompt": prompt.format(question=sample["question"], context=sample["context"]),
            #        "answer": sample["answer"]
            #    }
            #else:
            return {
                "prompt": prompt.format(question=sample["question"], context=sample["context"]),
                "answer": sample["answer"]
            }



    dataset = dataset.map(apply_prompt_template, fn_kwargs={"split":split,"raft":raft,"cot":cot},remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
