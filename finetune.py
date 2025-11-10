print ("Starting script..")

import torch
import os, json, re
import pandas as pd
from transformers import LlamaForCausalLM, AutoTokenizer
from llama_cookbook.configs import train_config as TRAIN_CONFIG
from transformers import BitsAndBytesConfig
from mtrag_utils import get_dataloader
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, PeftModel
from dataclasses import asdict
from llama_cookbook.configs import lora_config as LORA_CONFIG
import torch.optim as optim
from llama_cookbook.utils.train_utils import train
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from tqdm import tqdm
import argparse

import hf_login

# Huggingface cache dirs - do not change
os.environ['HF_HOME'] = '/blue/cis6930/team7/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = '/blue/cis6930/team7/hf_cache/'
os.environ['HF_DATASETS_CACHE'] = '/blue/cis6930/team7/hf_cache/'



print ('Starting')



def training(output_dir,model_name,raft,cot=0,num_epochs=10,batch_size=1,grad_steps=4,lr=3e-4,context_length=8192,lora_rank=4,lora_alpha=32,lora_dropout=0.1):

    """
    The main training function.
    :param raft: whether to use distractor documents
    :param cot: whether to use chain-of-thought answers for training
    :return: saves an output_dir containing peft weights; also outputs a json of loss/perplexity values in the output dir
    """

    # Example prompt to test qualitative output after model training
    if cot == 0:
        eval_prompt = """
        Question: How do I find a good lawyer to create my estate plan and will?

        context: [Don't feel like you have to jump at the first person whose name is given to you, Kirchick says. I think that people should interview two or more attorneys, accountants, trust officers, financial advisors and so on. According to financial planning experts, the average initial cost for the legal drafting of a will, living will and durable power of attorney documentation is between $500 and $1,200, depending on the family size and location. Continue to review your plan over time. Finally, your estate plan should never be a one and done thing, according to Henn. Every five to seven years, the documents should be readdressed to adapt to significant life events, tax law changes or even the addition of more children, he says. It is also important to keep tabs on your insurance policies and investments, as they all tie into the estate plan and can fluctuate based on the economic environment. If you have to make revisions, Henn says it will cost as much as it did to create the documents in the first place.]...[**What you need to know about [estate planning](http://money.cnn.com/magazines/moneymag/money101/lesson21/), including why you may need a will and assigning a power of attorney.** **1. No matter your net worth, it's important to have a basic estate plan in place.** Such a plan ensures that your family and financial goals are met after you die. **2. An estate plan has several elements.** They include: a will; assignment of power of attorney; and a living will or health-care proxy (medical power of attorney). For some people, a trust may also make sense. When putting together a plan, you must be mindful of both [federal and state laws](http://corlisslawgroup.com/) governing estates. **3. Taking inventory of your assets is a good place to start.** Your assets include your investments, retirement savings, insurance policies, and real estate or business interests. Ask yourself three questions: Whom do you want to inherit your assets? Whom do you want handling your financial affairs if you're ever incapacitated? Whom do you want making medical decisions for you if you become unable to make them for yourself? **4. Everybody needs a will.** A will tells the world exactly where you want your assets distributed when you die. It's also the best place to name guardians for your children. Dying without a will -- also known as dying intestate -- can be costly to your heirs and leaves you no say over who gets your assets. Even if you have a trust, you still need a will to take care of any holdings outside of that trust when you die. **5. Trusts aren't just for the wealthy.** Trusts are legal mechanisms that let you put conditions on how and when your assets will be distributed upon your death. They also allow you to reduce your estate and gift taxes and to distribute assets to your heirs without the cost, delay and publicity of probate court, which administers wills. Some also offer greater protection of your assets from creditors and lawsuits. **6.]...[Before jumping into the estate planning process, it's important to establish exactly what you want, and need, to happen after you die and relay those wishes to those around you. We find that the best transitions and financial transfers happen when all family members are involved in the [decision making](http://corlisslawgroup.com), says John Sweeney, executive vice president of retirement and investing strategies at Fidelity Investments. This way, after a loved one is gone, no one is squabbling over a couch or going, 'Why did person A get more than person B?' If wishes are laid out clearly while the individual is living, they can share the rationale behind the decisions. Focus on the basic estate plan components. Experts say life insurance, a will, a living will and a durable power of attorney are all important aspects of an estate plan that should be established at the start of the planning process. In the event of an untimely death, life insurance can replace lost earnings, which can be especially beneficial for younger individuals, says Bill Kirchick, a partner with Bingham McCutchen law firm in Boston. Young people can't afford to die, he says. They are going to lose a source of income if something happens to a young couple and they haven't had enough time to accumulate wealth from earnings to put aside in savings or a retirement plan. Also, the earlier you take out a life insurance policy, the more likely you are to be approved for reduced rates compared to older individuals. Utilize estate planning professionals. To draft these basic estate plans, experts recommend carefully selecting a team of professionals who will educate you and draft what you need based on your individual situation. Don't feel like you have to jump at the first person whose name is given to you, Kirchick says. I think that people should interview two or more attorneys, accountants, trust officers, financial advisors and so on. According to financial planning experts, the average initial cost for the legal drafting of a will, living will and durable power of attorney documentation is between $500 and $1,200, depending on the family size and location. Continue to review your plan over time.]...[Be careful when you say insurance -- these things are service plans. They provide you with specific services and discounts in exchange for a pre-determined fee. So you pay $299/year and get a will, telephone advice and similar services. Insurance, like liability insurance, guarantees compensation for specific losses. You can sometimes pay attorneys a retainer and get some discounts on services. This is only cost effective if you have enough work. These plans might make sense, depending on what you need.]...[Generally, it would be an accountant. Specifically in the case of very private (or unorganized, which is even worse) person - forensic accountant. Since there's no will - it will probably require a lawyer as well to gain access to all the accounts the accountant discovers. I would start with a good estate attorney, who in turn will hire a forensic accountant to trace the accounts.]

        Instructions: Given the question, context, and answer above, provide the answer. Please use the format of: ##Answer: (answer).

        """
    else:
        eval_prompt = """
           Question: How do I find a good lawyer to create my estate plan and will?
    
           context: [Don't feel like you have to jump at the first person whose name is given to you, Kirchick says. I think that people should interview two or more attorneys, accountants, trust officers, financial advisors and so on. According to financial planning experts, the average initial cost for the legal drafting of a will, living will and durable power of attorney documentation is between $500 and $1,200, depending on the family size and location. Continue to review your plan over time. Finally, your estate plan should never be a one and done thing, according to Henn. Every five to seven years, the documents should be readdressed to adapt to significant life events, tax law changes or even the addition of more children, he says. It is also important to keep tabs on your insurance policies and investments, as they all tie into the estate plan and can fluctuate based on the economic environment. If you have to make revisions, Henn says it will cost as much as it did to create the documents in the first place.]...[**What you need to know about [estate planning](http://money.cnn.com/magazines/moneymag/money101/lesson21/), including why you may need a will and assigning a power of attorney.** **1. No matter your net worth, it's important to have a basic estate plan in place.** Such a plan ensures that your family and financial goals are met after you die. **2. An estate plan has several elements.** They include: a will; assignment of power of attorney; and a living will or health-care proxy (medical power of attorney). For some people, a trust may also make sense. When putting together a plan, you must be mindful of both [federal and state laws](http://corlisslawgroup.com/) governing estates. **3. Taking inventory of your assets is a good place to start.** Your assets include your investments, retirement savings, insurance policies, and real estate or business interests. Ask yourself three questions: Whom do you want to inherit your assets? Whom do you want handling your financial affairs if you're ever incapacitated? Whom do you want making medical decisions for you if you become unable to make them for yourself? **4. Everybody needs a will.** A will tells the world exactly where you want your assets distributed when you die. It's also the best place to name guardians for your children. Dying without a will -- also known as dying intestate -- can be costly to your heirs and leaves you no say over who gets your assets. Even if you have a trust, you still need a will to take care of any holdings outside of that trust when you die. **5. Trusts aren't just for the wealthy.** Trusts are legal mechanisms that let you put conditions on how and when your assets will be distributed upon your death. They also allow you to reduce your estate and gift taxes and to distribute assets to your heirs without the cost, delay and publicity of probate court, which administers wills. Some also offer greater protection of your assets from creditors and lawsuits. **6.]...[Before jumping into the estate planning process, it's important to establish exactly what you want, and need, to happen after you die and relay those wishes to those around you. We find that the best transitions and financial transfers happen when all family members are involved in the [decision making](http://corlisslawgroup.com), says John Sweeney, executive vice president of retirement and investing strategies at Fidelity Investments. This way, after a loved one is gone, no one is squabbling over a couch or going, 'Why did person A get more than person B?' If wishes are laid out clearly while the individual is living, they can share the rationale behind the decisions. Focus on the basic estate plan components. Experts say life insurance, a will, a living will and a durable power of attorney are all important aspects of an estate plan that should be established at the start of the planning process. In the event of an untimely death, life insurance can replace lost earnings, which can be especially beneficial for younger individuals, says Bill Kirchick, a partner with Bingham McCutchen law firm in Boston. Young people can't afford to die, he says. They are going to lose a source of income if something happens to a young couple and they haven't had enough time to accumulate wealth from earnings to put aside in savings or a retirement plan. Also, the earlier you take out a life insurance policy, the more likely you are to be approved for reduced rates compared to older individuals. Utilize estate planning professionals. To draft these basic estate plans, experts recommend carefully selecting a team of professionals who will educate you and draft what you need based on your individual situation. Don't feel like you have to jump at the first person whose name is given to you, Kirchick says. I think that people should interview two or more attorneys, accountants, trust officers, financial advisors and so on. According to financial planning experts, the average initial cost for the legal drafting of a will, living will and durable power of attorney documentation is between $500 and $1,200, depending on the family size and location. Continue to review your plan over time.]...[Be careful when you say insurance -- these things are service plans. They provide you with specific services and discounts in exchange for a pre-determined fee. So you pay $299/year and get a will, telephone advice and similar services. Insurance, like liability insurance, guarantees compensation for specific losses. You can sometimes pay attorneys a retainer and get some discounts on services. This is only cost effective if you have enough work. These plans might make sense, depending on what you need.]...[Generally, it would be an accountant. Specifically in the case of very private (or unorganized, which is even worse) person - forensic accountant. Since there's no will - it will probably require a lawyer as well to gain access to all the accounts the accountant discovers. I would start with a good estate attorney, who in turn will hire a forensic accountant to trace the accounts.]
    
           Instructions: Given the question, context, and answer above, provide the answer. Please use the format of: ##Reason: (reason) ##Answer: (answer).
    
           """

    train_config = TRAIN_CONFIG()
    train_config.model_name = model_name
    train_config.num_epochs = num_epochs
    train_config.run_validation = True
    train_config.save_metrics = True
    train_config.gradient_accumulation_steps = grad_steps
    train_config.batch_size_training = batch_size
    train_config.lr = lr
    train_config.use_fast_kernels = True
    train_config.use_fp16 = True
    train_config.context_length = context_length # if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
    train_config.batching_strategy = "packing"
    train_config.output_dir = output_dir
    train_config.use_peft = True
    train_config.save_metrics=True
    train_config.weight_decay = 0.01 # L2 regularization

    lora_config = LORA_CONFIG()
    lora_config.r = lora_rank
    lora_config.lora_alpha = lora_alpha
    lora_config.lora_dropout = lora_dropout

    config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                device_map="auto",
                quantization_config=config,
                use_cache=False,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                torch_dtype=torch.float32, cache_dir='/blue/cis6930/team7/hf_cache/'
            )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader = get_dataloader(tokenizer, train_config,raft=raft,cot=cot,split="train")
    eval_dataloader = get_dataloader(tokenizer, train_config,raft=raft,cot=cot,split="val")

    peft_config = LoraConfig(**asdict(lora_config))


    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    """
    if raft == 0:
        steps_per_ep = 130
    elif raft == 3:
        steps_per_ep = 173
    else:
        steps_per_ep = 237
    """

    #scheduler = OneCycleLR(optimizer, max_lr=lr,steps_per_epoch=steps_per_ep,epochs=num_epochs,anneal_strategy='cos')
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    print ('Starting training')


    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run=None,

    )

    model.save_pretrained(train_config.output_dir)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    if cot == 0:
        output_token_count = 150
    else:
        output_token_count = 250

    model.eval()
    with torch.inference_mode():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=output_token_count)[0], skip_special_tokens=True)) # 150 is between 75th and 90th percentile of train response lengths

def inference(peft_output_dir,model_name,raft,cot=0,use_fast_kernels=True):

    """
    :param peft_output_dir: directory where PEFT/LORA weights are stored
    :param model_name: original model used in training
    :param use_fast_kernels: from the train config (True)
    :return: a jsonl file for evaluation using LLM-as-a-judge
    """

    #config = BitsAndBytesConfig(
    #    load_in_8bit=True,
    #)
    def clean(txt):
        txt = re.sub('\s+', ' ', txt)
        txt = re.sub('\n+', ' ', txt)
        txt = re.sub('\t+', ' ', txt)
        txt = re.sub(' +', ' ', txt)
        txt = txt.replace('"', '')

        return txt

    # Keep the test file fixed across all experiments
    test_dataset = "/blue/cis6930/team7/CIS6930_T7_MTRAG/data/task_C_baseline/test_df.tsv"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print ('Loading model')
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        #return_dict=True,
        device_map="auto",
        #quantization_config=config,
        use_cache=False,
        attn_implementation="sdpa" if use_fast_kernels else None,
        torch_dtype=torch.float16, cache_dir='/blue/cis6930/team7/hf_cache/'
    )

    print ('Loading PEFT params')
    model = PeftModel.from_pretrained(model,peft_output_dir)

    test_dataset = pd.read_csv(test_dataset, sep="\t")
    test_dataset = pd.DataFrame(test_dataset)

    if cot == 1:
        prompt = (
            f"Question: {{question}}\n\n context: {{context}}\n\n Instructions: Given the question, context, and answer above, provide a logical reasoning for that answer. Please use the format of: ##Reason: (reason) ##Answer: (answer). \n\n "
        )
    else:
        prompt = (
            f"Question: {{question}}\n\n context: {{context}}\n\n Instructions: Given the question, context, and answer above, provide the answer. Please use the format of: ##Answer: (answer). \n\n "
        )

    model.eval()

    print ('Generating predictions')

    with open("/blue/cis6930/team7/CIS6930_T7_MTRAG/outputs/" + "meta-llama_" + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.tsv','w') as w:
        w.write("conversation_id\tturn\tprediction\n")
        with torch.inference_mode():
            for _,row in tqdm(test_dataset.iterrows()):
                test_prompt = prompt.format(question=row["question"], context=row["context"])
                test_input = tokenizer(test_prompt, return_tensors="pt").to("cuda")
                test_output = tokenizer.decode(model.generate(**test_input, max_new_tokens=150)[0], skip_special_tokens=True)

                test_output = clean(test_output)
                test_output = re.split(r'##Answer: \(answer\).',test_output)[-1]

                w.write(row['conversation_id'] + '\t' + str(row['turn']) + '\t' + test_output.strip() + '\n')


def create_eval_jsonl(model_name,raft,cot=0):
    """

    :param test_output: predictions on test set (a tsv file)
    :return:
    """

    test_output = "/blue/cis6930/team7/CIS6930_T7_MTRAG/outputs/" + "meta-llama_" + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + '.tsv'
    json_file = "RAG.jsonl" # Task C file; already contains 5 documents using Elser retrieval
    preds = pd.read_csv(test_output,sep='\t')

    preds_jsonl = "/blue/cis6930/team7/CIS6930_T7_MTRAG/outputs/" + "meta-llama_" + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot) + ".jsonl"

    with open(preds_jsonl,'w') as w:
        with open(json_file, 'r') as f:
            recs = list(f)

        for r in recs:
            r = json.loads(r)
            #if r['conversation_id'] == "535ebd306304982e4e49cc989ce5b10b":
            #    print ('here')
            pred = preds.loc[(preds['conversation_id'] == r['conversation_id']) & (preds['turn'] == int(r['turn']))]
            if len(pred) == 0: continue

            #try:
            r['predictions'] = [{"text": pred['prediction'].tolist()[0].strip()}]
            #except Exception:
            #    print ('here')
            json.dump(r,w)
            w.write('\n')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raft', type=int, default=0,
                        help="number distractors to include, 0,3,6")
    parser.add_argument('--cot', type=int, default=0, help="include chain-of-thought answers or not for training; 0 or 1")
    parser.add_argument('--modelname', type=str, default="meta-llama/Meta-Llama-3.1-8B",
                        help="name of the LLM model from HF")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--gradsteps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--contextlength', type=int, default=8192, help="prompt context length")
    parser.add_argument('--lorarank', type=int, default=8)
    parser.add_argument('--loraalpha', type=int, default=32)
    parser.add_argument('--loradropout', type=int, default=0.1)

    args = parser.parse_args()

    raft = args.raft
    cot = args.cot
    model_name = args.modelname # should have a "/" in the modelname for HF models

    if raft not in [0,3,6]:
        print ('This must be either 0, 3, or 6 distractor documents')
        return

    if cot not in [0,1]:
        print ('This is a binary value')
        return

    print ('raft:' + str(raft) + ' cot:' + str(cot))

    num_epochs = args.epochs
    batch_size = args.batchsize
    grad_steps = args.gradsteps
    lr = args.lr
    context_length = args.contextlength
    lora_rank = args.lorarank
    lora_alpha =args.loraalpha
    lora_dropout = args.loradropout

    # PEFT output directory; will be overwritten every time.
    output_dir = "meta-llama_" + model_name.split('/')[-1] + "_raft_" + str(raft) + "_cot_" + str(cot)

    print ('Starting training')
    training(output_dir,model_name,raft,cot,num_epochs, batch_size, grad_steps, lr, context_length, lora_rank, lora_alpha, lora_dropout) # training

    print ('Starting inference')
    inference(output_dir,model_name, raft, cot) # inference

    print ('Creating jsonl predictions')
    create_eval_jsonl(model_name,raft,cot) # generates jsonl with predictions for LLM-as-a-judge


if __name__ == "__main__":
    main()




