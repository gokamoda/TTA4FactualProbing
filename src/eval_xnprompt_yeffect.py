import pandas as pd
import os

output_dirs = {
    # "t5-small": "2023-10-14/10-48-22",
    # "t5-small-gpt3": "2023-10-14/22-34-51",
    # "t5-large": "2023-10-14/10-47-44",
    # "t5-large-gpt3": "2023-10-15/02-06-44",
    # "t5-3b": "2023-10-14/00-58-42",
    # "t5-3b-gpt3": "2023-10-14/22-32-55",
    # "t5-11b": "2023-10-14/00-58-12",
    # "t5-11b-gpt3": "2023-10-14/22-31-20",
    # "flan-small": "2023-10-14/00-46-38",
    # "flan-xl": "2023-10-14/00-46-39",
    # "t0_3b": "2023-10-14/00-10-47",

    "T5-Small (Count)": "2023-10-18/17-59-55",
    "T5-Large (Count)": "2023-10-18/21-26-26",
    "T5-3B (Count)": "2023-10-18/21-28-37",
    "T5-11B (Count)": "2023-10-18/21-29-05",
}

ITER_NUM = 5
MAX_PROMPTS = 30
GPT3 = True

dfs = []
for model_name, output_dir in output_dirs.items():
    if os.path.exists(f'outputs/{output_dir}/agg_n_corrects.jsonl') and "flan" not in model_name:
        df = pd.read_json(f'outputs/{output_dir}/agg_n_corrects.jsonl', lines=True)
        dfs.append(df)
        print(df)
        continue
    n_corrects_model = []
    for nprompt in range(1, MAX_PROMPTS+1):

        if "gpt3" in model_name and nprompt>11:
            break

        if nprompt == 1 or nprompt == MAX_PROMPTS or ("gpt3" in model_name and nprompt==11):
            print('A')
            # try:
            print(f'outputs/{output_dir}/aggregated_generations/random_sample_0_{nprompt}.jsonl')
            df = pd.read_json(f'outputs/{output_dir}/aggregated_generations/random_sample_0_{nprompt}.jsonl', lines=True)[['pid', 'is_correct']]
            # except:
            #     print(f'../outputs/{output_dir}/aggregated_generations/random_sample_0_{nprompt}.jsonl')
            #     raise FileExistsError
            data = df.groupby('pid').sum('is_correct').to_dict()['is_correct']
            data['nprompt'] = nprompt
            n_corrects_model.append(data)    
        else:
            for seed in range(ITER_NUM):
                df = pd.read_json(f'outputs/{output_dir}/aggregated_generations/random_sample_{seed}_{nprompt}.jsonl', lines=True)
                assert df.shape[0] == 12500
                data = df.groupby('pid').sum('is_correct').to_dict()['is_correct']
                data['nprompt'] = nprompt
                n_corrects_model.append(data)   
            # n_corrects_model.append(n_corrects_tmp)
    
    df = pd.DataFrame(n_corrects_model)
    print(df)

    df.index.name = "nprompt"
    df['model'] = model_name
    df.to_json(f'outputs/{output_dir}/agg_n_corrects.jsonl', orient="records", lines=True)
    dfs.append(df)

df = pd.concat(dfs)
# df_to_save = df.reset_index(drop=False)
df.to_json("src/eval_xnprompt_yeffect_count.jsonl", orient="records", lines=True)