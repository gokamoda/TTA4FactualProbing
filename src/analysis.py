from glob import glob
import pandas as pd
from tqdm import tqdm

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


facts = pd.read_json('datasets/wikidata12500.jsonl', lines=True)
answers = facts['os']
p = facts['p']

def get_confidence(row):
    df_tmp = pd.DataFrame(row['candidates'])
    confidence = 1/df_tmp['normalized_score'].sum()
    return confidence

for model, folder in output_dirs.items():
    aggregation_results_path = glob(f'outputs/{folder}/aggregated_generations/*')
    for i in tqdm(range(len(aggregation_results_path))):
        result_path = aggregation_results_path[i]
        if not result_path.endswith('jsonl'):
            continue
        print(result_path)
        df = pd.read_json(result_path, lines=True)
        assert df.shape[0] == 12500, f'{result_path} has {df.shape[0]} rows'
        # if 'pid' in df.columns:
        #     continue

        # add qid
        df['pid'] = p

        # check answer
        df['answer'] = answers
        if "flan" in model:
            df['is_correct'] = df.apply(lambda row: row['top1']['generation'].lower() == row['answer'].lower(), axis=1)
        else:
            df['is_correct'] = df.apply(lambda row: row['top1']['generation'] == row['answer'], axis=1)

        # get confidence
        df['confidence'] = df.apply(get_confidence, axis=1)
        df['normalized_confidence'] = df['confidence'] / df['confidence'].max()

        df.to_json(result_path, orient='records', lines=True)
        del df