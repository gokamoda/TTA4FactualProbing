# Test-time Augmentation for Factual Probing


This repository contains code and data for the EMNLP 2023 Findings paper *Test-time Augmentation for Factual Probing* by Go Kamoda, Benjamin Heinzerling, Keisuke Sakaguchi and Kentaro Inui.

## Preparations
1. Clone this repository
   - `config/` contains configurations for experiments.
   - `datasets/` contains the dataset we used, which was extracted from WikiData.
   - `src/` contains codes for running experiments and analyses.
2. Install dependencies (use requirements.txt)

## Running Experiments
### Overview
- This repository use Hydra to pass arguments in CLI.
- Logs, results, and run settings will be stored into `outputs` directory.

### Details
- To run TTA on default setting, run the following code. It will 1) get datasset with original prompts, 2) augment prompts, 3) generate answer candidate, and 4) aggregate the generated answers.
  ```
  python src/main.py
  ```
  - The outputs would start like this:
    ```
    % python src/main.py
    dataset:
        name: wikidata12500
        path: datasets/wikidata12500.jsonl
    model:
        name: t5-small
        family: t5
        model_path: google/t5-small-ssm-nq
        device_map: auto
        generation_split: null
        generation_combine: null
        batch_size: 32
    augment_methods:
        word_swapping:
        - type: wordswap
            num_return_sequences: 4
            label: word_swapping_wordswap
        - type: wordnet
            num_return_sequences: 4
            label: word_swapping_wordnet
        back_translation:
        - target_language: fr
            num_return_sequences: 4
            label: back_translation_fr
        - target_language: ru
            num_return_sequences: 4
            label: back_translation_ru
        - target_language: de
            num_return_sequences: 4
            label: back_translation_de
        - target_language: es
            num_return_sequences: 4
            label: back_translation_es
        - target_language: jap
            num_return_sequences: 4
            label: back_translation_ja
        stopwords_filtering:
        - num_return_sequences: 1
            label: stopwords_filtering
    generation_args:
        do_sample: false
        output_scores: true
        return_dict_in_generate: true
        num_beams: 10
        num_return_sequences: 10
        max_new_tokens: 10
    aggregation_method:
        name: random_sample
        iter_num: 5
        function: sum
    logger:
        slack: false
    log_dir: outputs/2023-10-25/13-55-59
    filename: main.log

    2023-10-25 13:55:59,196/INFO_SLACK/__main__/main():29
    outputs/2023-10-25/13-55-59

    2023-10-25 13:55:59,245/INFO/Facts/__init__():26
    Facts: 12500

    2023-10-25 13:55:59,245/INFO/Facts/__init__():32
    columns check passed

    2023-10-25 13:55:59,250/INFO/Prompts/__init__():67
    augmenting with word_swapping with args: {'type': 'wordswap', 'num_return_sequences': 4, 'label': 'word_swapping_wordswap'}
    ```
- By default, the code will run using T5-Small model. To change the model, use something like the following
  ```
  python src/main.py model=t5-large
  ```
  See `config/model/` for avialable models. Change configurations according to your computation environment.

