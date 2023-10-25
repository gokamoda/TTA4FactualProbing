import os
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader


from modules.mylogger import init_logging
from modules.augmenter import augment_head, PromptDataset, generate
from modules.models import get_model



class Facts():
    def __init__(self, cfg: DictConfig, slack_handler) -> None:
        self.logger, _ = init_logging(
            'Facts',
            cfg.log_dir,
            cfg.filename,
            reset=False,
            slack=cfg.logger.slack,
            sh=slack_handler
        )

        self.df = pd.read_json(cfg.dataset.path, lines=True)
        # self.df = self.df.head(10)
        self.logger.info(f'Facts: {self.df.shape[0]}')

        # check columns
        columns = self.df.columns
        assert 'os' in columns, 'os column not found'
        assert 'prompt' in columns, 'prompt column not found'
        self.logger.info('columns check passed')

    def __str__(self) -> str:
        return self.df.shape
    
    def get_answer_dict(self):
        return self.df['os'].to_dict()
    
class Prompts():
    def __init__(self, cfg: DictConfig, facts: Facts, slack_handler) -> None:
        self.logger, _ = init_logging(
            'Prompts',
            cfg.log_dir,
            cfg.filename,
            reset=False,
            slack=cfg.logger.slack,
            sh=slack_handler
        )

        self.cache_dir = cfg.dataset.name
        
        # create dir if does not exist
        os.makedirs(f'cache/augmented_prompts/{self.cache_dir}', exist_ok=True)

        # get original prompts
        all_prompts = facts.df[['prompt']].copy()
        all_prompts.reset_index(inplace=True)
        all_prompts = all_prompts.rename(columns={'index': 'fact_id'})
        all_prompts['score'] = 1
        all_prompts['label'] = 'original'


        # get augmented prompts
        for method in cfg.augment_methods.keys():
            for args in cfg.augment_methods[method]:
                self.logger.info(f'augmenting with {method} with args: {args}')
                cache_path = f'cache/augmented_prompts/{self.cache_dir}/{args["label"]}.jsonl'
                print(cache_path)
                if os.path.exists(cache_path):
                    augmented_prompt_df = pd.read_json(cache_path, orient='records', lines=True)
                else:
                    augmented_prompt_df = augment_head(original_df=facts.df, method_name=method, args=args)
                    if augmented_prompt_df is not None:
                        augmented_prompt_df.to_json(cache_path, orient='records', lines=True)
                    print(augmented_prompt_df.head(5))
                all_prompts = pd.concat([all_prompts, augmented_prompt_df], ignore_index=True)
        all_prompts.reset_index(inplace=True)
        all_prompts = all_prompts.sort_values(by=['fact_id', 'index']).drop(columns=['index']).reset_index(drop=True)

        self.df = all_prompts
        self.logger.info(f'Done Augmentation. n_prompts: {self.df.shape[0]}')
    
    def _check_equal_num_prompts(self):
        prompts_count = self.df.groupby('fact_id').count()['prompt']
        assert prompts_count.var() == 0, 'number of prompts per fact_id is not consistent'
        return prompts_count.max()
    
    def _sample_prompts_random(self, seed, num_prompts):
        original_prompts = self.df[self.df['label'] == 'original']
        augmented_prompts = self.df[self.df['label'] != 'original']
        augmented_prompts = augmented_prompts.groupby('fact_id').sample(n=num_prompts-1, random_state=seed)
        sampled_prompts = pd.concat([original_prompts, augmented_prompts], ignore_index=False).sort_index()
        return sampled_prompts

class Generations():

    def __init__(
        self,
        cfg: DictConfig,
        facts: Facts,
        prompts: Prompts,
        slack_handler
    ):
        self.logger, _ = init_logging(
            'Generations',
            cfg.log_dir,
            cfg.filename,
            reset=False,
            slack=cfg.logger.slack,
            sh=slack_handler
        )

        if cfg.model.generation_split != None:
            self.logger.info("splitting generations")
            prompts.df = prompts.df.iloc[cfg.model.generation_split[0]:cfg.model.generation_split[1]]

        if cfg.model.generation_combine != None:
            self.df = self._combine_generations(cfg)
            self.df = self._add_fact_id_column(prompts.df, self.df)

            self.logger.info(f'Combined generations from {cfg.model.generation_combine}')
            self.logger.info(f'n_rows: {self.df.shape[0]}')
        else:
            generations = self._generate_candidates(cfg, facts, prompts)
            generations = self._add_fact_id_column(prompts.df, generations)
            generations.to_json(f'{cfg.log_dir}/generations.jsonl', orient='records', lines=True)
            self.df = generations

        self._check_equal_num_generations()



    def _check_max_new_tokens(self, cfg: DictConfig, facts_df: pd.DataFrame, tokenizer):
        df = facts_df.loc[:, ["os"]]
        df["len"] = df.apply(lambda row: len(tokenizer(row["os"], return_tensors="pt").input_ids[0]), axis=1)
        max_tokens = df["len"].max()
        assert max_tokens <= cfg.generation_args.max_new_tokens, f'max_tokens: {max_tokens}'

    def _combine_generations(self, cfg: DictConfig):
        generations = []
        for path in cfg.model.generation_combine:
            generations.append(pd.read_json(path+'/generations.jsonl', lines=True))
        df = pd.concat(generations, ignore_index=True)
        df = df.sort_values(by=['prompt_id']).reset_index(drop=True)
        return df

    def _generate_candidates(self, cfg: DictConfig, facts: Facts, prompts: Prompts):
        model, tokenizer = get_model(cfg.model)
        # self.logger.info(model.hf_device_map)    
        
        self._check_max_new_tokens(cfg, facts.df, tokenizer)

        # generate
        results = generate(
            texts=prompts.df.prompt.to_list(),
            model=model,
            tokenizer=tokenizer,
            generation_args=cfg.generation_args,
            batch_size=cfg.model.batch_size
        )
        prompt_ids = [x for x in prompts.df.index.to_list() for _ in range(cfg.generation_args.num_return_sequences)]
        df = pd.DataFrame({
            "prompt_id": prompt_ids,
            "generation": results["texts"],
            "score": results["scores"],
        })
        return df

    def _add_fact_id_column(self, prompt_df: pd.DataFrame, generation_df: pd.DataFrame):
        if 'fact_id' in generation_df.columns:
            return generation_df
        fact_id_dict = prompt_df[['fact_id']].to_dict()
        generation_df['fact_id'] = generation_df['prompt_id'].map(fact_id_dict['fact_id'])
        return generation_df
    
    def _check_equal_num_generations(self):
        gb = self.df.groupby('fact_id').count()
        generations_count = gb['generation']
        if generations_count.var() != 0:
            print("number of generations per fact_id is not consistent")
            print(gb.loc[gb['generation'] != generations_count.max()])
            print(generations_count.max())
        assert generations_count.var() == 0, 'number of generations per fact_id is not consistent'
        return generations_count.max()