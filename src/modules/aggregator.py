from omegaconf import DictConfig
import pandas as pd
from modules.df_classes import Facts, Prompts, Generations
from pprint import pprint

from concurrent.futures import ThreadPoolExecutor
from modules.mylogger import init_logging
# from pandarallel import pandarallel
# pandarallel.initialize()


class Aggregator():
    def __init__(self,
        cfg: DictConfig,
        facts: Facts,
        prompts: Prompts,
        generations: Generations,
        sh
    ) -> None:
        self.logger, _ = init_logging('Aggregator', cfg.log_dir, cfg.filename, reset=False, slack=True, sh=sh)
        
        self.gold_answer = facts.df['os']
        self.prompts = prompts
        self.generations = generations
        self.save_dir = cfg.log_dir + '/aggregated_generations'
        self.slack=cfg.logger.slack

        max_num_prompts = self.prompts._check_equal_num_prompts()
        self.logger.info(f'max_num_prompts: {max_num_prompts}')
        aggregation_functions = {
            'sum': self._aggregate,
            'count': self._aggregate_count
        }

        if cfg.aggregation_method.name == 'random_sample':
            self.aggregation_to_do = [
                (
                    self._random_sample,
                    {
                        'seed': 0,
                        'num_prompts': 1,
                        'sh': sh,
                        'aggregation_function': aggregation_functions[cfg.aggregation_method.function],

                    }
                )
            ]
            if max_num_prompts > 1:
                if max_num_prompts > 2:
                    self.aggregation_to_do += [
                        (   self._random_sample,
                            {
                                'seed': i,
                                'num_prompts': j,
                                'sh': sh,
                                'aggregation_function': aggregation_functions[cfg.aggregation_method.function],

                            }
                        )\
                        for i in range(cfg.aggregation_method.iter_num)\
                        for j in range(2, max_num_prompts)
                    ]
        
                self.aggregation_to_do.append(
                    (
                        self._random_sample,
                        {
                            'seed': 0,
                            'num_prompts': max_num_prompts,
                            'sh': sh,
                            'aggregation_function': aggregation_functions[cfg.aggregation_method.function],
                        }
                    )
                )

                # self.aggregation_to_do = [(
                #     self._random_sample,
                #     {
                #         'seed': 0,
                #         'num_prompts': 15,
                #         'sh': sh
                #     }
                # )]
        # print(self.aggregation_to_do)



    def aggregate_multi_thread(self):
        with ThreadPoolExecutor(max_workers=20) as executor:
            for func, args in self.aggregation_to_do:
                self.logger.info(f'aggregating with args: {args}')
                executor.submit(func, args)
        # for func, args in self.aggregation_to_do:
        #     self.logger.info(f'aggregating with args: {args}')
        #     func(args)


    def _random_sample(self, args: dict):
        logger, _ = init_logging(
            f'T_seed{args["seed"]}_nprompt{args["num_prompts"]}',
            self.save_dir,
            'threads.log',
            reset=False,
            slack=self.slack,
            sh=args['sh'])
        sampled_prompts = self.prompts._sample_prompts_random(seed=args['seed'], num_prompts=args['num_prompts'])
        sampled_generations = self.generations.df.loc[self.generations.df['prompt_id'].isin(sampled_prompts.index)]
        aggregated_results = args['aggregation_function'](sampled_generations, logger=logger)
        # aggregated_results = self._aggregate(sampled_generations, logger=logger)
        # aggregated_results = self._analyze(aggregated_results)
        aggregated_results.to_json(f'{self.save_dir}/random_sample_{args["seed"]}_{args["num_prompts"]}.jsonl', orient='records', lines=True)
        # aggregated_results.to_json(f'outputs/2023-10-14/00-10-47/aggregated_generations_count/random_sample_{args["seed"]}_{args["num_prompts"]}.jsonl', orient='records', lines=True)
        # pprint(aggregated_results.iloc[2042]['candidates'])
        logger.info_slack('saved')
        return 0
    
    def _aggregate(self, generations: pd.DataFrame, logger):
        logger.info(f'aggregating {generations.shape[0]} generations')

        def sum_scores(group):
            sum_result = group.groupby('generation')[['score']].sum('score')
            sum_result = sum_result.sort_values('score', ascending=False).reset_index()
            sum_result['normalized_score'] = sum_result['score'] / sum_result['score'].iloc[0]
            top1 = sum_result.iloc[0].to_dict()
            return pd.Series([top1, sum_result.to_dict('records')], index=['top1', 'candidates'])

        aggregation_result = generations.groupby('fact_id').apply(sum_scores)
        logger.info(f'aggregation_result shape: {aggregation_result.shape}')
        return aggregation_result
    
    def _aggregate_count(self, generations: pd.DataFrame, logger):
        logger.info(f'aggregating {generations.shape[0]} generations')

        def count_generations(group):
            group = group.sort_values('score', ascending=False).reset_index()
            group['priority'] = group.index
            count_result = group.groupby('generation').agg({'score': 'count', 'priority': 'min'})
            count_result = count_result.sort_values('priority', ascending=True).reset_index()
            count_result['normalized_score'] = count_result['score']
            top1 = count_result.iloc[0].to_dict()
            return pd.Series([top1, count_result.to_dict('records')], index=['top1', 'candidates'])

        aggregation_result = generations.groupby('fact_id').apply(count_generations)
        logger.info(f'aggregation_result shape: {aggregation_result.shape}')
        return aggregation_result
    
    def _analyze(self, aggregation_result: pd.DataFrame):
        # check answers
        aggregation_result['gold'] = self.gold_answer
        aggregation_result['is_correct'] = aggregation_result.apply(lambda row: row['top1']['generation'] == row['gold'], axis=1)

        # get confidence
        def get_confidence(row):
            df_tmp = pd.DataFrame(row['candidates'])
            confidence = 1/df_tmp['normalized_score'].sum()
            return confidence
        aggregation_result['confidence'] = aggregation_result.apply(get_confidence, axis=1)
        aggregation_result['normalized_confidence'] = aggregation_result['confidence'] / aggregation_result['confidence'].max()

        return aggregation_result



