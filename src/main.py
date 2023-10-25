import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from modules.mylogger import init_logging
from modules.df_classes import Facts, Prompts, Generations
from modules.aggregator import Aggregator


@hydra.main(version_base='1.2', config_path='../config', config_name='main')
def main(cfg: DictConfig):

    # hydra related settings
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    save_dir = 'outputs/' + '/'.join(output_dir.split('/')[-2:])
    with open_dict(cfg):
        cfg['log_dir'] = save_dir
        cfg['filename'] = 'main.log'
    print(OmegaConf.to_yaml(cfg))

    # create logger
    logger, slack_handler = init_logging(
        __name__,
        log_dir=save_dir,
        filename='main.log',
        reset=True,
        run_name=cfg.model.name,
        slack=cfg.logger.slack
    )
    logger.info_slack(save_dir)

    # prepare dataset
    facts = Facts(cfg, slack_handler)

    # augment dataset
    prompts = Prompts(cfg, facts, slack_handler)
    prompts.df.to_json(f'{save_dir}/prompts.jsonl',
                       orient='records', lines=True)

    # generate
    generations = Generations(cfg, facts, prompts, slack_handler)

    if cfg.model.generation_split == None:

        # aggregate
        aggregator = Aggregator(cfg, facts, prompts, generations, slack_handler)
        aggregator.aggregate_multi_thread()

    # print output dir
    logger.info(f'output_dir: {output_dir}')


if __name__ == '__main__':
    main()
