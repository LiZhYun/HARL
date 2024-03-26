"""Train an algorithm."""
import argparse
import json
import wandb
import socket
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "igcsac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument('--use_wandb', action='store_false', default=True,
                        help="Whether to use weights&biases, if not, use tensorboardX instead")
    
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["use_wandb"]:
        # init wandb
        wandb_config = {**algo_args["train"], **algo_args["model"], **algo_args["algo"], **env_args}
        project = "mujoco" if args["env"]=="mamujoco" else "StarCraft2v2"
        if project == "StarCraft2v2":
            groups = ['10gen_zerg', '10gen_protoss', '10gen_terran']
            if 'zerg' in env_args["map_name"]:
                group = groups[0]
            elif 'protoss' in env_args["map_name"]:
                group = groups[1]
            else:
                group = groups[2]
            wandb_config['units'] = "5v5"
        else:
            group = env_args["scenario"]

        
        wandb_config['algorithm_name'] = args["algo"]
        run = wandb.init(config=wandb_config,
                         project=project,
                         entity="zhiyuanli",
                         notes=socket.gethostname(),
                         name=str(args["algo"] +
                         "_seed" + str(algo_args["seed"]["seed"])),
                         group=group,
                        #  dir=str(run_dir),
                         job_type="training",
                         reinit=True,
                         tags=["iclr24"],)

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
