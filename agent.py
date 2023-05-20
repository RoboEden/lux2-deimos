from functools import wraps
import sys
import os.path as osp

work_directory = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(work_directory, "implementations"))

import numpy as np
import torch
import tree
from lux.config import EnvConfig
from lux.kit import GameState, obs_to_game_state

from implementations.env.parsers.action_parser_full_act import ActionParser
from implementations.env.parsers.feature_parser import FeatureParser
from implementations.policy.impl.multi_task_softmax_policy_impl import (_gen_pi, _sample_til_valid)
from implementations.policy.net import Net
from implementations.env.player import Player
from implementations.impl_config import EnvParam, ModelParam
import traceback

import json


def print_exc(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"torch.__version__={torch.__version__}", file=sys.stderr)
            print(e, file=sys.stderr)
            traceback.print_exc()
            raise

    return wrapper


class Agent():

    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.feature_parser = FeatureParser()
        self.action_parser = ActionParser()

        with open(osp.join(work_directory, "conf.json")) as f:
            conf = json.load(f)
        model_related = conf['model_related']
        model_param = ModelParam()
        if 'model_param' in model_related:
            model_param = ModelParam(**model_related['model_param'])
        self.net = Net(model_param)
        self.net.load_state_dict(torch.load(
            osp.join(work_directory, "model.pt"),
            map_location=torch.device('cpu'),
        ))

        self.early_setup_player = Player(player, env_cfg)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if EnvParam.rule_based_early_step:
            return self.early_setup_player.early_setup(step, obs, remainingOverageTime)
        else:
            return self.act(step, obs, remainingOverageTime)

    # @print_exc
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        for units in game_state.units.values():
            for u in units.values():
                u.action_queue = u.action_queue.tolist()

        feature = self.feature_parser._get_feature(game_state, self.player)
        va = self.action_parser.get_valid_actions(game_state, int(self.player[-1]))

        np2torch = lambda x, dtype: torch.tensor(x)[None].type(dtype)
        logp, value, action, entropy, _ = self.net.forward(
            np2torch(feature.global_feature, torch.float32),
            np2torch(feature.map_feature, torch.float32),
            tree.map_structure(lambda x: np2torch(x, torch.int16), feature.action_feature),
            tree.map_structure(lambda x: np2torch(x, torch.bool), va),
        )
        action = tree.map_structure(lambda x: x.detach().numpy()[0], action)

        action = self.action_parser._parse(game_state, self.player, action)

        return action
