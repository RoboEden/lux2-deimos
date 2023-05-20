from main import agent_fn
from luxai_s2 import LuxAI_S2
from argparse import Namespace
import dataclasses
import json
import numpy as np
import copy


def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj


if __name__ == '__main__':
    env = LuxAI_S2()
    obs = env.reset()
    env_cfg = dataclasses.asdict(env.env_cfg)

    state_obs = env.state.get_compressed_obs()
    obs = to_json(state_obs)
    while True:
        actions = {}
        for player in ['player_0', 'player_1']:
            observation = Namespace(
                step=env.env_steps,
                obs=json.dumps(obs),
                remainingOverageTime=60,
                player=player,
                info=None,
            )
            actions[player] = agent_fn(observation, dict(env_cfg=copy.deepcopy(env_cfg)))
        new_state_obs, rewards, dones, infos = env.step(actions)
        if dones['player_0']:
            break

        obs = to_json(env.state.get_change_obs(state_obs))
        state_obs = new_state_obs["player_0"]
