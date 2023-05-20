import numpy as np
import torch
import torch.nn as nn
from impl_config import ActDims, EnvParam, ModelParam, UnitActType, UnitActChannel
from torch.distributions import Beta, Categorical, Normal

BIG_NEG = -1e10


def sample_from_categorical(logits, va, action=None):
    n = logits.shape[0]
    if n > 0:
        logits = torch.where(va, logits, torch.tensor(BIG_NEG).type_as(logits))
        distribution = Categorical(logits=logits)
        action = distribution.sample() if action is None else action
        logp = distribution.log_prob(action)
        probs = distribution.probs
        zero = torch.tensor(0.).type_as(probs)
        entropy = -(torch.where(va, probs, zero) * distribution.logits).sum(-1)
    else:
        action = torch.zeros(torch.Size((0, )), device=logits.device, dtype=torch.long)
        logp = torch.ones(torch.Size((0, )), device=logits.device, dtype=torch.float32)
        entropy = torch.zeros(torch.Size((0, )), device=logits.device, dtype=torch.float32)
    return logp, action.long(), entropy


def sample_from_beta(param, action=None):
    alpha = nn.functional.softplus(param[..., 0]) + 0.001
    beta = nn.functional.softplus(param[..., 1]) + 0.001
    distribution = Beta(alpha, beta)
    action = distribution.sample() if action is None else action
    logp = distribution.log_prob(action)
    entropy = distribution.entropy()
    return logp, action, entropy


def sample_from_normal(param, action=None):
    mu = torch.sigmoid(param[..., 0])
    std = torch.sigmoid(param[..., 1]) * 2
    distribution = Normal(mu, std)
    action = distribution.sample() if action is None else action
    logp = distribution.log_prob(action)
    entropy = distribution.entropy()

    action = torch.clamp(action, 0., 1.)
    return logp, action, entropy


class ActorHead(nn.Module):
    amount_head_dim = {
        "beta": 2,
        "categorical": ActDims.amount,
        "normal": 2,
        "script": 1,
    }

    def __init__(self, model_param: ModelParam) -> None:
        super().__init__()
        self.model_param = model_param
        channel_sz = model_param.all_channel
        self.amount_distribution = self.model_param.amount_distribution
        self.amount_head_dim = ActorHead.amount_head_dim[self.amount_distribution]
        self.spawn_distribution = self.model_param.spawn_distribution
        self.spawn_amount_dim = ActorHead.amount_head_dim[self.spawn_distribution]

        # Output Head
        if not EnvParam.rule_based_early_step:
            self.bid = nn.Sequential(
                nn.Conv2d(channel_sz, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(EnvParam.map_size * EnvParam.map_size, ActDims.bid, bias=True),
            )
            self.spawn = nn.ModuleDict({
                "logits": nn.Conv2d(channel_sz, 1, kernel_size=1, bias=True),
                "water": nn.Conv2d(channel_sz, self.spawn_amount_dim, kernel_size=1, bias=True),
                "metal": nn.Conv2d(channel_sz, self.spawn_amount_dim, kernel_size=1, bias=True),
            })

        self.factory_act = nn.Linear(channel_sz, ActDims.factory_act, bias=True)

        self.unit_act_type = nn.Linear(channel_sz, ActDims.robot_act, bias=True)

        self.unit_act_move = nn.ModuleDict({
            "direction": nn.Linear(channel_sz, ActDims.direction, bias=True),
            "repeat": nn.Linear(channel_sz, ActDims.repeat, bias=True),
        })

        self.unit_act_transfer = nn.ModuleDict({
            "direction":
            nn.Linear(channel_sz, ActDims.direction, bias=True),
            "resource":
            nn.Sequential(
                nn.Linear(channel_sz, (ActDims.direction * ActDims.resource), bias=True),
                nn.Unflatten(1, (ActDims.direction, ActDims.resource)),
            ),
            "amount":
            nn.Sequential(
                nn.Linear(channel_sz, (ActDims.direction * ActDims.resource * self.amount_head_dim), bias=True),
                nn.Unflatten(1, (ActDims.direction, ActDims.resource, self.amount_head_dim)),
            ),
            "repeat":
            nn.Linear(channel_sz, ActDims.repeat, bias=True),
        })

        self.unit_act_pickup = nn.ModuleDict({
            "resource":
            nn.Linear(channel_sz, ActDims.resource, bias=True),
            "amount":
            nn.Sequential(
                nn.Linear(channel_sz, (ActDims.resource * self.amount_head_dim), bias=True),
                nn.Unflatten(1, (ActDims.resource, self.amount_head_dim)),
            ),
            "repeat":
            nn.Linear(channel_sz, ActDims.repeat, bias=True),
        })

        self.unit_act_dig = nn.ModuleDict({"repeat": nn.Linear(channel_sz, ActDims.repeat, bias=True)})

        self.unit_act_self_destruct = nn.ModuleDict({"repeat": nn.Linear(channel_sz, ActDims.repeat, bias=True)})

        self.unit_act_recharge = nn.ModuleDict({
            "amount": nn.Linear(channel_sz, ActDims.amount, bias=True),
            "repeat": nn.Linear(channel_sz, ActDims.repeat, bias=True),
        })

    def bid_actor(self, x, va, action=None):
        logp, action, entropy = sample_from_categorical(self.bid(x), va, action)
        return logp, action, entropy

    def spawn_actor(self, x, va, action=None):
        B = x.shape[0]
        mask = va.flatten(1).any(1)  # bool[B]

        logp = torch.zeros(B, device=x.device)
        entropy = torch.zeros(B, device=x.device)
        output_action = {
            "location": torch.zeros(B, device=x.device, dtype=torch.long),
            "water": torch.zeros(B, device=x.device),
            "metal": torch.zeros(B, device=x.device),
        }

        if mask.any():
            location_logp, location, location_entropy = sample_from_categorical(
                self.spawn.logits(x).flatten(1)[mask],
                va.flatten(1)[mask],
                action and action["location"][mask],
            )

            if self.spawn_distribution == "categorical":
                sample_amount_fn = lambda logits, action: sample_from_categorical(
                    logits, torch.tensor(True, device=x.device), action)
            elif self.spawn_distribution == "beta":
                sample_amount_fn = sample_from_beta
            elif self.spawn_distribution == "normal":
                sample_amount_fn = sample_from_normal
            elif self.spawn_distribution == "script":
                sample_amount_fn = lambda logits, action: (
                    torch.zeros((), device=logits.device),
                    torch.zeros((), device=logits.device),
                    torch.zeros((), device=logits.device),
                )
            else:
                raise NotImplementedError

            water_logp, water, water_entropy = sample_amount_fn(
                self.spawn.water(x).flatten(-2)[mask, ..., location],
                action and action["water"][mask],
            )

            metal_logp, metal, metal_entropy = sample_amount_fn(
                self.spawn.metal(x).flatten(-2)[mask, ..., location],
                action and action["metal"][mask],
            )

            logp[mask] += water_logp + metal_logp + location_logp
            entropy[mask] += water_entropy + metal_entropy + location_entropy

            output_action["location"][mask] = location
            output_action["water"][mask] = water.float()
            output_action["metal"][mask] = metal.float()

        return logp, output_action, entropy

    def factory_actor(self, x, va, action=None):
        logits = self.factory_act(x)
        logp, output_action, entropy = sample_from_categorical(logits, va, action)
        return logp, output_action, entropy

    def unit_actor(self, x, va, action=None):
        act_type_logp, act_type, act_type_entropy = sample_from_categorical(
            self.unit_act_type(x),
            va['act_type'],
            action[:, UnitActChannel.TYPE] if action is not None else None,
        )
        logp = act_type_logp
        entropy = act_type_entropy
        output_action = torch.zeros((x.shape[0], len(UnitActChannel)), device=x.device)

        actors = [
            self.move_actor,
            self.transfer_actor,
            self.pickup_actor,
            self.dig_actor,
            self.self_destruct_actor,
            self.recharge_actor,
            self.do_nothing_actor,
        ]
        for type, actor in zip(UnitActType, actors):
            mask = (act_type == type)
            move_logp, move_action, move_entropy = actor(
                x[mask],
                va[type.name.lower()][mask],
                action[mask] if action is not None else None,
            )
            logp[mask] += move_logp
            entropy[mask] += move_entropy
            output_action[mask] = move_action

        return logp, output_action, entropy

    def move_actor(self, x, va, action=None):
        n_units = x.shape[0]
        unit_idx = torch.arange(n_units, device=x.device)

        logp = torch.zeros(n_units, device=x.device)
        entropy = torch.zeros(n_units, device=x.device)
        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        # logits
        params = {name: layer(x) for name, layer in self.unit_act_move.items()}

        # action type
        output_action[:, UnitActChannel.TYPE] = UnitActType.MOVE
        output_action[:, UnitActChannel.N] = 1

        # direction
        direction_va = va.flatten(2).any(dim=-1)
        direction_logp, direction, direction_entropy = sample_from_categorical(
            params['direction'],
            direction_va,
            action[:, UnitActChannel.DIRECTION] if action is not None else None,
        )
        logp += direction_logp
        entropy += direction_entropy
        output_action[:, UnitActChannel.DIRECTION] = direction

        # repeat
        repeat_va = va[unit_idx, direction]
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            params['repeat'],
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )
        logp += repeat_logp
        entropy += repeat_entropy
        output_action[:, UnitActChannel.REPEAT] = repeat

        return logp, output_action, entropy

    def transfer_actor(self, x, va, action=None):
        n_units = x.shape[0]
        unit_idx = torch.arange(n_units, device=x.device)

        logp = torch.zeros(n_units, device=x.device)
        entropy = torch.zeros(n_units, device=x.device)
        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        # logits
        params = {name: layer(x) for name, layer in self.unit_act_transfer.items()}

        # action type
        output_action[:, UnitActChannel.TYPE] = UnitActType.TRANSFER
        output_action[:, UnitActChannel.N] = 1

        # direction
        direction_va = va.flatten(2).any(-1)
        direction_logp, direction, direction_entropy = sample_from_categorical(
            params['direction'],
            direction_va,
            action[:, UnitActChannel.DIRECTION] if action is not None else None,
        )
        logp += direction_logp
        entropy += direction_entropy
        output_action[:, UnitActChannel.DIRECTION] = direction

        # resource
        resource_va = va[unit_idx, direction].flatten(2).any(-1)
        resource_logp, resource, resource_entropy = sample_from_categorical(
            params['resource'][unit_idx, direction],
            resource_va,
            action[:, UnitActChannel.RESOURCE] if action is not None else None,
        )
        logp += resource_logp
        entropy += resource_entropy
        output_action[:, UnitActChannel.RESOURCE] = resource

        # amount
        params_amount = params['amount'][unit_idx, direction, resource]
        if self.model_param.amount_distribution == 'categorical':
            if action is not None:
                amount = action[:, UnitActChannel.AMOUNT]
                amount = (amount * self.amount_head_dim - 1).round()
            else:
                amount = None
            amount_logp, amount, amount_entropy = sample_from_categorical(
                params_amount,
                torch.tensor(True, device=x.device),
                amount,
            )
            amount = (amount + 1) / self.amount_head_dim
        elif self.model_param.amount_distribution == 'beta':
            amount_logp, amount, amount_entropy = sample_from_beta(
                params_amount,
                action[:, UnitActChannel.AMOUNT] if action is not None else None,
            )
        else:
            raise ValueError('Unknown amount distribution')
        logp += amount_logp
        entropy += amount_entropy
        output_action[:, UnitActChannel.AMOUNT] = amount

        # repeat
        repeat_va = va[unit_idx, direction, resource]
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            params['repeat'],
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )
        logp += repeat_logp
        entropy += repeat_entropy
        output_action[:, UnitActChannel.REPEAT] = repeat

        return logp, output_action, entropy

    def pickup_actor(self, x, va, action=None):
        n_units = x.shape[0]
        unit_idx = torch.arange(n_units, device=x.device)

        logp = torch.zeros(n_units, device=x.device)
        entropy = torch.zeros(n_units, device=x.device)
        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        # logits
        params = {name: layer(x) for name, layer in self.unit_act_pickup.items()}

        # action type
        output_action[:, UnitActChannel.TYPE] = UnitActType.PICKUP
        output_action[:, UnitActChannel.N] = 1

        # resource
        resource_va = va.flatten(2).any(-1)
        resource_logp, resource, resource_entropy = sample_from_categorical(
            params['resource'],
            resource_va,
            action[:, UnitActChannel.RESOURCE] if action is not None else None,
        )
        logp += resource_logp
        entropy += resource_entropy
        output_action[:, UnitActChannel.RESOURCE] = resource

        # amount
        params_amount = params['amount'][unit_idx, resource]
        if self.model_param.amount_distribution == 'categorical':
            if action is not None:
                amount = action[:, UnitActChannel.AMOUNT]
                amount = (amount * self.amount_head_dim - 1).round()
            else:
                amount = None
            amount_logp, amount, amount_entropy = sample_from_categorical(
                params_amount,
                torch.tensor(True, device=x.device),
                amount,
            )
            amount = (amount + 1) / self.amount_head_dim
        elif self.model_param.amount_distribution == 'beta':
            amount_logp, amount, amount_entropy = sample_from_beta(
                params_amount,
                action[:, UnitActChannel.AMOUNT] if action is not None else None,
            )
        else:
            raise ValueError('Unknown amount distribution')
        logp += amount_logp
        entropy += amount_entropy
        output_action[:, UnitActChannel.AMOUNT] = amount

        # repeat
        repeat_va = va[unit_idx, resource]
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            params['repeat'],
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )
        logp += repeat_logp
        entropy += repeat_entropy
        output_action[:, UnitActChannel.REPEAT] = repeat

        return logp, output_action, entropy

    def dig_actor(self, x, va, action=None):
        n_units = x.shape[0]
        logp = torch.zeros(n_units, device=x.device)
        entropy = torch.zeros(n_units, device=x.device)
        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        # logits
        params = {name: layer(x) for name, layer in self.unit_act_dig.items()}

        # action type
        output_action[:, UnitActChannel.TYPE] = UnitActType.DIG
        output_action[:, UnitActChannel.N] = 1

        # repeat
        repeat_va = va
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            params['repeat'],
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )
        logp += repeat_logp
        entropy += repeat_entropy
        output_action[:, UnitActChannel.REPEAT] = repeat

        return logp, output_action, entropy

    def self_destruct_actor(self, x, va, action=None):
        n_units = x.shape[0]
        logp = torch.zeros(n_units, device=x.device)
        entropy = torch.zeros(n_units, device=x.device)
        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        # logits
        params = {name: layer(x) for name, layer in self.unit_act_self_destruct.items()}

        # action type
        output_action[:, UnitActChannel.TYPE] = UnitActType.SELF_DESTRUCT
        output_action[:, UnitActChannel.N] = 1

        # repeat
        repeat_va = va
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            params['repeat'],
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )
        logp += repeat_logp
        entropy += repeat_entropy
        output_action[:, UnitActChannel.REPEAT] = repeat

        return logp, output_action, entropy

    def recharge_actor(self, x, va, action=None):
        n_units = x.shape[0]
        logp = torch.zeros(n_units, device=x.device)
        entropy = torch.zeros(n_units, device=x.device)
        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        # logits
        params = {name: layer(x) for name, layer in self.unit_act_recharge.items()}

        # action type
        output_action[:, UnitActChannel.TYPE] = UnitActType.RECHARGE
        output_action[:, UnitActChannel.N] = 1

        # repeat
        repeat_va = va
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            params['repeat'],
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )
        logp += repeat_logp
        entropy += repeat_entropy
        output_action[:, UnitActChannel.REPEAT] = repeat

        return logp, output_action, entropy

    def do_nothing_actor(self, x, va, action=None):
        n_units = x.shape[0]
        logp = torch.zeros(n_units, device=x.device)
        entropy = torch.zeros(n_units, device=x.device)
        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        # action type
        output_action[:, UnitActChannel.TYPE] = UnitActType.DO_NOTHING

        return logp, output_action, entropy

    # @torchsnooper.snoop()
    def forward(self, x, va, action=None):
        B, _, H, W = x.shape

        logp = torch.zeros(B, device=x.device)
        entropy = torch.zeros(B, device=x.device)
        output_action = {}

        def _gather_from_map(x, pos):
            return x[pos[0], ..., pos[1], pos[2]]

        def _put_into_map(emb, pos):
            shape = (B, ) + emb.shape[1:] + (H, W)
            map = torch.zeros(shape, dtype=emb.dtype, device=emb.device)
            map[pos[0], ..., pos[1], pos[2]] = emb
            return map

        # factory actor
        factory_pos = torch.where(va['factory_act'].any(1))
        factory_emb = _gather_from_map(x, factory_pos)
        factory_va = _gather_from_map(va['factory_act'], factory_pos)
        factory_action = action and _gather_from_map(action['factory_act'], factory_pos)

        factory_logp, factory_action, factory_entropy = self.factory_actor(
            factory_emb,
            factory_va,
            factory_action,
        )
        cnt = torch.zeros(B, device=x.device, dtype=torch.int32)
        cnt.scatter_add_(0, factory_pos[0], torch.ones_like(factory_pos[0], dtype=cnt.dtype))
        logp.scatter_add_(0, factory_pos[0], factory_logp / cnt[factory_pos[0]])
        entropy.scatter_add_(0, factory_pos[0], factory_entropy / cnt[factory_pos[0]])
        output_action['factory_act'] = _put_into_map(factory_action, factory_pos)

        # unit actor
        unit_act_type_va = torch.stack(
            [
                va['move'].flatten(1, 2).any(1),
                va['transfer'].flatten(1, 3).any(1),
                va['pickup'].flatten(1, 2).any(1),
                va['dig'].any(1),
                va['self_destruct'].any(1),
                va['recharge'].any(1),
                va['do_nothing'],
            ],
            axis=1,
        )
        unit_pos = torch.where(unit_act_type_va.any(1))
        unit_emb = _gather_from_map(x, unit_pos)
        unit_va = {
            'act_type': _gather_from_map(unit_act_type_va, unit_pos),
            'move': _gather_from_map(va['move'], unit_pos),
            'transfer': _gather_from_map(va['transfer'], unit_pos),
            'pickup': _gather_from_map(va['pickup'], unit_pos),
            'dig': _gather_from_map(va['dig'], unit_pos),
            'self_destruct': _gather_from_map(va['self_destruct'], unit_pos),
            'recharge': _gather_from_map(va['recharge'], unit_pos),
            'do_nothing': _gather_from_map(va['do_nothing'], unit_pos),
        }
        unit_action = action and _gather_from_map(action['unit_act'], unit_pos)
        unit_logp, unit_action, unit_entropy = self.unit_actor(
            unit_emb,
            unit_va,
            unit_action,
        )
        cnt = torch.zeros(B, device=x.device, dtype=torch.int32)
        cnt.scatter_add_(0, unit_pos[0], torch.ones_like(unit_pos[0], dtype=cnt.dtype))
        logp.scatter_add_(0, unit_pos[0], unit_logp / cnt[unit_pos[0]])
        entropy.scatter_add_(0, unit_pos[0], unit_entropy / cnt[unit_pos[0]])
        output_action['unit_act'] = _put_into_map(unit_action, unit_pos)

        # early step
        if not EnvParam.rule_based_early_step:
            bid_action = action and action['bid']
            bid_logp, bid_action, bid_entropy = self.bid_actor(x, va['bid'], bid_action)
            logp += bid_logp
            entropy += bid_entropy
            output_action['bid'] = bid_action

            spawn_action = action and action['factory_spawn']
            spawn_logp, spawn_action, spawn_entropy = self.spawn_actor(x, va['factory_spawn'], spawn_action)
            logp += spawn_logp
            entropy += spawn_entropy
            output_action['factory_spawn'] = spawn_action

        return logp, output_action, entropy