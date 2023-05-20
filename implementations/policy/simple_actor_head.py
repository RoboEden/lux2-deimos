from collections import OrderedDict
from .actor_head import (
    ActorHead,
    sample_from_categorical,
    sample_from_beta,
)
from impl_config import ModelParam, EnvParam, ActDims, UnitActChannel, UnitActType
from torch import nn
import torch


class SimpleActorHead(ActorHead):

    def __init__(self, model_param: ModelParam) -> None:
        nn.Module.__init__(self)
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

        self.unit_logit_dims = OrderedDict(
            type=ActDims.robot_act,
            direction=ActDims.direction,
            resource=ActDims.resource,
            amount=self.amount_head_dim,
            repeat=ActDims.repeat,
        )
        self.logit_ranges = OrderedDict()
        start = 0
        for k, d in self.unit_logit_dims.items():
            self.logit_ranges[k] = (start, start + d)
            start += d

        self.concated_logits = nn.Linear(channel_sz, sum(self.unit_logit_dims.values()), bias=True)

    def logits_dict(self, x):
        logits = self.concated_logits(x)
        return {k: logits[:, s:e] for k, (s, e) in self.logit_ranges.items()}

    def unit_actor(self, x, va, action=None):
        logp = 0
        entropy = 0
        output_action = torch.zeros((x.shape[0], len(UnitActChannel)), device=x.device)

        logits = self.logits_dict(x)

        # type
        act_type_logp, act_type, act_type_entropy = sample_from_categorical(
            logits["type"],
            va['act_type'],
            action[:, UnitActChannel.TYPE] if action is not None else None,
        )
        logp += act_type_logp
        entropy += act_type_entropy
        output_action[:, UnitActChannel.TYPE] = act_type

        # direction
        with torch.no_grad():  # prepare direction_va
            direction_va = torch.zeros_like(logits["direction"], dtype=torch.bool)
            move_mask = (act_type == UnitActType.MOVE)
            transfer_mask = (act_type == UnitActType.TRANSFER)
            direction_va[move_mask] = va['move'][move_mask].any(-1)
            direction_va[transfer_mask] = va['transfer'][transfer_mask].flatten(-2).any(-1)
            direction_va[torch.logical_not(move_mask | transfer_mask), 0] = True
        direction_logp, direction, direction_entropy = sample_from_categorical(
            logits["direction"],
            direction_va,
            action[:, UnitActChannel.DIRECTION] if action is not None else None,
        )
        logp += direction_logp
        entropy += direction_entropy
        output_action[:, UnitActChannel.DIRECTION] = direction

        # resource
        with torch.no_grad():
            resource_va = torch.zeros_like(logits["resource"], dtype=torch.bool)
            pickup_mask = (act_type == UnitActType.PICKUP)
            resource_va[transfer_mask] = va['transfer'][transfer_mask, direction[transfer_mask]].any(-1)
            resource_va[pickup_mask] = va['pickup'][pickup_mask].any(-1)
            pickup_or_transfer_mask = pickup_mask | transfer_mask
            resource_va[torch.logical_not(pickup_mask | transfer_mask), 0] = True
        resource_logp, resource, resource_entropy = sample_from_categorical(
            logits["resource"],
            resource_va,
            action[:, UnitActChannel.RESOURCE] if action is not None else None,
        )
        logp += resource_logp
        entropy += resource_entropy
        output_action[:, UnitActChannel.RESOURCE] = resource

        # amount
        params_amount = logits["amount"][pickup_or_transfer_mask]
        if self.model_param.amount_distribution == 'categorical':
            if action is not None:
                amount = action[pickup_or_transfer_mask, UnitActChannel.AMOUNT]
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
                action[pickup_or_transfer_mask, UnitActChannel.AMOUNT] if action is not None else None,
            )
        else:
            raise ValueError('Unknown amount distribution')
        logp[pickup_or_transfer_mask] += amount_logp
        entropy[pickup_or_transfer_mask] += amount_entropy
        output_action[pickup_or_transfer_mask, UnitActChannel.AMOUNT] = amount

        # repeat
        repeat_va = torch.zeros_like(logits["repeat"], dtype=torch.bool)
        repeat_va[(act_type == UnitActType.DIG) | move_mask] = True
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            logits["repeat"],
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )
        logp += repeat_logp
        entropy += repeat_entropy
        output_action[:, UnitActChannel.REPEAT] = repeat

        # n
        output_action[:, UnitActChannel.N] = 1

        return logp, output_action, entropy
