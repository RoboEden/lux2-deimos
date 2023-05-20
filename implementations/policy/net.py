import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from impl_config import ModelParam, ActDims
from .simple_actor_head import SimpleActorHead
from .actor_head import ActorHead
# import torchsnooper


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=True), nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=True), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=5, padding=2):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            nn.LeakyReLU(),
        )
        self.selayer = SELayer(out_channel)

    def forward(self, x):
        out = self.left(x)
        out = self.selayer(out)
        out = out + x
        out = F.leaky_relu(out)
        return out


class ActionPreprocess(nn.Module):

    def __init__(self, emb_dim, action_queue_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.action_queue_size = action_queue_size

        self.type_emb = nn.Embedding(ActDims.robot_act, emb_dim)
        self.direction_emb = nn.Embedding(ActDims.direction, emb_dim)
        self.resource_emb = nn.Embedding(ActDims.resource, emb_dim)
        self.repeat_emb = nn.Embedding(ActDims.repeat, emb_dim)

        self.conv = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(action_queue_size, 1, 1), bias=True),
            nn.Conv3d(1, 1, kernel_size=(1, 1, 6), bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, action_feature):
        b, x, y = torch.where(action_feature['unit_indicator'])
        type_emb = self.type_emb(action_feature['type'][b, x, y].long())
        direction_emb = self.direction_emb(action_feature['direction'][b, x, y].long())
        resource_emb = self.resource_emb(action_feature['resource'][b, x, y].long())
        repeat_emb = self.repeat_emb((action_feature['repeat'][b, x, y] != 0).long())
        amount_emb = action_feature['amount'][b, x, y]
        n_emb = action_feature['n'][b, x, y]

        U, Q, E = type_emb.shape

        # sum in queue dimension
        emb = torch.stack(
            [
                type_emb,
                direction_emb,
                resource_emb,
                amount_emb[..., None].expand(-1, -1, E),
                repeat_emb,
                n_emb[..., None].expand(-1, -1, E),
            ],
            dim=-1,
        )
        emb = self.conv(emb[:, None]).reshape(U, E)

        B, H, W = action_feature['unit_indicator'].shape
        map_emb = torch.zeros((B, E, H, W), dtype=emb.dtype, device=emb.device)
        map_emb[b, :, x, y] = emb

        return map_emb


class Net(nn.Module):

    def __init__(self, model_param: ModelParam, enable_amp=False):
        super(Net, self).__init__()
        self.model_param = model_param
        self.enable_amp = enable_amp

        all_channel = model_param.all_channel
        self.action_preprocess = ActionPreprocess(model_param.action_emb_dim, model_param.action_queue_size)

        self.global_fc = nn.Sequential(
            nn.Linear(model_param.global_feature_dims, model_param.global_emb_dim),
            nn.LeakyReLU(),
        )

        input_channel = model_param.global_emb_dim + model_param.map_channel + model_param.action_emb_dim
        self.map_conv = nn.Sequential(
            nn.Conv2d(input_channel, all_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(all_channel, all_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(all_channel, all_channel) for _ in range(model_param.n_res_blocks)])

        self.spectral_norm = nn.utils.spectral_norm(
            nn.Conv2d(all_channel, all_channel, kernel_size=1, stride=1, padding=0, bias=True))

        if self.model_param.actor_head == 'full':
            self.actor = ActorHead(self.model_param)
        elif self.model_param.actor_head == 'simple':
            self.actor = SimpleActorHead(self.model_param)
        else:
            raise NotImplementedError

        self.critic_head = nn.Sequential(
            nn.Conv2d(all_channel, all_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(all_channel, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if name[-len('weight'):] == 'weight':
                nn.init.orthogonal_(param, gain=1 / np.power(np.prod(param.shape), 1 / 5))
            elif name[-len('bias'):] == 'bias':
                nn.init.zeros_(param)

    def _init_critic_net(self):
        for name, param in self.critic_head.named_parameters():
            if name[-len('weight'):] == 'weight':
                nn.init.orthogonal_(param, gain=1 / np.power(np.prod(param.shape), 1 / 5))
            elif name[-len('bias'):] == 'bias':
                nn.init.zeros_(param)

    # @torchsnooper.snoop()
    def forward(self, global_feature, map_feature, action_feature, va, action=None, teacher_action=None):
        B, _, H, W = map_feature.shape

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            action_emb = self.action_preprocess(action_feature)

            global_emb = self.global_fc(global_feature)
            global_emb = global_emb[..., None, None].expand(-1, -1, H, W)

            global_emb = torch.cat([global_emb, map_feature, action_emb], dim=1)

            x = self.map_conv(global_emb)

            for block in self.res_blocks:
                x = block(x)

            x = self.spectral_norm(x)
        x = x.float()

        logp, action, entropy = self.actor(x, va, action)
        if teacher_action is not None:
            teacher_action_logp, action, entropy = self.actor(x, va, teacher_action)
        else:
            teacher_action_logp = None
        critic_value = self.critic_head(x)
        critic_value = torch.flatten(critic_value, start_dim=-3).mean(-1)

        return logp, critic_value, action, entropy, teacher_action_logp
