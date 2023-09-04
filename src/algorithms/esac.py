import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
import algorithms.equivariant_module as equim
from algorithms.sac import SAC

from escnn import gspaces
from e2cnn import nn as e2nn
from escnn import nn as esnn


class ESAC(SAC):
    # equivariant SAC
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        # use group equivariant CNN as shared encoder
        self.action_shape = action_shape
        self.equi_group = args.equi_group
        self.inv_group = args.inv_group
        if self.equi_group[0] == 'C':
            equi_gspace = gspaces.rot2dOnR2(N=int(self.equi_group[1]))
        elif self.equi_group[0] == 'D':
            equi_gspace = gspaces.flipRot2dOnR2(N=int(self.equi_group[1]))
        elif self.equi_group == 'None':
            equi_gspace = None

        if self.inv_group[0] == 'C':
            inv_gspace = gspaces.rot2dOnR2(N=int(self.inv_group[1]))
        elif self.inv_group[0] == 'D':
            inv_gspace = gspaces.flipRot2dOnR2(N=int(self.inv_group[1]))
        elif self.inv_group == 'None':
            inv_gspace = None
        shared_cnn = equim.InvEquiEncoder(obs_shape,
                equi_gspace, inv_gspace, num_layers=args.num_shared_layers, num_filters=args.num_filters).to(self.device)
        head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).to(self.device)
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim)
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim)
        )

        self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim,
                             args.actor_log_std_min, args.actor_log_std_max).to(self.device)
        self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).to(self.device)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )

        self.train()
        self.critic_target.train()
