import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
import algorithms.equivariant_module as equim
from algorithms.drq import DrQ


class EDrQ(DrQ):
    # equivariant DrQ [K=2,M=2]
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        # use group equivariant CNN as shared encoder
        shared_cnn = equim.sharedEquivariantEncoder(obs_shape=obs_shape,
                                                    n_out=args.esac_n_out, N=args.esac_N).to(self.device)
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