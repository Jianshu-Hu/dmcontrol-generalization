# Following modules are modified according to
# https://github.com/pointW/equi_rl/blob/iclr22/networks/equivariant_sac_net.py
import torch
from e2cnn import gspaces
from e2cnn import nn

import algorithms.modules as m


class sharedEquivariantEncoder(torch.nn.Module):
    """
    Equivariant Encoder. The input is a trivial representation with obs_channel channels.
    The output is a regular representation with n_out channels
    """
    def __init__(self, obs_shape, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_channel = obs_shape[0]
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.preprocess = torch.nn.Sequential(
            m.CenterCrop(size=84),
            m.NormalizeImg())
        self.conv = torch.nn.Sequential(
            # 84x84
            nn.R2Conv(nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 42x42
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 21x21
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 10x10
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 5x5
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # # 1x1
            # nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
            #           nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
            #           kernel_size=3, padding=0, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # # 1x1
        )
        self.out_shape = self._get_out_shape()

    def _get_out_shape(self):
        temp_obs = nn.GeometricTensor(torch.randn(*self.obs_shape).unsqueeze(0),
                                 nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        return self.conv(temp_obs).tensor.squeeze(0).shape

    def forward(self, obs):
        obs = self.preprocess(obs)
        geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        return self.conv(geo).tensor


# class EquivariantSACCritic(torch.nn.Module):
#     """
#     Equivariant SAC's invariant critic
#     """
#     def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
#         super().__init__()
#         self.obs_channel = obs_shape[0]
#         self.n_hidden = n_hidden
#         self.c4_act = gspaces.Rot2dOnR2(N)
#         # enc = sharedEquivariantEncoder(obs_channel=obs_shape[1])
#         self.img_conv = sharedEquivariantEncoder(self.obs_channel, n_hidden, initialize, N)
#         self.n_rho1 = 2 if N==2 else 1
#
#         self.critic_1 = torch.nn.Sequential(
#             # mixed representation including n_hidden regular representations (for the state),
#             # (action_dim-2) trivial representations (for the invariant actions)
#             # and 1 standard representation (for the equivariant actions)
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
#                       nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
#                       kernel_size=1, padding=0, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
#             nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
#                       nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
#                       kernel_size=1, padding=0, initialize=initialize),
#         )
#
#         self.critic_2 = torch.nn.Sequential(
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
#                       nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
#                       kernel_size=1, padding=0, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
#             nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
#                       nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
#                       kernel_size=1, padding=0, initialize=initialize),
#         )
#
#     def forward(self, obs, act):
#         batch_size = obs.shape[0]
#         obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
#         conv_out = self.img_conv(obs_geo)
#         dxy = act[:, 1:3]
#         inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
#         n_inv = inv_act.shape[1]
#         cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
#         cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
#         out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
#         out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
#         return out1, out2


# class EquivariantSACActor(SACGaussianPolicyBase):
#     """
#     Equivariant SAC's equivariant actor
#     """
#     def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
#         super().__init__()
#         assert obs_shape[1] in [128, 64]
#         self.obs_channel = obs_shape[0]
#         self.action_dim = action_dim
#         self.c4_act = gspaces.Rot2dOnR2(N)
#         enc = getEnc(obs_shape[1], enc_id)
#         self.n_rho1 = 2 if N==2 else 1
#         self.conv = torch.nn.Sequential(
#             enc(self.obs_channel, n_hidden, initialize, N),
#             nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
#                       # mixed representation including action_dim trivial representations (for the std of all actions),
#                       # (action_dim-2) trivial representations (for the mu of invariant actions),
#                       # and 1 standard representation (for the mu of equivariant actions)
#                       nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
#                       kernel_size=1, padding=0, initialize=initialize)
#         )
#
#     def forward(self, obs):
#         batch_size = obs.shape[0]
#         obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
#         conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
#         dxy = conv_out[:, 0:2]
#         inv_act = conv_out[:, 2:self.action_dim]
#         mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
#         log_std = conv_out[:, self.action_dim:]
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return mean, log_std


# if __name__ == "__main__":
#     obs = torch.randn(10, 9, 84, 84)
#     img_conv = sharedEquivariantEncoder(obs_shape=[9, 84, 84], n_out=128, initialize=True, N=4)
#     print(img_conv.out_shape)
#
#     # obs_geo = nn.GeometricTensor(obs, nn.FieldType(c4_act, obs_channel * [c4_act.trivial_repr]))
#     conv_out = img_conv(obs)
#     print(conv_out.size())