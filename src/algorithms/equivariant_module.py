# Following modules are modified according to
# https://github.com/QUVA-Lab/e2cnn/blob/master/examples/model.ipynb
import torch
import algorithms.modules as m

from escnn import gspaces
from e2cnn import nn as e2nn
from escnn import nn as esnn


class inv_equi_Encoder(torch.nn.Module):
    def __init__(self, obs_shape, gspace, inv, num_layers=11, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.gspace = gspace
        self.inv = inv

        self.preprocess = torch.nn.Sequential(
            # Need odd-sized inputs for stride=2 to preserve equivariance
            m.CenterCrop(size=84),
            m.NormalizeImg())

        self.in_type = esnn.FieldType(
            self.gspace, obs_shape[0] * [self.gspace.trivial_repr]
        )
        self.hid_type = esnn.FieldType(
            self.gspace, num_filters * [self.gspace.regular_repr]
        )

        self.layers = [esnn.R2Conv(self.in_type, self.hid_type, 3, stride=2)]
        for _ in range(1, num_layers):
            self.layers.append(esnn.ReLU(self.hid_type))
            self.layers.append(esnn.R2Conv(self.hid_type, self.hid_type, 3, stride=1))
        if inv:
            self.layers.append(esnn.GroupPooling(self.hid_type))
        self.conv_layers = esnn.SequentialModule(*self.layers)
        self.out_shape = self._get_out_shape()

    def _get_out_shape(self):
        input = torch.randn(*self.obs_shape).unsqueeze(0)
        return self.forward(input).squeeze(0).shape

    def forward(self, obs):
        input = self.preprocess(obs)
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = esnn.GeometricTensor(input, self.in_type)
        h = self.conv_layers(x)

        return h.tensor


class InvEquiEncoder(torch.nn.Module):
    def __init__(self, obs_shape, equi_gspace, inv_gspace, num_layers=11, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.obs_shape = obs_shape
        self.equi_gspace = equi_gspace
        self.inv_gspace = inv_gspace
        self.num_layers = num_layers
        self.num_filters = num_filters

        if equi_gspace is not None:
            # equivariant
            self.equi_encoder = inv_equi_Encoder(obs_shape, equi_gspace, inv=False, num_layers=num_layers, num_filters=num_filters)
        if inv_gspace is not None:
            # invariant
            self.inv_encoder = inv_equi_Encoder(obs_shape, inv_gspace, inv=True, num_layers=num_layers, num_filters=num_filters)

        self.out_shape = self._get_out_shape()

    def _get_out_shape(self):
        x = torch.randn(*self.obs_shape).unsqueeze(0)
        return self.forward(x).squeeze(0).shape

    def forward(self, obs):
        if self.inv_gspace is not None:
            h_inv = self.inv_encoder(obs)
            if self.equi_gspace is not None:
                h_equi = self.equi_encoder(obs)
                h_equi_inv = torch.concat((h_inv, h_equi), dim=1)
            else:
                h_equi_inv = h_inv
        else:
            if self.equi_gspace is not None:
                h_equi_inv = self.equi_encoder(obs)

        return h_equi_inv


class ExportEncoder(torch.nn.Module):
    def __init__(self, original_encoder):
        super().__init__()

        self.num_filters = original_encoder.num_filters
        self.equi_gspace = original_encoder.equi_gspace
        self.inv_gspace = original_encoder.inv_gspace

        self.preprocess = torch.nn.Sequential(
            # Need odd-sized inputs for stride=2 to preserve equivariance
            m.CenterCrop(size=84),
            m.NormalizeImg())

        if self.equi_gspace is not None:
            # export equivariant net
            self.equi_convnet = original_encoder.equi_encoder.conv_layers.export()
        if self.inv_gspace is not None:
            # export invariant net
            self.inv_convnet = original_encoder.inv_encoder.conv_layers.export()

    def forward(self, obs):
        obs = self.preprocess(obs)
        if self.inv_gspace is not None:
            h_inv = self.inv_convnet(obs)
            if self.equi_gspace is not None:
                h_equi = self.equi_convnet(obs)
                h_equi_inv = torch.concat((h_inv, h_equi), dim=1)
            else:
                h_equi_inv = h_inv
        else:
            if self.equi_gspace is not None:
                h_equi_inv = self.equi_convnet(obs)

        return h_equi_inv


# class sharedEquivariantEncoder(torch.nn.Module):
#     def __init__(self, obs_shape, n_out=128, initialize=True, N=4):
#         super().__init__()
#         self.obs_shape = obs_shape
#         self.obs_channel = obs_shape[0]
#         self.preprocess = torch.nn.Sequential(
#             m.CenterCrop(size=84),
#             m.NormalizeImg())
#
#         # the model is equivariant under rotations by 90 degrees, modelled by C4
#         self.r2_act = gspaces.Rot2dOnR2(N=N)
#
#         # the input image is a scalar field, corresponding to the trivial representation
#         in_type = e2nn.FieldType(self.r2_act, self.obs_channel*[self.r2_act.trivial_repr])
#
#         # we store the input type for wrapping the images into a geometric tensor during the forward pass
#         self.input_type = in_type
#
#         # convolution 1
#         # first specify the output type of the convolutional layer
#         # we choose 24 feature fields, each transforming under the regular representation of C4
#         out_type = e2nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
#         self.block1 = e2nn.SequentialModule(
#             # nn.MaskModule(in_type, 29, margin=1),
#             e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
#             # nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#
#         # convolution 2
#         # the old output type is the input type to the next layer
#         in_type = self.block1.out_type
#         # the output type of the second convolution layer are 48 regular feature fields of C4
#         out_type = e2nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
#         self.block2 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
#             # nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#         self.pool1 = e2nn.SequentialModule(
#             e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#         )
#
#         # convolution 3
#         # the old output type is the input type to the next layer
#         in_type = self.block2.out_type
#         # the output type of the third convolution layer are 48 regular feature fields of C4
#         out_type = e2nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
#         self.block3 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
#             # nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#
#         # convolution 4
#         # the old output type is the input type to the next layer
#         in_type = self.block3.out_type
#         # the output type of the fourth convolution layer are 96 regular feature fields of C4
#         out_type = e2nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
#         self.block4 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
#             # nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#         self.pool2 = e2nn.SequentialModule(
#             e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#         )
#
#         # convolution 5
#         # the old output type is the input type to the next layer
#         in_type = self.block4.out_type
#         # the output type of the fifth convolution layer are 96 regular feature fields of C4
#         out_type = e2nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
#         self.block5 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
#             # nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#
#         # convolution 6
#         # the old output type is the input type to the next layer
#         in_type = self.block5.out_type
#         # the output type of the sixth convolution layer are 64 regular feature fields of C4
#         out_type = e2nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])
#         self.block6 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
#             # nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#         self.pool3 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
#
#         self.gpool = e2nn.GroupPooling(out_type)
#
#         self.out_shape = self._get_out_shape()
#
#     def _get_out_shape(self):
#         return self.forward(torch.randn(*self.obs_shape).unsqueeze(0)).squeeze(0).shape
#
#     def forward(self, input: torch.Tensor):
#         input = self.preprocess(input)
#         # wrap the input tensor in a GeometricTensor
#         # (associate it with the input type)
#         x = e2nn.GeometricTensor(input, self.input_type)
#
#         # apply each equivariant block
#
#         # Each layer has an input and an output type
#         # A layer takes a GeometricTensor in input.
#         # This tensor needs to be associated with the same representation of the layer's input type
#         #
#         # The Layer outputs a new GeometricTensor, associated with the layer's output type.
#         # As a result, consecutive layers need to have matching input/output types
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.pool1(x)
#
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.pool2(x)
#
#         x = self.block5(x)
#         x = self.block6(x)
#
#         # pool over the spatial dimensions
#         x = self.pool3(x)
#
#         # pool over the group
#         x = self.gpool(x)
#
#         # unwrap the output GeometricTensor
#         # (take the Pytorch tensor and discard the associated representation)
#         x = x.tensor
#
#         return x

# class sharedEquivariantEncoder(torch.nn.Module):
#     """
#     Equivariant Encoder. The input is a trivial representation with obs_channel channels.
#     The output is a regular representation with n_out channels
#     """
#     def __init__(self, obs_shape, n_out=128, initialize=True, N=4):
#         super().__init__()
#         self.obs_shape = obs_shape
#         self.obs_channel = obs_shape[0]
#         self.c4_act = gspaces.Rot2dOnR2(N)
#         self.preprocess = torch.nn.Sequential(
#             m.CenterCrop(size=84),
#             m.NormalizeImg())
#         self.conv = torch.nn.Sequential(
#             # 84x84
#             nn.R2Conv(nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]),
#                       nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
#             # 42x42
#             nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
#             # 21x21
#             nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
#             # 10x10
#             nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
#             # 5x5
#             nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=1, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),
#
#             nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
#                       nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
#                       kernel_size=3, padding=0, initialize=initialize),
#             nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
#             nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
#             # # 1x1
#             # nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
#             #           nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
#             #           kernel_size=3, padding=0, initialize=initialize),
#             # nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
#             # # 1x1
#         )
#         self.out_shape = self._get_out_shape()
#
#     def _get_out_shape(self):
#         temp_obs = nn.GeometricTensor(torch.randn(*self.obs_shape).unsqueeze(0),
#                                  nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
#         return self.conv(temp_obs).tensor.squeeze(0).shape
#
#     def forward(self, obs):
#         obs = self.preprocess(obs)
#         geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
#         return self.conv(geo).tensor


