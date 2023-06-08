import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Iterable, List, Tuple
from copy import deepcopy


def _get_out_shape_cuda(in_shape, layers):
	x = torch.randn(*in_shape).cuda().unsqueeze(0)
	return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability"""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size in {84, 100}, f'unexpected size: {size}'
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
		if self.size == 84:
			p = 8
		return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


class RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		return self.projection(x)


class SODAMLP(nn.Module):
	def __init__(self, projection_dim, hidden_dim, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.mlp = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(x)


class SharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32):
		super().__init__()
		assert len(obs_shape) == 3
		self.num_layers = num_layers
		self.num_filters = num_filters

		self.layers = [CenterCrop(size=84), NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		for _ in range(1, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(obs_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class Encoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, projection):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		return self.projection(x)


class Actor(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
		super().__init__()
		self.encoder = encoder
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)
		self.mlp.apply(weight_init)

	def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
		x = self.encoder(x, detach)
		mu, log_std = self.mlp(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std


class QFunction(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.apply(weight_init)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)
		return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.Q1 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)

	def forward(self, x, action, detach=False):
		x = self.encoder(x, detach)
		return self.Q1(x, action), self.Q2(x, action)


class CURLHead(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder
		self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

	def compute_logits(self, z_a, z_pos):
		"""
		Uses logits trick for CURL:
		- compute (B,B) matrix z_a (W z_pos.T)
		- positives are all diagonal elements
		- negatives are all other elements
		- to compute loss use multiclass cross entropy with identity matrix for labels
		"""
		Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
		logits = torch.matmul(z_a, Wz)  # (B,B)
		logits = logits - torch.max(logits, 1)[0][:, None]
		return logits


class InverseDynamics(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = nn.Sequential(
			nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, action_shape[0])
		)
		self.apply(weight_init)

	def forward(self, x, x_next):
		h = self.encoder(x)
		h_next = self.encoder(x_next)
		joint_h = torch.cat([h, h_next], dim=1)
		return self.mlp(joint_h)


class SODAPredictor(nn.Module):
	def __init__(self, encoder, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = SODAMLP(
			encoder.out_dim, hidden_dim, encoder.out_dim
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(self.encoder(x))


# Modify PCGrad implemented in
# https://pytorch-optimizers.readthedocs.io/en/latest/_modules/pytorch_optimizer/optimizer/pcgrad.html#PCGrad
class PCGrad():
	"""Gradient Surgery for Multi-Task Learning.

	:param optimizer: OPTIMIZER: optimizer instance.
	:param reduction: str. reduction method.
	"""
	def __init__(self, optimizer, reduction='mean'):
		self.optimizer = optimizer
		self.reduction = reduction
		return

	@torch.no_grad()
	def reset(self):
		self.zero_grad()

	def zero_grad(self):
		return self.optimizer.zero_grad(set_to_none=True)

	def step(self):
		return self.optimizer.step()

	def set_grad(self, grads: List[torch.Tensor]):
		idx: int = 0
		for group in self.optimizer.param_groups:
			for p in group['params']:
				p.grad = grads[idx]
				idx += 1

	def retrieve_grad(self):
		"""Get the gradient of the parameters of the network with specific objective."""

		grad, shape, has_grad = [], [], []
		for group in self.optimizer.param_groups:
			for p in group['params']:
				if p.grad is None:
					shape.append(p.shape)
					grad.append(torch.zeros_like(p, device=p.device))
					has_grad.append(torch.zeros_like(p, device=p.device))
					continue
				shape.append(p.grad.shape)
				grad.append(p.grad.clone())
				has_grad.append(torch.ones_like(p, device=p.device))
		return grad, shape, has_grad

	def pack_grad(self, objectives):
		"""Pack the gradient of the parameters of the network for each objective.

		:param objectives: Iterable[nn.Module]. a list of objectives.
		:return: torch.Tensor. packed gradients.
		"""

		grads, shapes, has_grads = [], [], []
		for objective in objectives:
			self.optimizer.zero_grad(set_to_none=True)
			objective.backward(retain_graph=True)

			grad, shape, has_grad = self.retrieve_grad()

			grads.append(self.flatten_grad(grad))
			has_grads.append(self.flatten_grad(has_grad))
			shapes.append(shape)

		return grads, shapes, has_grads

	def project_conflicting(self, grads: List[torch.Tensor], has_grads: List[torch.Tensor]) -> torch.Tensor:
		"""Project conflicting.
		In our setting, the first task is the main task and th second task is the auxiliary task.

		:param grads: a list of the gradient of the parameters.
		:param has_grads: a list of mask represent whether the parameter has gradient.
		:return: torch.Tensor. merged gradients.
		"""
		shared: torch.Tensor = torch.stack(has_grads).prod(0).bool()
		pc_grad: List[torch.Tensor] = deepcopy(grads)
		# pc_grad[0]: g_main, pc_grad[1]: g_auxiliary
		g_dot = torch.dot(pc_grad[1], pc_grad[0])
		if g_dot < 0:
			pc_grad[1] -= g_dot * pc_grad[0]/(pc_grad[0].norm()**2)
		# for i, g_i in enumerate(pc_grad):
		# 	random.shuffle(grads)
		# 	for g_j in grads:
		# 		g_i_g_j = torch.dot(g_i, g_j)
		# 		if g_i_g_j < 0:
		# 			g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
		merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
		if self.reduction == 'mean':
			merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
		elif self.reduction == 'sum':
			merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
		else:
			exit('invalid reduction method')

		merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
		return merged_grad

	def pc_backward(self, objectives):
		"""Calculate the gradient of the parameters.

		:param objectives: Iterable[nn.Module]. a list of objectives.
		"""

		grads, shapes, has_grads = self.pack_grad(objectives)
		pc_grad = self.project_conflicting(grads, has_grads)
		pc_grad = self.un_flatten_grad(pc_grad, shapes[0])
		self.set_grad(pc_grad)

	def un_flatten_grad(self, grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
		"""Unflatten the gradient"""
		idx: int = 0
		un_flatten_grad: List[torch.Tensor] = []
		for shape in shapes:
			length = np.prod(shape)
			un_flatten_grad.append(grads[idx:idx + length].view(shape).clone())
			idx += length
		return un_flatten_grad

	def flatten_grad(self, grads: List[torch.Tensor]) -> torch.Tensor:
		"""Flatten the gradient."""
		return torch.cat([grad.flatten() for grad in grads])
