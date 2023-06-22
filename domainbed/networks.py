# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np
from torch.autograd import Function

# from domainbed.lib import misc
# from domainbed.lib import wide_resnet
from domainbed.lib import big_transfer
from domainbed.lib import vision_transformer
from domainbed.lib import mlp_mixer
from typing import List, Dict, Optional, Any, Tuple
import clip


class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, class_num: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
				nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, class_num),
            )

    def get_parameters(self):
        return self.parameters()
    
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['backbone'] == 'resnet18':
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet50':
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet18-BN':
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
            self.disable_bn = False
        elif hparams['backbone'] == 'resnet50-BN':
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = False

        if self.disable_bn:
            self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()
        if self.disable_bn:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.disable_bn:
            self.freeze_bn()
 
    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if hparams['backbone'] == 'clip':
    # if 'clip_' in hparams['backbone']:
        return CLIP_Featurizer(hparams)
    elif len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    # elif input_shape[1:3] == (28, 28):
        # return MNIST_CNN(input_shape)
    # elif input_shape[1:3] == (32, 32):
        # return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224) and hparams['backbone'] in ['resnet50', 'resnet18', 'resnet50-BN', 'resnet18-BN']:
        return ResNet(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'ViT-' in hparams['backbone']:
        return vision_transformer.ViT2(input_shape, hparams)
    # elif input_shape[1:3] == (224, 224) and hparams['backbone'] in ['B_16', 'B_32', 'L_16', 'L_32']:
        # return vision_transformer.ViT(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'dino' in hparams['backbone']:
        return vision_transformer.DINO(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'DeiT' in hparams['backbone']:
        return vision_transformer.DeiT(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'HViT' in hparams['backbone']:
        return vision_transformer.HybridViT(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'Mixer' in hparams['backbone']:
        return mlp_mixer.MLPMixer(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'BiT' in hparams['backbone']:
        return big_transfer.BiT(input_shape, hparams)
    else:
        raise NotImplementedError


class CLIP_Featurizer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        print(hparams['clip_backbone'])
        # RN50 -> 1024
        # RN50x4 -> 640
        # RN50x16 -> 768
        # RN101, ViT-B/32, ViT-B/16 -> 512
        if hparams['clip_backbone'] not in ['ViT-B/32', 'ViT-B/16', 'RN101']:
            raise ValueError('Unknown clip_backbone: {}'.format(hparams['clip_backbone']))
        
        print(f'Using {hparams["clip_backbone"]}...')
        self.clip_model = clip.load(hparams["clip_backbone"])[0].float()
        self.n_outputs = 512
    
    def forward(self, x):
        return self.clip_model.encode_image(x)


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

"""caualvae的networks"""
def dag_right_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret
    
def dag_left_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = weight.matmul(input)
        if bias is not None:
            output += bias
        ret = output
    return ret



class MaskLayer(nn.Module):
	def __init__(self, z_dim, concept=4,z1_dim=4):
		super().__init__()
		self.z_dim = z_dim
		self.z1_dim = z1_dim
		self.concept = concept
		
		self.elu = nn.ELU()
		self.net1 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim),
		)
		self.net2 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim),
		)
		self.net3 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
		  nn.Linear(32, z1_dim),
		)
		self.net4 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim)
		)
		self.net = nn.Sequential(
			nn.Linear(z_dim , 32),
			nn.ELU(),
			nn.Linear(32, z_dim),
		)
	def masked(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z
   
	def masked_sep(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z
   
	def mix(self, z):
		zy = z.view(-1, self.concept*self.z1_dim)
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			if self.concept ==4:
				zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
			elif self.concept ==3:
				zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
		else:
			if self.concept ==4:
				zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
			elif self.concept ==3:
				zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		if self.concept ==4:
			rx4 = self.net4(zy4)
			h = torch.cat((rx1,rx2,rx3,rx4), dim=1)
		elif self.concept ==3:
			h = torch.cat((rx1,rx2,rx3), dim=1)
		#print(h.size())
		return h
   
class Mix(nn.Module):
	def __init__(self, z_dim, concept, z1_dim):
		super().__init__()
		self.z_dim = z_dim
		self.z1_dim = z1_dim
		self.concept = concept
		
		self.elu = nn.ELU()
		self.net1 = nn.Sequential(
			nn.Linear(z1_dim , 16),
			nn.ELU(),
			nn.Linear(16, z1_dim),
		)
		self.net2 = nn.Sequential(
			nn.Linear(z1_dim , 16),
			nn.ELU(),
			nn.Linear(16, z1_dim),
		)
		self.net3 = nn.Sequential(
			nn.Linear(z1_dim , 16),
			nn.ELU(),
			nn.Linear(16, z1_dim),
		)
		self.net4 = nn.Sequential(
			nn.Linear(z1_dim , 16),
			nn.ELU(),
			nn.Linear(16, z1_dim),
		)

   
	def mix(self, z):
		zy = z.view(-1, self.concept*self.z1_dim)
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
		else:
			zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		rx4 = self.net4(zy4)
		h = torch.cat((rx1,rx2,rx3,rx4), dim=1) 
		#print(h.size())
		return h


class CausalLayer(nn.Module):
	def __init__(self, z_dim, concept=4,z1_dim=4):
		super().__init__()
		self.z_dim = z_dim
		self.z1_dim = z1_dim
		self.concept = concept
		
		self.elu = nn.ELU()
		self.net1 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim),
		)
		self.net2 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim),
		)
		self.net3 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
		  nn.Linear(32, z1_dim),
		)
		self.net4 = nn.Sequential(
			nn.Linear(z1_dim , 32),
			nn.ELU(),
			nn.Linear(32, z1_dim)
		)
		self.net = nn.Sequential(
			nn.Linear(z_dim , 128),
			nn.ELU(),
			nn.Linear(128, z_dim),
		)
   
	def calculate(self, z, v):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z, v
   
	def masked_sep(self, z, v):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z,v
   
	def calculate_dag(self, z, v):
		zy = z.view(-1, self.concept*self.z1_dim)
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
		else:
			zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		rx4 = self.net4(zy4)
		h = torch.cat((rx1,rx2,rx3,rx4), dim=1)
		#print(h.size())
		return h,v
   
   
class Attention(nn.Module):
  def __init__(self, in_features, bias=False):
    super().__init__()
    self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
    self.sigmd = torch.nn.Sigmoid()
    #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    #self.A = torch.zeros(in_features,in_features).to(device)
    
  def attention(self, z, e):
    a = z.matmul(self.M).matmul(e.permute(0,2,1))
    a = self.sigmd(a)
    #print(self.M)
    A = torch.softmax(a, dim = 1)
    e = torch.matmul(A,e)
    return e, A
    
class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features,i = False, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features,out_features)
        self.a = self.a
        #self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
        #self.a[1][2], self.a[1][3] = 1,1
        self.A = nn.Parameter(self.a)
        
        self.b = torch.eye(out_features)
        self.b = self.b
        self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(self.B.t(), x)
        return x
        
    def mask_u(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x
        
    def inv_cal(self, x,v):
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, self.I - self.A, self.bias)
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v

    def calculate_dag(self, x, v):
        #print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias) 
        #print(x.size())
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v
        
    def calculate_cov(self, x, v):
        #print(self.A)
        v = ut.vector_expand(v)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        #print(v)
        return x, v
        
    def calculate_gaussian_ini(self, x, v):
        print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
            v = v.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
        v = F.linear(v, torch.mul(torch.inverse(self.I - self.A),torch.inverse(self.I - self.A)), self.bias)
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
            v = v.permute(0,2,1).contiguous()
        return x, v
    def calculate_gaussian(self, x, v):
        print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
            v = v.permute(0,2,1)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
            v = v.permute(0,2,1).contiguous()
        return x, v
    #def encode_
    def forward(self, x):
        x = x * torch.inverse((self.A)+self.I)
        return x
      
class ConvEncoder(nn.Module):
	def __init__(self, out_dim=None):
		super().__init__()
		# init 96*96
		self.conv1 = torch.nn.Conv2d(3, 32, 4, 2, 1) # 48*48
		self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False) # 24*24
		self.conv3 = torch.nn.Conv2d(64, 1, 4, 2, 1, bias=False)
		#self.conv4 = torch.nn.Conv2d(128, 1, 1, 1, 0) # 54*44
   
		self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
		self.convm = torch.nn.Conv2d(1, 1, 4, 2, 1)
		self.convv = torch.nn.Conv2d(1, 1, 4, 2, 1)
		self.mean_layer = nn.Sequential(
			torch.nn.Linear(8*8, 16)
			) # 12*12
		self.var_layer = nn.Sequential(
			torch.nn.Linear(8*8, 16)
			)
		# self.fc1 = torch.nn.Linear(6*6*128, 512)
		self.conv6 = nn.Sequential(
			nn.Conv2d(3, 32, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(32, 32, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(32, 64, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(64, 64, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(64, 64, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(64, 256, 4, 1),
			nn.ReLU(True),
			nn.Conv2d(256,128 , 1)
		)

	def encode(self, x):
		x = self.LReLU(self.conv1(x))
		x = self.LReLU(self.conv2(x))
		x = self.LReLU(self.conv3(x))
		#x = self.LReLU(self.conv4(x))
		#print(x.size())
		hm = self.convm(x)
		#print(hm.size())
		hm = hm.view(-1, 8*8)
		hv = self.convv(x)
		hv = hv.view(-1, 8*8)
		mu, var = self.mean_layer(hm), self.var_layer(hv)
		var = F.softplus(var) + 1e-8
		#var = torch.reshape(var, [-1, 16, 16])
		#print(mu.size())
		return  mu, var
	def encode_simple(self,x):
		x = self.conv6(x)
		m,v = ut.gaussian_parameters(x, dim=1)
		#print(m.size())
		return m,v
class ConvDecoder(nn.Module):
	def __init__(self, out_dim = None):
		super().__init__()
   
		self.net6 = nn.Sequential(
				nn.Conv2d(16, 128, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(128, 64, 4),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(64, 64, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(64, 32, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(32, 32, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(32, 32, 4, 2, 1),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(32, 3, 4, 2, 1)
		)
   
	def decode_sep(self,x):
		return None
   
	def decode(self, z):
		z = z.view(-1, 16, 1, 1)
		z = self.net6(z)
		return z

class ConvDec(nn.Module):
  def __init__(self, out_dim = None):
    super().__init__()
    self.concept = 4
    self.z1_dim = 16
    self.z_dim = 64
    self.net1 = ConvDecoder()
    self.net2 = ConvDecoder()
    self.net3 = ConvDecoder()
    self.net4 = ConvDecoder()
    self.net5 = nn.Sequential(
			nn.Linear(16, 512),
  		nn.BatchNorm1d(512),
  		nn.Linear(512, 1024),
  		nn.BatchNorm1d(1024)
     )
    self.net6 = nn.Sequential(
  		nn.Conv2d(16, 128, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(128, 64, 4),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(64, 64, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(64, 32, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(32, 32, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(32, 32, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(32, 3, 4, 2, 1)
		)
        
  def decode_sep(self, z, u, y=None):
    z = z.view(-1, self.concept*self.z1_dim)
    zy = z if y is None else torch.cat((z, y), dim=1)
    zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
    rx1 = self.net1.decode(zy1)
    #print(rx1.size())
    rx2 = self.net2.decode(zy2)
    rx3 = self.net3.decode(zy3)
    rx4 = self.net4.decode(zy4)
    z = (rx1+rx2+rx3+rx4)/4
    return z
    
  def decode(self, z, u, y=None):
    z = z.view(-1, self.concept*self.z1_dim, 1, 1)
    z = self.net6(z)
    #print(z.size())
    
    return z

class Encoder(nn.Module):
	def __init__(self, z_dim, channel=4, y_dim=4):
		super().__init__()
		self.z_dim = z_dim
		self.y_dim = y_dim
		self.channel = channel
		self.fc1 = nn.Linear(self.channel*96*96, 300)
		self.fc2 = nn.Linear(300+y_dim, 300)
		self.fc3 = nn.Linear(300, 300)
		self.fc4 = nn.Linear(300, 2 * z_dim)
		self.LReLU = nn.LeakyReLU(0.2, inplace=True)
		self.net = nn.Sequential(
			nn.Linear(self.channel*96*96, 900),
			nn.ELU(),
			nn.Linear(900, 300),
			nn.ELU(),
			nn.Linear(300, 2 * z_dim),
		)

	def conditional_encode(self, x, l):
		x = x.view(-1, self.channel*96*96)
		x = F.elu(self.fc1(x))
		l = l.view(-1, 4)
		x = F.elu(self.fc2(torch.cat([x, l], dim=1)))
		x = F.elu(self.fc3(x))
		x = self.fc4(x)
		m, v = ut.gaussian_parameters(x, dim=1)
		return m,v

	def encode(self, x, y=None):
		xy = x if y is None else torch.cat((x, y), dim=1)
		xy = xy.view(-1, self.channel*96*96)
		h = self.net(xy)
		m, v = ut.gaussian_parameters(h, dim=1)
		#print(self.z_dim,m.size(),v.size())
		return m, v
   
   
class Decoder_DAG(nn.Module):
	def __init__(self, z_dim, concept, z1_dim, channel = 4, y_dim=0):
		super().__init__()
		self.z_dim = z_dim
		self.z1_dim = z1_dim
		self.concept = concept
		self.y_dim = y_dim
		self.channel = channel
		#print(self.channel)
		self.elu = nn.ELU()
		self.net1 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net2 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net3 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net4 = nn.Sequential(
			nn.Linear(z1_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
		self.net5 = nn.Sequential(
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
   
		self.net6 = nn.Sequential(
			nn.Linear(z_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 1024),
			nn.ELU(),
			nn.Linear(1024, 1024),
			nn.ELU(),
			nn.Linear(1024, self.channel*96*96)
		)
	def decode_condition(self, z, u):
		#z = z.view(-1,3*4)
		z = z.view(-1, 3*4)
		z1, z2, z3 = torch.split(z, self.z_dim//4, dim = 1)
		#print(u[:,0].reshape(1,u.size()[0]).size())
		rx1 = self.net1(torch.transpose(torch.cat((torch.transpose(z1, 1,0), u[:,0].reshape(1,u.size()[0])), dim = 0), 1, 0))
		rx2 = self.net2(torch.transpose(torch.cat((torch.transpose(z2, 1,0), u[:,1].reshape(1,u.size()[0])), dim = 0), 1, 0))
		rx3 = self.net3(torch.transpose(torch.cat((torch.transpose(z3, 1,0), u[:,2].reshape(1,u.size()[0])), dim = 0), 1, 0))
   
		h = self.net4( torch.cat((rx1,rx2, rx3), dim=1))
		return h

	def decode_mix(self, z):
		z = z.permute(0,2,1)
		z = torch.sum(z, dim = 2, out=None) 
		#print(z.contiguous().size())
		z = z.contiguous()
		h = self.net1(z)
		return h
   
	def decode_union(self, z, u, y=None):
		
		z = z.view(-1, self.concept*self.z1_dim)
		zy = z if y is None else torch.cat((z, y), dim=1)
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			zy1, zy2, zy3, zy4 = zy[:,0],zy[:,1],zy[:,2],zy[:,3]
		else:
			zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		rx4 = self.net4(zy4)
		h = self.net5((rx1+rx2+rx3+rx4)/4)
		return h,h,h,h,h
   
	def decode(self, z, u , y = None):
		z = z.view(-1, self.concept*self.z1_dim)
		h = self.net6(z)
		return h, h,h,h,h
    
	def decode_sep(self, z, u, y=None):
		z = z.view(-1, self.concept*self.z1_dim)
		zy = z if y is None else torch.cat((z, y), dim=1)
			
		if self.z1_dim == 1:
			zy = zy.reshape(zy.size()[0],zy.size()[1],1)
			if self.concept ==4:
				zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
			elif self.concept ==3:
				zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
		else:
			if self.concept ==4:
				zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
			elif self.concept ==3:
				zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		if self.concept ==4:
			rx4 = self.net4(zy4)
			h = (rx1+rx2+rx3+rx4)/self.concept
		elif self.concept ==3:
			h = (rx1+rx2+rx3)/self.concept
		
		return h,h,h,h,h
   
	def decode_cat(self, z, u, y=None):
		z = z.view(-1, 4*4)
		zy = z if y is None else torch.cat((z, y), dim=1)
		zy1, zy2, zy3, zy4 = torch.split(zy, 1, dim = 1)
		rx1 = self.net1(zy1)
		rx2 = self.net2(zy2)
		rx3 = self.net3(zy3)
		rx4 = self.net4(zy4)
		h = self.net5( torch.cat((rx1,rx2, rx3, rx4), dim=1))
		return h
   
   
class Decoder(nn.Module):
	def __init__(self, z_dim, y_dim=0):
		super().__init__()
		self.z_dim = z_dim
		self.y_dim = y_dim
		self.net = nn.Sequential(
			nn.Linear(z_dim + y_dim, 300),
			nn.ELU(),
			nn.Linear(300, 300),
			nn.ELU(),
			nn.Linear(300, 4*96*96)
		)

	def decode(self, z, y=None):
		zy = z if y is None else torch.cat((z, y), dim=1)
		return self.net(zy)

class Classifiercausalvae(nn.Module):
	def __init__(self, y_dim):
		super().__init__()
		self.y_dim = y_dim
		self.net = nn.Sequential(
			nn.Linear(784, 300),
			nn.ReLU(),
			nn.Linear(300, 300),
			nn.ReLU(),
			nn.Linear(300, y_dim)
		)

	def classify(self, x):
		return self.net(x)