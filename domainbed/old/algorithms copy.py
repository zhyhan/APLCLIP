# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from collections import OrderedDict
from typing import Tuple
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

import clip

_tokenizer = _Tokenizer()
ALGORITHMS = [
    'ERM',
    'FrozenERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'CLIP',
    'CoCoOpCLIP'
    'APLCLIP',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        #self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = clip.load(self.hparams['clip_backbone'])[0].float()

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print('Set self.clip_model.parameters.reguires_grad = False!')

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  # 
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        #self.prompt = torch.cat([clip.tokenize(f'a {ppt} cause a photo') for ppt in classnames]).to(self.device)
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        
    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}
    
    def predict(self, x):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)

class APLCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(APLCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        #  initial prompt.
        self.prompt_generator = CoCoOpPromptLearner(input_shape, num_classes, num_domains, hparams)
        self.tokenized_prompts = self.prompt_generator.tokenized_prompts
        self.dtype = self.clip_model.dtype
        self.text_encoder = TextEncoder(self.clip_model)
        self.prompt_generator.ctx.requires_grad = True
        self.discriminator = networks.DomainDiscriminator(in_feature=self.clip_model.text_projection.shape[1], hidden_size=1024).to('cuda')
        
        self.grl = networks.WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        #self.prompt_generator.meta_net.requires_grad = True
        self.optimizer = torch.optim.SGD(
            (list(self.prompt_generator.meta_net.parameters()) + [self.prompt_generator.ctx] + list(self.discriminator.get_parameters())),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        # self.optimizer_disc = torch.optim.SGD(
        #     self.discriminator.get_parameters(),
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
        # self.optimizer = torch.optim.SGD(
        #     [self.prompt_generator.ctx],
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
    def update(self, minibatches, unlabeled=None):

        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        image_features = [self.clip_model.encode_image(x) for x in all_x]

        image_features = torch.cat(image_features)

        all_acl = torch.cat([data[2].cuda().type_as(image_features) for data in minibatches])
        #all_acl = torch.zeros(image_features.size()[0], 3, 512).cuda()
        acl_features = torch.cat([all_acl, image_features.unsqueeze(dim=1)], dim = 1)


        #acl_features = acl_features.mean(dim=0, keepdim=True) #consider the computation speed
        ctx, image_features = self.prompt_generator(acl_features, ctx_only=True)

        N = 32*3 #每个样本都要算一个独立的text encoder, 虽然这样会减慢速度。
        #N = 1
        prefix = self.prompt_generator.token_prefix.expand(N, -1, -1, -1) # [N, n_cls, 1, dim]
        suffix = self.prompt_generator.token_suffix.expand(N, -1, -1, -1)
        ctx = ctx.expand(self.prompt_generator.n_cls, -1, -1, -1) #expand to class
        ctx = ctx.permute(1, 0, 2, 3)

        prompts = torch.cat([
            prefix,
            ctx,
            suffix
        ], dim=-2)

        prompts = prompts.reshape(N * self.prompt_generator.n_cls, -1, self.prompt_generator.ctx_dim)

        tokenized_prompts = self.prompt_generator.tokenized_prompts
        tokenized_prompts = tokenized_prompts.repeat(N, 1)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        #TODO: design a good merge algorithm
        true_class_text_feature = text_features.reshape(N, self.prompt_generator.n_cls, -1)
        true_class_text_feature = torch.cat([true_class_text_feature[i,all_y[i],:].unsqueeze(0) for i in range(true_class_text_feature.size()[0])])

        disc_input = image_features + true_class_text_feature

        disc_input = self.grl(disc_input)
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64)
            for i, (x, y, z) in enumerate(minibatches)
        ]).to('cuda')
        disc_loss = F.cross_entropy(disc_out, disc_labels)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        classification_loss = F.cross_entropy(logits_per_image, all_y) 
        #通过设计margin loss，增加每个类prompt的context差距，进而学习到更具有相关的prompt token。
        #第一步：找出正确的true_class_text_feature
        #第二步：计算true_class_text_feature与其它text_feature的距离，计算平均距离，让这个距离越大越好
        #第三步：计算损失
        true_class_text_feature = true_class_text_feature / true_class_text_feature.norm(dim=-1, keepdim=True)
        distences_per_image = true_class_text_feature@text_features.t()
        dist_loss = distences_per_image.mean()


        loss = classification_loss + 0.1*disc_loss - 0.01*dist_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"class_loss": classification_loss.item(), "disc_loss": disc_loss.item(), "dist_loss": dist_loss.item()}
     
    def predict(self, x, z):
        image_feature = self.clip_model.encode_image(x)
        #all_acl = torch.zeros(x.size()[0], 3, 512).cuda()
        all_acl = z.cuda().type_as(image_feature)
        acl_features = torch.cat([all_acl, image_feature.unsqueeze(dim=1)], dim = 1)
        prompts, image_feature = self.prompt_generator(acl_features)

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.prompt_generator.tokenized_prompts

        logit_scale = self.clip_model.logit_scale.data.exp()
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_feature):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        return logits


class CoCoOpPromptLearner(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CoCoOpPromptLearner, self).__init__(input_shape, num_classes, num_domains, hparams)
        dtype = self.clip_model.dtype
        n_ctx = 4
        ctx_init = "a_photo_of_a"
        #ctx_init = "cause_the_presence_of"
        ctx_position = 'end'

        classnames =  hparams['class_names']
        n_cls = len(classnames)
        
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        embed_dim = self.clip_model.text_projection.shape[1]

        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = torch.cat([clip.tokenize(p) for p in ctx_init]).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = "cause_the_presence_of"
        prompt_prefix = prompt_prefix.replace("_", " ")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.prompt_prefix = prompt_prefix

        #To optimize
        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(
           nn.Linear(embed_dim, embed_dim // 16),
           nn.ReLU(inplace=True),
           nn.Linear(embed_dim // 16, ctx_dim)
        )
        #self.meta_net = nn.Linear(embed_dim, embed_dim)
        

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        #prompts = [name + " " + "causes" for name in classnames]
        prompts = [name + " " + prompt_prefix + " " + ctx_init + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1+5, :])  # SOS 句子起始标识符
        self.register_buffer("token_suffix", embedding[:, 1+5+n_ctx:, :])  # CLS, EOS #句子结束表示符

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.dtype = dtype
        self.ctx_dim = ctx_dim
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts
    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([self.clip.tokenize(p) for p in prompts]).to(self.device)

        with torch.no_grad():
            embedding = self.clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts

    def forward(self, im_features, ctx_only=False):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        acps = [] #anticausal prompts                     # (n_ctx, ctx_dim)
        for i in range(self.n_ctx):
            acp = self.meta_net(im_features[:, i, :]) 
            acps.append(acp.unsqueeze(1))
        bias = torch.cat(acps, dim=1)      # (batch, n_ctx, ctx_dim)
        #bias = torch.cat([self.meta_net(im_features[:, i, :]) for i in range(self.n_ctx)], dim=1)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        #print(ctx_shifted.size())
        image_features = acps[-1].squeeze(1)
        if ctx_only:
            return ctx_shifted, image_features # don't expand to n_cls, optimize one ctx for all classes
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts, image_features

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
