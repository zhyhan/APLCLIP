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


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)#TODO: 如何不微调CLIP的特征提取器呢？
        
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)


class FrozenERM(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FrozenERM, self).__init__(input_shape, num_classes, num_domains,
                                hparams)

        for param in self.featurizer.parameters():
            param.requires_grad = False
        print('Set self.model.parameters.reguires_grad = False!')

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )


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
        #self.prompt_generator.meta_net.requires_grad = True
        self.optimizer = torch.optim.SGD(
            (list(self.prompt_generator.meta_net.parameters()) + [self.prompt_generator.ctx]),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        # self.optimizer = torch.optim.SGD(
        #     [self.prompt_generator.ctx],
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        #all_x = [data[0].cpu().float() for data in minibatches]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        #anticausal information size = [batch, 4, 512]
        all_acl = torch.cat([data[1].cuda().float() for data in minibatches])
        #all_y = torch.cat([data[1].cpu().long() for data in minibatches])
        #  encode image for each domain and concat them
        image_features = [self.clip_model.encode_image(x) for x in all_x]

        #anticausal information size = [batch, 4, 512]
        image_features = torch.cat(image_features)

        image_feature_avg = image_features.mean(dim=0, keepdim=True)

        acl_features = torch.cat([all_acl, image_features


        ctx = self.prompt_generator(image_feature_avg, ctx_only=True)

        N = 1
        prefix = self.prompt_generator.token_prefix.expand(N, -1, -1, -1) # [N, n_cls, 1, dim]
        suffix = self.prompt_generator.token_suffix.expand(N, -1, -1, -1)
        ctx = ctx.expand(self.prompt_generator.n_cls, -1, -1, -1)
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

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(logits_per_image, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}
    
    def predict(self, x):
        image_feature = self.clip_model.encode_image(x)
        prompts = self.prompt_generator(image_feature)

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
        prompt_prefix = ctx_init
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.prompt_prefix = prompt_prefix

        #To optimize
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized
        self.meta_net = nn.Sequential(
           nn.Linear(embed_dim, embed_dim // 16),
           nn.ReLU(inplace=True),
           nn.Linear(embed_dim // 16, ctx_dim)
        )

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

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
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        if ctx_only:
            return ctx_shifted # don't expand to n_cls, optimize one ctx for all classes
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

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