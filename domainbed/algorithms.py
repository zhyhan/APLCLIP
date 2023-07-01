# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from scipy.stats import beta

import torch.distributed as dist


import copy
import numpy as np

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR


from domainbed.clip import clip

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
    'DPLCLIP',
    'CMSAN',
    'MetricSoftmax',
    'MetricSoftmaxAlign',
    'MetricSoftmaxAlignPatch',
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
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = clip.load(self.hparams['clip_backbone'])[0].float()

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        #print('Set self.clip_model.parameters.reguires_grad = False!')

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  # 
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        
    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}
    
    def predict(self, x):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)
     

class CoCoOpCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CoCoOpCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
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
        #all_y = torch.cat([data[1].cpu().long() for data in minibatches])
        #  encode image for each domain and concat them
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        image_features = torch.cat(image_features)

        image_feature_avg = image_features.mean(dim=0, keepdim=True)
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

# rename to DPL (Domain Prompt Learning)
class DPLCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams, sentence_prompt=False):
        super(DPLCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)

        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        
        if sentence_prompt:
            print('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in hparams['class_names']]
        else:
            classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]
        
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.network.apply(init_weights)
        
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
            
    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_features]
        image_features = torch.cat(image_features)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]

        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
        
        #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
        # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
        text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])
            
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(logits_per_image, all_y)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}


    def _get_text_features(self, domain_feature, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection      
        return text_features

    def predict(self, x):
        image_feature = self.clip_model.encode_image(x)
        
        domain_feature = self.network(image_feature)
        mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.hparams['class_names']), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)
        
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()

#MetricSoftmax is a baseline method that fine-tune the methods 
class MetricSoftmax(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MetricSoftmax, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dtype = self.clip_model.dtype

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        #self.discriminator = networks.MLP(self.clip_model.text_projection.shape[1], num_domains, self.hparams)
        self.register_buffer('update_count', torch.tensor([0]))

        params_to_update = [param for name, param in self.featurizer.named_parameters() if name.startswith('clip_model.visual.transformer.resblocks.11')]

        self.gen_opt = torch.optim.AdamW(#only update little parameters, knowledge vit structures
            params_to_update,
            lr=self.hparams["lr_g"],
            betas=(self.hparams['beta1'], 0.9))
        
        # self.disc_opt = torch.optim.AdamW(
        #     list(self.discriminator.parameters()),
        #     lr=self.hparams["lr_d"],
        #     betas=(self.hparams['beta1'], 0.5))
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]

        self.gen_scheduler = CosineAnnealingLR(self.gen_opt, T_max=5000, eta_min=1e-8, last_epoch=-1, verbose=False)
        #self.disc_scheduler = CosineAnnealingLR(self.disc_opt, T_max=5000, eta_min=1e-7, last_epoch=-1, verbose=False)

        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        self.class_embeddings = self.clip_model.encode_text(self.prompt)
        self.text_features = self.class_embeddings / self.class_embeddings.norm(dim=-1, keepdim=True)

    def update(self, minibatches, unlabeled=None):
        self.update_count += 1
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        image_features = self.featurizer(all_x)
        #disc_input = image_features
        #disc_out = self.discriminator(disc_input)
        #disc_labels = torch.cat([
        #    torch.full((x.shape[0], ), i, dtype=torch.int64)
        #    for i, (x, y, z) in enumerate(minibatches)
        #]).to('cuda')
        #disc_loss = F.cross_entropy(disc_out, disc_labels)
        #disc_softmax = F.softmax(disc_out, dim=1)
        #input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
        #    [disc_input], create_graph=True)[0]
        #grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        #disc_loss += self.hparams['grad_penalty'] * grad_penalty

        #d_steps_per_g = self.hparams['d_steps_per_g_step']

        # if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):#偶数
        #     self.disc_opt.zero_grad()
        #     disc_loss.backward()
        #     self.disc_opt.step()
        #     self.disc_scheduler.step()

        #     return {'disc_loss': disc_loss.item()}
        #else:#奇数
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #text_features = self.class_embeddings / self.class_embeddings.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()#self.clip_model.logit_scale.exp() = 100

        classification_loss = F.cross_entropy(logits_per_image, all_y)
        #gen_loss = (classification_loss +
        #            (self.hparams['lambda'] * -disc_loss))
        gen_loss = classification_loss
        #self.disc_opt.zero_grad()
        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()
        #adjust learning rate as CLIPOOD

        self.gen_scheduler.step()
        
        return {"class_loss": classification_loss.item(), "lr": self.gen_opt.param_groups[0]['lr']}
     
    def predict(self, x):
        image_feature = self.featurizer(x)
        #all_acl = torch.zeros(x.size()[0], 3, 512).cuda()

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        
        logits_per_image = self.clip_model.logit_scale.exp() * image_feature @ self.text_features.t()
        return logits_per_image

#this is a baseline finetuning image encoder + dann
class MetricSoftmaxAlign(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MetricSoftmaxAlign, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dtype = self.clip_model.dtype
        self.featurizer = self.clip_model.visual
        self.discriminator = networks.DomainDiscriminator(self.clip_model.text_projection.shape[1], num_domains)
        self.register_buffer('update_count', torch.tensor([0]))
        #print(self.featurizer)
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = True

        params_to_update = [param for name, param in self.featurizer.named_parameters() if name.startswith('transformer.resblocks.11')]

        self.gen_opt = torch.optim.AdamW(#only update little parameters, knowledge vit structures
            params_to_update,
            lr=self.hparams["lr_g"],
            betas=(self.hparams['beta1'], 0.9))
        
        self.disc_opt = torch.optim.Adam(
            list(self.discriminator.parameters()),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))
        
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]

        self.gen_scheduler = CosineAnnealingLR(self.gen_opt, T_max=5000, eta_min=1e-8, last_epoch=-1, verbose=False)
        #self.disc_scheduler = CosineAnnealingLR(self.disc_opt, T_max=5000, eta_min=1e-5, last_epoch=-1, verbose=False)

        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        self.class_embeddings = self.clip_model.encode_text(self.prompt)
        self.text_features = self.class_embeddings / self.class_embeddings.norm(dim=-1, keepdim=True)

    def update(self, minibatches, unlabeled=None):
        self.update_count += 1
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        image_features = self.featurizer(all_x)
        disc_input = image_features
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
           torch.full((x.shape[0], ), i, dtype=torch.int64)
           for i, (x, y) in enumerate(minibatches)
        ]).to('cuda')
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
           [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):#偶数
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            #self.disc_scheduler.step()

            return {'disc_loss': disc_loss.item()}
        else:#奇数
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #text_features = self.class_embeddings / self.class_embeddings.norm(dim=-1, keepdim=True)
            logits_per_image = self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()#self.clip_model.logit_scale.exp() = 100

            classification_loss = F.cross_entropy(logits_per_image, all_y)
            gen_loss = (classification_loss +
                       (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()

            self.gen_scheduler.step()
            
            return {"class_loss": classification_loss.item(), "disc_loss": disc_loss.item(), "lr": self.gen_opt.param_groups[0]['lr']}
     
    def predict(self, x):
        image_feature = self.featurizer(x)
        #all_acl = torch.zeros(x.size()[0], 3, 512).cuda()

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        
        logits_per_image = self.clip_model.logit_scale.exp() * image_feature @ self.text_features.t()
        return logits_per_image
    
class MetricSoftmaxAlignPatch(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MetricSoftmaxAlignPatch, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dtype = self.clip_model.dtype
        self.featurizer = self.clip_model.visual
        self.discriminator = networks.DomainDiscriminator(in_feature=self.clip_model.text_projection.shape[1], hidden_size=1024, class_num=num_domains).to('cuda')
        self.grl = networks.WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.register_buffer('update_count', torch.tensor([0]))
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = True
            #print(name)

        self.ema = BetaEMA(self.featurizer)
        self.ema.register()

        self.gen_opt = torch.optim.AdamW(#only update little parameters, knowledge vit structures
            list(self.featurizer.parameters()),
            lr=self.hparams["lr_g"],
            betas=(self.hparams['beta1'], 0.9))
        
        self.disc_opt = torch.optim.Adam(
            list(self.discriminator.get_parameters()),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]

        self.gen_scheduler = CosineAnnealingLR(self.gen_opt, T_max=5000, eta_min=1e-8, last_epoch=-1, verbose=False)
        #self.disc_scheduler = CosineAnnealingLR(self.disc_opt, T_max=5000, eta_min=1e-5, last_epoch=-1, verbose=False)
        self.disc_scheduler = CosineAnnealingLR(self.gen_opt, T_max=5000, eta_min=1e-5, last_epoch=-1, verbose=False)
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        self.class_embeddings = self.clip_model.encode_text(self.prompt)
        self.text_features = self.class_embeddings / self.class_embeddings.norm(dim=-1, keepdim=True)

    def update(self, minibatches, unlabeled=None):
        self.update_count += 1
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_index = torch.cat([data[2].cuda() for data in minibatches])

        image_features = self.featurizer(all_x, all_index, mask=True)

        #discriminator
        disc_input = self.grl(image_features)
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
           torch.full((x.shape[0], ), i, dtype=torch.int64)
           for i, (x, _, _) in enumerate(minibatches)
        ]).to('cuda')
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()#self.clip_model.logit_scale.exp() = 100

        classification_loss = F.cross_entropy(logits_per_image, all_y)
        gen_loss = classification_loss + disc_loss

        self.disc_opt.zero_grad()
        self.gen_opt.zero_grad()

        gen_loss.backward()

        self.disc_opt.step()
        self.gen_opt.step()
        self.gen_scheduler.step()
        self.disc_scheduler.step()
        self.ema.update()
        return {"class_loss": classification_loss.item(), "disc_loss": disc_loss.item()}
     
    def predict(self, x, z):
        image_feature, _ = self.featurizer(x, z)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_feature @ self.text_features.t()
        return logits_per_image

class DecayEMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.step = 0

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.step += 1
        decay = self.decay * self.step
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class CMSAN(CLIP):
    #Conditional_Maked_Siamese_Alignment_Networks
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CMSAN, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dtype = self.clip_model.dtype
        self.featurizer = self.clip_model.visual
        self.discriminator = networks.DomainDiscriminator(in_feature=self.clip_model.text_projection.shape[1], hidden_size=1024, class_num=num_domains).to('cuda')
        self.grl = networks.WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.register_buffer('update_count', torch.tensor([0]))
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = True
            #print(name)

        self.ema = BetaEMA(self.featurizer)
        self.ema.register()

        self.gen_opt = torch.optim.AdamW(#only update little parameters, knowledge vit structures
            list(self.featurizer.parameters()),
            lr=self.hparams["lr_g"],
            betas=(self.hparams['beta1'], 0.9))
        
        self.disc_opt = torch.optim.Adam(
            list(self.discriminator.get_parameters()),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]

        self.gen_scheduler = CosineAnnealingLR(self.gen_opt, T_max=5000, eta_min=1e-8, last_epoch=-1, verbose=False)
        #self.disc_scheduler = CosineAnnealingLR(self.disc_opt, T_max=5000, eta_min=1e-5, last_epoch=-1, verbose=False)
        self.disc_scheduler = CosineAnnealingLR(self.gen_opt, T_max=5000, eta_min=1e-5, last_epoch=-1, verbose=False)
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        self.class_embeddings = self.clip_model.encode_text(self.prompt)
        self.text_features = self.class_embeddings / self.class_embeddings.norm(dim=-1, keepdim=True)
        self.klloss = nn.KLDivLoss()
        self.softmax = nn.Softmax(dim=-1)


    def update(self, minibatches, unlabeled=None):
        self.update_count += 1
        all_x_anchor = torch.cat([data[0].cuda().float() for data in minibatches])
        all_x_target = torch.cat([data[1].cuda().float() for data in minibatches])
        all_y = torch.cat([data[2].cuda().long() for data in minibatches])
        all_index = torch.cat([data[3].cuda() for data in minibatches])

        image_features_target = self.featurizer(all_x_target, all_index, mask=False)
        image_features_anchor = self.featurizer(all_x_anchor, all_index, mask=True)
        
        disc_input_anchor = self.grl(image_features_anchor)
        disc_out_anchor = self.discriminator(disc_input_anchor)
        disc_labels = torch.cat([
           torch.full((x.shape[0], ), i, dtype=torch.int64)
           for i, (x,_,_,_) in enumerate(minibatches)
        ]).to('cuda')
        dloss_anchor = F.cross_entropy(disc_out_anchor, disc_labels)

        #anchor model loss
        image_features_target = image_features_target / image_features_target.norm(dim=-1, keepdim=True)
        logits_per_image_target = self.clip_model.logit_scale.exp() * image_features_target @ self.text_features.t()#self.clip_model.logit_scale.exp() = 100

        image_features_anchor = image_features_anchor / image_features_anchor.norm(dim=-1, keepdim=True)
        logits_per_image_anchor = self.clip_model.logit_scale.exp() * image_features_anchor @ self.text_features.t()#self.clip_model.logit_scale.exp() = 100

        closs = F.cross_entropy(logits_per_image_target, all_y)
        softmax_per_image_anchor = self.softmax(logits_per_image_anchor)
        softmax_per_image_target = self.softmax(logits_per_image_target)
        bloss = (-softmax_per_image_target * torch.log(softmax_per_image_anchor)).sum(dim=1).mean()

        #mloss = torch.mean(torch.sum(torch.log(logits_per_image_anchor**(-logits_per_image_target)), dim=1))#cross entropy loss TODO
        #compute me-max regularizer
        #rloss = 0.
        #avg_probs = torch.mean(logits_per_image_anchor, dim=0)
        #rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))#todo
        gen_loss = closs + bloss + dloss_anchor#+ rloss

        self.disc_opt.zero_grad()
        self.gen_opt.zero_grad()

        gen_loss.backward()

        self.disc_opt.step()
        self.gen_opt.step()
        self.gen_scheduler.step()
        self.disc_scheduler.step()
        self.ema.update()
        return {"closs": closs.item(), "dloss_anchor": dloss_anchor.item(), "bloss": bloss.item()}
     
    def predict(self, x, z):
        image_feature = self.featurizer(x, z)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_feature @ self.text_features.t()
        return logits_per_image

    
class DecayEMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.step = 0

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.step += 1
        decay = self.decay * self.step
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class BetaEMA:
    def __init__(self, model, n_steps=5000, alpha=0.5, beta=0.5):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.n_steps = n_steps
        self.shadow = {}
        self.backup = {}
        self.step = 0
        self.weights_sum = 0

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.step += 1
        weight = beta.pdf((self.step+0.5) / (self.n_steps+1), self.alpha, self.beta)
        self.weights_sum += weight
        self.previous_weight  = (self.weights_sum - weight) / self.weights_sum
        self.current_weight = weight/self.weights_sum
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.previous_weight * self.shadow[name] +  self.current_weight * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
