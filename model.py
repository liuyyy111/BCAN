# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""
import copy

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
# from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.checkpoint import checkpoint


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(norm + eps) + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8, keep_dim=False):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim, keepdim=keep_dim)
    w1 = torch.norm(x1, 2, dim, keepdim=keep_dim)
    w2 = torch.norm(x2, 2, dim, keepdim=keep_dim)
    if keep_dim:
        return w12 / (w1 * w2).clamp(min=eps)
    else:
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze(-1)


def func_attention(query, context, g_sim, opt, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    opt: parameters
    """
    batch_size, queryL, sourceL = context.size(
        0), query.size(1), context.size(1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * opt.lambda_softmax)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if opt.correct_type == 'equal':
        re_attn = correct_equal(attn, query, context, sourceL, g_sim)
    elif opt.correct_type == 'prob':
        re_attn = correct_prob(attn, query, context, sourceL, g_sim)
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)

    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    if torch.isnan(weightedContext).any():
        print('ddd')
    return weightedContext, re_attn


def correct_equal(attn, query, context, sourceL, g_sim):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    re_attn = (g_sim - 0.3).unsqueeze(1).unsqueeze(2) * attn
    attn_sum = torch.sum(re_attn, dim=-1, keepdim=True)
    re_attn = re_attn / attn_sum

    cos1 = cosine_similarity(torch.bmm(re_attn, context), query, dim=-1, keep_dim=True)
    cos1 = torch.where(cos1 == 0, cos1.new_full(cos1.shape, 1e-8), cos1)
    re_attn1 = focal_equal(re_attn, query, context, sourceL)

    cos = cosine_similarity(torch.bmm(re_attn1, context), query, dim=-1, keep_dim=True)
    cos = torch.where(cos == 0, cos.new_full(cos.shape, 1e-8), cos)

    delta = cos - cos1
    delta = torch.where(delta == 0, delta.new_full(delta.shape, 1e-8), delta)

    re_attn2 = delta * re_attn1
    attn_sum = torch.sum(re_attn2, dim=-1, keepdim=True)
    re_attn2 = re_attn2 / attn_sum

    re_attn2 = focal_equal(re_attn2, query, context, sourceL)
    # re_attn2 = focal_equal(attn, query, context, sourceL)
    return re_attn2


def focal_equal(attn, query, context, sourceL):
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))

    # Step 3: reassign attention
    tmp_attn = fattn * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    return re_attn


def correct_prob(attn, query, context, sourceL, g_sim):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """
    d = g_sim - 0.3
    d = torch.where(d == 0, d.new_full(d.shape, 1e-8), d)
    re_attn = d.unsqueeze(1).unsqueeze(2) * attn
    attn_sum = torch.sum(re_attn, dim=-1, keepdim=True)
    re_attn = re_attn / attn_sum

    cos1 = cosine_similarity(torch.bmm(re_attn, context), query, dim=-1, keep_dim=True)
    cos1 = torch.where(cos1 == 0, cos1.new_full(cos1.shape, 1e-8), cos1)
    re_attn1 = focal_prob(re_attn, query, context, sourceL)

    cos = cosine_similarity(torch.bmm(re_attn1, context), query, dim=-1, keep_dim=True)
    cos = torch.where(cos == 0, cos.new_full(cos.shape, 1e-8), cos)

    delta = cos - cos1
    delta = torch.where(delta == 0, delta.new_full(delta.shape, 1e-8), delta)

    re_attn2 = delta * re_attn1
    attn_sum = torch.sum(re_attn2, dim=-1, keepdim=True)
    re_attn2 = re_attn2 / attn_sum

    re_attn2 = focal_prob(re_attn2, query, context, sourceL)
    if torch.isnan(re_attn2).any():
        print("ddd")
    return re_attn2


def focal_prob(attn, query, context, sourceL):
    batch_size, queryL, sourceL = context.size(
        0), query.size(1), context.size(1)

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size * queryL, sourceL, 1)
    xj = xj.view(batch_size * queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size * queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi).clamp(min=1e-8)
    term2 = xj * xj_confi
    funcF = torch.sum(term1 - term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))

    # Step 3: reassign attention
    tmp_attn = fattn * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    if torch.isnan(re_attn).any():
        print("ddd")
    return re_attn


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    img_enc = EncoderImagePrecomp(img_dim, embed_size, no_imgnorm)

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)
        features_mean = torch.mean(features, 1)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features, features_mean


def encoder_text(vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, no_txtnorm=False):
    txt_enc = EncoderText(vocab_size, word_dim, embed_size, num_layers, use_bi_gru, no_txtnorm)

    return txt_enc


class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2
        cap_emb_mean = torch.mean(cap_emb, 1)
        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
            cap_emb_mean = l2norm(cap_emb_mean, dim=1)
        return cap_emb, cap_len, cap_emb_mean


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, scores):
        # compute image-sentence score matrix

        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self, opt):
        super(SCAN, self).__init__()
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = encoder_text(opt.vocab_size, opt.word_dim,
                                    opt.embed_size, opt.num_layers,
                                    use_bi_gru=True,
                                    no_txtnorm=opt.no_txtnorm)
        # self.tag_enc = EncoderTag(opt)
        # self.fusion = Fusion(opt)
        # self.txt_gat = GAT(opt.embed_size, 1)
        self.opt = opt
        self.Eiters = 0

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Forward
        # tag = pos[:,:,4]
        # tag_prob = pos[:,:,5]
        # pos = pos[:,:,:4]
        img_emb, img_mean = self.img_enc(images)
        # tag_emb = self.tag_enc(tag)

        # img_emb = self.fusion(img_emb, tag_emb)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens, cap_mean = self.txt_enc(captions, lengths)
        return img_emb, img_mean, cap_emb, cap_lens, cap_mean

    def forward_sim(self, img_emb, img_mean, cap_emb, cap_len, cap_mean, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        scores = self.xattn_score(img_emb, img_mean, cap_emb, cap_len, cap_mean)

        return scores

    def forward(self, images, captions, lengths, ids=None, *args):
        # compute the embeddings
        lengths = lengths.cpu().numpy().tolist()
        img_emb, img_mean, cap_emb, cap_lens, cap_mean = self.forward_emb(images, captions, lengths)
        scores = self.forward_sim(img_emb, img_mean, cap_emb, cap_lens, cap_mean)
        return scores

    def xattn_score(self, images, img_mean, captions, cap_lens, cap_mean):
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        g_sims = cap_mean.mm(img_mean.t())
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            g_sim = g_sims[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # Focal attention in text-to-image direction
            # weiContext: (n_image, n_word, d)
            weiContext, _ = func_attention(cap_i_expand, images, g_sim, self.opt)
            t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
            t2i_sim = t2i_sim.mean(dim=1, keepdim=True)

            # Focal attention in image-to-text direction
            # weiContext: (n_image, n_word, d)
            weiContext, _ = func_attention(images, cap_i_expand, g_sim, self.opt)
            i2t_sim = cosine_similarity(images, weiContext, dim=2)
            i2t_sim = i2t_sim.mean(dim=1, keepdim=True)

            # Overall similarity for image and text

            sim = t2i_sim + i2t_sim

            similarities.append(sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        if self.training:
            similarities = similarities.transpose(0, 1)

        return similarities

    def visualize(self, images, img_mean, captions, cap_lens, cap_mean):
        g_sim = cap_mean.mm(img_mean.t())[0]
        n_word = cap_lens
        cap_i = captions[:n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(1, 1, 1)
        img_i = images.unsqueeze(0)
        img_i_expand = img_i.repeat(1, 1, 1)
        # Focal attention in text-to-image direction
        # weiContext: (n_image, n_word, d)
        weiContext, t2i_attn = func_attention(cap_i_expand, img_i_expand, g_sim, self.opt)


        # Focal attention in image-to-text direction
        # weiContext: (n_image, n_word, d)
        weiContext, i2t_attn = func_attention(img_i_expand, cap_i_expand, g_sim, self.opt)


        return t2i_attn.squeeze(0), i2t_attn.squeeze(0)