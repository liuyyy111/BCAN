# -----------------------------------------------------------
# "BCAN++: Cross-modal Retrieval With Bidirectional Correct Attention Network"
# Yang Liu, Hong Liu, Huaqiu Wang, Fanyang Meng, Mengyuan Liu*
#
# ---------------------------------------------------------------
"""BCAN model"""
import copy

import torch
import torch.nn as nn
import torch.nn.init
import torchtext
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


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
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()
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
    # GCU process
    d = g_sim - 0.3
    d = torch.where(d == 0, d.new_full(d.shape, 1e-8), d)
    re_attn = d.unsqueeze(1).unsqueeze(2) * attn
    attn_sum = torch.sum(re_attn, dim=-1, keepdim=True)
    re_attn = re_attn / attn_sum
    cos1 = cosine_similarity(torch.bmm(re_attn, context), query, dim=-1, keep_dim=True)
    cos1 = torch.where(cos1 == 0, cos1.new_full(cos1.shape, 1e-8), cos1)
    re_attn1 = focal_equal(re_attn, query, context, sourceL)

    # LCU process
    cos = cosine_similarity(torch.bmm(re_attn1, context), query, dim=-1, keep_dim=True)
    cos = torch.where(cos == 0, cos.new_full(cos.shape, 1e-8), cos)
    delta = cos - cos1
    delta = torch.where(delta == 0, delta.new_full(delta.shape, 1e-8), delta)
    re_attn2 = delta * re_attn1
    attn_sum = torch.sum(re_attn2, dim=-1, keepdim=True)
    re_attn2 = re_attn2 / attn_sum
    re_attn2 = focal_equal(re_attn2, query, context, sourceL)
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
    # GCU process
    d = g_sim - 0.3
    d = torch.where(d == 0, d.new_full(d.shape, 1e-8), d)
    re_attn = d.unsqueeze(1).unsqueeze(2) * attn
    attn_sum = torch.sum(re_attn, dim=-1, keepdim=True)
    re_attn = re_attn / attn_sum
    cos1 = cosine_similarity(torch.bmm(re_attn, context), query, dim=-1, keep_dim=True)
    cos1 = torch.where(cos1 == 0, cos1.new_full(cos1.shape, 1e-8), cos1)
    re_attn1 = focal_prob(re_attn, query, context, sourceL)

    # LCU process
    cos = cosine_similarity(torch.bmm(re_attn1, context), query, dim=-1, keep_dim=True)
    cos = torch.where(cos == 0, cos.new_full(cos.shape, 1e-8), cos)
    delta = cos - cos1
    delta = torch.where(delta == 0, delta.new_full(delta.shape, 1e-8), delta)
    re_attn2 = delta * re_attn1
    attn_sum = torch.sum(re_attn2, dim=-1, keepdim=True)
    re_attn2 = re_attn2 / attn_sum
    re_attn2 = focal_prob(re_attn2, query, context, sourceL)
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


def encoder_text(word2idx, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, no_txtnorm=False):
    txt_enc = EncoderText(word2idx, vocab_size, word_dim, embed_size, num_layers, use_bi_gru, no_txtnorm)

    return txt_enc


class EncoderText(nn.Module):

    def __init__(self, word2idx, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights(word2idx)

    def init_weights(self, word2idx):
        # self.embed.weight.data.uniform_(-0.1, 0.1)

        wemb = torchtext.vocab.GloVe(cache="D:/data/.vector_cache")

        # quick-and-dirty trick to improve word-hit rate
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace('-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

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


''' Visual self-attention module '''


class V_single_modal_atten(nn.Module):
    """
    Single Visual Modal Attention Network.
    """

    def __init__(self, image_dim, embed_dim, dropout_rate=0.4, img_region_num=36):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(V_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space

        self.fc2 = nn.Linear(image_dim, embed_dim)  # embed memory to common space
        self.fc2_2 = nn.Linear(embed_dim, embed_dim)

        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        self.fc4 = nn.Linear(image_dim, embed_dim)  # embed attentive feature to common space

        self.embedding_1 = nn.Sequential(self.fc1, nn.BatchNorm1d(img_region_num), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_2 = nn.Sequential(self.fc2, nn.BatchNorm1d(embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_2_2 = nn.Sequential(self.fc2_2, nn.BatchNorm1d(embed_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, v_t, m_v):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_v = self.embedding_1(v_t)

        if m_v.size()[-1] == v_t.size()[-1]:
            W_v_m = self.embedding_2(m_v)
        else:
            W_v_m = self.embedding_2_2(m_v)

        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_v.size()[1], 1)

        h_v = W_v.mul(W_v_m)

        a_v = self.embedding_3(h_v)
        a_v = a_v.squeeze(2)
        weights = self.softmax(a_v)

        v_att = ((weights.unsqueeze(2) * v_t)).sum(dim=1)

        # l2 norm
        v_att = l2norm(v_att, -1)

        return v_att, weights


class T_single_modal_atten(nn.Module):
    """
    Single Textual Modal Attention Network.
    """

    def __init__(self, embed_dim, dropout_rate=0.4):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(T_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed memory to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        self.embedding_1 = nn.Sequential(self.fc1, nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_2 = nn.Sequential(self.fc2, nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, u_t, m_u):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_u = self.embedding_1(u_t)

        W_u_m = self.embedding_2(m_u)
        W_u_m = W_u_m.unsqueeze(1).repeat(1, W_u.size()[1], 1)

        h_u = W_u.mul(W_u_m)

        a_u = self.embedding_3(h_u)
        a_u = a_u.squeeze(2)
        weights = self.softmax(a_u)

        u_att = ((weights.unsqueeze(2) * u_t)).sum(dim=1)

        # l2 norm
        u_att = l2norm(u_att, -1)

        return u_att, weights


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

    def __init__(self, word2idx, opt):
        super(SCAN, self).__init__()
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = encoder_text(word2idx, opt.vocab_size, opt.word_dim,
                                    opt.embed_size, opt.num_layers,
                                    use_bi_gru=True,
                                    no_txtnorm=opt.no_txtnorm)

        self.V_self_atten_enhance = V_single_modal_atten(opt.embed_size, opt.embed_size)
        self.T_self_atten_enhance = T_single_modal_atten(opt.embed_size)

        self.opt = opt
        self.Eiters = 0

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Forward
        img_emb, img_mean = self.img_enc(images)
        cap_emb, cap_lens, cap_mean = self.txt_enc(captions, lengths)

        img_mean, _ = self.V_self_atten_enhance(img_emb, img_mean)
        cap_mean, _ = self.T_self_atten_enhance(cap_emb, cap_mean)
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

            # t2i process
            # weiContext: (n_image, n_word, d)
            weiContext, _ = func_attention(cap_i_expand, images, g_sim, self.opt)
            t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
            t2i_sim = t2i_sim.mean(dim=1, keepdim=True)

            # i2t process
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
