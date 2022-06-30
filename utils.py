import torch
import os
import pandas as pd
import math
import numpy as np
import dendropy
from Bio import SeqIO
from torch.autograd import Function
import geoopt.manifolds.poincare.math as pmath
import geoopt
import time


def get_seq_length(args):
    backbone_seq_file = args.backbone_seq_file
    backbone_tree_file = args.backbone_tree_file
    seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    args.sequence_length = len(list(seq.values())[0])
    tree = dendropy.Tree.get(path=backbone_tree_file, schema='newick')
    num_nodes = len(tree.leaf_nodes())
    if args.embedding_size == -1:
        args.embedding_size = 2 ** math.floor(math.log2(10 * num_nodes ** (1 / 2)))


def distance_portion(nodes1, nodes2, mode):
    if len(nodes1.shape) == 1:
        nodes1 = nodes1.unsqueeze(0)
    if len(nodes2.shape) == 1:
        nodes2 = nodes2.unsqueeze(0)
    n1 = len(nodes1)
    n2 = len(nodes2)
    nodes1 = nodes1.view(n1, 1, -1)
    nodes2 = nodes2.view(1, n2, -1)
    if mode == 'ms':
        return torch.sum((nodes1 - nodes2) ** 2, dim=-1)
    elif mode == 'L2':
        # breakpoint()
        return (torch.sum((nodes1 - nodes2) ** 2, dim=-1) + 1e-6).sqrt()
    elif mode == 'L1':
        return torch.sum(abs(nodes1 - nodes2), dim=-1)
    elif mode == 'cosine':
        return 1 - torch.nn.functional.cosine_similarity(nodes1, nodes2, dim=-1)
    elif mode == 'tan':
        cosine = torch.nn.functional.cosine_similarity(nodes1, nodes2, dim=-1)
        return (1 - cosine ** 2) / (cosine + 1e-9)
    elif mode == 'hyperbolic':
        nodes1 = nodes1.squeeze(1)
        nodes2 = nodes2.squeeze(0)
        # breakpoint()
        # nodes1 = nodes1 / (nodes1.norm(-1, keepdim=True) / (1 - 1e-4))
        # nodes2 = nodes2 / (nodes2.norm(-1, keepdim=True) / (1 - 1e-4))
        # return hyperbolic_dist(nodes1, nodes2)
        # ms_x = (nodes1 ** 2).sum(-1)
        # ms_y = (nodes2 ** 2).sum(-1)
        # d = - ((1 + ms_x)*(1 + ms_y)).sqrt() + (nodes1 * nodes2).sum(-1)
        # d = torch.acosh(-d + 1e-15)
        # return d
        return hyp_dist(nodes1, nodes2)


def project_hyperbolic(x):
    N, d = x.shape[0], x.shape[1]

    hnorm_x = torch.norm(x, dim=1, p=2, keepdim=True)

    s1 = torch.cosh(hnorm_x)
    s2 = torch.div(torch.sinh(hnorm_x), hnorm_x)

    e = torch.zeros(N, d + 1).to(x.device)
    e[:, 0] = 1

    zero_col = torch.zeros(N, 1).to(x.device)
    z = torch.cat((zero_col, x), 1)
    z = torch.mul(s1, e) + torch.mul(s2, z)  # hyperbolic embeddings
    return z


def hyp_dist(embeddings1, embeddings2=None):
    if embeddings2 is None:
        x1 = x2 = project_hyperbolic(embeddings1)
    else:
        x1, x2 = project_hyperbolic(embeddings1), project_hyperbolic(embeddings2)
    d = x1.shape[1] - 1
    H = torch.eye(d + 1, d + 1).to(x1.device)
    H[0, 0] = -1
    N1, N2 = x1.shape[0], x2.shape[0]
    G = torch.matmul(torch.matmul(x1, H), torch.transpose(x2, 0, 1))
    G[G >= -1] = -1
    return torch.acosh(-G)


def hyp_dist_tmp(embeddings, formula, return_coord=True):
    # N = embeddings.weight.size()[0]
    # x = embeddings( torch.arange(0, N) )
    # D = torch.cdist(x, x, p = 2)
    # D = torch.pow(D, 2)
    # print(embeddings.weight)

    # norm_x = 1-torch.sum(x**2, dim=-1, keepdim=True)

    # D = torch.div(D, norm_x)
    # D = torch.transpose(D, 0, 1)
    # D = torch.div(D, norm_x)
    # D = D[torch.triu(torch.ones(N,N),diagonal = 1) == 1]
    # D = torch.acosh(1+2*D)
    # breakpoint()
    N = embeddings.size()[0]
    d = embeddings.size()[1]
    x = embeddings
    hnorm_x = torch.norm(x, dim=1, p=2, keepdim=True)

    s1 = torch.cosh(hnorm_x)
    s2 = torch.div(torch.sinh(hnorm_x), hnorm_x)

    e = torch.zeros(N, d + 1).to(embeddings.device)
    e[:, 0] = 1
    H = torch.eye(d + 1, d + 1).to(embeddings.device)
    H[0, 0] = -1

    zero_col = torch.zeros(N, 1).to(embeddings.device)
    z = torch.cat((zero_col, x), 1)
    z = torch.mul(s1, e) + torch.mul(s2, z)  # hyperbolic embeddings

    G = torch.matmul(torch.matmul(z, H), torch.transpose(z, 0, 1))
    # G = G[torch.triu(torch.ones(N, N), diagonal=1) == 1]
    # G = G - 1e-5
    # tmp = G[torch.triu(torch.ones(N, N), diagonal=1)==1]
    # print((tmp >= -1).sum() / (N * N - N))
    if formula == 1:
        G[G >= -1] = -1  # log this
        return torch.acosh(-G), (G[np.triu(np.ones((N, N)), 1) == 1] >= -1).sum() / (N-1)**2
    else:
        return -G - 1, (G[np.triu(np.ones((N, N)), 1) == 1] >= -1).sum() / (N-1)**2


def loss(D, embeddings, lr, scale, formula):  # loss(self, triple_ids, similarities): # commented by Puoya
    """Computes the HypHC loss.
    Args:
        triple_ids: B x 3 tensor with triple ids
        similarities: B x 3 tensor with pairwise similarities for triples
                      [s12, s13, s23]
    """
    # commented by Puoya
    # e1 = self.embeddings(triple_ids[:, 0])
    # e2 = self.embeddings(triple_ids[:, 1])
    # e3 = self.embeddings(triple_ids[:, 2])
    # e1 = self.normalize_embeddings(e1)
    # e2 = self.normalize_embeddings(e2)
    # e3 = self.normalize_embeddings(e3)
    # d_12 = hyp_lca(e1, e2, return_coord=False)
    # d_13 = hyp_lca(e1, e3, return_coord=False)
    # d_23 = hyp_lca(e2, e3, return_coord=False)
    # lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
    # weights = torch.softmax(lca_norm / self.temperature, dim=-1)
    # w_ord = torch.sum(similarities * weights, dim=-1, keepdim=True)
    # total = torch.sum(similarities, dim=-1, keepdim=True) - w_ord
    # return torch.mean(total)
    N = D.shape[0]
    distances, num_one = hyp_dist_tmp(embeddings, formula)
    distances = distances[np.triu(np.ones((N, N)), 1) == 1]
    # distances = euc_dist(self.embeddings)
    D_vec = D[np.triu(np.ones((N, N)), 1) == 1]
    D_vec = torch.tensor(D_vec)
    if formula == 2:
        D_vec = torch.cosh(D_vec * 1 / scale) - 1
    r = 1
    # r = 0.39073709165622195/5.873781526199998
    # print(flag)
    # if flag:
    if formula == 1:
        cost = torch.sqrt(torch.mean(torch.pow(torch.div(scale * r * distances, D_vec + 1e-12) - 1, 2)))
#        cost = torch.mean(torch.pow(torch.div(scale * r * distances, D_vec + 1e-12) - 1, 2))



#divd_sqrt = (scale * r * distances/D_vec) ** (1/2)
        #cost = torch.mean(D_vec**(-3/2) * ((divd_sqrt - 1) ** 2)).sqrt()
#        weight = 1 / (D_vec ** 2)
#        dev = ((scale * r * distances)**0.5 - D_vec**0.5)**2
#        cost = torch.mean(weight * dev)
    else:
        cost = torch.sqrt(torch.mean(torch.pow(torch.div(distances, D_vec) - 1, 2)))
    a = torch.mean(torch.div(r * distances, D_vec+1e-12) + 10 ** (-10))
    b = torch.mean(torch.pow(torch.div(r * distances, D_vec+1e-12), 2) + 10 ** (-10))
    scale = (1 - lr) * scale + lr * (a.item() / b.item())
    # print(flag)
    # print(scale)
    # cost =  torch.sqrt(torch.mean( torch.pow( torch.div(distances, D_vec)-1, 2) ) )
    # cost = torch.sqrt( torch.mean ( torch.pow(scale*distances- D_vec,2) ) )
    # grad = torch.mean(torch.mul( torch.div(distances,D_vec), torch.div(scale*distances,D_vec)-1))

    return cost, distances, scale, num_one


def jc_dist(seqs1_c, seqs2, names1, names2):
    seqs1_tmp = np.zeros(seqs1_c.shape)
    seqs2_tmp = np.zeros(seqs2.shape)
    seqs1_tmp[seqs1_c == 'A'] = 0
    seqs1_tmp[seqs1_c == 'C'] = 1
    seqs1_tmp[seqs1_c == 'G'] = 2
    seqs1_tmp[seqs1_c == 'T'] = 3
    seqs1_tmp[seqs1_c == '-'] = 4
    seqs2_tmp[seqs2 == 'A'] = 0
    seqs2_tmp[seqs2 == 'C'] = 1
    seqs2_tmp[seqs2 == 'G'] = 2
    seqs2_tmp[seqs2 == 'T'] = 3
    seqs2_tmp[seqs2 == '-'] = 4
    seqs1_c = seqs1_tmp
    seqs2 = seqs2_tmp

    n2, l = seqs2.shape[0], seqs2.shape[-1]
    seqs2 = seqs2.reshape(1, n2, -1)
    hamming_dist = []
    for i in range(math.ceil(len(seqs1_c) / 1000)):
        seqs1 = seqs1_c[i * 1000: (i + 1) * 1000]
        n1 = seqs1.shape[0]
        seqs1 = seqs1.reshape(n1, 1, -1)
        # breakpoint()
        non_zero = np.logical_and(seqs1 != 4, seqs2 != 4)
        hd = (seqs1 != seqs2) * non_zero
        hd = np.count_nonzero(hd, axis=-1)
        hamming_dist.append(hd / np.count_nonzero(non_zero, axis=-1))
    hamming_dist = np.concatenate(hamming_dist, axis=0)
    jc = - 3 / 4 * np.log(1 - 4 / 3 * hamming_dist)
    jc_df = pd.DataFrame(dict(zip(names2, jc)))
    jc_df.index = names1
    return jc_df


def distance(nodes1, nodes2, mode):
    # node1: query
    # node2: backbone
    dist = []
    # np.save('query_emb.npy', np.array(nodes1.cpu()))
    # np.save('backbone_emb.npy', np.array(nodes2.cpu()))
    for i in range(math.ceil(len(nodes1) / 1000.0)):
        dist.append(distance_portion(nodes1[i * 1000: (i + 1) * 1000], nodes2, mode))
    return torch.cat(dist, dim=0)


def mse_loss(model_dist, true_dist, weighted_method, hyperbolic=False):
    assert model_dist.shape == true_dist.shape
    if weighted_method == 'ols':
        return ((model_dist - true_dist) ** 2).mean()
    elif weighted_method == 'fm':
        weight = 1 / (true_dist + 1e-5) ** 2
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'be':
        weight = 1 / (true_dist + 1e-5)
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_fm':
        weight = 1 / (true_dist + 1e-5) ** 2
        true_dist = torch.sqrt(true_dist)
        # if hyperbolic:
        #     # breakpoint()
        #     true_dist = true_dist ** 2
        #     weight = 1 / (torch.acosh(true_dist) + 1e-4) ** 2
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_be':
        true_dist = torch.sqrt(true_dist)
        weight = 1 / (true_dist + 1e-5)
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_ols':
        true_dist = torch.sqrt(true_dist)
        weight = 1
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_sqrt':
        true_dist = torch.sqrt(true_dist)
        weight = 1 / (torch.sqrt(true_dist) + 1e-5)
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_four':
        true_dist = torch.sqrt(true_dist)
        weight = 1 / (true_dist + 1e-5) ** 4
        return ((model_dist - true_dist) ** 2 * weight).mean()


def process_seq(self_seq, args, isbackbone, need_mask=False):
    L = len(list(self_seq.values())[0])
    names = list(self_seq.keys())
    seqs = np.zeros([4, len(self_seq), L])
    if need_mask:
        mask = np.ones([1, len(self_seq), L])
    raw_seqs = [np.array(self_seq[k].seq).reshape(1, -1) for k in self_seq]
    raw_seqs = np.concatenate(raw_seqs, axis=0)
    seqs[0][raw_seqs == 'A'] = 1
    seqs[1][raw_seqs == 'C'] = 1
    seqs[2][raw_seqs == 'G'] = 1
    seqs[3][raw_seqs == 'T'] = 1

    # R
    idx = raw_seqs == 'R'
    seqs[0][idx] = 1 / 2
    seqs[2][idx] = 1 / 2

    # Y
    idx = raw_seqs == 'Y'
    seqs[1][idx] = 1 / 2
    seqs[3][idx] = 1 / 2

    # S
    idx = raw_seqs == 'S'
    seqs[1][idx] = 1 / 2
    seqs[2][idx] = 1 / 2

    # W
    idx = raw_seqs == 'W'
    seqs[0][idx] = 1 / 2
    seqs[3][idx] = 1 / 2

    # K
    idx = raw_seqs == 'K'
    seqs[2][idx] = 1 / 2
    seqs[3][idx] = 1 / 2

    # M
    idx = raw_seqs == 'M'
    seqs[0][idx] = 1 / 2
    seqs[1][idx] = 1 / 2

    # B
    idx = raw_seqs == 'B'
    seqs[1][idx] = 1 / 3
    seqs[2][idx] = 1 / 3
    seqs[3][idx] = 1 / 3

    # D
    idx = raw_seqs == 'D'
    seqs[0][idx] = 1 / 3
    seqs[2][idx] = 1 / 3
    seqs[3][idx] = 1 / 3

    # H
    idx = raw_seqs == 'H'
    seqs[0][idx] = 1 / 3
    seqs[1][idx] = 1 / 3
    seqs[3][idx] = 1 / 3

    # V
    idx = raw_seqs == 'V'
    seqs[0][idx] = 1 / 3
    seqs[1][idx] = 1 / 3
    seqs[2][idx] = 1 / 3

    seqs[:, raw_seqs == '-'] = args.gap_encode
    seqs[:, raw_seqs == 'N'] = args.gap_encode

    if need_mask:
        mask[:, raw_seqs == '-'] = 0
        mask[:, raw_seqs == 'N'] = 0
        mask = np.transpose(mask, axes=(1, 0, 2))

    seqs = np.transpose(seqs, axes=(1, 0, 2))
    if args.replicate_seq and (isbackbone or args.query_dist):
        df = pd.DataFrame(columns=['seqs'])
        df['seqs'] = df['seqs'].astype(object)
        df['seqs'] = list(seqs)
        df['names'] = names
        df = df.set_index('names')
        df = df.groupby(by=lambda x: x.split('_')[0]).sum(numeric_only=False)
        seqs = np.concatenate([i.reshape(1, 4, -1) for i in df['seqs'].values])
        seqs /= (seqs.sum(1, keepdims=True) + 1e-8)
        comb_names = list(df.index)
        if need_mask:
            mask_df = pd.DataFrame(columns=['masks'])
            mask_df['masks'] = mask_df['masks'].astype(object)
            mask_df['masks'] = list(mask)
            mask_df['names'] = names
            mask_df = mask_df.set_index('names')
            mask_df = mask_df.groupby(by=lambda x: x.split('_')[0]).sum(numeric_only=False)
            mask_df = mask_df.loc[comb_names]
            mask = np.concatenate([i.reshape(1, 1, -1) for i in mask_df['masks'].values])
        names = comb_names

    if need_mask:
        return names, torch.from_numpy(seqs), torch.from_numpy(mask).bool()
#    return names, torch.from_numpy(seqs)
    return names, torch.from_numpy(seqs)

def get_embeddings(seqs, model, mask=None):
    encodings = []
    for i in range(math.ceil(len(seqs) / 2000.0)):
        if not (mask is None):
            encodings_tmp = model(seqs[i * 2000: (i + 1) * 2000].double(), mask=mask[i * 2000: (i + 1) * 2000]).detach()
        else:
            encodings_tmp = model(seqs[i * 2000: (i + 1) * 2000].double()).detach()
        encodings.append(encodings_tmp)
    encodings = torch.cat(encodings, dim=0)
    return encodings

def save_depp_dist(model, args, recon_model=None):
    t1 = time.time()
    if model is not None:
        model.eval()
        args.replicate_seq = model.hparams.replicate_seq
        args.distance_ratio = model.hparams.distance_ratio
        args.gap_encode = model.hparams.gap_encode
        args.jc_correct = model.hparams.jc_correct
        args.distance_mode = model.hparams.distance_mode
    elif recon_model is not None:
        args.replicate_seq = recon_model.hparams.replicate_seq
        args.distance_ratio = recon_model.hparams.distance_ratio
        args.gap_encode = recon_model.hparams.gap_encode
        args.jc_correct = recon_model.hparams.jc_correct

    print('processing data...')
    backbone_seq_file = args.backbone_seq_file
    query_seq_file = args.query_seq_file
    dis_file_root = os.path.join(args.outdir)
    # args.distance_ratio = float(1.0 / float(args.embedding_size) / 10 * float(args.distance_alpha))
    #args.replicate_seq = model.hparams.replicate_seq
    print('jc_correct', args.jc_correct)
    if args.jc_correct:
        args.jc_ratio = model.hparams.jc_ratio
    if not os.path.exists(dis_file_root):
        os.makedirs(dis_file_root, exist_ok=True)

    backbone_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    query_seq = SeqIO.to_dict(SeqIO.parse(query_seq_file, "fasta"))

    if args.jc_correct:
        backbone_seq_names, backbone_seq_names_raw, backbone_seq_tensor, backbone_raw_array = \
            process_seq(backbone_seq, args, isbackbone=True)
        query_seq_names, query_seq_names_raw, query_seq_tensor, query_raw_array = \
            process_seq(query_seq, args, isbackbone=False)
    else:
        # breakpoint()
        if not (recon_model is None):
            if (args.recon_backbone_emb is None) or (args.backbone_id is None) or (args.backbone_gap is None):
                backbone_seq_names, backbone_seq_tensor, backbone_mask = process_seq(backbone_seq, args, isbackbone=True, need_mask=True)
                torch.save(backbone_mask, f'{dis_file_root}/backbone_gap.pt')
            else:
                backbone_seq_names = torch.load(args.backbone_id)
                backbone_mask = torch.load(args.backbone_gap)
            query_seq_names, query_seq_tensor, query_mask = process_seq(query_seq, args, isbackbone=False, need_mask=True)
        else:
            if (args.backbone_emb is None) or (args.backbone_id is None):
                backbone_seq_names, backbone_seq_tensor = process_seq(backbone_seq, args, isbackbone=True)
            else:
                backbone_seq_names = torch.load(args.backbone_id)
            query_seq_names, query_seq_tensor = process_seq(query_seq, args, isbackbone=False)
    if model is not None:
        for param in model.parameters():
            param.requires_grad = False
    if recon_model is not None:
        for param in recon_model.parameters():
            param.requires_grad = False
    print('finish data processing!')
    print(f'{len(backbone_seq_names)} backbone sequences')
    print(f'{len(query_seq_names)} query sequence(s)')
    print(f'calculating embeddings...')
    if not (model is None):
        if (args.backbone_emb is None) or (args.backbone_id is None):
            backbone_encodings = get_embeddings(backbone_seq_tensor, model)
        else:
            backbone_encodings = torch.load(args.backbone_emb)
        query_encodings = get_embeddings(query_seq_tensor, model)
    #torch.save(query_encodings, f'{dis_file_root}/query_embeddings.pt')
    #torch.save(query_seq_names, f'{dis_file_root}/query_names.pt')
    #torch.save(backbone_encodings, f'{dis_file_root}/backbone_embeddings.pt')
    #torch.save(backbone_seq_names, f'{dis_file_root}/backbone_names.pt')

    if not (recon_model is None):
        if (args.recon_backbone_emb is None) or (args.backbone_id is None) or (args.backbone_gap is None):
            recon_backbone_encodings = get_embeddings(backbone_seq_tensor, recon_model, backbone_mask)
        else:
            recon_backbone_encodings = torch.load(args.recon_backbone_emb)
        recon_query_encodings = get_embeddings(query_seq_tensor, recon_model, query_mask)
        torch.save(recon_backbone_encodings, f'{dis_file_root}/recon_backbone_embeddings.pt')

    print(f'finish embedding calculation!')
    print(f'calculating distance matrix...')
    t2 = time.time()
    #print('calculate embeddings', t2 - t1)

    # query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio
    if model:
        query_dist = distance(query_encodings, backbone_encodings, args.distance_mode)
        if not args.distance_mode == 'hyperbolic':
            query_dist = query_dist * args.distance_ratio
        else:
            query_dist = query_dist * model.c
        print(model.c, args.distance_mode)
        # if 'square_root' in args.weighted_method:
        #     query_dist = query_dist ** 2

    if recon_model:
        gap_portion = 1 - query_mask.int().sum(-1) / query_mask.shape[-1]
        recon_query_dist = distance(recon_query_encodings, recon_backbone_encodings, args.distance_mode)
        if not args.distance_mode == 'hyperbolic':
            recon_query_dist = recon_query_dist * args.distance_ratio
        else:
            recon_query_dist = recon_query_dist * model.c
        # if 'square_root' in args.weighted_method:
        #     recon_query_dist = recon_query_dist ** 2
        if model:
            query_dist = query_dist * (1 - gap_portion) + recon_query_dist * gap_portion
        else:
            query_dist = recon_query_dist

    t3 = time.time()
    #print('calculate distance', t3 - t2)
    query_dist = np.array(query_dist)
    query_dist[query_dist < 1e-3] = 0
    data_origin = dict(zip(query_seq_names, list(query_dist.astype(str))))
    data_origin = "\t" + "\t".join(backbone_seq_names) + "\n" + \
                  "\n".join([str(k) + "\t"+ "\t".join(data_origin[k]) for k in data_origin]) + "\n"
    t4 = time.time()
    #print('convert string', t4 - t3)
    with open(os.path.join(dis_file_root, f'depp.csv'), 'w') as f:
        f.write(data_origin)
    t5 = time.time()
    #print('save string', t5 - t4)
    # data_origin = pd.DataFrame.from_dict(data_origin, orient='index', columns=backbone_seq_names)

    if args.query_dist:
        idx = data_origin.index
        data_origin = data_origin[idx]
    # data_origin.to_csv(os.path.join(dis_file_root, f'depp.csv'), sep='\t')
    # if not os.path.isdir(f'{args.outdir}/depp_tmp'):
    #     os.makedirs(f'{args.outdir}/depp_tmp')
    # with open(f'{args.outdir}/depp_tmp/seq_name.txt', 'w') as f:
    #     f.write("\n".join(query_seq_names) + '\n')
    print('original distanace matrix saved!')
    print("take {:.2f} seconds".format(t5-t1))


class Distance(Function):
    # @staticmethod
    # def grad(x, v, sqnormx, sqnormv, sqdist, eps):
    #     alpha = (1 - sqnormx)
    #     beta = (1 - sqnormv)
    #     z = 1 + 2 * sqdist / (alpha * beta)
    #     a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2))\
    #         .unsqueeze(-1).expand_as(x)
    #     a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    #     z = torch.sqrt(torch.pow(z, 2) - 1)
    #     z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
    #     d = 4 * a / z.expand_as(x)
    #     d_p = ((1 - sqnormx) ** 2 / 4).unsqueeze(-1) * d
    #     return d_p

    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        # print('z.shape', z.shape)
        a = (sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2)
        # print('a.shape', a.shape)
        # print('v.shape', v.shape, alpha.shape)
        # breakpoint()
        a = a.unsqueeze(-1) * x - v / alpha.unsqueeze(-1)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps)
        d = 4 * a / z.unsqueeze(-1)
        # print('d.shaped', d.shape)
        d_p = ((1 - sqnormx) ** 2 / 4).unsqueeze(-1) * d
        return d_p

    @staticmethod
    def forward(ctx, u, v, eps):
        # breakpoint()
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None


def mobius_linear(
        input,
        weight,
        bias=None,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        c=1.0,
):
    if hyperbolic_input:
        output = pmath.mobius_matvec(weight, input, c=c)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, c=c)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, c=c)
        output = pmath.mobius_add(output, bias, c=c)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, c=c)
    output = pmath.project(output, c=c)
    return output


class MobiusLinear(torch.nn.Linear):
    def __init__(
            self,
            *args,
            hyperbolic_input=True,
            hyperbolic_bias=True,
            nonlin=None,
            c=1.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ball = manifold = geoopt.PoincareBall(c=c)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=c)
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() / 4, c=c))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            c=self.ball.c,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info
