import multiprocessing

import gc
import os

import numpy as np
import time

from scipy.spatial.distance import cdist

g = 1000000000


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            rank = (-sim[i, :]).argsort()
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        if ref in rank:
            rank_index = np.where(rank == ref)[0][0]
        else:
            rank_index = 15000
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set


def eval_alignment_by_sim_mat(embed1, embed2, top_k, nums_threads, csls=0, accurate=False, output=True):
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, csls, nums_threads)
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            print(
                "\n hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                      t_mrr,
                                                                                      time.time() - t))
        else:
            print("\nhits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    return t_prec_set, hits1


def cal_rank_by_div_embed(frags, dic, sub_embed, embed, top_k, accurate, is_euclidean):
    mean = 0
    mrr = 0
    num = np.array([0 for k in top_k])
    mean1 = 0
    mrr1 = 0
    num1 = np.array([0 for k in top_k])
    if is_euclidean:
        sim_mat = sim_hander_ou(sub_embed, embed)
    else:
        sim_mat = np.matmul(sub_embed, embed.T)
    prec_set = set()
    aligned_e = None
    for i in range(len(frags)):
        ref = frags[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank

        if dic is not None and dic.get(ref, -1) > -1:
            e2 = dic.get(ref)
            sim_mat[i, e2] += 1.0
            rank = (-sim_mat[i, :]).argsort()
            aligned_e = rank[0]
            assert ref in rank
            rank_index = np.where(rank == ref)[0][0]
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1
            # del rank
        else:
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1

        prec_set.add((ref, aligned_e))
    del sim_mat
    return mean, mrr, num, mean1, mrr1, num1, prec_set


def eval_alignment_by_div_embed(embed1, embed2, top_k, nums_threads, selected_pairs=None, accurate=False,
                                is_euclidean=False):
    def pair2dic(pairs):
        if pairs is None or len(pairs) == 0:
            return None
        dic = dict()
        for i, j in pairs:
            if i not in dic.keys():
                dic[i] = j
        assert len(dic) == len(pairs)
        return dic

    t = time.time()
    dic = pair2dic(selected_pairs)
    ref_num = embed1.shape[0]
    t_num = np.array([0 for k in top_k])
    t_mean = 0
    t_mrr = 0
    t_num1 = np.array([0 for k in top_k])
    t_mean1 = 0
    t_mrr1 = 0
    t_prec_set = set()
    frags = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(frags))
    reses = list()
    for frag in frags:
        reses.append(pool.apply_async(cal_rank_by_div_embed, (frag, dic, embed1[frag, :],
                                                              embed2, top_k, accurate, is_euclidean)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, mean1, mrr1, num1, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += num
        t_mean1 += mean1
        t_mrr1 += mrr1
        t_num1 += num1
        t_prec_set |= prec_set

    assert len(t_prec_set) == ref_num

    acc = t_num / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if accurate:
        print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc,
                                                                                                   t_mean, t_mrr,
                                                                                                   time.time() - t))
    else:
        print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    if selected_pairs is not None and len(selected_pairs) > 0:
        acc1 = t_num1 / ref_num * 100
        for i in range(len(acc1)):
            acc1[i] = round(acc1[i], 2)
        t_mean1 /= ref_num
        t_mrr1 /= ref_num
        hits1 = acc1[0]
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc,
                                                                                                       t_mean, t_mrr,
                                                                                                       time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    gc.collect()
    return t_prec_set, hits1


def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values


def CSLS_sim(sim_mat1, k, nums_threads):
    # sorted_mat = -np.partition(-sim_mat1, k, axis=1) # -np.sort(-sim_mat1)
    # nearest_k = sorted_mat[:, 0:k]
    # sim_values = np.mean(nearest_k, axis=1)

    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values


def sim_handler(embed1, embed2, k, nums_threads):
    sim_mat = np.matmul(embed1, embed2.T)
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, nums_threads)
    csls2 = CSLS_sim(sim_mat.T, k, nums_threads)
    # for i in range(sim_mat.shape[0]):
    #     for j in range(sim_mat.shape[1]):
    #         sim_mat[i][j] = 2 * sim_mat[i][j] - csls1[i] - csls2[j]
    # return sim_mat
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat


def sim_hander_ou(embed1, embed2):
    return -cdist(embed1, embed2, metric='euclidean')


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def nearest_neighbor_sampling(emb, ids, K):
    t = time.time()
    neg = []
    distance = pairwise_distances(emb[ids], emb[ids])
    for idx in range(ids.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg.append(ids[indices[1: K + 1]])
    neg = torch.cat(tuple(neg), dim=0)
    return neg


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def generate_neighbours(entity_embeds1, entity_list1, entity_embeds2, entity_list2, neighbors_num, threads_num=4):
    ent_frags = div_list(np.array(entity_list1), threads_num)
    ent_frag_indexes = div_list(np.array(range(len(entity_list1))), threads_num)

    pool = multiprocessing.Pool(processes=len(ent_frags))
    results = list()
    for i in range(len(ent_frags)):
        results.append(pool.apply_async(find_neighbours,
                                        args=(ent_frags[i],
                                              entity_embeds1[ent_frag_indexes[i], :],
                                              np.array(entity_list2),
                                              entity_embeds2,
                                              neighbors_num)))

    pool.close()
    pool.join()

    dic = dict()
    for res in results:
        dic = merge_dic(dic, res.get())

    del results
    gc.collect()
    return dic


def find_neighbours(frags, sub_embed1, entity_list2, embed2, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed1, embed2.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list2[neighbors_index].tolist()
        dic[frags[i]] = neighbors
    return dic
