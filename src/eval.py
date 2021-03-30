from text_utils import *
from utils import *
import torch.nn.functional as F
from tqdm import trange
from math import floor, ceil


def get_hit_k(match_id: Tensor, link: Tensor, src=0, k_list=(1, 3, 5, 10), ignore=None, start=""):
    trg = 1 - src
    total = link.size(1)
    if ignore is not None:
        match_id[ignore] = torch.ones_like(match_id[ignore], device=match_id.device, dtype=torch.long) * -1
        ignore_sum = ignore.clone()
        ignore_sum[link[src]] = False
        print(start + "total ignore:", ignore.sum(), ", valid ignore", ignore_sum.sum())
        total = total - ignore_sum.sum()
    print(start + "total is ", total)
    match_id = match_id[link[src]]
    link: Tensor = link[trg]
    hitk_result = {}
    for k in k_list:
        if k > match_id.size(1):
            break
        match_k = match_id[:, :k]
        link_k = link.view(-1, 1).expand(-1, k)
        hit_k = (match_k == link_k).sum().item()
        hitk_result['hits@{}'.format(k)] = hit_k / total
        print("{2}hits@{0} is {1}".format(k, hit_k / total, start))
    return hitk_result


def bi_csls_matrix(sim_matrix0, sim_matrix1, k=10, return2=True) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    dist0, _ = torch.topk(sim_matrix0, k)
    dist1, _ = torch.topk(sim_matrix1, k)
    if return2:
        return csls_impl(sim_matrix0, dist0, dist1), csls_impl(sim_matrix1, dist1, dist0)
    return csls_impl(sim_matrix0, dist0, dist1)


def csls_impl(sim_matrix, dist0, dist1) -> Tensor:
    dist0 = dist0.mean(dim=1).view(-1, 1).expand_as(sim_matrix)
    dist1 = dist1.mean(dim=1).view(1, -1).expand_as(sim_matrix)
    sim_matrix = sim_matrix * 2 - dist0 - dist1
    return sim_matrix


def get_csls_sim(sim_matrix: Tensor, dist0: Tensor, dist1: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    k = dist1.size(1)
    sim_matrix = csls_impl(sim_matrix, dist0, dist1)
    r, rid = torch.topk(sim_matrix, k, dim=1)
    return r, rid, sim_matrix


def get_cos_sim(src_emb: Tensor, trg_emb: Tensor, k_ent=10) -> Tuple[Tensor, Tensor, Tensor]:
    sim = cosine_sim(src_emb, trg_emb)
    return get_topk_sim(sim, k_ent)


def get_topk_sim(sim: Tensor, k_ent=10) -> Tuple[Tensor, Tensor, Tensor]:
    return torch.topk(sim, k=k_ent) + (sim,)


@torch.no_grad()
def get_mrr(link: Tensor, sim_matrix: Tensor, which=0, batch_size=4096, start="\t"):
    all = link.size(1)
    curr = 0
    mrr = torch.tensor(0.).to(link.device)
    while curr < all:
        begin, end = curr, min(curr + batch_size, all)
        curr = end
        src, trg = link[which, begin:end], link[1 - which, begin:end]
        sim = sim_matrix[src]
        sim = torch.argsort(sim, dim=1, descending=True)
        sim = torch.argmax((sim == trg.view(-1, 1)).to(torch.long), dim=1, keepdim=False)
        mrr += (1.0 / (sim + 1).to(float)).sum()
    mrr /= all
    print("{0}MRR is {1}".format(start, mrr))
    return mrr.item()


@torch.no_grad()
def evaluate_sim_matrix(link, sim_x2y, sim_y2x=None, ignore=(None, None), start="\t", no_csls=True, mrr=False):
    start_outer = start
    start = start + start
    device = link.device
    sim_x2y = sim_x2y.to(device)
    if sim_x2y.is_sparse:
        sim_x2y = sim_x2y.to_dense()
    MRR = 'MRR'
    match_sim0, match_id0, sim_matrix0 = get_topk_sim(sim_x2y)
    result = get_hit_k(match_id0, link, 0, ignore=ignore[0], start=start)
    if mrr:
        result[MRR] = get_mrr(link, sim_matrix0, 0, start=start)

    if sim_y2x is not None:
        sim_y2x = sim_y2x.to(device)
        if sim_y2x.is_sparse:
            sim_y2x = sim_y2x.to_dense()
        match_sim1, match_id1, sim_matrix1 = get_topk_sim(sim_y2x)

        result_rev = get_hit_k(match_id1, link, 1, ignore=ignore[1], start=start)
        result_rev[MRR] = get_mrr(link, sim_matrix1, 1, start=start)
        if no_csls:
            return result, result_rev
        print(start_outer + '------csls')
        match_sim0, match_id0, sim_matrix0 = get_csls_sim(sim_matrix0, match_sim0, match_sim1)
        match_sim1, match_id1, sim_matrix1 = get_csls_sim(sim_matrix1, match_sim1, match_sim0)

        result_csls_0 = get_hit_k(match_id0, link, 0, ignore=ignore[0], start=start)
        result_csls_0[MRR] = get_mrr(link, sim_matrix0, 0, start=start)

        result_csls_1 = get_hit_k(match_id1, link, 1, ignore=ignore[1], start=start)
        result_csls_1[MRR] = get_mrr(link, sim_matrix1, 1, start=start)
        return result, result_rev, result_csls_0, result_csls_1
    else:
        return result


def evaluate_embeds(src_emb, trg_emb, link, mapping=None, no_csls=True, rev=False, mrr=False):
    print('------------')
    if mapping is None:
        src_R, trg_R = src_emb, trg_emb
    else:
        src_R = F.linear(src_emb, mapping[0])
        trg_R = F.linear(trg_emb, mapping[1])

    return evaluate_sim_matrix(link, cosine_sim(src_R, trg_emb),
                               cosine_sim(trg_R, src_emb) if rev else None,
                               no_csls=no_csls, mrr=mrr)


@torch.no_grad()
def sparse_acc(sp_sim: Tensor, link: Tensor, device='cpu'):
    # ind, val, sz = split_sp(sp_sim)

    print('Total link is', link.size(1))
    # add_ind = torch.arange(min(sp_sim.size()))
    # add_val = torch.tensor(0.0001, dtype=torch.float32, device=link.device).expand_as(add_ind)
    # add_ind = torch.stack([add_ind, add_ind])
    # add = ind2sparse(add_ind, sp_sim.size(), values=add_val)
    # print(add.size(), add._values().size())
    # sp_sim = sp_sim + add
    sp_sim, link = apply(lambda x: x.to(device), sp_sim, link)
    print(sp_sim.size(), sp_sim._indices().size(), sp_sim._values().size())
    pred = matrix_argmax(sp_sim).view(-1)
    acc: Tensor = pred[link[0]] == link[1]
    print('calculate acc complete')
    return (acc.sum() / acc.numel()).item()


# @torch.no_grad()
# def get_mrr(link: Tensor, sim_matrix: Tensor, which=0, batch_size=4096, start="\t"):
#     all = link.size(1)
#     curr = 0
#     mrr = torch.tensor(0.).to(link.device)
#     while curr < all:
#         begin, end = curr, min(curr + batch_size, all)
#         curr = end
#         src, trg = link[which, begin:end], link[1 - which, begin:end]
#         sim = sim_matrix[src]
#         sim = torch.argsort(sim, dim=1, descending=True)
#         sim = torch.argmax((sim == trg.view(-1, 1)).to(torch.long), dim=1, keepdim=False)
#         mrr += (1.0 / (sim + 1).to(float)).sum()
#     mrr /= all
#     print("{0}MRR is {1}".format(start, mrr))
#     return mrr.item()

def truncated_mrr(topks: Tensor, link: Tensor, fail=None):
    src, trg = link
    if fail is None:
        fail = int(topks.size(0) / 2)
    rank = (topks[src] == trg.view(-1, 1)).to(torch.long) * 2
    rank_dummy = torch.ones([rank.size(0), 1]).to(topks.device)
    rank = torch.cat([rank, rank_dummy], dim=1)
    dummy_pos = rank.size(1) - 1
    rank = torch.argmax(rank, dim=-1, keepdim=False)
    rank[rank == dummy_pos] = fail
    mrr = (1.0 / (rank + 1).to(torch.float)).mean().item()
    print("MRR is {}".format(mrr))
    return mrr


# @torch.no_grad()
# def sparse_matrix_batcher(sp_mat: Tensor, batch_size):
#     all_len = sp_mat.size(0)
#     trg_len = sp_mat.size(1)
#     sp_mat = sp_mat.coalesce()
#     ind, val = sp_mat._indices(), sp_mat._values()
#     total_elms = val.numel()
#     guess_bs = int(ceil(total_elms / all_len)) * batch_size
#     ret_begin = 0
#     ret_end = min(ret_begin + batch_size, all_len)
#     curr_inds = []
#     curr_vals = []
#     for i_batch in range(0, total_elms, guess_bs):
#         i_end = min(i_batch + guess_bs, total_elms)
#         curr_ind = ind[:, i_batch:i_end]
#         curr_val = val[i_batch:i_end]
#

@torch.no_grad()
def sparse_top_k(sp_sim: Tensor, link: Tensor, device='cuda', needed=(1, 5, 50), batch_size=512):
    # assert device == 'cpu'
    sp_sim, link = apply(lambda x: x.to(device), sp_sim, link)
    all_len = sp_sim.size(0)
    link_len = link.size(1)
    trg_len = sp_sim.size(1)
    # all_link = -1 * torch.ones(all_len).to(device)
    # all_link[link[0]] = link[1]
    topks = []
    for i_batch in trange(0, all_len, batch_size):
        i_end = min(all_len, i_batch + batch_size)
        curr_topk = resize_sparse(filter_which(sp_sim, ind_0=([torch.ge, torch.lt], [i_batch, i_end])),
                                  [i_end - i_batch, trg_len], [-i_batch, 0]).to_dense().topk(needed[-1])[1]
        topks.append(curr_topk)
    topks = torch.cat(topks)

    results = get_hit_k(topks, link, k_list=needed)
    results['MRR'] = truncated_mrr(topks, link)
    return results
