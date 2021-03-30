# from fml.functional import sinkhorn
from utils import *
import torch
import text_utils as tu
import emb_loader as emb
from multiprocessing import pool, cpu_count
from functools import partial
from text_utils import EntTokenInfo
from torch_geometric.utils import softmax
import faiss
import dto
import secrets
from tqdm import tqdm, trange
import subprocess
from text_utils import global_level_semantic_sim

try:
    from pykeops.torch import LazyTensor
except:
    LazyTensor = None


def matrix_sinkhorn(pred_or_m, expected=None, a=None, b=None):
    from fml.functional import sinkhorn
    device = pred_or_m.device
    if expected is None:
        M = view3(pred_or_m).to(torch.float32)
        m, n = tuple(pred_or_m.size())
    else:
        m = pred_or_m.size(0)
        n = expected.size(0)
        M = cosine_distance(pred_or_m, expected)
        M = view3(M)

    if a is None:
        a = torch.ones([1, m], requires_grad=False, device=device)
    else:
        a = a.to(device)

    if b is None:
        b = torch.ones([1, n], requires_grad=False, device=device)
    else:
        b = b.to(device)
    P = sinkhorn(a, b, M, 1e-3, max_iters=300, stop_thresh=1e-3)
    return view2(P)


def lazy_topk(xs, xt, k=1, req='both') -> Union[Tuple[Tensor, Tensor], Tensor]:
    x_s = xs.unsqueeze(-2)  # [..., n_s, 1, d]
    x_t = xt.unsqueeze(-3)  # [..., 1, n_t, d]
    x_s, x_t = LazyTensor(x_s), LazyTensor(x_t)
    S_ij = (-x_s * x_t).sum(dim=-1)
    if req == 'k':
        return S_ij.Kmin(k, dim=-1, backend='auto')
    elif req == 'argk':
        return S_ij.argKmin(k, dim=-1, backend='auto')
    elif req == 'both':
        return S_ij.Kmin(k, dim=1, backend='auto'), S_ij.argKmin(k, dim=1, backend='auto')


def calc_topk_sim(xs, xt, k=1, which=0, batch_size=2048, split=False, lazy=False):
    slen, tlen = xs.size(0), xt.size(0)
    if lazy and LazyTensor is not None:
        val, ind = lazy_topk(xs, xt, k, req='both')
    else:
        vals, inds = [], []
        for i in range(0, slen, batch_size):
            sim = cosine_sim(xs[i:min(i + batch_size, slen)], xt)
            val, ind, _ = remain_topk_sim(sim, k=k, split=True, dim=which)
            vals.append(val)
            inds.append(ind)
        val, ind = torch.cat(vals, dim=0), torch.cat(inds, dim=0)
    return topk2spmat(val, ind, [slen, tlen], which, xs.device, split).to(torch.float32)


@torch.no_grad()
def approximate_sim(src: Tensor, mapping: Tensor, trg: Tensor, rank=1000, niter=2,
                    keep_k=100, batch_size=5000):
    srclen = src.size(-1)
    trglen = trg.size(-1)
    tgm = spspmm(src.t(), mapping)
    us, ss, vs = torch.svd_lowrank(tgm, rank, niter)
    vs = vs.t()
    print('calculate svd complete')
    # [N, R] [R] [R, T]
    result = None

    def merge(a, b):
        if a is None:
            return b
        return torch.sparse_coo_tensor(
            torch.cat([a._indices(), b._indices()], 1),
            torch.cat([a._values(), b._values()], 0),
            [a.size(0) + b.size(0), a.size(1)]
        )

    for i_batch in range(0, srclen, batch_size):
        i_end = min(i_batch + batch_size, srclen)
        batched_tgm = us[i_batch:i_end].mm(torch.diag(ss)).mm(vs)
        val, ind = batched_tgm.topk(dim=-1, k=keep_k)
        batched_tgm = topk2spmat(val, ind, batched_tgm.size(), 0, batched_tgm.device)
        batched_tgm = spspmm(batched_tgm, trg)
        # save gpu memory
        result = merge(result, batched_tgm)
        if i_batch % 10 * batch_size == 0:
            print('batch', i_batch, 'complete, result size', result._values().size())

    return result


@torch.no_grad()
def token_level_similarity(src_w2e: Tensor, trg_w2e: Tensor, src_word_x: Tensor, trg_word_x: Tensor, sparse_k=1,
                           dense_mm=False, do_sinkhorn=False):
    # sim: Tensor = cosine_sim(src_word_x, trg_word_x)

    if sparse_k is None:
        # print(src_w2e.size(), sim.size(), trg_w2e.size())
        sim = cosine_sim(src_word_x, trg_word_x)
        tgm = spmm(src_w2e.t(), sim)
        tgm = spmm(trg_w2e.t(), tgm.t()).t()
    else:
        sim = calc_topk_sim(src_word_x, trg_word_x, k=1)
        print(sim.dtype)
        print('token similarity complete')
        if dense_mm:
            tgm = src_w2e.t().to_dense().mm(sim.to_dense())
            tgm = tgm.mm(trg_w2e.to_dense())
        else:
            # tgm = approximate_sim(src_w2e, sim, trg_w2e)
            print('src * sim')
            tgm = spspmm(src_w2e.t(), sim)
            print('src * sim * trg')
            print(tgm._values().numel())
            # u, s, v = torch.svd_lowrank(tgm.coalesce())
            # print(u.size(), s.size(), v.size())
            # tgm = spmm(u, spmm(torch.diag(s), spmm(v.t(), trg_w2e)))
            tgm = spspmm(tgm, trg_w2e)
        print('tgm complete')
    if do_sinkhorn:
        tgm = sinkhorn_process(tgm)
    return dense_to_sparse(tgm)


def sinkhorn_process(M: Tensor):
    if M.is_sparse:
        M = M.to_dense()
    return dense_to_sparse(matrix_sinkhorn(1 - masked_minmax(M)))


# def batch_tgm()


def get_ent_token_info(ent1: Dict[str, int], ent2: Dict[str, int],
                       device='cuda', save_prefix='ei_', **kwargs) \
        -> Tuple[EntTokenInfo, EntTokenInfo]:
    ent1_list, ent2_list = tu.remove_prefix_to_list(ent1), \
                           tu.remove_prefix_to_list(ent2)

    loader = emb.EmbeddingLoader(kwargs.get('model', 'bert-base-multilingual-cased'), device)

    e1info = EntTokenInfo(ent1_list, *tu.get_name_feature_map(ent1_list, loader, **kwargs))
    e2info = EntTokenInfo(ent2_list, *tu.get_name_feature_map(ent2_list, loader, **kwargs))
    tokenizer = loader.tokenizer
    e1info.tf_idf = tu.get_tf_idf(e1info.words, e1info.ents, tokenizer)
    e1info.save(save_prefix + str(1))
    e2info.tf_idf = tu.get_tf_idf(e2info.words, e2info.ents, tokenizer)
    e2info.save(save_prefix + str(2))

    return e1info, e2info


def union(mp, sa, sb, hf, now):
    if now in hf[0]:
        return set()
    nowb = mp[now]
    if nowb in hf[1]:
        return set()
    return set((a, b) for a in sa[now] for b in sb[nowb])


from datasketch import MinHashLSH, MinHash


def makeset(ent_list, num_perm):
    sets = [set(ei.split('_')) for ei in ent_list]
    print('build MinHash')
    for s in tqdm(sets):
        m = MinHash(num_perm)
        for d in s:
            m.update(d.encode('utf-8'))
        yield m


def minhash_select_pairs(e1: Iterable[MinHash], e2: Iterable[str],
                         begin_with=0, threshold=0.5, num_perm=128,
                         redis_port=6379):
    storage_config = {
        'type': 'redis',
        'basename': 'unique_name_{}'.format(secrets.token_hex(5)).encode('utf-8'),
        'redis': {'host': 'localhost', 'port': redis_port},
    }
    # eq, eb = map(makeset, [e1, e2])
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, storage_config=storage_config)
    # wait for redis
    # import redis
    # while redis.
    print('build LSH index')
    for i, e in enumerate(makeset(e2, num_perm)):
        lsh.insert(str(i + begin_with), e)

    query_result = []
    print('query LSH')
    for i, e in enumerate(tqdm(e1)):
        query_result.append([x for x in map(int, lsh.query(e))])

    lsh.keys._redis.flushall(True)
    print('total pairs', sum(map(len, query_result)))
    print('expand result into oo pairs')
    pairs = []
    for i, r in enumerate(tqdm(query_result)):
        pairs += [(i, v) for v in r]

    time.sleep(1)
    return pairs


def sparse_string_sim(ent1, ent2, batch_size=1000000, num_perm=128, *args, **kwargs) -> Tensor:
    e1, e2 = tu.remove_prefix_to_list(ent1, punc=tu.PUNC), tu.remove_prefix_to_list(ent2, punc=tu.PUNC)
    len_all = len(e2)
    # pairs = []
    edit_dists = []
    idxs = []
    minhashs = [x for x in tqdm(makeset(e1, num_perm))]
    for i_batch in range(0, len_all, batch_size):
        batch_end = min(len_all, i_batch + batch_size)
        pairs = minhash_select_pairs(minhashs, e2[i_batch:batch_end], i_batch, num_perm=num_perm, *args, **kwargs)
        print(i_batch, 'complete')
        update_time_logs('minhash_select_pairs')
        edit_dist, idx = tu.selected_edit_distance(e1, e2, pairs)
        edit_dists.append(torch.from_numpy(edit_dist))
        idxs.append(torch.tensor(idx))
    update_time_logs('get_lev_dist')
    return ind2sparse(torch.cat(idxs).t(), [len(e1), len(e2)], values=torch.cat(edit_dists))


@torch.no_grad()
def sparse_semantic_sim(e1info: EntTokenInfo, e2info: EntTokenInfo, device: torch.device = 'cuda',
                        filter_token_cnt=None) -> Tensor:
    e1tf_idf, e2tf_idf = e1info.get_tf_idf(filter_tokens=filter_token_cnt), \
                         e2info.get_tf_idf(filter_tokens=filter_token_cnt)
    e1tf_idf, e2tf_idf, e1_emb, e2_emb = \
        apply(lambda x: x.to(device), e1tf_idf, e2tf_idf, e1info.emb, e2info.emb)

    apply(lambda x: print(x.dtype, x.device), e1tf_idf, e2tf_idf, e1info.emb, e2info.emb)
    apply(lambda x: print(x._values().numel()), e1tf_idf, e2tf_idf)
    print('get tf-idf complete')
    embs = apply(lambda x: x.cpu(), e1_emb, e2_emb)
    del e1info, e2info
    token_sim = tu.get_batch_sim(embs, 1, split=False).to(device)
    # token_sim = topk2spmat(*reversed(tu.get_batch_sim(embs, 1)), [embs[0].size(0), embs[1].size(0)], device=device)
    del embs
    return spspmm(spspmm(e1tf_idf.t(), token_sim), e2tf_idf)

    # torch.Size([15000, 15000]) torch.Size([2, 132540052]) torch.Size([132540052])
    # torch.Size([15000, 15000]) torch.Size([2, 132540052]) torch.Size([132540052])
    # tensor(0.8831)


@torch.no_grad()
def get_bert_maxpooling_embs(ent1: Dict[str, int], ent2: Dict[str, int], encode_batch_sz=2048,
                             model='bert-base-multilingual-cased', device='cuda'):
    sent_list = tu.remove_prefix_to_list(ent1), \
                tu.remove_prefix_to_list(ent2)
    bert = emb.BERT(model)
    bert.to(device)
    print('BERT max pooling embedding')
    return [bert.pooled_encode_batched(lst, batch_size=encode_batch_sz,
                                       save_gpu_memory=True, layer=1)
            for lst in sent_list]
