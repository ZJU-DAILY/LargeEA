import regex
import string
from utils import *
from tqdm import tqdm
import faiss

import gc

try:
    from Levenshtein import ratio
except:
    print('holy shit')

from multiprocessing import Pool
import multiprocessing
from functools import partial

import collections
import os.path as osp

PREFIX = r'http(s)?://[a-z\.]+/[^/]+/'


class EntTokenInfo:
    ents: Dict[str, int]
    words: List[str]
    emb: Tensor
    w2e: List[Set[int]]
    e2w: List[List[int]]

    def __init__(self, ents, words, emb, w2e, e2w):
        self.ents = ents
        self.words = words
        self.emb = emb
        self.w2e = w2e
        self.e2w = e2w
        self.tf_idf = None

    @staticmethod
    def load(path, *args, **kwargs):
        return torch.load(path, *args, **kwargs)

    def save(self, path, *args, **kwargs):
        torch.save(self, path, *args, **kwargs)

    def get_tf_idf(self, filter_eps=None, filter_tokens=None):
        result = to_torch_sparse(self.tf_idf, device='cpu') \
            .to(torch.float32).coalesce().t()
        # self.emb.device
        if filter_eps is not None:
            result = filter_which(result.clone(), val=(torch.lt, filter_eps))
        if filter_tokens is not None:
            tokens = self.filter_tokens(filter_tokens)
            result = filter_which(result.clone(), ind_0=(torch.ne, tokens))
        return result

    def ent_cnt(self):
        return len(self.ents)

    def word_cnt(self):
        return len(self.words)

    @staticmethod
    def static_high_freq_words(w2e, word_list, k=25, verbose=False):
        lens = [len(es) for es in w2e]
        lens = np.array(lens)
        words = np.argsort(lens)

        if verbose:
            for i in words[-k:]:
                print(word_list[i], 'has freq of', lens[i])
        return set(words[-k:])

    @staticmethod
    def static_punc_tokens(word_list, punc=None, verbose=False):
        if punc is None:
            punc = get_punctuations()
        occurred_punc = set()
        for i, word in enumerate(word_list):
            if word in punc:
                occurred_punc.add(i)
        if verbose:
            print('all puncs is:', [word_list[i] for i in occurred_punc])

        return occurred_punc

    def filter_tokens(self, k=25, verbose=False):
        return self.static_high_freq_words(self.w2e, self.words, k, verbose) \
            .union(self.static_punc_tokens(self.words))


def get_punctuations():
    # zh = zhon.hanzi.punctuation
    en = string.punctuation
    zh = ""
    puncs = set()
    for i in (zh + en):
        puncs.add(i)
    puncs.remove('_')
    return puncs


PUNC = get_punctuations()


def remove_punc(str, punc=None):
    if punc is None:
        punc = PUNC
    if punc == '':
        return str
    return ''.join([' ' if i in punc else i for i in str])


def remove_prefix_to_list(entity_dict: {}, prefix=PREFIX, punc='') -> []:
    # punc = get_punctuations()
    tmp_dict = {}
    entity_list = []
    p = regex.compile(prefix)
    for ent in entity_dict.keys():
        res = p.search(ent)
        if res is None:
            entity_list.append(remove_punc(ent, punc))
        else:
            _, end = res.span()
            entity_list.append(remove_punc(ent[end:], punc))

        tmp_dict[entity_list[-1]] = entity_dict[ent]
        # print(ent, entity_list[-1])
    entity_list = sorted(entity_list, key=lambda x: entity_dict[x] if x in entity_dict else tmp_dict[x])
    return entity_list


def normalize_vectors(embeddings, center=False):
    if center:
        embeddings -= torch.mean(embeddings, dim=0, keepdim=True)
    embeddings /= torch.linalg.norm(embeddings, dim=1, keepdim=True)
    return embeddings


def get_count(words, ent_lists, binary=True):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(vocabulary=words, lowercase=False, tokenizer=lambda x: x.split(), binary=binary)
    return vectorizer.fit_transform(ent_lists)


def get_tf_idf(words, ent_lists, bert_tokenizer=None):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    if bert_tokenizer is None:
        tokenizer = lambda x: x.split()
    else:
        tokenizer = lambda x: bert_tokenizer.tokenize(x)
    vectorizer = CountVectorizer(vocabulary=words, lowercase=False, tokenizer=tokenizer, binary=False)
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(ent_lists)
    tfidf = transformer.fit_transform(X)
    return tfidf


def get_fasttext_aligned_vectors(words, device, lang):
    embs = {}
    with open(osp.join("aligned_vectors", 'wiki.{}.align.vec'.format(lang)), 'r') as f:
        for i, line in enumerate(f):
            info = line.strip().split(' ')
            if len(info) > 300:
                embs[info[0]] = torch.tensor([float(x) for x in info[1:]])
            else:
                embs['**UNK**'] = torch.tensor([float(x) for x in info])
    # for word in words:
    #     word = word.replace('#', '')
    #     if word not in embs:
    #         print("fasttext unk:", word)
    word_embeds = [embs.get(word.replace('#', '').lower(), embs['**UNK**']) for word in words]

    return torch.stack(word_embeds, dim=0).to(device)


def tokenize(sent, tokenizer):
    if tokenizer is None:
        return sent.split()
    else:
        return tokenizer.tokenize(sent)


def get_name_feature_map(sents, embedding_loader=None, device='cuda',
                         batch_size=1024, use_fasttext=False, lang=None,
                         **kwargs):
    word_id_map = {}
    entity2word = []
    word2entity = collections.defaultdict(set)
    tokenizer = None if embedding_loader is None else embedding_loader.tokenizer

    for ent_id, sent in enumerate(sents):
        entity2word.append([])
        for word in tokenize(sent, tokenizer):
            word_id_map, word_id = add_cnt_for(word_id_map, word)
            entity2word[-1].append(word_id)
            word2entity[word_id].add(ent_id)
    word2entity = [word2entity[i] for i in range(len(word_id_map))]
    words = mp2list(word_id_map)
    # print("----print high freq tokens")
    # EntTokenInfo.static_high_freq_words(word2entity, words, verbose=True)
    # print('----print puncs')
    # EntTokenInfo.static_punc_tokens(words, verbose=True)
    # print("----print end")

    if use_fasttext:
        if isinstance(lang, str):
            embeddings = get_fasttext_aligned_vectors(words, device, lang)
        else:
            embeddings = torch.cat([get_fasttext_aligned_vectors(words, device, lang) for lang in lang], dim=1)
    else:
        lens = []
        print('all ents to embed is', len(sents))
        print('all words to average is', len(words))
        print('batch size is', batch_size)
        embeddings = None
        for i in tqdm(range(0, len(sents), batch_size)):
            embed, length = embedding_loader.get_embed_list(sents[i:min(i + batch_size, len(sents))], True)
            if embeddings is None:
                embed_size = embed.size(-1)
                embeddings = torch.zeros([len(words), embed_size], device=device, dtype=torch.float)
            for j in range(batch_size):
                if i + j >= len(sents):
                    break
                index = torch.tensor(entity2word[i + j], device=device, dtype=torch.long)
                embeddings[index] += embed[j][:len(entity2word[i + j])]
        if kwargs.get('size_average', True):
            sizes = torch.tensor([len(i) for i in word2entity]).to(device)
            embeddings /= sizes.view(-1, 1)

    if kwargs.get('normalize', True):
        embeddings = normalize_vectors(embeddings, kwargs.get('center', True))
    return words, embeddings.to(device), word2entity, entity2word


# CPM
def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )


def reduce(tensor, reduction='mean', dim=0):
    if reduction == 'mean':
        return torch.mean(tensor, dim)
    if reduction == 'max':
        return torch.max(tensor, dim)[0]
    if reduction == 'min':
        return torch.min(tensor, dim)[0]
    if reduction == 'sum':
        return torch.sum(tensor, dim)
    if 'p_mean_' in reduction:
        p = float(reduction.split('_')[-1])
        vals = tensor.cpu().numpy()
        return torch.from_numpy(gen_mean(vals, p).real).to(tensor.device)


def embed_word2entity(ent2word, word_emb, reduction='max') -> Tensor:
    ent_emb = []
    for ent in ent2word:
        ent_emb.append(reduce(word_emb[torch.tensor(ent, device=word_emb.device)], reduction).squeeze())
    return torch.stack(ent_emb, dim=0)


def cpm_embedding(ent2word, words, cpm_types, models=('en', 'fr')):
    word_vec = [get_fasttext_aligned_vectors(words, 'cuda', lang) for lang in models]
    cpms = torch.cat([embed_word2entity(ent2word, vec, ty) for vec in word_vec for ty in cpm_types], dim=1)
    return cpms.to('cpu')


def edit_dist_of(sent0, sent1, item):
    x, y = item
    return ratio(sent0[x], sent1[y])


# def fast_topk(k, xq, xb):
#     xq, xb = map(lambda x: norm_process(x).cpu().numpy(), [xq, xb])
#     index = faiss.IndexFlat(xq.shape[1])
#     index.add(xb)
#     val, ind = index.search(xq, k)
#     return 1 - val, ind
from fuse import naive_sim_fuser


def faiss_search_impl(emb_q, emb_id, emb_size, shift, k=50, search_batch_sz=50000, gpu=True):
    index = faiss.IndexFlat(emb_size)
    if gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(emb_id)
    print('Total index =', index.ntotal)
    vals, inds = [], []
    for i_batch in tqdm(range(0, len(emb_q), search_batch_sz)):
        val, ind = index.search(emb_q[i_batch:min(i_batch + search_batch_sz, len(emb_q))], k)
        val = torch.from_numpy(val)
        val = 1 - val
        vals.append(val)
        inds.append(torch.from_numpy(ind) + shift)
        # print(vals[-1].size())
        # print(inds[-1].size())
    del index, emb_id, emb_q
    vals, inds = torch.cat(vals), torch.cat(inds)
    return vals, inds


@torch.no_grad()
def global_level_semantic_sim(embs, k=50, search_batch_sz=50000, index_batch_sz=500000
                              , split=False, norm=True, gpu=True):
    print('FAISS number of GPUs=', faiss.get_num_gpus())
    size = [embs[0].size(0), embs[1].size(0)]
    emb_size = embs[0].size(1)
    if norm:
        embs = apply(norm_process, *embs)
    emb_q, emb_id = apply(lambda x: x.cpu().numpy(), *embs)
    del embs
    gc.collect()
    vals, inds = [], []
    total_size = emb_id.shape[0]
    for i_batch in range(0, total_size, index_batch_sz):
        i_end = min(total_size, i_batch + index_batch_sz)
        val, ind = faiss_search_impl(emb_q, emb_id[i_batch:i_end], emb_size, i_batch, k, search_batch_sz, gpu)
        vals.append(val)
        inds.append(ind)

    vals, inds = torch.cat(vals, dim=1), torch.cat(inds, dim=1)
    print(vals.size(), inds.size())

    return topk2spmat(vals, inds, size, 0, torch.device('cpu'), split)


def get_batch_sim(embed, topk=50, split=True):
    # embed = self.get_gnn_embed()
    size = apply(lambda x: x.size(0), *embed)
    # x2y_val, x2y_argmax = fast_topk(2, embed[0], embed[1])
    # y2x_val, y2x_argmax = fast_topk(2, embed[1], embed[0])
    # ind, val = filter_mapping(x2y_argmax, y2x_argmax, size, (x2y_val, y2x_val), 0)
    spmat = global_level_semantic_sim(embed, k=topk, gpu=False).to(embed[0].device)
    if split:
        return spmat._indices(), spmat._values()
    else:
        return spmat


def selected_edit_distance(sent0, sent1, needed, batch_size=100000):
    # needed = sorted(needed, key=lambda x: x[0])
    x = np.empty([len(needed)])
    cpu = multiprocessing.cpu_count()
    print('cpu has', cpu, 'cores')
    pool = Pool(processes=cpu)
    for i in tqdm(range(0, len(needed), batch_size)):
        x[i: i + batch_size] = pool.map(partial(edit_dist_of, sent0, sent1), needed[i:i + batch_size])
    return x, needed


def pairwise_edit_distance(sent0, sent1, to_tensor=True):
    x = np.empty([len(sent0), len(sent1)], np.float)
    print(multiprocessing.cpu_count())
    pool = Pool(processes=multiprocessing.cpu_count())
    for i, s0 in enumerate(sent0):
        if i % 5000 == 0:
            print("edit distance --", i, "complete")
        x[i, :] = pool.map(partial(ratio, s0), sent1)
        # for j, s1 in enumerate(sent1):
        #     x[i, j] = distance(s0, s1)

    if to_tensor:
        return (torch.from_numpy(x).to(torch.float))
