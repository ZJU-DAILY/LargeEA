from partition import *
import text_utils
from models import ModelWrapper


def get_bi_mapping(src2trg, trg2src, lens) -> Tensor:
    srclen, trglen = lens
    with torch.no_grad():
        i = torch.arange(srclen, device=src2trg.device).to(torch.long)
        return trg2src[src2trg[i]] == i


def filter_mapping(src2trg: Tensor, trg2src: Tensor, lens, values: Tuple[Tensor, Tensor], th):
    vals2t, valt2s = values
    added_s = np.zeros(lens[0], dtype=np.int)
    added_t = np.zeros(lens[1], dtype=np.int)
    pair_s, pair_t = [], []
    val = []
    for i in range(lens[0]):
        j = src2trg[i, 0]
        # print(now)
        if added_t[j] == 1 or i != trg2src[j, 0]:
            continue
        gap_x2y = vals2t[i, 0] - vals2t[i, 1]
        gap_y2x = valt2s[j, 0] - valt2s[j, 1]
        if gap_y2x < th or gap_x2y < th:
            continue
        added_s[i] = 1
        added_t[j] = 1
        pair_s.append(i)
        pair_t.append(j)
        val.append(vals2t[i, 0])

    return torch.tensor([pair_s, pair_t]), torch.tensor(val)


def rearrange_ids(nodes, merge: bool, *to_map):
    ent_mappings = [{}, {}]
    rel_mappings = [{}, {}]
    ent_ids = [[], []]
    shift = 0
    for w, node_set in enumerate(nodes):
        for n in node_set:
            ent_mappings[w], nn, shift = add_cnt_for(ent_mappings[w], n, shift)
            ent_ids[w].append(nn)
        shift = len(ent_ids[w]) if merge else 0
    mapped = []
    shift = 0
    curr = 0
    for i, need in enumerate(to_map):
        now = []
        if len(need) == 0:
            mapped.append([])
            continue
        is_triple = len(need[0]) == 3
        for tu in need:
            if is_triple:
                h, t = ent_mappings[curr][tu[0]], ent_mappings[curr][tu[-1]]
                rel_mappings[curr], r, shift = add_cnt_for(rel_mappings[curr], tu[1], shift)
                now.append((h, r, t))
            else:
                now.append((ent_mappings[0][tu[0]], ent_mappings[1][tu[-1]]))
        mapped.append(now)
        curr += is_triple
        if not merge:
            shift = 0
    rel_ids = [list(rm.values()) for rm in rel_mappings]

    return ent_mappings, rel_mappings, ent_ids, rel_ids, mapped


def make_assoc(maps, src_len, trg_len, merge):
    assoc = np.empty(src_len + trg_len, dtype=np.int)
    shift = 0 if merge else 1
    shift = shift * src_len
    for idx, ent_mp in enumerate(maps):
        for k, v in ent_mp.items():
            assoc[v + idx * shift] = k
    return torch.tensor(assoc)


def filter_ent_list(id_map, ent_collection):
    id_ent_mp = {}
    if isinstance(ent_collection, dict):
        for ent, i in ent_collection.items():
            if i in id_map:
                id_ent_mp[ent] = id_map[i]
        return id_ent_mp
    else:
        for i, ent in enumerate(ent_collection):
            if i in id_map:
                id_ent_mp[ent] = id_map[i]

        return sorted(id_ent_mp.keys(), key=lambda x: id_ent_mp[x])


class SelectedCandidates:
    def __init__(self, pairs, e1, e2):
        self.total_len = len(pairs)
        pairs = np.array(pairs).T
        self.pairs = pairs
        self.ent_maps = rearrange_ids(pairs, False)[0]
        self.assoc = make_assoc(self.ent_maps, *([self.total_len] * 2), False)
        self.shift = self.total_len
        self.ents = [x for x in map(filter_ent_list, self.ent_maps, [e1, e2])]
        self.sz = [len(e1), len(e2)]
        pass

    def convert_sim_mat(self, sim):
        # selected sim(dense) to normal sim(sparse)
        ind, val = sim._indices(), sim._values()
        assoc = self.assoc.to(sim.device)
        ind = torch.stack(
            [assoc[ind[0]],
             assoc[ind[1] + self.shift]]
        )
        return ind2sparse(ind, self.sz, values=val)

    @torch.no_grad()
    def filter_sim_mat(self, sim):
        # '''
        # filter normal sim with selected candidates
        def build_filter_array(sz, nodes, device):
            a = torch.zeros(sz).to(torch.bool).to(device)
            a[torch.from_numpy(nodes).to(device)] = True
            a = torch.logical_not(a)
            ret = torch.arange(sz).to(device)
            ret[a] = -1
            return ret

        ind, val = sim._indices(), sim._values()
        ind0, ind1 = map(lambda x, xsz, xn: build_filter_array(xsz, xn, x.device)[x],
                         ind, sim.size(), self.pairs)

        remain = torch.bitwise_and(ind0 >= 0, ind1 >= 0)
        return ind2sparse(ind[:, remain], sim.size(), values=val[remain])


class AlignmentBatch:
    def __init__(self, triple1, triple2, src_nodes, trg_nodes, train_pairs, test_pairs, backbone='eakit', *args,
                 **kwargs):
        self.backbone = backbone
        self.merge = True
        print("Batch info: ", '\n\t'.join(map(lambda x, y: '='.join(map(str, [x, y])),
                                              ['triple1', 'triple2', 'srcNodes', 'trgNodes',
                                               'trainPairs', 'testPairs'],
                                              map(len,
                                                  [triple1, triple2, src_nodes, trg_nodes, train_pairs, test_pairs]))))
        try:
            self.ent_maps, self.rel_maps, ent_ids, rel_ids, \
            [t1, t2, train_ill, test_ill] = rearrange_ids([src_nodes, trg_nodes], self.merge,
                                                          triple1, triple2, train_pairs, test_pairs)
            # dev_split = int(dev_ratio * len(train_ill))
            # train_ill, dev_ill = train_ill[dev_split:], train_ill[:dev_split]
            self.test_ill = test_ill
            self.train_ill = train_ill
            self.test_pairs = test_pairs
            self.len_src, self.len_trg = len(src_nodes), len(trg_nodes)
            self.shift = len(src_nodes)
            self.assoc = make_assoc(self.ent_maps, self.len_src, self.len_trg, self.merge)
            if self.backbone == 'eakit':
                self.alignment_data = AlignmentData(ent_ids, rel_ids, *map(np.array, [train_ill, test_ill]),
                                                    t1 + t2, args=kwargs['args'])

            elif self.backbone == 'rrea' or self.backbone == 'gcn-align':
                self.model = ModelWrapper(self.backbone,
                                          triples=t1 + t2,
                                          link=torch.tensor(test_ill).t(),
                                          ent_sizes=[len(ids) for ids in ent_ids],
                                          rel_sizes=[x for x in map(len, rel_ids)],
                                          device='cuda',
                                          dim=200,
                                          )

                self.model.update_trainset(np.array(self.train_ill).T)
            else:
                raise NotImplementedError
        except:
            self.skip = True
            print('skip batch')

        update_time_logs('build_batch_info')

    @staticmethod
    def get_ei(triple):
        return torch.tensor([[t[0], t[-1]] for t in triple]).t()

    @staticmethod
    def get_et(triple):
        return torch.tensor([t[1] for t in triple])

    @property
    def test_set(self):
        return torch.tensor(self.test_pairs).t()

    @torch.no_grad()
    def get_sim_mat(self, all_embeds, size):
        if isinstance(all_embeds, tuple):
            embeds = all_embeds
        else:
            embeds = [all_embeds[:self.shift], all_embeds[self.shift:]]
        ind, val = text_utils.get_batch_sim(embeds)
        ind = torch.stack(
            [self.assoc[ind[0]],
             self.assoc[ind[1] + self.shift]]
        )
        return ind2sparse(ind, size, values=val)


def batch_sampler(data: EAData, src_split=30, trg_split=100, top_k_corr=5, which=0, share_triples=True,
                  backbone='rrea', random=False, *args, **kwargs):
    def place_triplets(triplets, nodes_batch):
        batch = defaultdict(list)
        node2batch = {}
        for i, nodes in enumerate(nodes_batch):
            for n in nodes:
                node2batch[n] = i
        removed = 0
        for h, r, t in triplets:
            h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
            if h_batch == t_batch and h_batch >= 0:
                batch[h_batch].append((h, r, t))
            else:
                removed += 1
        print('split triplets complete, total {} triplets removed'.format(removed))

        return batch, removed

    def make_pairs(src, trg, mp):
        return list(filter(lambda p: p[1] in trg, [(e, mp[e]) for e in set(filter(lambda x: x in mp, src))]))

    time_now = time.time()
    metis = Partition(data)

    src_nodes, trg_nodes, src_train, trg_train = metis.random_partition(which, src_split, trg_split, share_triples) \
        if random else metis.partition(which, src_split, trg_split, share_triples)

    update_time_logs('graph_partition')
    triple1_batch, removed1 = place_triplets(data.triples[which], src_nodes)
    triple2_batch, removed2 = place_triplets(data.triples[1 - which], trg_nodes)
    add_logs('triple1_removed', removed1)
    add_logs('triple2_removed', removed2)
    update_time_logs('place_triples')
    corr = torch.from_numpy(overlaps(
        [set(metis.train_map[which][i] for i in s) for s in src_train],
        [set(s) for s in trg_train]
    ))
    mapping = metis.train_map[which]
    corr_val, corr_ind = map(lambda x: x.numpy(), corr.topk(top_k_corr))
    update_time_logs('get_corr_batch')
    # corr_ind = corr_ind.numpy()
    print('partition complete, time=', time.time() - time_now)
    for src_id, src_corr in enumerate(corr_ind):
        ids1, train1 = src_nodes[src_id], src_train[src_id]
        train2, ids2, triple2 = [], [], []
        corr_rate = 0.
        for trg_rank, trg_id in enumerate(src_corr):
            train2 += trg_train[trg_id]
            ids2 += trg_nodes[trg_id]
            triple2 += triple2_batch[trg_id]
            corr_rate += corr_val[src_id][trg_rank]
        ids1, ids2, train1, train2 = map(set, [ids1, ids2, train1, train2])
        print('Train corr=', corr_rate)
        yield AlignmentBatch(triple1_batch[src_id], triple2,
                             ids1, ids2, make_pairs(train1, train2, mapping),
                             make_pairs(ids1, ids2, mapping),
                             backbone=backbone,
                             *args, **kwargs)
