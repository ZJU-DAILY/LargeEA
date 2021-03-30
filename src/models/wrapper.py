from utils import *
from models.models import Encoder, Decoder
from torch_scatter import scatter_mean
from torch_geometric.utils import add_self_loops, remove_self_loops
from tqdm import tqdm
from eval import evaluate_sim_matrix
from .hyperparams import *
import torch.nn as nn


class PyTorchModelWrapper(nn.Module):
    def __init__(self, name, **kwargs):
        super(PyTorchModelWrapper, self).__init__()
        # framework parameters
        self.device = kwargs.get('device', 'cuda')
        self.dim = kwargs.get('dim', 200)
        self.edge_index = kwargs['ei']
        self.edge_type = kwargs['et']
        ent_sizes = kwargs.get('ent_sizes', [15000, 15000])
        rel_sizes = kwargs.get('rel_sizes', [500, 500])
        dim = self.dim
        self.ent_num = ent_sizes[0] + ent_sizes[1]
        self.rel_num = rel_sizes[0] + rel_sizes[1]
        self.ent_split = ent_sizes[0]
        self.rel_split = rel_sizes[0]
        # load model specific parameters
        param = PARAMS.get(name, {})

        self.hiddens = param.get('hiddens', (dim, dim, dim))
        self.dim = dim = self.hiddens[0]
        self.heads = param.get('heads', (1, 1, 1))
        self.feat_drop = param.get('feat_drop', 0.2)
        self.attn_drop = param.get('attn_drop', 0.)
        self.negative_slope = param.get('negative_slope', 0.)
        self.update = param.get('update', 10)
        self.dist = param.get('dist', 'manhattan')
        self.lr = param.get('lr', 0.005)
        self.share = param.get('share', False)
        sampling = param.get('sampling', ['N'])
        k = param.get('k', [25])
        margin = param.get('margin', [1])
        alpha = param.get('alpha', [1])

        encoder_name = param.get('encoder', None)
        decoder_names = param.get('decoder', ['align'])

        self.encoder = Encoder(name, self.hiddens, self.heads, F.elu, self.feat_drop, self.attn_drop,
                               self.negative_slope, False) if encoder_name else None

        knowledge_decoder = []
        for idx, decoder_name in enumerate(decoder_names):
            knowledge_decoder.append(Decoder(decoder_name, params={
                "e_num": self.ent_num,
                "r_num": self.rel_num,
                "dim": self.hiddens[-1],
                "feat_drop": self.feat_drop,
                "train_dist": self.dist,
                "sampling": sampling[idx],
                "k": k[idx],
                "margin": margin[idx],
                "alpha": alpha[idx],
                "boot": False,
                # pass other useful parameters to Decoder
            }))
        self.knowledge_decoder = nn.ModuleList(knowledge_decoder)

        self.cached_sample = {}
        self.preprocessing()
        self.init_emb(encoder_name, decoder_names, margin, self.ent_num, self.rel_num, self.device)

    @torch.no_grad()
    def preprocessing(self):
        edge_index0, edge_index1 = self.edge_index
        edge_index1 = edge_index1 + self.ent_split
        self.edge_index = torch.cat([edge_index0, edge_index1], dim=1)

        rel0, rel1 = self.edge_type
        rel1 = rel1 + self.rel_split
        self.rel = torch.cat([rel0, rel1], dim=0)

        total = self.edge_index.size(1)
        ei, et = apply(lambda x: x.cpu().numpy(), self.edge_index, self.rel)
        self.triples = [(ei[0][i], et[i], ei[1][i]) for i in range(total)]
        self.edge_index = add_self_loops(remove_self_loops(self.edge_index)[0])[0].t()
        self.ids = [
            set(range(0, self.ent_split)),
            set(range(self.ent_split, self.ent_num))
        ]

    def init_emb(self, encoder_name, decoder_names, margin, ent_num, rel_num, device):
        e_scale, r_scale = 1, 1
        if not encoder_name:
            if decoder_names == ["rotate"]:
                r_scale = r_scale / 2
            elif decoder_names == ["hake"]:
                r_scale = (r_scale / 2) * 3
            elif decoder_names == ["transh"]:
                r_scale = r_scale * 2
            elif decoder_names == ["transr"]:
                r_scale = self.hiddens[0] + 1
        self.ent_embeds = nn.Embedding(ent_num, self.hiddens[0] * e_scale).to(device)
        self.rel_embeds = nn.Embedding(rel_num, int(self.hiddens[0] * r_scale)).to(device)
        if decoder_names == ["rotate"] or decoder_names == ["hake"]:
            ins_range = (margin[0] + 2.0) / float(self.hiddens[0] * e_scale)
            nn.init.uniform_(tensor=self.ent_embeds.weight, a=-ins_range, b=ins_range)
            rel_range = (margin[0] + 2.0) / float(self.hiddens[0] * r_scale)
            nn.init.uniform_(tensor=self.rel_embeds.weight, a=-rel_range, b=rel_range)
            if decoder_names == ["hake"]:
                r_dim = int(self.hiddens[0] / 2)
                nn.init.ones_(tensor=self.rel_embeds.weight[:, r_dim: 2 * r_dim])
                nn.init.zeros_(tensor=self.rel_embeds.weight[:, 2 * r_dim: 3 * r_dim])
        else:
            nn.init.xavier_normal_(self.ent_embeds.weight)
            nn.init.xavier_normal_(self.rel_embeds.weight)
        if "alignea" in decoder_names or "mtranse_align" in decoder_names or "transedge" in decoder_names:
            self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight, p=2, dim=1)
            self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight, p=2, dim=1)
        # elif "transr" in decoder_names:
        #     assert self.args.pre != ""
        #     self.ent_embeds.weight.data = torch.from_numpy(np.load(self.args.pre + "_ins.npy")).to(device)
        #     self.rel_embeds.weight[:, :self.hiddens[0]].data = torch.from_numpy(
        #         np.load(self.args.pre + "_rel.npy")).to(device)
        self.enh_ins_emb = self.ent_embeds.weight.cpu().detach().numpy()
        self.mapping_ins_emb = None

    def run_test(self, pair):
        npy_embeds = apply(lambda x: x.cpu().numpy(), *self.get_embed())
        npy_sim = sim(*npy_embeds, metric=self.dist, normalize=True, csls_k=10)
        evaluate_sim_matrix(pair, torch.from_numpy(npy_sim).to(self.device))

    def get_embed(self):
        emb = self.ent_embeds.weight
        if self.encoder:
            self.encoder.eval()
            emb = self.encoder.forward(self.edge_index, emb, None)
        embs = apply(norm_embed, *[emb[:self.ent_split], emb[self.ent_split:]])
        return embs

    def refresh_cache(self):
        self.cached_sample = {}

    def share_triples(self, pairs, triples):
        ill = {k: v for k, v in pairs}
        new_triple = []
        for (h, r, t) in triples:
            if h in ill:
                h = ill[h]
            if t in ill:
                t = ill[t]
            new_triple.append((h, r, t))
        return list(set(new_triple))

    def gen_sparse_graph_from_triples(self, triples, ins_num, with_r=False):
        edge_dict = {}
        for (h, r, t) in triples:
            if h != t:
                if (h, t) not in edge_dict:
                    edge_dict[(h, t)] = []
                    edge_dict[(t, h)] = []
                edge_dict[(h, t)].append(r)
                edge_dict[(t, h)].append(-r)
        if with_r:
            edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            r_ij = [abs(r) for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            edges = np.array(edges, dtype=np.int32)
            values = np.array(values, dtype=np.float32)
            r_ij = np.array(r_ij, dtype=np.float32)
            return edges, values, r_ij
        else:
            edges = [[h, t] for (h, t) in edge_dict]
            values = [1 for (h, t) in edge_dict]
        # add self-loop
        edges += [[e, e] for e in range(ins_num)]
        values += [1 for e in range(ins_num)]
        edges = np.array(edges, dtype=np.int32)
        values = np.array(values, dtype=np.float32)
        return edges, values, None

    def train1step(self, it, pairs, opt):
        pairs = torch.stack([pairs[0], pairs[1] + self.ent_split])
        pairs = pairs.t().cpu().numpy()
        triples = self.triples
        ei, et = self.edge_index, self.edge_type
        if self.share:
            triples = self.share_triples(pairs, triples)
            ei = self.gen_sparse_graph_from_triples(triples, self.ent_num)

        for decoder in self.knowledge_decoder:
            self._train(it, opt, self.encoder, decoder, ei, triples, pairs,
                        self.ids, self.ent_embeds.weight, self.rel_embeds.weight)

    def _train(self, it, opt, encoder, decoder, edges, triples, ills, ids, ins_emb,
               rel_emb):
        device = self.device
        if encoder:
            encoder.train()
        decoder.train()
        losses = []
        if "pos_" + decoder.print_name not in self.cached_sample or it % self.update == 0:
            if decoder.name in ["align", "mtranse_align", "n_r_align"]:
                self.cached_sample["pos_" + decoder.print_name] = ills.tolist()
                self.cached_sample["pos_" + decoder.print_name] = np.array(
                    self.cached_sample["pos_" + decoder.print_name])
            else:
                self.cached_sample["pos_" + decoder.print_name] = triples
            np.random.shuffle(self.cached_sample["pos_" + decoder.print_name])
            # print("train size:", len(self.cached_sample["pos_"+decoder.print_name]))

        train = self.cached_sample["pos_" + decoder.print_name]
        train_batch_size = len(train)
        for i in range(0, len(train), train_batch_size):
            pos_batch = train[i:i + train_batch_size]

            if (decoder.print_name + str(
                    i) not in self.cached_sample or it % self.update == 0) and decoder.sampling_method:
                self.cached_sample[decoder.print_name + str(i)] = decoder.sampling_method(pos_batch, triples, ills, ids,
                                                                                          decoder.k, params={
                        "emb": self.enh_ins_emb,
                        "metric": self.dist,
                    })

            if decoder.sampling_method:
                neg_batch = self.cached_sample[decoder.print_name + str(i)]

            opt.zero_grad()
            if decoder.sampling_method:
                neg = torch.LongTensor(neg_batch).to(device)
                if neg.size(0) > len(pos_batch) * decoder.k:
                    pos = torch.LongTensor(pos_batch).repeat(decoder.k * 2, 1).to(device)
                elif hasattr(decoder.func, "loss") and decoder.name not in ["rotate", "hake", "conve", "mmea",
                                                                            "n_transe"]:
                    pos = torch.LongTensor(pos_batch).to(device)
                else:
                    pos = torch.LongTensor(pos_batch).repeat(decoder.k, 1).to(device)
            else:
                pos = torch.LongTensor(pos_batch).to(device)

            if encoder:
                enh_emb = encoder.forward(edges, ins_emb, None)
            else:
                enh_emb = ins_emb

            self.enh_ins_emb = enh_emb[
                0].cpu().detach().numpy() if encoder and encoder.name == "naea" else enh_emb.cpu().detach().numpy()
            if decoder.name == "n_r_align":
                rel_emb = ins_emb

            if decoder.sampling_method:
                pos_score = decoder.forward(enh_emb, rel_emb, pos)
                neg_score = decoder.forward(enh_emb, rel_emb, neg)
                target = torch.ones(neg_score.size()).to(device)

                loss = decoder.loss(pos_score, neg_score, target) * decoder.alpha
            else:
                loss = decoder.forward(enh_emb, rel_emb, pos) * decoder.alpha

            loss.backward()

            opt.step()
            losses.append(loss.item())

        return np.mean(losses)


def default(*args, **kwargs):
    pass


class ModelWrapper:
    def __init__(self, name, **kwargs):
        print('Model name is', name)
        if name in ['mraea', 'rrea']:
            from .rrea.rrea import TFModelWrapper
            self.tf = True
            self.model = TFModelWrapper(name, **kwargs)
        elif name == 'gcn-align':
            from .gcn_align import GCNAlignWrapper
            self.tf = True
            self.model = GCNAlignWrapper(**kwargs)
        else:
            self.tf = False
            self.device = kwargs.get('device', 'cuda')
            self.model = PyTorchModelWrapper(name, **kwargs).to(self.device)
            self._update_devset(kwargs['link'])

    def __getattr__(self, item):
        SHARED_METHODS = ['update_trainset',
                          'update_devset',
                          'train1step',
                          'test_train_pair_acc',
                          'get_curr_embeddings',
                          'mraea_iteration'
                          ]
        if item in SHARED_METHODS:
            if self.tf:
                if hasattr(self.model, item):
                    return object.__getattribute__(self.model, item)
                return default
            else:
                return object.__getattribute__(self, '_' + item)
        else:
            return self.__getattribute__(item)

    def _update_trainset(self, pairs, append=False):
        self.train_pair = torch.from_numpy(pairs).to(self.device)

    def _update_devset(self, pairs, append=False):
        self.dev_pair = pairs.to(self.device)

    def default_sgd(self):
        if not hasattr(self, '_default_sgd'):
            self._default_sgd = optim.RMSprop(self.model.parameters(), lr=self.model.lr)
        return self._default_sgd

    def _train1step(self, epoch=75, sgd=None):
        if sgd is None:
            sgd = self.default_sgd()
        self.model.refresh_cache()
        for it in tqdm(range(epoch)):
            self.model.train1step(it, self.train_pair, sgd)

        self.model.run_test(self.dev_pair)

    def _test_train_pair_acc(self):
        pass

    def _get_curr_embeddings(self, device=None):
        if device is None:
            device = self.device
        return apply(lambda x: x.detach().to(device), *self.model.get_embed())
