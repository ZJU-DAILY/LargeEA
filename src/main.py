import text_sim as text
from dataset import *
import torch
from eval import sparse_acc, evaluate_embeds, sparse_top_k
from collections import defaultdict
from tqdm import tqdm
from EAKit import AlignmentData, Experiment
# from torch.utils.tensorboard import SummaryWriter
import argparse, logging, random, time
from sampler import *
import numba.cuda
from fuse import *
import time

def train(batch: AlignmentBatch, device: torch.device = 'cuda', **kwargs):
    args = kwargs['args']

    if hasattr(batch, 'skip'):
        return None
    elif hasattr(batch, 'alignment_data'):
        d = batch.alignment_data
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        writer = SummaryWriter("_runs/%s_%s" % (args.data_dir.split("/")[-1], args.log))
        logger.info(args)

        logger.info(d)

        experiment = Experiment(logger, device, writer, d, args=args)

        t_total = time.time()
        embed = experiment.train_and_eval()
        logger.info("optimization finished!")
        logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))
        return embed
    elif hasattr(batch, 'model'):
        model = batch.model
        try:
            for it in range(args.it_round):
                model.train1step(args.epoch)
                if it < args.it_round - 1:
                    model.mraea_iteration()
            return model.get_curr_embeddings('cpu')
        except Exception as e:
            print('TF error', str(e))
            return None
        #  TODO
        pass
    else:
        raise NotImplementedError


import copy

import multiprocessing as mulp


def run_batched_ea(data: EAData, src_split, trg_split, topk, args):
    # data = LargeScaleEAData('raw_data', 'de')
    # data.save('dedata')
    # dataset = LargeScaleEAData.load('dedata')
    print('read data complete')

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    curr_sim = None
    for batch in tqdm(batch_sampler(data, src_split, trg_split, topk, random=args.random_split, backbone=args.model,
                                    args=copy.deepcopy(args))):
        # Load Data
        embed = train(batch, device, args=copy.deepcopy(args))
        update_time_logs('train_on_batch')
        if embed is None:
            print('batch skipped')
            continue
        sim = batch.get_sim_mat(embed, data.size())
        update_time_logs('get_sim_mat_on_batch')
        print('acc=', sparse_acc(sim, batch.test_set, device='cpu'))
        update_time_logs('eval_on_batch')
        curr_sim = sim if curr_sim is None else curr_sim + sim
        update_time_logs('fuse_sim')

    return curr_sim


# class Fuse(torch.nn.Module):
#     def __init__(self, sims):
#         super().__init__()
#         self.sims = sims
#         self.weight = torch.nn.Parameter(torch.randn(len(sims)), requires_grad=True)
#
#     def forward(self, pairs):


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
    parser.add_argument("--val", type=float, default=0.0, help="valid set rate")
    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding")
    parser.add_argument("--pre", default="", help="pre-train embedding dir (only use in transr)")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--check", type=int, default=5, help="check point")
    parser.add_argument("--update", type=int, default=5, help="number of epoch for updating negtive samples")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early", action="store_true", default=False,
                        help="whether to use early stop")  # Early stop when the Hits@1 score begins to drop on the validation sets, checked every 10 epochs.
    parser.add_argument("--share", action="store_true", default=False, help="whether to share ill emb")
    parser.add_argument("--swap", action="store_true", default=False, help="whether to swap ill in triple")

    parser.add_argument("--bootstrap", action="store_true", default=False, help="whether to use bootstrap")
    parser.add_argument("--start_bp", type=int, default=9, help="epoch of starting bootstrapping")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold of bootstrap alignment")

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--hiddens", type=str, default="100,100,100",
                        help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--attn_drop", type=float, default=0, help="dropout rate for gat layers")

    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use: . min = 1")
    parser.add_argument("--sampling", type=str, default="N", help="negtive sampling method for each decoder")
    parser.add_argument("--k", type=str, default="25", help="negtive sampling number for each decoder")
    parser.add_argument("--margin", type=str, default="1",
                        help="margin for each margin based ranking loss (or params for other loss function)")
    parser.add_argument("--alpha", type=str, default="1", help="weight for each margin based ranking loss")
    parser.add_argument("--feat_drop", type=float, default=0, help="dropout rate for layers")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")

    parser.add_argument("--train_dist", type=str, default="euclidean",
                        help="distance function used in train (inner, cosine, euclidean, manhattan)")
    parser.add_argument("--test_dist", type=str, default="euclidean",
                        help="distance function used in test (inner, cosine, euclidean, manhattan)")

    parser.add_argument("--csls", type=int, default=0, help="whether to use csls in test (0 means not using)")
    parser.add_argument("--rerank", action="store_true", default=False, help="whether to use rerank in test")

    # My arguments
    parser.add_argument('--dataset', type=str, default='small')
    parser.add_argument('--lang', type=str, default='fr')
    parser.add_argument('--phase', type=int, default=4)
    parser.add_argument('--src_split', type=int, default=-1)
    parser.add_argument('--trg_split', type=int, default=-1)
    parser.add_argument('--topk_corr', type=int, default=1)
    parser.add_argument('--it_round', type=int, default=1)
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs to train")
    parser.add_argument('--model', type=str, default='rrea')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--openea', action='store_true', default=False)
    parser.add_argument('--unsup', action='store_true', default=False)
    parser.add_argument('--eval_which', type=str, default='sgtnf')
    parser.add_argument("--random_split", action="store_true", default=False, help="whether to use random split")
    parser.add_argument("--save_folder", type=str, default='tmp4')
    return parser.parse_args()


default_save_prefix = ''
save_prefix = ''

default_folder = 'tmp4/'


def no_prefix_sim_name(phase, data, lang, which, is_train):
    return default_folder + default_save_prefix + 'sim.phase.{0}.data.{1}.lang.{2}.src.{3}.train.{4}'. \
        format(phase, data, lang, which, is_train)


def sim_filename(phase, data, lang, which, is_train):
    return default_folder + save_prefix + 'sim.phase.{0}.data.{1}.lang.{2}.src.{3}.train.{4}'. \
        format(phase, data, lang, which, is_train)


def save_sim_phase(sim, *args):
    torch.save(sim, sim_filename(*args))


def load_sims_phase(phases, *args):
    sims = []
    for p in phases:
        try:
            print('load', sim_filename(p, *args))
            sims.append(torch.load(sim_filename(p, *args)))
        except FileNotFoundError:
            print('load', no_prefix_sim_name(p, *args))
            try:
                sims.append(torch.load(no_prefix_sim_name(p, *args)))
            except FileNotFoundError:
                print('load failed')
                sims.append(None)
    return tuple(sims)


def get_semi_link(phases=[1, 3], param=[1., 0.05], *args):
    # args = list(args)
    # args[2] = 0
    x2y_fused = naive_sim_fuser(load_sims_phase(phases, *args), param, device='cpu')
    # args[2] = 1
    # y2x_fused = naive_sim_fuser(load_sims_phase(phases, *args), param, device='cpu')
    y2x_fused = x2y_fused
    x2y_fused = matrix_argmax(x2y_fused, 1)
    y2x_fused = matrix_argmax(y2x_fused, 0)
    mask = get_bi_mapping(x2y_fused, y2x_fused, [x2y_fused.numel(), y2x_fused.numel()])
    return torch.stack(
        [torch.arange(x2y_fused.numel()),
         x2y_fused]
    ).t()[mask].numpy()


def log_information(data: str, lang: str, phase: int, model: str = '', prefix='',
                    result='No result',
                    split: int = 0, random=False, semi_pairs: int = 0, epoch=0, openea=False):
    if phase == 1:
        model = 'BERT-maxpooling'
    elif phase == 2:
        model = 'FAISS-TopK'
    elif phase == 3:
        model = 'MinHash-LSH-Lev'
    if phase == 0:
        sampling = 'random' if random else 'metis'
        graph_based = '\n' + argprint(samping=sampling, k_parts=split, semi_pairs=semi_pairs, epoch=epoch)
    else:
        graph_based = ''
    with open(default_folder + '_results_on_{0}_{1}{2}'
            .format(data, lang, {False: "", True: "_OpenEA"}[openea]), 'a') as f:
        f.write('Run {4} {0} on {1}-{2} {3}\n'.format(model, data, lang, graph_based, prefix))
        f.write('Result is: {}\n'.format(result))
        f.write('Logs :{}\n'.format(argprint(**all_logs)))
        f.write('Time usages:\n{}'.format('\n'.join(time_logs)))
        f.write('\n_______________\n')


if __name__ == '__main__':
    begin = time.time()
    args = get_args()
    if args.src_split < 0:
        args.src_split = args.trg_split = dict(small=5, medium=10, large=20)[args.dataset]
    if args.epoch < 0:
        args.epoch = {'rrea': 100, 'gcn-align': 300}.get(args.model, 100)
    if args.unsup:
        default_folder = 'unsup/'
        add_logs('unsup', 'ok')
    try:
        d = EAData.load(default_folder + 'dataset_{0}_{1}'.format(args.dataset, args.lang))
    except:
        if args.dataset == 'large':
            d = LargeScaleEAData('../mkdata/', args.lang, False, unsup=args.unsup)
        else:
            d = OpenEAData('../OpenEA_dataset_v1.1/EN_{0}_{1}K_V1/'
                           .format(args.lang.upper(), dict(small='15', medium='100')[args.dataset]), unsup=args.unsup)
        d.save(default_folder + 'dataset_{0}_{1}'.format(args.dataset, args.lang))
    # args=
    update_time_logs('load_data')
    save_prefix = args.save_prefix
    phase = args.phase
    data = args.dataset
    lang = args.lang
    print(argprint(lang=lang, phase=phase, save_prefix=save_prefix, data=data, rdm=args.random_split))
    which = 0
    is_train = 0
    # train_candidates = SelectedCandidates(d.train, *d.ents)
    # all_candidates = [test_candidates, train_candidates]
    if args.openea:
        from run_openea import run_openea

        try:
            result = run_openea(args, d)
        except Exception as e:
            result = str(e)
        log_information(data, lang, -1, 'OpenEA-' + args.model, args.save_prefix, result, openea=True)
        exit(0)

    if phase == 0:
        try:
            semi = get_semi_link([2, 3], [1., 0.05], data, lang, which, 0)
            total_semi = len(semi)
            d.train = semi

        except:
            total_semi = 0
        update_time_logs('get_semi_link')
        print('total semi pairs', total_semi)
        stru_sim = run_batched_ea(d, args.src_split, args.trg_split, args.topk_corr, args)
        save_sim_phase(stru_sim, phase, data, lang, which, 0)
        update_time_logs('save_curr_sim')
        result = sparse_acc(stru_sim, d.ill(d.test, 'cpu'))
        print('acc is', result)
        log_information(data, lang, phase, args.model, args.save_prefix, result, args.src_split, args.random_split,
                        total_semi, args.epoch)
        exit(0)

    if phase == 1:
        candidates = SelectedCandidates(d.test, *d.ents)
        update_time_logs('build_candidates')
        torch.save((candidates, text.get_bert_maxpooling_embs(*candidates.ents)),
                   default_folder + 'bert_result_data_{0}_lang_{1}_train_{2}'.format(data, lang, is_train))

        update_time_logs('bert_embedding_of_names')

        log_information(data, lang, phase, prefix=args.save_prefix)
        exit(0)

    if phase in range(2, 4):

        # for is_train, candidates in enumerate(all_candidates):
        if phase == 2:
            candidates, bert_result = torch.load(
                default_folder + 'bert_result_data_{0}_lang_{1}_train_{2}'.format(data, lang, is_train))
            sim = text.global_level_semantic_sim(bert_result)
            update_time_logs('faiss_topk_search')
            # sim = text.sparse_semantic_sim(*text.get_ent_token_info(*candidates.ents), filter_token_cnt=500)
        elif phase == 3:
            candidates = SelectedCandidates(d.test, *d.ents)
            update_time_logs('build_candidates')
            sim = text.sparse_string_sim(*candidates.ents)
            update_time_logs('build_sparse_string_sim')
        sim = candidates.convert_sim_mat(sim)
        update_time_logs('convert_sim')
        save_sim_phase(sim, phase, data, lang, which, is_train)
        update_time_logs('save_sim')
        del candidates
        # try:
        #     result = sparse_top_k(sim, d.ill(d.test))
        # except:
        result = sparse_acc(sim, d.ill(d.test))
        update_time_logs('get_hits_mrr')
        log_information(data, lang, phase, prefix=args.save_prefix, result=result)

    if phase == 4:
        # stru_sim, global_sim, token_sim, string_sim = load_sims_phase(range(phase), data, lang, which, 0)
        # stru_sim = sparse_softmax(sparse_softmax(stru_sim) +
        #                           ind2sparse(d.ill(d.train, 'cpu'), stru_sim.size(), dtype=stru_sim.dtype))
        # token_sim = sparse_softmax(token_sim)

        global_sim, string_sim, stru_sim = load_sims_phase([2, 3, 0], data, lang, which, 0)
        update_time_logs('load_sims')
        if stru_sim is not None:
            candidates = SelectedCandidates(d.test, *d.ents)
            stru_sim = candidates.filter_sim_mat(stru_sim)
        update_time_logs('filter_sim_mat')
        global_sim = global_sim
        string_sim = string_sim * 0.05
        stru_sim = stru_sim
        sims = [global_sim, string_sim]
        fused_name = naive_sim_fuser(sims, device='cpu')
        update_time_logs('fuse_name_sims')
        fused_all = naive_sim_fuser([stru_sim, fused_name], device='cpu')
        # fused = weighted_sim_fuser(d.ill(d.train), d.ill(d.test), sims, device='cpu')
        # sims = [global_sim, token_sim, stru_sim, string_sim, fused]
        # sims = [stru_sim, global_sim, string_sim, fused]
        sims = dict(stru_sim_sel=stru_sim, global_sim_sel=global_sim,
                    text_sim_sel=string_sim, name_sim_sel=fused_name,
                    fused_all_sel=fused_all)
        for key, sim in sims.items():
            if sim is not None and key[0] in args.eval_which:
                # print(result)
                result = sparse_top_k(sim, d.ill(d.test, 'cpu'))
                update_time_logs('get_hits_mrr_of_{}'.format(key))
                log_information(data, lang, phase, key, args.save_prefix, result)
    print('phase {0} complete, time consumed:{1}'.format(phase, time.time() - begin))
# 2900s 46.1g
# 848.5645837783813s 44g
# 1016.9110007286072
# 6918.654584407806
