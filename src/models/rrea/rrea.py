# %%

import warnings

warnings.filterwarnings('ignore')

import keras
from tqdm import *
from .utils import *
from .CSLS import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from .layer import NR_GraphAttention
from .mraea.model import get_mraea_model
from utils import *
from tensorflow.python.client import device_lib

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.disable_v2_behavior()

# print(device_lib.list_local_devices())
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_WARNINGS"] = 'off'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# your code

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


class TFModelWrapper(object):
    def __init__(self, name='rrea', lang='en_fr', **kwargs):
        self.ent_sizes = kwargs['ent_sizes']
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = sess
        # self.epoch = epoch
        # %%

        self.lang = lang
        # self.construct_adj(pth=pth)
        self.construct_adj(**kwargs)
        self.load_pair(**kwargs)
        self.train_pair = []
        # %%

        self.node_size = self.adj_features.shape[0]
        self.rel_size = self.rel_features.shape[1]
        self.triple_size = len(self.adj_matrix)
        self.batch_size = self.node_size
        default_params = dict(
            dropout_rate=0.30,
            node_size=self.node_size,
            rel_size=self.rel_size,
            n_attn_heads=1,
            depth=2,
            gamma=3,
            node_hidden=100,
            rel_hidden=100,
            triple_size=self.triple_size,
            batch_size=self.batch_size
        )

        get_model = dict(rrea=self.get_trgat, mraea=get_mraea_model)
        self.model, self.get_emb = get_model[name](**default_params)
        # self.model.summary()
        self.initial_weights = self.model.get_weights()

    def load_pair(self, **kwargs):
        self.update_devset(kwargs['link'].cpu().numpy())

    def construct_adj(self, triples=None, ent_sizes=None, rel_sizes=None, **kwargs):

        # ei = apply(lambda x: x.t().cpu().numpy(), *ei)
        # et = apply(lambda x: x.cpu().numpy(), *et)
        # triples = []
        # ent_begin, rel_begin = 0, 0
        # for i, edge_index in enumerate(ei):
        #     rel = et[i]
        #     triples += [(ht[0] + ent_begin, rel[j] + rel_begin, ht[1] + ent_begin)
        #                 for j, ht in enumerate(edge_index)]
        #     ent_begin += ent_sizes[i]
        #     rel_begin += rel_sizes[i]
        entsz = ent_sizes[0] + ent_sizes[1]
        relsz = rel_sizes[0] + rel_sizes[1]
        adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples, entsz, relsz)

        # return
        self.adj_matrix, self.r_index, self.r_val, self.adj_features, self.rel_features = \
            adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features
        self.adj_matrix = np.stack(self.adj_matrix.nonzero(), axis=1)
        self.rel_matrix, self.rel_val = np.stack(self.rel_features.nonzero(), axis=1), self.rel_features.data
        self.ent_matrix, self.ent_val = np.stack(self.adj_features.nonzero(), axis=1), self.adj_features.data

    def convert_ent_id(self, ent_ids, which=0):

        return ent_ids
        # if which == 1:
        #     # return [val + self.ent_sizes[0] for val in ent_ids]
        #     return ent_ids + self.ent_sizes[0]
        # else:
        #     return ent_ids

    def convert_rel_id(self, rel_ids, which=0):
        raise NotImplementedError()

    def append_pairs(self, old_pair, new_pair):
        if len(old_pair) == 0:
            return new_pair
        px, py = set(), set()
        for e1, e2 in old_pair:
            px.add(e1)
            py.add(e2)
        filtered = []
        for e1, e2 in new_pair:
            if e1 not in px and e2 not in py:
                filtered.append([e1, e2])
        if len(filtered) == 0:
            return old_pair
        filtered = np.array(filtered)
        return np.concatenate([filtered, old_pair], axis=0)

    def update_trainset(self, pairs, append=False):
        trainset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]

        curr_pair = np.array(trainset).T
        if append:
            if append == 'REVERSE':
                self.train_pair = self.append_pairs(curr_pair, self.train_pair)
            else:
                self.train_pair = self.append_pairs(self.train_pair, curr_pair)
        else:
            self.train_pair = curr_pair
        # print('srs-iteration-update-train-pair')

    def update_devset(self, pairs):
        # pairs = [pairs[0], pairs[1]]
        devset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]
        self.dev_pair = np.array(devset).T

    def get_curr_embeddings(self, device=None):
        vec = self.get_embedding()
        vec = np.array(vec)
        sep = self.ent_sizes[0]
        vecs = vec[:sep], vec[sep:]
        vecs = apply(torch.from_numpy, *vecs)
        tf.compat.v1.reset_default_graph()
        return vecs if device is None else apply(lambda x: x.to(device), *vecs)

    def train1step(self, epoch=75):
        # self.test_train_pair_acc()
        # print("iteration %d start." % turn)
        verbose = 20
        if epoch > 100:
            verbose = epoch // 5
        for i in range(epoch):
            train_set = self.get_train_set()
            inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, train_set]
            inputs = [np.expand_dims(item, axis=0) for item in inputs]
            self.model.train_on_batch(inputs, np.zeros((1, 1)))
            # if (i + 1) % verbose == 0:
            #     self.CSLS_test()
        self.CSLS_test()

    def mraea_iteration(self):
        print('mraea-iteration-update-train-pair')
        vec = self.get_embedding()
        # np.random.shuffle(rest_set_1)
        rest_set_1 = [e1 for e1, e2 in self.dev_pair]
        rest_set_2 = [e2 for e1, e2 in self.dev_pair]
        # np.random.shuffle(rest_set_2)
        for e1, e2 in self.train_pair:
            e1, e2 = int(e1), int(e2)
            if e1 in rest_set_1:
                rest_set_1.remove(e1)
            if e2 in rest_set_2:
                rest_set_2.remove(e2)

        new_pair = []
        Lvec = np.array([vec[e] for e in rest_set_1])
        Rvec = np.array([vec[e] for e in rest_set_2])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
        B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
        A = sorted(list(A))
        B = sorted(list(B))

        for a, b in A:
            if B[b][1] == a:
                new_pair.append([rest_set_1[a], rest_set_2[b]])
        print("generate new semi-pairs: %d." % len(new_pair))

        self.train_pair = np.concatenate([self.train_pair, np.array(new_pair)], axis=0)

    def test_train_pair_acc(self):
        pred = set((int(k), int(v)) for k, v in self.train_pair)
        actual = set((int(k), int(v)) for k, v in self.dev_pair)
        print('train pair={0}, dev pair={1}'.format(len(self.train_pair), len(self.dev_pair)))

        tp = len(pred.intersection(actual))
        fp = len(pred.difference(actual))
        fn = len(actual.difference(pred))
        # 😀
        print("tp={0}, fp={1}, fn={2}".format(tp, fp, fn))
        prec = float(tp) / float(tp + fp)
        recall = float(tp) / float(tp + fn)
        f1 = 2 * prec * recall / (prec + recall)
        print('prec={0}, recall={1}, f1={2}'.format(prec, recall, f1))
        return {
            'precision': prec,
            'recall': recall,
            'f1-score': f1,
            'confusion_matrix': (tp, fp, fn)
        }

    def get_embedding(self):
        inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        return self.get_emb.predict_on_batch(inputs)

    def test(self, wrank=None):
        vec = self.get_embedding()
        return get_hits(vec, self.dev_pair, wrank=wrank)

    def CSLS_test(self, thread_number=16, csls=10, accurate=True):
        if len(self.dev_pair) == 0:
            print('EVAL--No dev')
            return
        vec = self.get_embedding()
        Lvec = np.array([vec[e1] for e1, e2 in self.dev_pair])
        Rvec = np.array([vec[e2] for e1, e2 in self.dev_pair])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
        return None

    def get_train_set(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        negative_ratio = batch_size // len(self.train_pair) + 1
        train_set = np.reshape(np.repeat(np.expand_dims(self.train_pair, axis=0), axis=0, repeats=negative_ratio),
                               newshape=(-1, 2))
        np.random.shuffle(train_set)
        train_set = train_set[:batch_size]
        train_set = np.concatenate([train_set, np.random.randint(0, self.node_size, train_set.shape)], axis=-1)
        return train_set

    def get_trgat(self, node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0.,
                  gamma=3, lr=0.005, depth=2, **kwargs):
        adj_input = Input(shape=(None, 2))
        index_input = Input(shape=(None, 2), dtype='int64')
        val_input = Input(shape=(None,))
        rel_adj = Input(shape=(None, 2))
        ent_adj = Input(shape=(None, 2))

        ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
        rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

        def avg(tensor, size):
            adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
            adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                                  dense_shape=(node_size, size))
            adj = tf.compat.v1.sparse_softmax(adj)
            return tf.compat.v1.sparse_tensor_dense_matmul(adj, tensor[1])

        opt = [rel_emb, adj_input, index_input, val_input]
        ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
        rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])

        encoder = NR_GraphAttention(node_size, activation="relu",
                                    rel_size=rel_size,
                                    depth=depth,
                                    attn_heads=n_attn_heads,
                                    triple_size=triple_size,
                                    attn_heads_reduction='average',
                                    dropout_rate=dropout_rate)

        out_feature = Concatenate(-1)([encoder([ent_feature] + opt), encoder([rel_feature] + opt)])
        out_feature = Dropout(dropout_rate)(out_feature)

        alignment_input = Input(shape=(None, 4))
        find = Lambda(lambda x: K.gather(reference=x[0], indices=K.cast(K.squeeze(x[1], axis=0), 'int32')))(
            [out_feature, alignment_input])

        def align_loss(tensor):
            def _cosine(x):
                dot1 = K.batch_dot(x[0], x[1], axes=1)
                dot2 = K.batch_dot(x[0], x[0], axes=1)
                dot3 = K.batch_dot(x[1], x[1], axes=1)
                max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
                return dot1 / max_

            def l1(ll, rr):
                return K.sum(K.abs(ll - rr), axis=-1, keepdims=True)

            def l2(ll, rr):
                return K.sum(K.square(ll - rr), axis=-1, keepdims=True)

            l, r, fl, fr = [tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], tensor[:, 3, :]]
            loss = K.relu(gamma + l1(l, r) - l1(l, fr)) + K.relu(gamma + l1(l, r) - l1(fl, r))
            return tf.compat.v1.reduce_sum(loss, keep_dims=True) / self.batch_size

        loss = Lambda(align_loss)(find)
        inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
        train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
        train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.RMSprop(lr))

        feature_model = keras.Model(inputs=inputs, outputs=out_feature)
        return train_model, feature_model
