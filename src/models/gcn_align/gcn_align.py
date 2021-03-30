from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf

from .utils import *
from .metrics import *
from .models import GCN_Align


class GCNAlignWrapper:
    def __init__(self, **kwargs):
        # Set random seed
        seed = 12306
        np.random.seed(seed)
        tf.set_random_seed(seed)
        tf.disable_eager_execution()
        # Settings
        # flags = tf.app.flags
        # FLAGS = seFLAGS
        # self.DEFINE_string('lang', 'zh_en', 'Dataset string.')  # 'zh_en', 'ja_en', 'fr_en'
        # self.DEFINE_float('learning_rate', 20, 'Initial learning rate.')
        # self.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
        # self.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
        # self.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
        # self.DEFINE_integer('k', 5, 'Number of negative samples for each positive seed.')
        # self.DEFINE_float('beta', 0.9, 'Weight for structure embeddings.')
        # self.DEFINE_integer('se_dim', 200, 'Dimension for SE.')
        # self.DEFINE_integer('ae_dim', 100, 'Dimension for AE.')
        # self.DEFINE_integer('seed', 3, 'Proportion of seeds, 3 means 30%')

        # Load data
        # adj, ae_input, train, test = load_data(self.lang)
        self.init_inputs(**kwargs)
        self.k = 5
        self.dim = 200
        self.dropout = 0.
        self.gamma = 3.0
        self.lr = 20
        # Some preprocessing
        self.support = [preprocess_adj(self.adj)]
        num_supports = 1
        model_func = GCN_Align
        k = self.k
        # e = ae_input[2][0]

        # Define placeholders
        # ph_ae = {
        #     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        #     'features': tf.sparse_placeholder(tf.float32),  # tf.placeholder(tf.float32),
        #     'dropout': tf.placeholder_with_default(0., shape=()),
        #     'num_features_nonzero': tf.placeholder_with_default(0, shape=())
        # }
        self.ph_se = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder_with_default(0, shape=()),
            'ill': tf.placeholder(tf.int32, shape=(None, 2)),
            't': tf.placeholder(tf.float32),
        }

        # Create model
        # model_ae = model_func(ph_ae, input_dim=ae_input[2][1], output_dim=self.ae_dim, ILL=train, sparse_inputs=True,
        #                       featureless=False, logging=True)
        self.model_se = GCN_Align(self.ph_se, input_dim=self.ent_total, output_dim=self.dim, ILL=None,
                                  sparse_inputs=False,
                                  featureless=True,
                                  logging=True,
                                  gamma=self.gamma,
                                  k=self.k,
                                  lr=self.lr)
        # Initialize session
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        self.sess = sess
        pass

    def init_inputs(self, **kwargs):

        self.ent_nums = kwargs.get('ent_sizes', [15000, 15000])
        self.rel_nums = kwargs.get('rel_sizes', [500, 500])
        self.ent_total = sum(self.ent_nums)
        self.rel_total = sum(self.rel_nums)
        self.ent_split = self.ent_nums[0]
        self.rel_split = self.rel_nums[0]
        # ei, et = list(kwargs['ei']), list(kwargs['et'])
        # ei[1] = ei[1] + self.ent_split
        # et[1] = et[1] + self.rel_split
        # import torch
        # triples = torch.cat([torch.stack([ei[i][0], et[i], ei[i][1]]).t() for i in range(2)], dim=0).cpu().numpy()
        triples = np.array(kwargs['triples'])
        self.adj = get_weighted_adj(self.ent_total, triples)
        self.update_devset(kwargs['link'].cpu().numpy())

    def convert_ent_id(self, ent_ids, which=0):
        # if which == 1:
        #     # return [val + self.ent_sizes[0] for val in ent_ids]
        #     return ent_ids + self.ent_split
        # else:
        return ent_ids

    def convert_rel_id(self, rel_ids, which=0):
        # if which == 1:
        #     # return [val + self.ent_sizes[0] for val in ent_ids]
        #     return rel_ids + self.ent_split
        # else:
        return rel_ids

    def append_pairs(self, old_pair, new_pair):
        px, py = set(), set()
        for e1, e2 in old_pair:
            px.add(e1)
            py.add(e2)
        filtered = []
        for e1, e2 in new_pair:
            if e1 not in px and e2 not in py:
                filtered.append([e1, e2])
        return np.concatenate([np.array(filtered), old_pair], axis=0)

    def update_trainset(self, pairs, append=False):
        trainset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]

        curr_pair = np.array(trainset).T
        self.train_pair = self.append_pairs(self.train_pair, curr_pair) if append else curr_pair
        print('srs-iteration-update-train-pair')

    def update_devset(self, pairs):
        devset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]
        self.dev_pair = np.array(devset).T

    def get_curr_embeddings(self, device=None):
        import torch
        from utils import apply
        feed_dict_se = construct_feed_dict(1.0, self.support, self.ph_se, train=self.train_pair)

        vec_se = self.sess.run(self.model_se.outputs, feed_dict=feed_dict_se)
        vec = np.array(vec_se)
        sep = self.ent_split
        vecs = vec[:sep], vec[sep:]
        vecs = apply(torch.from_numpy, *vecs)

        # tf.compat.v1.disable_eager_execution()
        return vecs if device is None else apply(lambda x: x.to(device), *vecs)

    def train1step(self, epoch):

        cost_val = []
        train = self.train_pair
        k = self.k
        e = self.ent_total
        sess = self.sess
        ph_se = self.ph_se
        support = self.support
        model_se = self.model_se
        t = len(train)
        L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
        neg_left = L.reshape((t * k,))
        L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
        neg2_right = L.reshape((t * k,))

        # Train model
        for epoch in range(epoch):
            if epoch % 10 == 0:
                neg2_left = np.random.choice(e, t * k)
                neg_right = np.random.choice(e, t * k)
            # Construct feed dictionary
            # feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
            # feed_dict_ae.update({ph_ae['dropout']: self.dropout})
            # feed_dict_ae.update(
            #     {'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left,
            #      'neg2_right:0': neg2_right})
            feed_dict_se = construct_feed_dict(1.0, support, ph_se, train=train)
            feed_dict_se.update({self.ph_se['dropout']: self.dropout})
            feed_dict_se.update(
                {'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left,
                 'neg2_right:0': neg2_right})
            # Training step
            # outs_ae = sess.run([model_ae.opt_op, model_ae.loss], feed_dict=feed_dict_ae)
            outs_se = sess.run([model_se.opt_op, model_se.loss], feed_dict=feed_dict_se)
            # cost_val.append((outs_ae[1], outs_se[1]))

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "SE_train_loss=",
                  "{:.5f}".format(outs_se[1]))

        print("Optimization Finished!")
        pass
