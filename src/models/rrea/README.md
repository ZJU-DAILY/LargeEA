# Relational Reflection Entity Alignment (https://arxiv.org/pdf/2008.07962.pdf)

## Datasets

The datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align).

* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;

## Environment

* Anaconda>=4.3.30
* Python>=3.5
* Keras>=2.2.4
* Tensorflow>=1.13.1
* Scipy
* Numpy
* tqdm

## Acknowledgement

We refer to the codes of these repos: keras-gat, GCN-Align, TransEdge. Thanks for their great contributions!