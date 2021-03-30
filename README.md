# LargeEA : Aligning Entities between Large-scale Knowledge Graphs [Scalable Data Science]

If there is any problem on reproduction, please create an issue in [The GitHub repo page](https://github.com/ZJU-DBL/LargeEA)


## Requirements

pytorch>=1.7.0

tensorflow>=2.4.1 (required for RREA)

faiss

transformers

datasketch[redis]

...

A full list of required packages is located in ``src/requirements.txt``

## Datasets 
The OpenEA dataset is provided by [OpenEA](https://github.com/nju-websoft/OpenEA)

Our DBP1M dataset is avaliable at [Google Drive](https://drive.google.com/file/d/15jeGD-6pVGlqI5jCn7KJfGIER6AeoQ-L/view?usp=sharing)

First download and unzip dataset files, place them to the project root folder:

    unzip OpenEA_dataset_v1.1.zip
    unzip mkdata.zip


The __dataset__ (small for IDS15K, medium for IDS100K, large for DBP1M) and  __lang__ (fr or de)parameter controls which benchmark to use.
For example, in src folder

    python main.py --dataset small --lang de

runs on OpenEA EN_FR_15K_V1 dataset.

## Run

Make sure the folder for results is created:

    cd src/
    mkdir tmp4

### Name Channel

First get the BERT embeddings of all entities

    python main.py --phase 1 --dataset large --lang fr 

Then calculate TopK sims based on BERT:

    python main.py --phase 2 --dataset large --lang fr 

Finally the string-based similarity(this requires a redis server listening  localhost:6379):

    python main.py --phase 3 --dataset large --lang fr 

### Structure Channel

The structure channel uses result of name channel to get name-based seeds. Make sure run name channel first.

To run RREA model: 

    python main.py --phase 0 --dataset large --lang fr --model rrea --epoch 100 


### Channel Fusion and Eval

    python main.py --phase 4  --dataset large --lang fr 


## Acknowledgements

We use the code of 
[MRAEA](https://github.com/MaoXinn/MRAEA),
[RREA](https://github.com/MaoXinn/RREA), 
[GCN-Align](https://github.com/1049451037/GCN-Align),
[DGMC](https://github.com/rusty1s/deep-graph-matching-consensus),
[AttrGNN](https://github.com/thunlp/explore-and-evaluate),
[OpenEA](https://github.com/nju-websoft/OpenEA),
[EAKit](https://github.com/THU-KEG/EAKit),
[SimAlign](https://github.com/cisnlp/simalign).


We also provide the modified version of OpenEA in order to run experiments on RTX3090 GPU:[OpenEA-TF2](https://github.com/joker-xii/OpenEA-TF2)