import os
import torch
from transformers import *

from tqdm import tqdm
from utils import *
import numpy as np

os.environ["KMP_WARNINGS"] = 'off'


# LOG = get_logger(__name__)


class EmbeddingLoader(object):
    TR_Models = {
        'bert-base-uncased': (BertModel, BertTokenizer),
        'bert-base-cased': (BertModel, BertTokenizer),
        'bert-base-multilingual-cased': (BertModel, BertTokenizer),
        'bert-base-multilingual-uncased': (BertModel, BertTokenizer),
        'xlm-mlm-100-1280': (XLMModel, XLMTokenizer),
        'roberta-base': (RobertaModel, RobertaTokenizer),
        'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer),
        'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer),
    }

    @staticmethod
    def pretrained_models(model):
        return EmbeddingLoader.TR_Models[model]

    @staticmethod
    def get_tokenizer(model: str, *args, **kwargs):
        return EmbeddingLoader.TR_Models[model][1].from_pretrained(*args, **kwargs)

    def __init__(self, model: str = "xlm-roberta-base", device=torch.device('cuda'), layer=1):
        self.model = model
        self.device = device
        self.layer = layer
        self.emb_model = None
        self.tokenizer = None
        TR_Models = self.TR_Models
        if model in TR_Models:
            model_class, tokenizer_class = TR_Models[model]
            self.emb_model = model_class.from_pretrained(model, output_hidden_states=True)
            self.emb_model.eval()
            self.emb_model.to(self.device)
            self.tokenizer = tokenizer_class.from_pretrained(model, do_lower_case=False)
            print("Initialized the EmbeddingLoader with model: {}".format(self.model))
        else:
            if os.path.isdir(model):
                # try to load model with auto-classes
                config = AutoConfig.from_pretrained(model, output_hidden_states=True)
                self.emb_model = AutoModel.from_pretrained(model, config=config)
                self.emb_model.eval()
                self.emb_model.to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model)
                print("Initialized the EmbeddingLoader from path: {}".format(self.model))
            else:
                raise ValueError("The model '{}' is not recognised!".format(model))

    def get_embed_list(self, sent_batch: List[List[str]], get_length=False) \
            -> Union[torch.Tensor, Tuple[Tensor, Tensor]]:
        if self.emb_model is not None:
            with torch.no_grad():
                if not isinstance(sent_batch[0], str):
                    inputs = self.tokenizer(sent_batch, is_split_into_words=True, padding=True, truncation=True,
                                            return_tensors="pt", return_length=get_length)
                    # print('---', inputs['input_ids'].size())
                else:
                    inputs = self.tokenizer(sent_batch, is_split_into_words=False, padding=True, truncation=True,
                                            return_tensors="pt", return_length=get_length)

                inputs = dict(inputs.to(self.device))
                if get_length:
                    lengths = inputs['length']
                    del inputs['length']
                if self.layer is None:
                    outputs = self.emb_model(**inputs)[0]
                else:
                    outputs = self.emb_model(**inputs)[2][self.layer]
                # return outputs
                if get_length:
                    return outputs[:, 1:-1, :], lengths
                else:
                    return outputs
        else:
            return None


def average_embeds_over_words(bpe_vectors: np.ndarray, word_tokens_pair: List[List[str]]) -> List[np.array]:
    print('average begin')
    w2b_map = []
    for i in range(len(word_tokens_pair)):
        cnt = 0
        w2b_map.append([])
        for wlist in word_tokens_pair[i]:
            w2b_map[-1].append([])
            for x in wlist:
                w2b_map[-1][-1].append(cnt)
                cnt += 1
    # cnt = 0
    # w2b_map.append([])
    # for wlist in word_tokens_pair[1]:
    #     w2b_map[1].append([])
    #     for x in wlist:
    #         w2b_map[1][-1].append(cnt)
    #         cnt += 1

    new_vectors = []
    for l_id in range(len(word_tokens_pair)):
        w_vector = []
        for word_set in w2b_map[l_id]:
            w_vector.append(bpe_vectors[l_id][word_set].sum(0))
        new_vectors.append(np.array(w_vector))
    print('average complete')
    return new_vectors


def minus_mask(inputs, input_lens, mask_type='max'):
    # Inputs shape = (batch_size, sent_len, embed_dim)
    # input_len shape = [batch_sie]
    # max_len scalar
    assert inputs.shape[0] == input_lens.shape[0]
    assert len(input_lens.shape) == 1
    assert len(inputs.shape) == 3
    device = inputs.device

    max_len = torch.max(input_lens)
    batch_size = inputs.shape[0]
    mask = torch.arange(max_len).expand(batch_size, max_len).to(device)
    mask = mask >= input_lens.view(-1, 1)
    mask = mask.float()
    mask = mask.reshape(-1, max_len, 1)
    if mask_type == 'max':
        mask = mask * 1e-30
        inputs = inputs + mask
    elif mask_type == "mean":
        inputs = inputs - mask * inputs
    return inputs


class BERT(object):
    # For entity alignment, the best layer is 1
    def __init__(self, model='bert-base-cased', pool='max'):
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=False)
        self.model = BertModel.from_pretrained(model, output_hidden_states=True)
        self.model.eval()
        self.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.cls_token_id = self.tokenizer.encode(self.tokenizer.cls_token)[0]
        self.sep_token_id = self.tokenizer.encode(self.tokenizer.sep_token)[0]
        self.device = 'cpu'
        self.pool = pool

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def pooled_encode_batched(self, sentences, batch_size=512, layer=1, save_gpu_memory=False):
        # Split the sentences into batches and further encode
        sent_batch = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        outputs = []
        for batch in tqdm(sent_batch):
            out = self.pooled_bert_encode(batch, layer)
            if save_gpu_memory:
                out = out.cpu()
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def pooled_bert_encode(self, sentences, layer=1):
        required_layer_hidden_state, sent_lens = self.bert_encode(sentences, layer)
        required_layer_hidden_state = minus_mask(required_layer_hidden_state, sent_lens.to(self.device), self.pool)
        # Max pooling
        # required_layer_hidden_state, indices = torch.max(required_layer_hidden_state, dim=1, keepdim=False)
        if self.pool == 'max':
            required_layer_hidden_state, indices = torch.max(required_layer_hidden_state, dim=1, keepdim=False)
        elif self.pool == 'mean':
            required_layer_hidden_state = torch.mean(required_layer_hidden_state, dim=1, keepdim=False)
        # print(indices.cpu().tolist())
        # input()
        return required_layer_hidden_state

    def bert_encode(self, sentences, layer=1):
        # layer: output the max pooling over the designated layer hidden state

        # Limit batch size to avoid exceed gpu memory limitation
        sent_num = len(sentences)
        # assert sent_num <= 512
        strs = sentences
        # The 382 is to avoid exceed bert's maximum seq_len and to save memory
        sentences = [[self.cls_token_id] + self.tokenizer.encode(sent)[:382] + [self.sep_token_id] for sent in
                     sentences]
        sent_lens = [len(sent) for sent in sentences]
        max_len = max(sent_lens)
        sent_lens = torch.tensor(sent_lens)
        sentences = torch.tensor([sent + (max_len - len(sent)) * [self.pad_token_id] for sent in sentences]).to(
            self.device)
        with torch.no_grad():
            result = self.model(sentences)
            if isinstance(result, tuple):
                last_hidden_state, _, all_hidden_state = result
            else:
                last_hidden_state, all_hidden_state = \
                    result["last_hidden_state"], result["hidden_states"]

        # assert len(all_hidden_state) == 13
        if layer is None:
            required_layer_hidden_state = last_hidden_state
        else:
            required_layer_hidden_state = all_hidden_state[layer]
        return required_layer_hidden_state, sent_lens
        # inputs = self.tokenizer(sentences, is_split_into_words=False, padding=True, truncation=True,
        #                         return_tensors="pt", return_length=True)
        # length = inputs['length']
        # inputs = dict(inputs.to(self.device))
        # del inputs['length']
        #
        # required_layer_hidden_state = self.model(**inputs)[2][layer]
        # return required_layer_hidden_state, length
