import json
import tensorflow as tf
import numpy as np
from model import Model

def main(config):
    with tf.device(config.device):
        if config.mode == 'train':
            train(config)
        
def train(config):
    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, 'train', data_filter=data_filter)
    update_config(config, train_data)

    _config_debug(config)

    word2vec_dict = train_data.shared['lower_word2vec']
    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat

    with tf.device('/gpu:0'):
        model = Model(config)



def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2

def update_config(config, data_set):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_para_size = 0
    data=data_set.data
    shared=data_set.shared
    for idx in data_set.valid_idxs:
        rx=data['*x'][idx]
        q=data['q'][idx]
        sents=shared['x'][rx[0]][rx[1]]
        config.max_para_size = max(config.max_para_size, sum(map(len, sents)))
        config.max_num_sents = max(config.max_num_sents, len(sents))
        config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
        config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
        if len(q) > 0:
            config.max_ques_size = max(config.max_ques_size, len(q))
            config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    if config.mode == 'train':
        config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
        config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
        config.max_para_size = min(config.max_para_size, config.para_size_th)
    config.max_word_size = min(config.max_word_size, config.word_size_th)
    config.word_emb_size = len(next(iter(data_set.shared['word2vec'].values())))
    config.word_vocab_size = len(data_sets.shared['word2idx'])

class DataSet:        
    def __init__(self,data,data_type,shared,valid_idxs):
        self.data = data
        self.data_tpye=data_type
        self.shared=shared
        total_num_examples=len(iter(next(self.data.values())))
        self.valid_idxs=range(total_num_examples)
        self.num_examples=len(self.valid_idxs)
    
def read_data(config, data_type, data_filter=None):
    with open(r'C:\Users\User\CS224n\bi-att-flow-master\data\squad\shared_train.json','r')as f:
        shared=json.load(f)
    with open(r'C:\Users\User\CS224n\bi-att-flow-master\data\squad\data_train.json','r')as f:
        data=json.load(f)

    num_examples = len(next(iter(data.values())))
    mask = []
    keys = data.keys()
    values = data.values()
    for vals in zip(*values):
        each = {key: val for key,val in zip(keys, vals)}
        mask.append(data_filter(each, shared))
    valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from train".format(len(valid_idxs), num_examples))

    word2vec_dict = shared['lower_word2vec']
    word_counter = shared['lower_word_counter']
    shared['word2idx'] = {word:idx+2 for idx, word in enumerate(word for word, count in word_counter.items()
                                                                if count > config.word_count_th and word not in word2vec_dict)}
    shared['word2idx']['-NULL-'] = 0
    shared['word2idx']['-UNK-'] = 1

    #create new word2idx and word2vec
    new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
    shared['new_word2idx'] = new_word2idx_dict
    idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
    new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
    shared['new_emb_mat'] = new_emb_mat

    data_set = DataSet(data, 'train', shared = shared, valid_idxs = valid_idxs)
    return data_set
    
      
def get_squad_data_filter(config):
    def data_filter(data_point,shared):
        rx, q, y = (data_point[key] for key in ('*x','q','y'))
        x = shared['x']
        
        for start, stop in y:
            if start[0] >= config.num_sents_th:
                return False
            if start[0] != stop[0]:
                return False
            if stop[1] >= config.sent_size_th:
                return False

        return True
    return data_filter



