import numpy as np
import tensorflow as tf
from general import get_initializer

class Model:
    def __init__(self,config):
        self.config = config

        batch_size, max_num_sents, max_sent_size, max_ques_size, word_vocab_size,\
                    max_word_size = config.batch_size, config.max_num_sents,\
                    config.max_sent_size, config.max_ques_size, config.word_vocab_size,\
                    config.max_word_size
        
        #forward inputs
        self.x = tf.placeholder(dtype='int32',[batch_size,None,None],name='x')
        self.x_mask = tf.placeholder(dtype='bool',[batch_size,None,None],name='x_mask')
        self.q = tf.placeholder(dtype='int32',[batch_size,None,None],name='q')
        self.q_mask = tf.placeholder(dtype='bool',[batch_size,None,None],name='q_mask')
        self.y = tf.placeholder(dtype='bool',[batch_size,None,None],name='y')
        self.y2 = tf.placeholder(dtype='bool',[batch_size,None,None],name='y2')
        self.new_emb_mat = tf.placeholder(dtype='float',[None, config.word_emb_size],name='new_emb_mat')
        
        #define dict
        self.tensor_dict = {}

        #forward outputs
        self.logits = None

        #loss outputs
        self.loss = None

        self.build_forward()
        self.build_loss()

    def build_forward(self):
        config = self.config
        JX = tf.shape(self.x)[2] #max_sent_size: most words in a context
        JQ = tf.shape(self.q)[1] #max_ques_size: most words in a question
        M =tf.shape(self.x)[1] #max_num_sents: 1
        dw = config.word_emb_size

        with tf.variable_scope('emb'):
            with tf.variable_scope('emb_var'), tf.device('/cpu:0'):
                if config.mode == 'train':
                    word_emb_mat = tf.get_variable(name='word_emb_mat',dtype='float',shape=[config.word_vocab_size,dw],initializer=get_initializer(config.emb_mat))

                word_emb_mat = tf.concat([word_emb_mat,new_emb_mat],axis=0)

            with tf.name_scope('word'):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)
                self.tensor_dict['x'] = Ax ##[batch_size, max_num_sents, max_sent_size, d]
                self.tensor_dict['q'] = Aq ##[batch_size, max_ques_size, d]
            xx = Ax
            qq = Aq
            self.tensor_dict['xx'] = xx
            self.tensor_dict['qq'] = qq

            cell = tf.contrib.rnn.BasicLSTMCell(d)
            x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), axis=2)
            q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), axis=1)

        with tf.variable_scope('contextualizing'):
            q_inputs = flatten(qq,2)
            q_seq_len = tf.cast(flatten(q_len, 0), 'int64')
            x_inputs = flatten(xx,2)
            x_seq_len = tf.cast(flatten(x_len ,0), 'int64')
            fw_state, bw_state = cell.zero_state(batch_size, dtype=tf.float32),cell.zero_state(batch_size, dtype=tf.float32)
            (fw_u,bw_u),(output_fw,output_bw) = tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=q_inputs,sequence_length=q_seq_len,initial_state_fw=fw_state,\
                                                                                initial_state_bw=bw_state)
            tf.get_variable_scope().reuse_variables()
            (fw_x,bw_x),(output_fw_x,output_bw_x) = tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=x_inputs,sequence_length=x_seq_len,initial_state_fw=output_fw,\
                                                                                initial_state_bw=output_bw)
            Q_t = tf.concat([fw_u,bw_u],axis=2)
            D_t = tf.concat([fw_x,bw_x],axis=2)
            Q = tf.concat([Q_t,tf.zeros([batch_size,1,Q_t.shape[2]])],axis=1) ##[N,JQ+1,h]
            D = tf.concat([D_t,tf.zeros([batch_size,1,D_t.shape[2]])],axis=1) ##[N,JX+1,h]

            Q = tf.reshape(Q,[batch_size*Q.shape[1],Q.shape[2]])
            D = tf.reshape(D,[batch_size*D.shape[1],D.shape[2]])

        with tf.variable_scope('Weight'):
            b = tf.get_variable('b', [2*config.hidden_size,], initializer=tf.contrib.layers.xavier_initializer())
            wq = tf.get_variable('wq',[2*config.hidden_size,2*config.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            
            Q = tf.matmul(Q,wq)+b
            Q = tf.tanh(Q) ##[N*(JQ+1),h]

            wl = tf.get_variable('wl',[2*config.hidden_size,2*config.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            L = tf.matmul(D,wl)
            L = tf.matmul(L,Q,transpose_b=True)

            L_Q = tf.reshape(L,[L.shape[0],batch_size,-1]) ##[N*(JX+1),N,JQ+1]
            AQ = tf.nn.softmax(L_Q) ##[N*(JX+1),N,JQ+1]
            AQ = tf.reshape(AQ,[L.shape[0],-1])
            
            L_D = tf.transpose(L)
            L_D = tf.reshape(L_D,[L.shape[0],batch_size,-1]) ##[N*(JQ+1),N,JX+1]
            AD = tf.nn.softmax(L_D) ##[N*(JQ+1),N,JX+1]
            AD = tf.reshape(AD,[L.shape[0],-1])
            
            CQ = tf.matmul(D,AQ,transpose_a=True) ##[h,N*(JQ+1)]
            CD = tf.concat([tf.transpose(Q),CQ],axis=0) 
            CD = tf.matmul(CD,AD) ##[2h,N*(JX+1)]
            CD = tf.concat([tf.transpose(D),CD],axis=0)
            
                
