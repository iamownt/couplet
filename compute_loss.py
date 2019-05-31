# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:22:23 2019

@author: ownt
"""

te_path = r'D:\MS\other source\couplet_generation\data\couplet\train\in.txt'
ta_path = r'D:\MS\other source\couplet_generation\data\couplet\train\out.txt'

lis = []
with open(te_path, "r", encoding='utf-8') as source_file:
    c = source_file.readlines()

lis = []    
for source in c:
    source_ints = dataflow.encode_text(source.strip(), vocab_indices).append(1)
    lis.append(source_ints)






    tf.reset_default_graph()

    print('loading model...')
    in_seq_holder = tf.placeholder(tf.int32, shape=[1, None], name='in_seq')
    in_seq_len_holder = tf.placeholder(tf.int32, shape=[1], name='in_seq_len')
    test_model = model.Seq2Seq()
    vocabs = dataflow.read_vocab(dataflow.vocabular)
    vocab_indices = dict((c, i) for i, c in
                         enumerate(vocabs))
    voc_size = len(vocabs)
    test_model.build_infer(in_seq_holder, in_seq_len_holder,
                           voc_size,
                           dataflow.hidden_unit, dataflow.layers, name_scope='infer')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = model.Saver(sess)
    saver.load(dataflow.init_path, scope_name='infer', del_scope=True)

i = 0  
los = []
with open(te_path, "r", encoding='utf-8') as source_file:    
    with open(ta_path, "r", encoding='utf-8') as target_file:
        source, target = source_file.readline(), target_file.readline()
        while source and target:
            in_source_seq = dataflow.encode_text(source.strip(), vocab_indices)+[1]
            in_source_len = len(in_source_seq)
            in_target_seq = dataflow.encode_text(target.strip(), vocab_indices)
            in_target_len = len(in_target_seq)
            output = sess.run(test_model.abc,
                              feed_dict={
                                      in_seq_holder: [in_source_seq],
                                      in_seq_len_holder: [in_source_len]})
            
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output.rnn_output[0][:-1],
                                                                  labels=in_target_seq)
            loss = tf.reduce_mean(cost)
            i+=1
            los.append(sess.run(loss))
            source, target = source_file.readline(), target_file.readline()
            if i >= 100:
                break
    