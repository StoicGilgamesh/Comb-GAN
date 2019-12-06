#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:05:06 2018

@author: kaygudo
"""

import tensorflow as tf
import numpy as np

#a=np.random.dirichlet(np.ones(2),size=1)
#a=a.astype(np.float32)
#print("INITIAL DISTRIBUTION: ",a)

with tf.variable_scope("generator_comb") as scope:

    w1_comb = tf.get_variable(name='w1_comb', initializer=0.01,trainable=True)
    w1_comb=tf.cast(w1_comb,tf.float32)

    w2_comb = tf.get_variable(name='w2_comb', initializer=0.99,trainable=True)
    w2_comb=tf.cast(w2_comb,tf.float32)
    
def gen_comb(Z, hidden_units_g, seq_len, batch_size, num_generated_features):
    
    g_sample1 = generator1(Z,hidden_units_g, seq_len, batch_size, num_generated_features)
    g_sample2 = generator2(Z,hidden_units_g, seq_len, batch_size, num_generated_features)
    
    G_sample=tf.add(tf.multiply(w1_comb,g_sample1),tf.multiply(w2_comb,g_sample2))
    
    return G_sample


def sample_Z(batch_size, seq_length, latent_dim):
    
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))

    return sample


def GAN_loss_DISC( Z, hidden_units_g, g_sample1, g_sample2,X, hidden_units_d, seq_len, batch_size, num_generated_features):
    
    G_sample= gen_comb( Z, hidden_units_g, seq_len, batch_size, num_generated_features)
    
    D_real, D_logit_real  = discriminator(X, hidden_units_d, seq_len, batch_size)
    D_fake, D_logit_fake = discriminator(G_sample, hidden_units_d, seq_len, batch_size, reuse=True)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)

    D_loss = D_loss_real + D_loss_fake

    return D_loss,D_logit_fake


def GAN_loss_GEN(D_logit_fake):
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)
    return G_loss


def GAN_solvers(D_loss, G_loss, learning_rate):

    discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    generator_vars1 = [v for v in tf.trainable_variables() if v.name.startswith('generator1')]#################################################
    generator_vars2 = [v for v in tf.trainable_variables() if v.name.startswith('generator2')]#################################################
    generator_vars_comb = [v for v in tf.trainable_variables() if v.name.startswith('generator_comb')]

    D_loss_mean_over_batch = tf.reduce_mean(D_loss)
    D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss_mean_over_batch, var_list=discriminator_vars)
    
    G_loss_mean_over_batch = tf.reduce_mean(G_loss)
    G_solver1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss_mean_over_batch, var_list=generator_vars1)
    G_solver2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss_mean_over_batch, var_list=generator_vars2)###################################

    
    G_solver_comb = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss_mean_over_batch, var_list=generator_vars_comb[0])#################################
    
    return D_solver, G_solver1, G_solver2, G_solver_comb



###################### DIRTY CODE ###############################################################################
    
def generator1(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None,learn_scale=True):


    with tf.variable_scope("generator1") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_G_initializer = tf.truncated_normal_initializer()
            b_out_G_initializer = tf.truncated_normal_initializer()
            scale_out_G_initializer = tf.constant_initializer(value=1.0)
            lstm_initializer = None
            bias_start = 1.0
        else:
            W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
            b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
            try:
                scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
            except KeyError:
                scale_out_G_initializer = tf.constant_initializer(value=1)
                assert learn_scale
            lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
            bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)

        inputs = z

        cell=tf.nn.rnn_cell.LSTMCell(hidden_units_g,initializer=lstm_initializer,state_is_tuple=True,reuse=reuse)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length]*batch_size,
            inputs=inputs)
        
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
        
        
    return output_3d


def generator2(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None,learn_scale=True):


    with tf.variable_scope("generator2") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_G_initializer = tf.truncated_normal_initializer()
            b_out_G_initializer = tf.truncated_normal_initializer()
            scale_out_G_initializer = tf.constant_initializer(value=1.0)
            lstm_initializer = None
            bias_start = 1.0
        else:
            W_out_G_initializer = tf.constant_initializer(value=parameters['generator2/W_out_G:0'])
            b_out_G_initializer = tf.constant_initializer(value=parameters['generator2/b_out_G:0'])
            try:
                scale_out_G_initializer = tf.constant_initializer(value=parameters['generator2/scale_out_G:0'])
            except KeyError:
                scale_out_G_initializer = tf.constant_initializer(value=1)
                assert learn_scale
            lstm_initializer = tf.constant_initializer(value=parameters['generator2/rnn/lstm_cell/weights:0'])
            bias_start = parameters['generator2/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G2', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G2', shape=num_generated_features, initializer=b_out_G_initializer)

        inputs = z

        cell=tf.nn.rnn_cell.LSTMCell(hidden_units_g,initializer=lstm_initializer,state_is_tuple=True,reuse=reuse)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length]*batch_size,
            inputs=inputs)
        
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
        
        
    return output_3d

###################### DIRTY CODE ###############################################################################

def discriminator(x, hidden_units_d, seq_length, batch_size, reuse=False, batch_mean=False):
    
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                initializer=tf.truncated_normal_initializer())
        b_out_D = tf.get_variable(name='b_out_D', shape=1,
                initializer=tf.truncated_normal_initializer())

        inputs = x
        # add the average of the inputs to the inputs (mode collapse?
        if batch_mean:
            mean_over_batch = tf.stack([tf.reduce_mean(x, axis=0)]*batch_size, axis=0)
            inputs = tf.concat([x, mean_over_batch], axis=2)
        
        cell=tf.nn.rnn_cell.LSTMCell(hidden_units_d,state_is_tuple=True,reuse=reuse)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=inputs)
#        logit_final = tf.matmul(rnn_outputs[:, -1], W_final_D) + b_final_D
        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D

        output = tf.nn.sigmoid(logits)
        
    return output, logits


def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    if labels is None:
        return samples[start_pos:end_pos], None
    else:
        if type(labels) == tuple: # two sets of labels
            assert len(labels) == 2
            return samples[start_pos:end_pos], labels[0][start_pos:end_pos], labels[1][start_pos:end_pos]
        else:
            assert type(labels) == np.ndarray
            return samples[start_pos:end_pos], labels[start_pos:end_pos]



def train_epoch(epoch, samples, labels, sess, Z, X, D_loss, G_loss, D_solver, G_solver1, G_solver2,G_solver_comb,
                batch_size, D_rounds, G_rounds, seq_length, 
                latent_dim, num_generated_features, cond_dim):

    for batch_idx in range(0, int(len(samples) / batch_size) - (D_rounds + (cond_dim > 0)*G_rounds), D_rounds + (cond_dim > 0)*G_rounds):
        # update the discriminator
        for d in range(D_rounds):
            
            X_mb, Y_mb = get_batch(samples, batch_size, batch_idx + d, labels)
            Z_mb = sample_Z(batch_size, seq_length, latent_dim)
            
            _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb})    

        # update the generator
        for g in range(G_rounds):

            
            _1 = sess.run([G_solver1,G_solver_comb],feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim)})           
            _2 = sess.run(G_solver2,feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim)})
            
            
#            G_sample1 = Combmodel_rnn_gan.generator1(sample_Z(batch_size, seq_length, latent_dim),hidden_units_g, seq_len, batch_size, num_generated_features)
#            G_sample2 = Combmodel_rnn_gan.generator2(sample_Z(batch_size, seq_length, latent_dim),hidden_units_g, seq_len, batch_size, num_generated_features)

#            _ = sess.run(G_solver_comb,feed_dict={G: gen_comb(g_sample1,g_sample2,Z, hidden_units_g, seq_len, batch_size, num_generated_features)})
            
            
#            _ = sess.run/G_solver_comb
    generator_vars_comb = [v for v in tf.trainable_variables() if v.name.startswith('generator_comb')]

    
    combW1=sess.run(generator_vars_comb[0])
    combW1=1-combW1
    assign_combW1=tf.assign(generator_vars_comb[1],combW1)
    sess.run(assign_combW1)
        
    a = tf.nn.softmax([generator_vars_comb[0],generator_vars_comb[1]])
    

    a = sess.run(a)


    assign1 = tf.assign(generator_vars_comb[0], a[0])
    assign2 = tf.assign(generator_vars_comb[1], a[1])
    sess.run([assign1,assign2])
    
    print('COMB WEIGTHS: ',sess.run(generator_vars_comb))
#    print('GEN2 WEIGTHS: ',sess.run(generator_vars2[2]))
#    print('GEN1 WEIGTHS: ',sess.run(generator_vars1[2]))

        
    D_loss_curr, G_loss_curr= sess.run([D_loss, G_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, seq_length, latent_dim)})
    D_loss_curr = np.mean(D_loss_curr)
    G_loss_curr = np.mean(G_loss_curr)
    
    return D_loss_curr, G_loss_curr





def Save_Parameters(identifier, sess):

    dump_path = './experiments/parameters/' + identifier + '.npy'
    model_parameters = dict()
    for v in tf.trainable_variables():
        model_parameters[v.name] = sess.run(v)
    np.save(dump_path, model_parameters)
    print('Recorded', len(model_parameters), 'parameters to', dump_path)
    return True

def load_parameters(identifier):

    load_path = './experiments/parameters/' + identifier + '.npy'
    model_parameters = np.load(load_path).item()
    return model_parameters


def Generate_Samples(identifier, epoch, seq_length,latent_dim, num_samples, hidden_units_g, num_generated_features, Z_samples=None):

    print('Sampling', num_samples, 'samples from', identifier, 'at epoch', epoch)
    # get the parameters, get other variables
    parameters = load_parameters(identifier + '_' + str(epoch))
    # create placeholder, Z samples
    Z = tf.placeholder(tf.float32, [num_samples, seq_length, latent_dim])
#    CG = tf.placeholder(tf.float32, [num_samples, settings['cond_dim']])
    if Z_samples is None:
        Z_samples = sample_Z(num_samples, seq_length, latent_dim)
    else:
        assert Z_samples.shape[0] == num_samples
    # create the generator (GAN or CGAN)

    G_samples = generator(Z, hidden_units_g, seq_length, 
                              num_samples, num_generated_features, 
                              reuse=True, parameters=parameters)

    # sample from it 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        real_samples = sess.run(G_samples, feed_dict={Z: Z_samples})

    tf.reset_default_graph()
    return real_samples
