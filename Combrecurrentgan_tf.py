# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:20:49 2018

@author: CHHRGUP
"""


import tensorflow as tf
import numpy as np
from math import ceil
from sklearn import preprocessing
import Combmodel_rnn_gan
from time import time
import random
import mmd

split=0.9
cols=["Close","Volume"]

seq_len=45

normalise = False
batch_size =34
hidden_units_g=128
hidden_units_d=128
dropout_rate=0.25
latent_dim=5
num_signals=2
num_generated_features=1
learning_rate=0.0002
num_epochs=1000
D_rounds=5
G_rounds=3
identifier='test'
shuffle= True

log_dir="./experiments/"

def split(samples, proportions, normalise=False, scale=False, labels=None, random_seed=None):
    """
    Return train/validation/test split.
    """
    if random_seed != None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    assert np.sum(proportions) == 1
    n_total = samples.shape[0]
    n_train = ceil(n_total*proportions[0])
    n_test = ceil(n_total*proportions[2])
    n_vali = n_total - (n_train + n_test)
    # permutation to shuffle the samples
    shuff = np.random.permutation(n_total)
    train_indices = shuff[:n_train]
    vali_indices = shuff[n_train:(n_train + n_vali)]
    test_indices = shuff[(n_train + n_vali):]
    # TODO when we want to scale we can just return the indices
    assert len(set(train_indices).intersection(vali_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(vali_indices).intersection(test_indices)) == 0
    # split up the samples
    train = samples[train_indices]
    vali = samples[vali_indices]
    test = samples[test_indices]
    
    if labels is None:
        return train, vali, test
    else:
        print('Splitting labels...')
        if type(labels) == np.ndarray:
            train_labels = labels[train_indices]
            vali_labels = labels[vali_indices]
            test_labels = labels[test_indices]
            labels_split = [train_labels, vali_labels, test_labels]
        elif type(labels) == dict:
            # more than one set of labels!  (weird case)
            labels_split = dict()
            for (label_name, label_set) in labels.items():
                train_labels = label_set[train_indices]
                vali_labels = label_set[vali_indices]
                test_labels = label_set[test_indices]
                labels_split[label_name] = [train_labels, vali_labels, test_labels]
        else:
            raise ValueError(type(labels))
        return train, vali, test, labels_split
    

def faulty(predict_label=False):
    """
    Load the eICU data for the extreme-value prediction task
    """
    data = np.load('faulty_full.npy')
    
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
        
    pdf=None
    labels=None
    
    return data,pdf,labels,min_max_scaler


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

samples, pdf, labels, min_max_scaler =  faulty()

num_samples= samples.shape[0]
        
train, vali, test = split(samples, [0.8, 0.1, 0.1])
samples = np.reshape(samples,(samples.shape[0],samples.shape[1],1))
train = np.reshape(train,(train.shape[0],train.shape[1],1))
test = np.reshape(test,(test.shape[0],test.shape[1],1))
vali = np.reshape(vali,(vali.shape[0],vali.shape[1],1))

train_labels, vali_labels, test_labels = None, None, None


labels = dict()
labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels

samples = dict()
samples['train'], samples['vali'], samples['test'] = train, vali, test

Z = tf.placeholder(tf.float32, [batch_size, seq_len, latent_dim])
X = tf.placeholder(tf.float32, [batch_size, seq_len, num_generated_features])

g_sample1 = tf.placeholder(tf.float32, [batch_size, seq_len, latent_dim])
g_sample2 = tf.placeholder(tf.float32, [batch_size, seq_len, latent_dim])

D_loss,D_logit_fake = Combmodel_rnn_gan.GAN_loss_DISC( Z, hidden_units_g, X, hidden_units_d, seq_len, batch_size, num_generated_features)
G_loss = Combmodel_rnn_gan.GAN_loss_GEN(D_logit_fake)

D_solver, G_solver1,G_solver2,G_solver_comb = Combmodel_rnn_gan.GAN_solvers(D_loss, G_loss, learning_rate)##############################G_solver_comb

vis_freq = max(14000//num_samples, 1)
vis_Z = Combmodel_rnn_gan.sample_Z(batch_size, seq_len, latent_dim)


######### MMD #############################################################

eval_freq = max(7000//num_samples, 1)

heuristic_sigma_training = mmd.median_pairwise_distance(samples['vali'])
best_mmd2_so_far = 1000

batch_multiplier = 5000//batch_size
eval_size = batch_multiplier*batch_size
eval_eval_size = int(0.2*eval_size)
eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_len, num_generated_features])
eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_len, num_generated_features])
n_sigmas = 2
sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
mmd2, that = mmd.mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
with tf.variable_scope("SIGMA_optimizer"):
    sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])

sigma_opt_iter = 2000
sigma_opt_thresh = 0.001
sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

######### MMD #############################################################


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

trace = open( identifier + '.trace.txt', 'w')
trace.write('epoch time D_loss G_loss\n')

t0 = time()
best_epoch = 0
gen_waves1=[]
gen_waves2=[]
dloss=[]
gloss=[]
mmd_save=[]
cond_dim=0


print('epoch\ttime\tD_loss\tG_loss\tmmd2')

for epoch in range(num_epochs):
    D_loss_curr, G_loss_curr = Combmodel_rnn_gan.train_epoch(epoch, samples['train'], labels['train'],
                                        sess, Z, X,D_loss, G_loss,D_solver, G_solver1,G_solver2, G_solver_comb,
                                        batch_size, D_rounds, G_rounds, seq_len, 
                                        latent_dim, num_generated_features, cond_dim)
        
    
#    if epoch % vis_freq == 0:

#        vis_sample1 = sess.run(G_sample1, feed_dict={Z: vis_Z})
#        vis_sample2 = sess.run(G_sample2, feed_dict={Z: vis_Z})
#        gen_waves1.append(vis_sample1)
#        gen_waves2.append(vis_sample2)
        #plotting.visualise_at_epoch(vis_sample, epoch, identifier, num_epochs,resample_rate_in_min=15)
        
    t = time() - t0
           
    dloss.append(D_loss_curr)
    gloss.append(G_loss_curr)
        ## save trace
    trace.write(' '.join(map(str, [epoch, t, D_loss_curr, G_loss_curr])) + '\n')
    

#    # MMD2
#    if epoch % eval_freq == 0:
#        ## how many samples to evaluate with?
#        eval_Z = Combmodel_rnn_gan.sample_Z(eval_size, seq_len, latent_dim)
#        eval_sample = np.empty(shape=(eval_size, seq_len, num_generated_features))
#        for i in range(batch_multiplier):
#            
#            eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size]})
#        eval_sample = np.float32(eval_sample)
#        eval_real = np.float32(samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier*batch_size), :, :])
#       
#        eval_eval_real = eval_real[:eval_eval_size]
#        eval_test_real = eval_real[eval_eval_size:]
#        eval_eval_sample = eval_sample[:eval_eval_size]
#        eval_test_sample = eval_sample[eval_eval_size:]
#        
#        ## MMD
#        sess.run(tf.variables_initializer(sigma_opt_vars))
#        sigma_iter = 0
#        that_change = sigma_opt_thresh*2
#        old_that = 0
#
#        mmd21, that_np = sess.run(mmd.mix_rbf_mmd2_and_ratio(eval_test_real, eval_test_sample,biased=False, sigmas=sigma))
       
    print('%d\t%.2f\t%.4f\t%.4f\t%.4f\t' % (epoch, t, D_loss_curr, G_loss_curr,0))
        
    
    if epoch % 50 == 0:
        Combmodel_rnn_gan.Save_Parameters(identifier + '_' + str(epoch), sess)

       



gen_waves1=np.asarray(gen_waves1)
gen_waves2=np.asarray(gen_waves2)
np.save('genwaves1.npy',gen_waves1)
np.save('genwaves2.npy',gen_waves2)
np.save('D_LOSS.npy',dloss)
np.save('G_LOSS.npy',gloss)
#np.save('mmd.npy',mmd_save)

trace.flush()
Combmodel_rnn_gan.Save_Parameters(identifier + '_' + str(epoch), sess)


a1=np.load('genwaves1.npy')
a2=np.load('genwaves2.npy')
for i in range(epoch-1):
    a1=a1[i,:,:,0]
    a2=a2[i,:,:,0]
    unscaled1=min_max_scaler.inverse_transform(a1)
    unscaled2=min_max_scaler.inverse_transform(a2)
    np.save('zwei/genwaves_transf1'+ str(i),unscaled1)
    np.save('zwei/genwaves_transf2'+ str(i),unscaled2)






