#!/usr/bin/python
# coding=utf-8


import numerous
import tensorflow as tf

from numerous.framework.sparse_parameter import SparseParameter
from numerous.framework.dense_parameter import DenseParameter
from numerous.optimizers.adam import Adam
from numerous.optimizers.ftrl import FTRL
from numerous.distributions.uniform import Uniform

from numerous.estimator import Estimator

import math
import sys


# dense embedding
position_emb_table = tf.get_variable('position_embedding',
                                     [20, 8],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(0,1))
tf_embedding = tf.nn.embedding_lookup(position_emb_table, tf.constant([1,2]))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print(sess.run(tf_embedding))


# numerous embedding
slot_id = map(str,[1,2,3])
emb_w, x_data = numerous.framework.SparseEmbedding(model_slice="embedding",
                                                    embedding_dim=8,
                                                   optimizer=Adam(rho1=0.9, rho2=0.999, eps=0.0001),
                                                    distribution=Uniform(left=0, right=1),
                                                    slot_ids=slot_id)

numerous_embedding_1, numerous_embedding_2, numerous_embedding_3 = [tf.matmul(x, emb_w) for x in x_data]

print("!!!")
print(emb_w)
print(x_data)

print(tf_embedding.shape)
print(emb_w.shape)
#print(x_data.shape)
#numerous_embedding = tf.reshape(numerous_embedding_1, (-1, len(slot_id), 8))
print(numerous_embedding_1.shape)

numerous.training.Task(model_name="emb_tmp",
                       worker_thread=4,
                       worker_async_thread=2,
                       server_thread=4)

numerous.training.Saver("dir:1", complete_hour="06", always_complete=1,
                        expire_time=0, expired_detection_interval=1200)

estimator = numerous.estimator.Estimator()

numerous.optimizers.OptimizerContext().set_optimizer(optimizer=Adam(rho1=0.9, rho2=0.999, eps=0.0001))

ctx = numerous.get_default_context()
# Hooks for estimator
train_hook = numerous.estimator.LegacyTrainDataHook()
# local_auc_hook = numerous.estimator.TensorFlowAUCHook(self.label, self.pred)
# print_hook = PrintHook({"mse loss": self.loss})
print_hook_dict = {}

tmp = 1010
tmp = tf.Print(tmp, [tmp], "tmp:", first_n=100, summarize=200)
print_hook_dict["tmp::: "] = tmp
emb_w = tf.Print(emb_w, [emb_w], "emb_w:", first_n=100, summarize=200)
print_hook_dict["emb_w::: "] = emb_w
x_data = tf.Print(x_data, [x_data], "x_data:", first_n=100, summarize=200)
print_hook_dict["x_data::: "] = x_data
numerous_embedding = tf.Print(numerous_embedding_1, [numerous_embedding_1], "numerous_embedding_1:", first_n=100, summarize=200)
print_hook_dict["numerous_embedding_1::: "] = numerous_embedding

mix_emb = tf.concat((tf_embedding, numerous_embedding_1), -1)
mix_emb = tf.Print(mix_emb, [mix_emb], "mix_emb:", first_n=100, summarize=200)
print_hook_dict["mix_emb::: "] = mix_emb


print_hook = numerous.estimator.PrintHook(print_hook_dict)

logits = tf.constant([1.,2.])
label = tf.constant([1.,2.])
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label,
                                                               name="logloss_tmp")

loss_metrics = numerous.metrics.DistributeMean(values=loss, name="loss")
# Metric hook for multi-task auc
metric_hook = numerous.estimator.MetricHook(report_every_n_batch=50)
metric_hook.add_metric(loss_metrics)

target = numerous.optimizers.OptimizerContext().minimize(loss)
target.set_epoch(0)

estimator.train(target=target, hooks=[train_hook, print_hook, metric_hook])