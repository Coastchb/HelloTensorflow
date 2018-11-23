# -*- coding:utf-8 -*- 
# @Time		:2018/11/23 2:58 PM
# @Author	:Coast Cao

import tensorflow as tf

### Question 1: What if run classifier.train() several times? (the checkpoint)

train_file = "iris_training.csv"
test_file = "iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
BATCH_SIZE = 32
STEPS = 1000

def parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop("Species")
    return features, label

def get_input_fn(file_path, batch_size):
    dataset = tf.data.TextLineDataset(file_path).skip(1)
    dataset = dataset.map(parse_line)
    #print(dataset)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

def get_feature_column():
    feature_columns = []
    for key in CSV_COLUMN_NAMES[:-1]:
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    return feature_columns

#get_input_fn(train_file,BATCH_SIZE)

def get_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, feature_columns=params['feature_column'])
    for i in params['hidden_units']:
        net = tf.layers.dense(net, i, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    predicts = tf.argmax(logits, 1)

    if(mode == tf.estimator.ModeKeys.PREDICT):
        predictions = {
            'class_ids': predicts[:,tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicts)
    merics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if(mode == tf.estimator.ModeKeys.EVAL):
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=merics)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

classifier = tf.estimator.Estimator(
    model_fn=get_model_fn,
    params= {
        'feature_column': get_feature_column(),
        'hidden_units': [3,3],
        'n_classes': 3
    },
    model_dir= 'model/'
)

### Question 1
classifier.train(input_fn=lambda:get_input_fn(train_file, BATCH_SIZE), steps=STEPS)
classifier.evaluate(input_fn=lambda:get_input_fn(test_file,BATCH_SIZE), steps=STEPS)