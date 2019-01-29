# -*- coding:utf-8 -*- 
# @Time		:2018/11/23 2:58 PM
# @Author	:Coast Cao

import tensorflow as tf

### test:
###     (1) create two models for training and predict with the same structure and save directory, but differrent configuration
###     (2) add hook and do evaluation while training

### Quesiont:
### Q1: what if steps=100 set instead of max_steps=STEPS set?
### Q2: how to do evaluation after training for specified steps, and then continue to train?

train_file = "iris_training.csv"
test_file = "iris_test.csv"
predict_file = "iris_predict.csv"

FEATURE_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth']
LABEL_NAME = ['Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
CSV_TYPES_PREDICT = [[0.0], [0.0], [0.0], [0.0]]
BATCH_SIZE = 32
STEPS = 1000

def parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(FEATURE_NAMES, fields[:4]))
    label = fields[4]
    return features, label

def parse_line_predict(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES_PREDICT)
    features = dict(zip(FEATURE_NAMES, fields))
    return features

def get_input_fn(file_path, batch_size):
    def _input_fn(mode):
        dataset = tf.data.TextLineDataset(file_path).skip(1)
        if (mode == tf.estimator.ModeKeys.PREDICT):
            dataset = dataset.map(parse_line_predict)
        else:
            dataset = dataset.map(parse_line)
        if(mode == tf.estimator.ModeKeys.TRAIN):
            dataset = dataset.shuffle(1000).repeat()
        dataset = dataset.batch(batch_size)
        return dataset
    return _input_fn

def get_feature_column():
    feature_columns = []
    for key in FEATURE_NAMES:
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    return feature_columns

def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, feature_columns=params['feature_column'])
    for i in params['hidden_units']:
        net = tf.layers.dense(net, i, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    predicts = tf.argmax(logits, 1)

    if(mode == tf.estimator.ModeKeys.PREDICT):
        tf.summary.histogram("pre_predict", predicts)
        predictions = {
            'class_ids': predicts[:,tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    with tf.variable_scope("inference") as scope:
        loss = tf.identity(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits), name="loss")
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicts)
    metrics = {'accuracy_eval': accuracy}            # loss cannot add to metrics
    tf.summary.scalar("loss_eval",loss)
    tf.summary.histogram("pre_eval", predicts)

    '''
    # can't write summary out within model_fn
    # manually write summary out
    with tf.summary.FileWriter("outputs/") as fw:
        with tf.Session() as sess:
            s = sess.run(tf.summary.histogram("pre_eval_out", predicts))
            fw.add_summary(s, global_step=tf.train.get_global_step())
    '''

    if(mode == tf.estimator.ModeKeys.EVAL):
        #tf.summary.scalar('accuracy_eval', accuracy[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    tf.summary.scalar('accuracy_train', accuracy[1])
    tf.summary.scalar('loss_train', loss)
    tf.summary.histogram('pre_train', predicts)
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def get_model_fn(mode):
    if(mode in ["train", "eval", "predict"]):
        return model_fn
    else:
        raise RuntimeError("Invalid mode:%s" % mode)

run_config = tf.estimator.RunConfig(
    model_dir='model/',
    save_checkpoints_steps=100,
    save_summary_steps=100,
    log_step_count_steps=100
    )

classifier_train = tf.estimator.Estimator(
    model_fn = get_model_fn("train"),
    params= {
        'feature_column': get_feature_column(),
        'hidden_units': [3,3],
        'n_classes': 3,
        #'summary_writer': summary_writer
    },
    config=run_config
)
'''
classifier_eval = tf.estimator.Estimator(
    model_fn = get_model_fn("eval"),
    params = {
        'feature_column': get_feature_column(),
        'hidden_units': [3, 3],
        'n_classes': 3,
        # 'summary_writer': summary_writer
    },
    model_dir = 'model/'
)
'''

class Hook(tf.train.SessionRunHook):
    def __init__(self,item_keys,eval_interval):
        if(type(item_keys)==list):
            self._keys = item_keys
        else:
            raise RuntimeError("Fetch tensors must be a list")

        self._interval = tf.train.SecondOrStepTimer(every_steps=eval_interval)

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        print("###### Session begins ######")
        print(self._global_step_tensor)
        print(self._global_step_tensor.name)
        print("######")
        if(self._global_step_tensor is None):
            raise RuntimeError("Global step should be set before session is created.")

    def after_create_session(self, session, coord):
        print("New session created.")
    def before_run(self, run_context):
        #print("before running...")
        return tf.train.SessionRunArgs([self._global_step_tensor.name] + self._keys)

    def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
        #print(run_values.results)
        global_step = run_values.results[0]
        #print("global_step:%d" % global_step)
        if(self._interval.should_trigger_for_step(global_step)):
            '''
            time, steps = self._interval.update_last_triggered_step(global_step)
            print("Training step=%d, loss=%.3f" % (global_step, run_values.results[1]))
            run_context.request_stop()
            '''
            time, steps = self._interval.update_last_triggered_step(global_step)
            #print(time)
            #print(steps)
            if(time is not None):
                print("Training step=%d, loss=%.3f" % (global_step, run_values.results[1]))
                #run_context.request_stop()

   # def end(self):
   #     print("Seesion ended.")

while True:
    # Q1
    train_rets = classifier_train.train(input_fn=get_input_fn(train_file, BATCH_SIZE),  hooks=[Hook(["inference/loss:0"],100)], max_steps=STEPS)
    print("on train:")
    print(train_rets)

    eval_rets = classifier_train.evaluate(input_fn=get_input_fn(test_file,BATCH_SIZE),steps=1)
    print("\non eval:")
    print(eval_rets)
    step = eval_rets['global_step']
    if(step >= STEPS):
        break

### or just use clasifier_train
classifier_predict = tf.estimator.Estimator(
    model_fn = get_model_fn("predict"),
    params={
        'feature_column': get_feature_column(),
        'hidden_units': [3, 3],
        'n_classes': 3,
        # 'summary_writer': summary_writer
    },
    config=run_config

)
pre_rets = classifier_predict.predict(input_fn=get_input_fn(predict_file, batch_size=BATCH_SIZE)) #, predict_keys=["class_ids"])
print("\non predict:")
print(pre_rets)

all_pre_ids = []
for item in pre_rets:
    print(item)
    all_pre_ids.append(item['class_ids'][0])
print(all_pre_ids)

# manually write summary out
with tf.summary.FileWriter("outputs/") as fw:
    with tf.Session() as sess:
        s = sess.run(tf.summary.histogram("pre_ids", all_pre_ids))
        fw.add_summary(s, global_step=tf.train.get_global_step())
