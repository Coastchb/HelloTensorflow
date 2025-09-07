import tensorflow as tf

Flags = tf.flags

Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')

FLAGS = Flags.FLAGS


def print_configuration_op(FLAGS):
    print('My Configurations:')
    # pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value = value.value
        if type(value) == float:
            print(' %s:\t %f' % (name, value))
        elif type(value) == int:
            print(' %s:\t %d' % (name, value))
        elif type(value) == str:
            print(' %s:\t %s' % (name, value))
        elif type(value) == bool:
            print(' %s:\t %s' % (name, value))
        else:
            print('%s:\t %s' % (name, value))
    # for k, v in sorted(FLAGS.__dict__.items()):
    # print(f'{k}={v}\n')
    print('End of configuration')


def main(argv):
    print_configuration_op(FLAGS)


if __name__ == '__main__':
    tf.app.run()
