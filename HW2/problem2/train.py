import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

# how to use: 
# python train.py

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'shakespeare', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 200, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 50, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 512, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 3, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.8, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', 'shakespeare_train.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 2000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 100, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')
tf.flags.DEFINE_string('input_file_vali', 'shakespeare_valid.txt', 'utf8 encoded text file')

def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)

    with codecs.open(FLAGS.input_file_vali, encoding='utf-8') as f_v:
        text_v = f_v.read()
    # converter_v = TextConverter(text_v, FLAGS.max_vocab)
    # converter_v.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr_v = converter.text_to_arr(text_v)
    g_v = batch_generator(arr_v, FLAGS.num_seqs, FLAGS.num_steps)


    # print(converter.vocab_size)
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                g_v
                )


if __name__ == '__main__':
    tf.app.run()
