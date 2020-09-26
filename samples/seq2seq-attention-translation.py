# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
from numpy import ndarray
import os
import io
import time

# Download the file
class EngFraDataset:
    def download(self):
        path_to_zip = tf.keras.utils.get_file(
        'fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
        extract=True)

        path_to_file = os.path.dirname(path_to_zip)+"/fra.txt"
        return path_to_file

    # Converts the unicode file to ascii
    def unicode_to_ascii(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self,w):
        w = self.unicode_to_ascii(w.lower().strip())
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    def create_dataset(self, path, num_examples):
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
        return zip(*word_pairs)

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
        return tensor, lang_tokenizer

    def load_data(self, path=None, num_examples=None):
        if path is None:
            path = self.download()
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path, num_examples)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def convert(self, lang, tensor):
        for t in tensor:
            if t!=0:
                print ("%d ----> %s" % (t, lang.index_word[t]))


class Encoder(keras.Model):
    '''
    encoder
    '''
    def __init__(
        self,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        **kwargs
    ):
        '''encoder'''
        super(Encoder, self).__init__(**kwargs)
        #self.input_shape=[input_length]
        self.vocab_size = vocab_size
        self.word_vect_size = word_vect_size
        self.recurrent_units = recurrent_units

        self.embedding = keras.layers.Embedding(vocab_size, word_vect_size)
        self.rnn = keras.layers.GRU(recurrent_units,
            return_state=True,return_sequences=True,)
        #self.rnn = keras.layers.LSTM(recurrent_units,
        #    return_state=True,return_sequences=True,)
        #self.rnn = keras.layers.SimpleRNN(recurrent_units,
        #    return_state=True,return_sequences=True,)

    def call(
        self,
        inputs: ndarray,
        training: bool,
        initial_state: tuple=None,
        **kwargs) -> tuple:
        '''forward'''
        wordvect = self.embedding(inputs)
        states = self.rnn(wordvect,training=training,initial_state=initial_state)
        outputs = states.pop(0)
        return (outputs, states)


class Decoder(keras.Model):
    '''
    Decoder
    '''
    def __init__(
        self,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        **kwargs
        ):
        '''
        Decoder
        '''
        super(Decoder, self).__init__(**kwargs)
        #self.input_shape=[input_length]
        self.vocab_size = vocab_size
        self.word_vect_size = word_vect_size
        self.recurrent_size = recurrent_units

        self.embedding = keras.layers.Embedding(vocab_size, word_vect_size)
        self.rnn = keras.layers.GRU(recurrent_units,
            return_state=True,
            return_sequences=True,)
        #self.rnn = keras.layers.LSTM(recurrent_units,
        #    return_state=True,
        #    return_sequences=True,)
        #self.rnn = keras.layers.SimpleRNN(recurrent_units,
        #    return_state=True,
        #    return_sequences=True,)
        self.attention = keras.layers.Attention()
        self.concat = keras.layers.Concatenate()
        self.dense = keras.layers.Dense(vocab_size)

    def call(
        self,
        inputs: ndarray,
        training: bool,
        initial_state: tuple=None,
        enc_outputs=None,
        **kwargs) -> tuple:
        '''forward'''
        wordvect = self.embedding(inputs)
        states = self.rnn(wordvect,training=training,initial_state=initial_state)
        outputs = states.pop(0)
        context_vector = self.attention([outputs,enc_outputs])
        outputs = self.concat([outputs,context_vector])
        outputs = self.dense(outputs)
        return (outputs,states)


class Seq2seq(keras.Model):

    def __init__(
        self,
        input_length=None,
        input_vocab_size=None,
        output_length=None,
        target_vocab_size=None,
        word_vect_size=8,
        recurrent_units=256,
        start_voc_id=0,
        **kwargs
    ):
        '''
        input_length: input sequence length
        input_vocab_size: vocabulary dictionary size for input sequence
        output_length: output sequence length
        target_vocab_size: vocabulary dictionary size for target sequence
        word_vect_size: word vector size of embedding layer
        recurrent_units: units of the recurrent layer
        dense_units: units of the full connection layer
        start_voc_id: vocabulary id of start word in input sequence
        '''
        super(Seq2seq, self).__init__(**kwargs)
        self.encoder = Encoder(
            input_length,
            input_vocab_size,
            word_vect_size,
            recurrent_units,
            **kwargs
        )
        self.decoder = Decoder(
            output_length,
            target_vocab_size,
            word_vect_size,
            recurrent_units,
            **kwargs
        )
        #self.out = keras.layers.Activation('softmax')
        self.start_voc_id = start_voc_id
        self.output_length = output_length

    def shiftSentence(
        self,
        sentence: ndarray,
        ) -> ndarray:
        '''shift target sequence to learn'''
        shape = tf.shape(sentence)
        batchs = shape[0]
        start_id = tf.expand_dims(tf.repeat([self.start_voc_id],repeats=[batchs]), 1)
        seq = sentence[:,:-1]
        result = tf.concat([start_id,seq],1)
        return result

    def call(
        self,
        inputs,
        training=None,
        mask=None,
        ):
        '''forward step'''
        train_data = inputs
        inputs,trues = train_data
        #print('--------forward step---------')
        #print(inputs)
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            print(train_data)
            raise Exception('error')
        enc_outputs,states = self.encoder(inputs,training)
        dec_inputs = self.shiftSentence(trues)

        outputs,dummy = self.decoder(dec_inputs,training,
            initial_state=states,enc_outputs=enc_outputs)
        #
        #outputs = self.out(outputs)
        return outputs

    def train_step(
        self,
        train_data: tuple
    ) -> dict:
        '''train step callback'''
        inputs,trues = train_data

        with tf.GradientTape() as tape:
            #print('====================-')
            #print(trues)
            outputs = self(train_data,training=True)
            loss = self.compiled_loss(
                trues,outputs,
                regularization_losses=self.losses)

        variables = self.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        self.compiled_metrics.update_state(trues, outputs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        The logic for one evaluation step.
        """
        x, y = data

        y_pred = self(data, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def translate(
        self,
        sentence: ndarray) -> ndarray:
        '''translate sequence'''
        input_length = sentence.size
        sentence = sentence.reshape([1,input_length])
        enc_outputs,states=self.encoder(sentence,training=True)
        voc_id = self.start_voc_id
        target_sentence =[]
        for i in range(self.output_length):
            inp = np.array([[voc_id]])
            predictions, states = self.decoder(inp,
                training=False,initial_state=states,enc_outputs=enc_outputs)
            voc_id = np.argmax(predictions)
            target_sentence.append(voc_id)

        return np.array(target_sentence)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


epochs = 30#30;
batch_size = 128;
word_vect_size=256#256
recurrent_units=256#1024
num_examples=10000#30000

print("embedding_dim: ",word_vect_size)
print("units: ",recurrent_units)

dataset = EngFraDataset()

print("Generating data...")
input_tensor, target_tensor, inp_lang, targ_lang = dataset.load_data(num_examples=num_examples)
corpus_size = len(input_tensor)
print("Total questions:", corpus_size)
#input_voc,target_voc,input_dic,target_dic=dataset.dicts()

# Explicitly set apart 10% for validation data that we never train over.
#split_at = len(questions) - len(questions) // 10
#(x_train, x_val) = questions[:split_at], questions[split_at:]
#(y_train, y_val) = answers[:split_at], answers[split_at:]

input_length  = input_tensor.shape[1] #DIGITS*2 + 1;
output_length = target_tensor.shape[1] #DIGITS + 1;
print("Input length =",input_length)
print("Output length=",output_length)

print("Training Data:")
print(input_tensor.shape)
print(target_tensor.shape)

seq2seq = Seq2seq(
    input_length=input_length,
    input_vocab_size=len(inp_lang.word_index)+1,
    output_length=output_length,
    target_vocab_size=len(targ_lang.word_index)+1,
    start_voc_id=targ_lang.word_index['<start>'],
    word_vect_size=word_vect_size,
    recurrent_units=recurrent_units,
)


print("Compile model...")
seq2seq.compile(
    #loss='sparse_categorical_crossentropy',
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss=loss_function,
    optimizer='adam',
    metrics=['accuracy'],
    )

print("Train model...")
history = seq2seq.fit(
    input_tensor,
    target_tensor,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

for i in range(10):
    idx = np.random.randint(0,len(input_tensor))
    question = input_tensor[idx]
    input = question.reshape(1,input_length)

    #input = keras.utils.to_categorical(
    #    input.reshape(input.size,),
    #    num_classes=len(input_voc)
    #    ).reshape(input.shape[0],input.shape[1],len(input_voc))

    #predict = model.predict(input)
    #predict_seq = np.argmax(predict[0].reshape(output_length,len(target_dic)),axis=1)
    predict_seq = seq2seq.translate(question);

    predict_str = dataset.seq2str(predict_seq,target_voc)
    question_str = dataset.seq2str(question,input_voc)
    answer_str = dataset.seq2str(answers[idx],target_voc)
    correct = '*' if predict_str==answer_str else ' '
    print('%s=%s : %s %s' % (question_str,predict_str,correct,answer_str))

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()
