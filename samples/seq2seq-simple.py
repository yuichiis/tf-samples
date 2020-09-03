# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pickle
#from packaging import version
from distutils.version import LooseVersion, StrictVersion
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

print('tensorflow: %s, keras: %s' % (tf.__version__, keras.__version__))
if StrictVersion(tf.__version__) < StrictVersion("2.2.0"):
    raise Exception('tensorflow v2.2.0 or later is required.')

class Encoder(keras.Model):
    '''
    encoder
    '''
    def __init__(
        self,
        rnn: str,
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
        self.rnnName = rnn
        if rnn=='simple':
            self.rnn = keras.layers.SimpleRNN(
                recurrent_units,
                return_state=True,
                )
        elif rnn=='lstm':
            self.rnn = keras.layers.LSTM(
                recurrent_units,
                return_state=True,
                )
        elif rnn=='gru':
            self.rnn = keras.layers.GRU(
                recurrent_units,
                return_state=True,
                )
        else:
            raise Exception('unknown rnn type: '+rnn)

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
        rnn: str,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        dense_units: int,
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
        self.dense_units = dense_units

        self.embedding = keras.layers.Embedding(vocab_size, word_vect_size)
        self.rnn_name = rnn
        if rnn=='simple':
            self.rnn = keras.layers.SimpleRNN(
                recurrent_units,
                return_state=True,
                return_sequences=True,
            )
        elif rnn=='lstm':
            self.rnn = keras.layers.LSTM(
                recurrent_units,
                return_state=True,
                return_sequences=True,
            )
        elif rnn=='gru':
            self.rnn = keras.layers.GRU(
                recurrent_units,
                return_state=True,
                return_sequences=True,
            )
        else:
            raise Exception('unknown rnn type: '+rnn)

        self.dense = keras.layers.Dense(dense_units)

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
        outputs = self.dense(outputs)
        return (outputs,states)


class Seq2seq(keras.Model):

    def __init__(
        self,
        rnn=None,
        input_length=None,
        input_vocab_size=None,
        target_vocab_size=None,
        word_vect_size=8,
        recurrent_units=256,
        dense_units=256,
        start_voc_id=0,
        **kwargs
    ):
        '''
        rnn: 'simple' or 'lstm'
        input_length: input sequence length
        input_vocab_size: vocabulary dictionary size for input sequence
        target_vocab_size: vocabulary dictionary size for target sequence
        word_vect_size: word vector size of embedding layer
        recurrent_units: units of the recurrent layer
        dense_units: units of the full connection layer
        start_voc_id: vocabulary id of start word in input sequence
        '''
        super(Seq2seq, self).__init__(**kwargs)
        self.encoder = Encoder(
            rnn,
            input_length,
            input_vocab_size,
            word_vect_size,
            recurrent_units,
            **kwargs
        )
        self.decoder = Decoder(
            rnn,
            input_length,
            target_vocab_size,
            word_vect_size,
            recurrent_units,
            dense_units,
            **kwargs
        )
        #self.out = keras.layers.Activation('softmax')
        self.start_voc_id = start_voc_id

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
        mask=None
        ):
        '''forward step'''
        train_data = inputs
        inputs,trues = train_data
        #print('--------forward step---------')
        #print(inputs)
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            print(train_data)
            raise Exception('error')
        dummy,states = self.encoder(inputs,training)
        dec_inputs = self.shiftSentence(trues)
        outputs,dummy = self.decoder(dec_inputs,training,initial_state=states)
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
            outputs = self.call(train_data,training=True)
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
        dmy,states=self.encoder(sentence,training=True)
        voc_id = self.start_voc_id
        target_sentence =[]
        for i in range(input_length):
            inp = np.array([[voc_id]])
            predictions, states = self.decoder(inp,training=False,initial_state=states)
            voc_id = np.argmax(predictions)
            target_sentence.append(voc_id)

        return np.array(target_sentence)


class DecHexDataset:

    def __init__(self):
        self.vocab_input = ['@','0','1','2','3','4','5','6','7','8','9',' ']
        self.vocab_target = ['@','0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F',' ']
        self.dict_input = dict(zip(self.vocab_input,range(len(self.vocab_input))))
        self.dict_target = dict(zip(self.vocab_target,range(len(self.vocab_target))))

    def dicts(self) -> tuple:
        return (
            self.vocab_input,
            self.vocab_target,
            self.dict_input,
            self.dict_target,
        )

    def generate(
        self,
        corp_size: int,
        length: int) -> tuple:
        '''generate random sequence'''
        sequence = np.zeros([corp_size,length],dtype=np.int32)
        target = np.zeros([corp_size,length],dtype=np.int32)
        numbers = np.random.choice(corp_size,corp_size)
        for i in range(corp_size):
            num = numbers[i]
            dec = str(num)
            hex = '%x' % num
            self.str2seq(
                dec,
                self.dict_input,
                sequence[i])
            self.str2seq(
                hex,
                self.dict_target,
                target[i])

        return (sequence,target)


    def str2seq(
        self,
        input_string: str,
        word_dic: dict,
        output_seq: ndarray) -> None:
        '''translate string to sequence'''
        sseq = list(input_string.upper())
        seq_len = len(sseq)
        sp = word_dic[' ']
        bufsz=output_seq.size
        for i in range(bufsz):
            if i<seq_len:
                output_seq[i] = word_dic[sseq[i]]
            else:
                output_seq[i]=sp

    def seq2str(
        self,
        input_seq: ndarray,
        word_dic: dict
        ) -> str:
        '''translate string to sequence'''
        output_str = ''
        bufsz = input_seq.size
        for i in range(bufsz):
            output_str = output_str + word_dic[input_seq[i]]
        return output_str

    def translate(
        self,
        model: Seq2seq,
        input_str: str) -> str:
        '''translate sentence'''
        inputs = np.zeros([1,self.length],dtype=np.int32)
        self.str2seq(
            input_str,
            self.dict_input,
            inputs[0])
        target = model.translate(inputs)
        return self.seq2str(
            target,
            self.vocab_target
            )

    def loadData(
        self,
        corp_size: int,
        path: str=None) -> ndarray:
        '''load dataset'''
        self.length = len(str(corp_size))
        if path is None:
            path='dec2hex-dataset.pkl'

        if os.path.exists(path):
            with open(path,'rb') as fp:
                dataset = pickle.load(fp)
        else:
            dataset = self.generate(corp_size,self.length)
            with open(path,'wb') as fp:
                pickle.dump(dataset,fp)
        return dataset


#rnn = 'simple'
#rnn = 'lstm'
rnn = 'gru'
corp_size = 100
test_size = 10
dataset = DecHexDataset()
dec_seq,hex_seq=dataset.loadData(corp_size)
train_inputs = dec_seq[0:corp_size-test_size]
train_target = hex_seq[0:corp_size-test_size]
test_inputs = dec_seq[corp_size-test_size:corp_size]
test_target = hex_seq[corp_size-test_size:corp_size]
input_length = train_inputs.shape[1]
iv,tv,input_dic,target_dic=dataset.dicts()
input_vocab_size = len(input_dic)
target_vocab_size = len(target_dic)
batch_size=128
print('rnn type=%s' % rnn)
print('train,test: %d,%d' % (train_inputs.shape[0],test_inputs.shape[0]))
print('['+dataset.seq2str(train_inputs[0],dataset.vocab_input)+']=>['
    +dataset.seq2str(train_target[0],dataset.vocab_target)+']\n')

seq2seq = Seq2seq(
    rnn=rnn,
    input_length=input_length,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    start_voc_id=dataset.dict_target['@'],
    word_vect_size=16,
    recurrent_units=512,
    dense_units=512,
)

seq2seq.compile(
    #loss='sparse_categorical_crossentropy',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'],
    )
history = seq2seq.fit(
    train_inputs,
    train_target,
    epochs=5,
    batch_size=batch_size,
    validation_data=(
        test_inputs,test_target)
    )

samples = ['10','255','1024']
for sequence in samples:
    target = dataset.translate(
        seq2seq,sequence)
    print('[%s]=>[%s]' % (sequence,target))

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()
