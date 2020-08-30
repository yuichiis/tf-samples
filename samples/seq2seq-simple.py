# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pickle
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import numpy as np
import numpy.ndarray
import matplotlib.pyplot as plt

print(keras.__version__)

class Encoder(keras.Model):
    '''
    encoder
    '''
    def __init__(
        rnn: str,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int
    ):
        '''enconder'''
        self.input_shape=[input_length]
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
        else:
            raise Exception('unknown rnn type: '+rnn)

    def call(
        inputs: ndarray,
        training: bool,
        initalStates: tuple=None,
        **kwarg) -> tuple:
        '''foward'''
        wordvect = self.embedding(inputs,training)
        outputs,states=srlf.rnn(wordvect,training,initalStates)
        return (outputs,states)


class Decoder(keras.Model):
    '''
    Decoder
    '''
    def __init__(
        rnn: str,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        dense_units: int
    )
        '''
        Decoder
        '''
        self.input_shape=[input_length]
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
                return_sequence=True,
            )
        elif rnn=='lstm':
            self.rnn = ketas.layers.LSTM(
                recurrent_units,
                return_state=True,
                return_sequence=True,
            )
        else:
            raise Exception('unknown rnn type: '+rnn)
        
        self.dense = self.layers.Dense(dense_units)
        
    def call(
        inputs: ndarray,
        training: bool,
        initalStates: tuple=None,
        **kwargs) -> tuple:
        '''foward'''
        wordvect = self.embedding(inputs,training)
        outputs,states=self.rnn(wordvect,training,initalStates)
        outputs=self.dense(outputs,training)
        return (outputs,states)
        

class Seq2seq(keras.Model):

    def __init__(
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
        self.out = keras.layers.Activation('softmax')
        self.start_voc_id = start_voc_id

    def shiftSentence(
        ndarray sentence) -> ndarray:
        '''shift target sequence to learn'''
        result = np.zerosLike(sentence)
        result[:,1:] = sequence[:,:-2]
        result[:,0] = self.start_voc_id
        return result
    }

    def call(
        inputs: ndarray,
        trues: ndarray=None,
        training: bool=None
        ) -> ndarray:
        '''forward step'''
        [dummy,states] = self.encoder(inputs,training,null)
        dec_inputs = self.shiftSentence(trues);
        [outputs,dummy] = self.decoder(dec_inputs,training,states)
        outputs = self.out(outputs,training)
        return outputs
    
    def train_step(self,
        train_data: tuple
    ) -> dict:
        '''train step callback'''
        inputs,trues = train_data
        
        with tf.GradientTape() as tape:
            outputs = self(inputs,trues,True)
            loss = self.compiled_loss(
                trues,outputs,
                regularization_losses=self.losses)
                
        variables = self.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        self.compiled_metrics.update_state(trues, outputs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        
    def translate(self,
        sentence: ndarray) -> ndarray:
        '''translate sequence'''
        input_length = sentence.size
        sentence = sentence.reshape([1,input_length])
        dmy,states=self.encoder(sentence,training=True)
        voc_id = self.start_voc_id
        target_sentence =[]
        for i in range(input_length):
            inp = np.array([[$vocId]])
            predictions,states = self.decoder(inp,training=False,states)
            voc_id = np.argmax(predictions)
            target_sentence.append(voc_id)
        
        return np.array(target_sentence)


class DecHexDataset:

    def __init__(self):
        self.vocab_input = ['@','0','1','2','3','4','5','6','7','8','9',' ']
        self.vocab_target = ['@','0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F',' ']
        self.dict_input = dict((v,k) for k,v in self.vocab_input.items())
        self.dict_target = dict((v,k) for k,v in self.vocab_target.items())

    def dicts(self) -> tuple:
        return (
            self.vocab_input,
            self.vocab_target,
            self.dict_input,
            self.dict_target,
        )

    def generate(self,
        corp_size: int,
        length: int) -> tuple:
        '''generate random sequence'''
        sequence = np.zeros([corp_size,length])
        target = np.zeros([corp_size,length])
        numbers = np.random.choice(corp_size,corp_size)
        for i in range(corp_size):
            num = numbers[i];
            dec = str(num);
            hex = hex(num);
            self.str2seq(
                dec,
                self.dict_input,
                sequence[i]);
            self.str2seq(
                hex,
                self.dict_target,
                target[i])
        
        return (sequence,target)
    
    
    
    def str2seq(
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
        input_seq: ndarray,
        word_dic: dict
        ) -> str:
        '''translate string to sequence'''
        outout_str = ''
        bufsz=input_seq.size
        for i in range(bufsz):
            output_str += word_dic[input_buf[i]]
            
        return output_str

    def translate(
        model: Seq2seq,
        input_str: str) -> str:
        '''translate sentence'''
        inputs = np.zeros([1,self.length])
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
                pickle.dump(dataset)
        return dataset


rnn = 'lstm';
corp_size = 10000;
test_size = 100;
dataset = DecHexDataset()
dec_seq,hex_seq=dataset.loadData(corp_size)
train_inputs = dec_seq[[0,corp_size-test_size-1]]
train_target = hex_seq[[0,corp_size-test_size-1]]
test_input = dec_seq[[corp_size-test_size,corp_size-1]]
test_target = hex_seq[[corp_size-test_size,corp_size-1]]
input_length = train_inputs.shape[1]
iv,tv,input_dic,target_dic=dataset.dicts()
input_vocab_size = len(input_dic)
target_vocab_size = len(target_dic)

print('['+dataset.seq2str(train_inputs[0],dataset.vocab_input)+']=>['+dataset.seq2str(train_target[0],dataset.vocab_target)+']\n')

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
    optimizer=keras.optimizers.Adam(),
    )
$history = $seq2seq->fit($train_inputs,$train_target,
    ['epochs'=>5,'batch_size'=>128,'validation_data'=>[$test_input,$test_target]]);

$samples = ['10','255','1024'];
foreach ($samples as $value) {
    $target = $dataset->translate(
        $seq2seq,$value);
    echo "[$value]=>[$target]\n";
}
