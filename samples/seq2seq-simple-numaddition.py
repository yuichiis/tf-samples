import os
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

# Parameters for the model and dataset.
#TRAINING_SIZE = 50000
TRAINING_SIZE = 5000
DIGITS = 2
REVERSE = True

# Generate the data
class NumAdditionDataset:

    def __init__(
            self,
            corpus_max: int,
            digits: int,
            reverse=True):
        self.vocab_input  = ['0','1','2','3','4','5','6','7','8','9','+',' ']
        self.vocab_target = ['0','1','2','3','4','5','6','7','8','9','+',' ']
        self.dict_input = dict(zip(self.vocab_input,range(len(self.vocab_input))))
        self.dict_target = dict(zip(self.vocab_target,range(len(self.vocab_target))))
        self.corpus_max = corpus_max
        self.digits = digits
        self.input_length = digits*2+1
        self.output_length = digits+1
        self.reverse = reverse

    def dicts(self) -> tuple:
        return (
            self.vocab_input,
            self.vocab_target,
            self.dict_input,
            self.dict_target,
        )

    def generate(self) -> tuple:
        '''generate random sequence'''
        max_num = pow(10,self.digits)
        max_sample = max_num ** 2 - 1
        numbers = np.random.choice(
            max_sample,max_sample,replace=False)
        questions = {}
        size = 0
        for i in range(max_sample):
            num = numbers[i]
            x1,x2 = sorted(divmod(num,max_num))
            question = '%d+%d' % (x1,x2)
            if questions.get(question) is not None:
                continue
            questions[question] = '%d' % (x1+x2)
            size += 1
            if size >= self.corpus_max:
                break
        numbers = None
        sequence = np.zeros([size,self.input_length],dtype=np.int32)
        target = np.zeros([size,self.output_length],dtype=np.int32)
        i = 0
        for question,answer in questions.items():
            question = question + " " * (self.input_length - len(question))
            answer = answer + " " * (self.output_length - len(answer))
            #if self.reverse:
            #    question = question[::-1]
            self.str2seq(
                question,
                self.dict_input,
                sequence[i])
            self.str2seq(
                answer,
                self.dict_target,
                target[i])
            i += 1
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

    #def translate(
    #    self,
    #    model,
    #    input_str: str) -> str:
    #    '''translate sentence'''
    #    inputs = np.zeros([1,self.input_length],dtype=np.int32)
    #    self.str2seq(
    #        input_str,
    #        self.dict_input,
    #        inputs[0])
    #    outputs = model.predict(inputs)
    #    return self.seq2str(
    #        np.argmax(outputs[0],axis=1)),
    #        self.vocab_target)

    def load_data(
        self,
        path: str=None) -> ndarray:
        '''load dataset'''
        if path is None:
            path='numaddition-dataset.pkl'

        if os.path.exists(path):
            with open(path,'rb') as fp:
                dataset = pickle.load(fp)
        else:
            dataset = self.generate()
            with open(path,'wb') as fp:
                pickle.dump(dataset,fp)
        return dataset


# All the numbers, plus sign and space for padding.
input_length  = DIGITS*2 + 1
output_length = DIGITS + 1

dataset = NumAdditionDataset(TRAINING_SIZE,DIGITS,REVERSE)

print("Generating data...")
questions,answers = dataset.load_data()
corpus_size = len(questions)
print("Total questions:", corpus_size)
input_voc,target_voc,input_dic,target_dic=dataset.dicts()

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(questions) - len(questions) // 10
(x_train, x_val) = questions[:split_at], questions[split_at:]
(y_train, y_val) = answers[:split_at], answers[split_at:]

print("Training Data:")
print(x_train.shape)
print(y_train.shape)

print("Validation Data:")
print(x_val.shape)
print(y_val.shape)

# Build the model

print("Build model...")

model = keras.Sequential([
    layers.Embedding(len(input_dic), 16),
    # Encoder
    layers.GRU(128,go_backwards=REVERSE),
    # Expand to answer length and peeking hidden states
    layers.RepeatVector(output_length),
    # Decoder
    layers.GRU(128, return_sequences=True),
    # Output
    layers.Dense(
        len(target_dic),
        #activation='softmax',
    ),
])

model.compile(
    #loss="sparse_categorical_crossentropy",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"])
model.summary()

# Train the model

epochs = 30
batch_size = 32

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
)

for i in range(10):
    idx = np.random.randint(0,len(questions))
    question = questions[idx]
    input = question.reshape(1,input_length)
    predict = model.predict(input)
    predict_seq = np.argmax(predict[0].reshape(output_length,len(target_dic)),axis=1)
    predict_str = dataset.seq2str(predict_seq,target_voc)
    question_str = dataset.seq2str(question,input_voc)
    answer_str = dataset.seq2str(answers[idx],target_voc)
    #if dataset.reverse:
    #    question_str = question_str[::-1]
    correct = '*' if predict_str==answer_str else ' '
    print('%s=%s : %s %s' % (question_str,predict_str,correct,answer_str))

plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend();
plt.title('seq2seq-numaddition')
plt.show()
