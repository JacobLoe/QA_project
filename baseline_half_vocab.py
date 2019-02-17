# on server: 'screen' ,then start script
# use 'strg+a d' to return to terminal
# use 'screen -r' to return to screen

import numpy as np
import json
import os

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import LSTM,GRU
from keras.layers import Bidirectional
from keras.utils import plot_model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
###############################################################
os.environ['CUDA_VISIBLE_DEVICES']='1'
################################################################
path = 'models/baseline_half_vocab'
# rnn parameters
context_hidden_size = 100 #100 is the standard
question_hidden_size = 100 #100 is the standard
BATCH_SIZE = 512 #for the training on the GPU this to be has to very large, otherwise the GPU is used very inefficiently
EPOCHS = 200

#glove embedding parameters
GLOVE_DIR = '../glove/glove.6B.100d.txt'
EMBEDDING_DIM = 100
EVAL_SPLIT = 0.2
#################################################################
#open SQuAD-dataset and extract the relevant data from the json-file
#to a easier readable/accessible dictionary
with open('SQuAD/train-v2.0.json') as file:
    train=json.load(file)
train_context=[]
train_question=[]
train_answer=[]
train_new={'context':train_context,'question':train_question,'answer':train_answer}
for j,data in enumerate(train['data']):
    for i,paragraph in enumerate(data['paragraphs']):
        context=paragraph['context']
        for qas in paragraph['qas']:
            #create a dataset with only the answerable questions
            if (qas['is_impossible']==False):
                train_new['context'].append(context)
                train_new['question'].append(qas['question'])
                train_new['answer'].append(qas['answers'][0]['text'])
####################################################################
#concatenate the data in one vector for preprocessing
train_all=[]
for line in train_new['context']:
    train_all.append(line)
for line in train_new['question']:
    train_all.append(line)
for line in train_new['answer']:
    train_all.append(line)
####################################################################
vocab = set()
for text in train_all:
    vocab |= set(text_to_word_sequence(text))
vocab = sorted(vocab)
vocab_size =round((len(vocab) + 1)/2)
print(np.shape(vocab))
#####################################################################
#prepare the data to use as input of the rnn
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_all)
context_sequences = tokenizer.texts_to_sequences(train_new['context'])
question_sequences = tokenizer.texts_to_sequences(train_new['question'])
answer_sequences = tokenizer.texts_to_sequences(train_new['answer'])

max_len_context=max(map(len,context_sequences))
max_len_question=max(map(len,question_sequences))
max_len_answer=max(map(len,answer_sequences))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

context = pad_sequences(context_sequences, maxlen=max_len_context)
question = pad_sequences(question_sequences, maxlen=max_len_question)
answer = pad_sequences(answer_sequences, maxlen=max_len_answer)

print('Shape of context tensor:', context.shape)
print('Shape of question tensor:', question.shape)
print('Shape of answer tensor:', answer.shape)
##########################################################################
# split the data into a training set and an evaluation set
indices = np.arange(context.shape[0])
np.random.shuffle(indices)
context = context[indices]
question = question[indices]
answer = answer[indices]
num_eval_samples = int(EVAL_SPLIT * context.shape[0])

x_train_context = context[:-num_eval_samples]
x_train_question = question[:-num_eval_samples]
y_train_answer = answer[:-num_eval_samples]

x_eval_context = context[-num_eval_samples:]
x_eval_question = question[-num_eval_samples:]
y_eval_answer = answer[-num_eval_samples:]
##########################################################################
#FIX_ME: add glove download
# https://nlp.stanford.edu/projects/glove/
#get glove embeddings
embeddings_index = {}
f = open(GLOVE_DIR)#os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
###########################################################################
#extract the glove-embedding to a matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
###########################################################################
#create non-trainable embedding layers
# for the context and the question each
context_embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_len_context,
                            trainable=False,
                            name='context_embedding')
question_embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_len_question,
                            trainable=False,
                            name='question_embedding')
###########################################################################
#function for creating an attention layer
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
###########################################################################
print('Build model...')

context_layer = layers.Input(shape=(max_len_context,), dtype='int32',name='Context_input')
embedded_context = context_embedding_layer(context_layer)
bidirec_context = Bidirectional(GRU(context_hidden_size, 
                                     recurrent_dropout=0.2, 
                                     dropout=0.2, return_sequences=True))(embedded_context)
attention_context = AttentionWithContext()(bidirec_context)

question_layer = layers.Input(shape=(max_len_question,), dtype='int32',name='Question_input')
embedded_question = question_embedding_layer(question_layer)
bidirec_question = Bidirectional(GRU(context_hidden_size, 
                                     recurrent_dropout=0.2, 
                                     dropout=0.2, return_sequences=True))(embedded_question)
attention_question = AttentionWithContext()(bidirec_question)

merged = layers.concatenate([attention_context,attention_question])

preds = layers.Dense(max_len_answer, activation='softmax',name='Output_layer')(merged) #dimensions of dense layer have to to the same as the answer dimensions

model = Model([context_layer, question_layer], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('successfully built the model')
print(model.summary())
##########################################################################
print('Training')
model.fit([x_train_context, x_train_question], y_train_answer,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)

print('Evaluation')
loss, acc = model.evaluate([x_eval_context, x_eval_question], y_eval_answer,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
############################################################################
print('save model')
plot_model(model, to_file=path+'/model.png')
model.save_weights(path+'model.h5')
model_json = model.to_json()
with open(path+"baseline_model.json",'w') as json_file:
    json_file.write(model_json)
