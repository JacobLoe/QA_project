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
from keras.layers import Input,Dense,LSTM,GRU
from keras.layers import Bidirectional
from keras.utils import plot_model
os.environ['CUDA_VISIBLE_DEVICES']='0'
###################################################################
path='models/baseline/'
# rnn parameters
hidden_size = 100 #100 is the standard
batch_size = 512 #for the training on the GPU this to be has to very large, otherwise the GPU is used very inefficiently
epochs = 400

size=10000

#glove embedding parameters
glove_dir = '../glove/glove.6B.100d.txt'
embedding_dim = 100
######################################################################
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
            #add a bos and eos token to the target
            if (qas['is_impossible']==False):
                a=context.lower()
                b=qas['question'].lower()
                c=qas['answers'][0]['text'].lower()
                
                train_new['context'].append('\t'+a+'\n')
                train_new['question'].append('\t'+b+'\n')
                train_new['answer'].append('\t'+c+'\n')
print('context ',len(train_new['context']))
print('question',len(train_new['question']))
print('answer',len(train_new['answer']))
########################################################################
context=train_new['context'][:size]
question=train_new['question'][:size]
answer=train_new['answer'][:size]
########################################################################
# https://towardsdatascience.com/nlp-sequence-to-sequence-networks-part-1-processing-text-data-d141a5643b72
# Create word dictionaries :
context_words=set()
for line in context:
    for word in line.split():
        if word not in context_words:
            context_words.add(word)
    
question_words=set()
for line in question:
    for word in line.split():
        if word not in question_words:
            question_words.add(word)
            
answer_words=set()
for line in answer:
    for word in line.split():
        if word not in answer_words:
            answer_words.add(word)
############################################################################
# get lengths and sizes :
len_context_vocab = len(context_words)
len_question_vocab = len(question_words)
len_answer_vocab = len(answer_words)

max_context_len = max([len(line.split()) for line in context])
max_question_len = max([len(line.split()) for line in question])
max_answer_len = max([len(line.split()) for line in answer])

len_context = len(context)
len_question = len(question)
len_answer = len(answer)

print('vocab ',len_context_vocab,len_question_vocab,len_answer_vocab)
print('max len ',max_context_len,max_question_len,max_answer_len)
print('data len ',len_context,len_question,len_answer)
##############################################################################
# Get lists of words :
input_context_words = sorted(list(context_words))
input_question_words = sorted(list(question_words))
target_answer_words = sorted(list(answer_words))

context_token_to_int = dict()
context_int_to_token = dict()

question_token_to_int = dict()
question_int_to_token = dict()

answer_token_to_int = dict()
answer_int_to_token = dict()

#Tokenizing the words ( Convert them to numbers ) :
for i,token in enumerate(input_context_words):
    context_token_to_int[token] = i
    context_int_to_token[i]     = token

for i,token in enumerate(input_question_words):
    question_token_to_int[token] = i
    question_int_to_token[i]     = token
    
for i,token in enumerate(target_answer_words):
    answer_token_to_int[token] = i
    answer_int_to_token[i]     = token
    
#print(len(context_token_to_int),len(context_int_to_token))
###############################################################################
# initiate numpy arrays to hold the data that our seq2seq model will use:
encoder_input_context = np.zeros(
    (len_context, max_context_len),
    dtype='float32')
encoder_input_question = np.zeros(
    (len_question, max_question_len),
    dtype='float32')
decoder_input_answer = np.zeros(
    (len_answer, max_answer_len),
    dtype='float32')
decoder_target_answer = np.zeros(
    (len_answer, max_answer_len, len_answer_vocab),
    dtype='float32')
print('data shape ',np.shape(encoder_input_context),np.shape(encoder_input_question),np.shape(decoder_input_answer),np.shape(decoder_target_answer))
####################################################################################
# Process samples, to get input, output, target data:
for i, (input_context, input_question,target_answer) in enumerate(zip(context,question,answer)):
    for t, word in enumerate(input_context.split()):
        encoder_input_context[i, t] = context_token_to_int[word]
        
    for t, word in enumerate(input_question.split()):
        encoder_input_question[i, t] = question_token_to_int[word]
        
    for t, word in enumerate(target_answer.split()):
        # decoder_target_answer is ahead of decoder_input_answer by one timestep
        decoder_input_answer[i, t] = answer_token_to_int[word]
        if t > 0:
            # decoder_target_answer will be ahead by one timestep
            # and will not include the start character.
            decoder_target_answer[i, t - 1, answer_token_to_int[word]] = 1.
##################################################################################
#FIX_ME: add glove download
# https://nlp.stanford.edu/projects/glove/
#get glove embeddings
embeddings_index = {}
f = open(glove_dir)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
#############################################################################
#extract the glove-embedding to a matrix
context_embedding_matrix = np.zeros((len_context_vocab, embedding_dim))
for word, i in context_token_to_int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        context_embedding_matrix[i] = embedding_vector

question_embedding_matrix = np.zeros((len_question_vocab, embedding_dim))
for word, i in question_token_to_int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        question_embedding_matrix[i] = embedding_vector

answer_embedding_matrix = np.zeros((len_answer_vocab, embedding_dim))
for word, i in answer_token_to_int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        answer_embedding_matrix[i] = embedding_vector
print(np.shape(context_embedding_matrix),np.shape(question_embedding_matrix),np.shape(answer_embedding_matrix))
############################################################################
# Define an input sequence and process it.
context_encoder_inputs = Input(shape=(None,))
context_x = Embedding(len_context_vocab, embedding_dim,weights=[context_embedding_matrix],
                      trainable=False)(context_encoder_inputs)
context_x, context_state_h, context_state_c = LSTM(embedding_dim,
                           return_state=True)(context_x)
context_encoder_states = [context_state_h, context_state_c]

question_encoder_inputs = Input(shape=(None,))
question_x = Embedding(len_question_vocab, embedding_dim,weights=[question_embedding_matrix],
                       trainable=False)(question_encoder_inputs)
question_x, question_state_h, question_state_c = LSTM(embedding_dim,
                           return_state=True)(question_x)
question_encoder_states = [question_state_h, question_state_c]

state_h=layers.Concatenate()([context_state_h,question_state_h])
state_c=layers.Concatenate()([context_state_c,question_state_c])
concat_states=[state_h,state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(len_answer_vocab, embedding_dim,weights=[answer_embedding_matrix],trainable=False)(decoder_inputs)
x = LSTM(embedding_dim*2, return_sequences=True)(x, initial_state=concat_states)
decoder_outputs = Dense(len_answer_vocab, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([context_encoder_inputs,question_encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
###########################################################################
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.fit([encoder_input_context,encoder_input_question, decoder_input_answer], decoder_target_answer,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
###############################################################################
print('save model')
if not os.path.isdir(path):
    os.makedirs(path)
model.save_weights(path+str('baseline_model.h5')) #save weights
model_json = model.to_json()
with open(path+str('baseline_model.json'),'w') as json_file:
    json_file.write(model_json)
