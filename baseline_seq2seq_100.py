# on server: 'screen' ,then start script
# use 'strg+a d' to return to terminal
# use 'screen -r' to return to screen
import numpy as np
import json
import os

from keras.layers.embeddings import Embedding
from keras.layers import Concatenate
from keras.models import Model
from keras.layers import Input,Dense,LSTM
from keras.utils import plot_model
os.environ['CUDA_VISIBLE_DEVICES']='0'

import process_data as pd
#########################################################################
# https://towardsdatascience.com/nlp-sequence-to-sequence-networks-part-1-processing-text-data-d141a5643b72
path='models/'
# rnn parameters
hidden_size = 100 #100 is the standard
batch_size = 512 #for the training on the GPU this to be has to very large, otherwise the GPU is used very inefficiently
epochs = 400

size=10000

#glove embedding parameters
glove_dir = '../glove/glove.6B.100d.txt'
embedding_dim = 100
########################################################################
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
                
                train_new['context'].append(a)
                train_new['question'].append(b)
                train_new['answer'].append('START_ '+c+' _END')
print(len(train_new['context']))
print(len(train_new['question']))
print(len(train_new['answer']))
############################################################################
context=train_new['context'][:size]
question=train_new['question'][:size]
answer=train_new['answer'][:size]
data=[context,question,answer]
input_data=pd.process_data(data)
############################################################################
context_encoder_input=input_data['encoder_input']['context_encoder_input']
question_encoder_input=input_data['encoder_input']['question_encoder_input']
answer_decoder_input=input_data['decoder_input']['answer_decoder_input']
answer_decoder_target=input_data['decoder_input']['answer_decoder_target']

context_len_vocab=input_data['len_vocab']['context_len_vocab']
question_len_vocab=input_data['len_vocab']['question_len_vocab']
answer_len_vocab=input_data['len_vocab']['answer_len_vocab']

context_token_to_int=input_data['token_to_int']['context_token_to_int']
question_token_to_int=input_data['token_to_int']['question_token_to_int']
answer_token_to_int=input_data['token_to_int']['answer_token_to_int']

answer_int_to_token=input_data['int_to_token']['answer_int_to_token']
#############################################################################
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
###############################################################################
#extract the glove-embedding to a matrix
context_embedding_matrix = np.zeros((context_len_vocab, embedding_dim))
for word, i in context_token_to_int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        context_embedding_matrix[i] = embedding_vector

question_embedding_matrix = np.zeros((question_len_vocab, embedding_dim))
for word, i in question_token_to_int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        question_embedding_matrix[i] = embedding_vector

answer_embedding_matrix = np.zeros((answer_len_vocab, embedding_dim))
for word, i in answer_token_to_int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        answer_embedding_matrix[i] = embedding_vector
print(np.shape(context_embedding_matrix),np.shape(question_embedding_matrix),np.shape(answer_embedding_matrix))
#################################################################################################################
#https://medium.com/@dev.elect.iitd/neural-machine-translation-using-word-level-seq2seq-model-47538cba8cd7
# encoder
context_encoder_inputs = Input(shape=(None,))
context_embedding_layer = Embedding(context_len_vocab, 
                        embedding_dim,weights=[context_embedding_matrix],trainable=False)
context_embedding=context_embedding_layer(context_encoder_inputs)

context_decoder_lstm = LSTM(embedding_dim,return_state=True)
context_x, context_state_h, context_state_c = context_decoder_lstm(context_embedding)
context_encoder_states = [context_state_h, context_state_c]


question_encoder_inputs = Input(shape=(None,))
question_embedding_layer = Embedding(question_len_vocab, 
                    embedding_dim,weights=[question_embedding_matrix],trainable=False)
question_embedding=question_embedding_layer(question_encoder_inputs)

question_decoder_lstm = LSTM(embedding_dim,return_state=True)
question_x, question_state_h, question_state_c = question_decoder_lstm(question_embedding)
question_encoder_states = [question_state_h, question_state_c]


encoder_state_h=Concatenate()([context_state_h,question_state_h])
encoder_state_c=Concatenate()([context_state_c,question_state_c])
concat_encoder_states=[encoder_state_h,encoder_state_c]

# decoder #################################
decoder_inputs = Input(shape=(None,))
answer_embedding_layer = Embedding(answer_len_vocab, 
                             embedding_dim,weights=[answer_embedding_matrix])
answer_embedding = answer_embedding_layer(decoder_inputs)

decoder_lstm = LSTM(embedding_dim*2, return_sequences=True,return_state=True)
decoder_lstm_output,_,_ = decoder_lstm(answer_embedding, initial_state=concat_encoder_states)

decoder_dense = Dense(answer_len_vocab, activation='softmax')
decoder_output = decoder_dense(decoder_lstm_output)

model = Model([context_encoder_inputs,question_encoder_inputs, decoder_inputs], decoder_output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])
model.summary()
#############################################################################################
model.fit([context_encoder_input,
           question_encoder_input, 
           answer_decoder_input], 
          answer_decoder_target,
          batch_size=batch_size,
          epochs=epochs,)
#############################################################################################
print('save model')
if not os.path.isdir(path):
    os.makedirs(path)
model.save_weights(path+str('baseline_model.h5')) #save weights
model_json = model.to_json()
with open(path+str('baseline_model.json'),'w') as json_file:
    json_file.write(model_json)
