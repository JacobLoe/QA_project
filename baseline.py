# on server: 'screen' ,then start script
# use 'strg+a d' to return to terminal
# use 'screen -r' to return to screen
import numpy as np
import json
import os
import math

from keras.models import model_from_json
os.environ['CUDA_VISIBLE_DEVICES']='1'

import preprocess_data as ppd
import train_slices as ts

from tqdm import tqdm
##########################################
path='models/'
# rnn parameters
hidden_size = 100 
batch_size = 100 
epochs = 25

size=10000

#glove embedding parameters
glove_dir = '../glove/glove.6B.100d.txt'
embedding_dim = 100
############################################
#open SQuAD-dataset and extract the relevant data from the json-file
#to an easier readable/accessible dictionary
with open('SQuAD/train-v2.0.json') as data_file:
    train=json.load(data_file)
train_qid=[]
train_context=[]
train_question=[]
train_answer=[]
train_new={'context':train_context,'question':train_question,'answer':train_answer,'qid':train_qid}
for j,data in enumerate(train['data']):
    for i,paragraph in enumerate(data['paragraphs']):
        context=paragraph['context']
        for qas in paragraph['qas']:
            #create a dataset with only the answerable questions
            #add a bos and eos token to the target
            if (qas['is_impossible']==False):
                c=qas['answers'][0]['text'].lower()
                
                train_new['qid'].append(qas['id'])
                train_new['context'].append(context.lower())
                train_new['question'].append(qas['question'].lower())
                train_new['answer'].append('START_ '+c+' _END')
            else:
                train_new['qid'].append(qas['id'])
                train_new['context'].append(context.lower())
                train_new['question'].append(qas['question'].lower())
                train_new['answer'].append('START_ '+str(qas['answers'])+' _END')
################################################################
data_info=ppd.get_data_info([train_new['context'],
                             train_new['question'],
                             train_new['answer']])

#get glove embeddings
print('getting the glove embeddings')
embeddings_index = {}
f = open(glove_dir)
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

#extract the glove-embedding to a matrix
context_embedding_matrix = np.zeros((data_info['len_context_vocab'], embedding_dim))
for word, i in data_info['context_token_to_int'].items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        context_embedding_matrix[i] = embedding_vector

question_embedding_matrix = np.zeros((data_info['len_question_vocab'], embedding_dim))
for word, i in data_info['question_token_to_int'].items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        question_embedding_matrix[i] = embedding_vector

answer_embedding_matrix = np.zeros((data_info['len_answer_vocab'], embedding_dim))
for word, i in data_info['answer_token_to_int'].items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        answer_embedding_matrix[i] = embedding_vector
############################################################
# train model
for slice_size in range(10,math.ceil(len(train_new['context'])/size)):
    ts.train_slices([train_new['context'],train_new['question'],train_new['answer']],
                    data_info,
                    [context_embedding_matrix,question_embedding_matrix,answer_embedding_matrix],
                    hidden_size,
                    embedding_dim,
                    batch_size,
                    epochs,
                    slice_size,
                    size,
                    path)
######################################################################
# load the trained encoder and decoder model
print('start inference')
with open(path+'encoder_model.json', 'r') as encoder_json_file:
    loaded_model_json = encoder_json_file.read()
    encoder_model = model_from_json(loaded_model_json)
encoder_model.load_weights(path+'encoder_model.h5')
encoder_json_file.close()
    
with open(path+'decoder_model.json', 'r') as decoder_json_file:
    loaded_model_json = decoder_json_file.read()
    decoder_model = model_from_json(loaded_model_json)
decoder_model.load_weights(path+'decoder_model.h5')
decoder_json_file.close()

# answer question
qid_to_answer_dict={}
for slice_size in range(math.ceil(len(train_new['context'])/size)):
    print('inference on part %s of the dataset' % slice_size)
    input_data=ppd.process_data([train_new['context'][size*slice_size:size*(slice_size+1)],
                                 train_new['question'][size*slice_size:size*(slice_size+1)],
                                 train_new['answer'][size*slice_size:size*(slice_size+1)]]
                                ,data_info)
    for seq_index in tqdm(range(len(train_new['context'][size*slice_size:size*(slice_size+1)]))):
        decoded_sentence = ppd.decode_sequence(input_data['encoder_input']['context_encoder_input'][seq_index:seq_index+1],
                                                input_data['encoder_input']['question_encoder_input'][seq_index:seq_index+1],
                                                data_info['answer_token_to_int'],
                                                data_info['answer_int_to_token'],
                                                encoder_model,
                                                decoder_model)
        qid_to_answer_dict[train_new['qid'][seq_index+(slice_size*size)]]=decoded_sentence
# write the answers to a .json-file
print('write answer to json')
with open(path+'answers.json', 'w') as json_file:
    json.dump(qid_to_answer_dict, json_file)
json_file.close()
