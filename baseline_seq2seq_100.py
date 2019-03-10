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

import preprocess_data as ppd
import make_RNN_models as mrm

#os.environ['CUDA_VISIBLE_DEVICES']='0'
#########################################################################
# https://towardsdatascience.com/nlp-sequence-to-sequence-networks-part-1-processing-text-data-d141a5643b72
path='models/'
# rnn parameters
hidden_size = 100 #100 is the standard
batch_size = 100 #for the training on the GPU this to be has to very large, otherwise the GPU is used very inefficiently
epochs = 50

size=10000

#glove embedding parameters
glove_dir = '../glove/glove.6B.100d.txt'
embedding_dim = 100
########################################################################
#open SQuAD-dataset and extract the relevant data from the json-file
#to a easier readable/accessible dictionary
with open('SQuAD/train-v2.0.json') as file:
    train=json.load(file)
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
                a=context.lower()
                b=qas['question'].lower()
                c=qas['answers'][0]['text'].lower()
                
                train_new['qid'].append(qas['id'])
                train_new['context'].append(a)
                train_new['question'].append(b)
                train_new['answer'].append('START_ '+c+' _END')
############################################################################
#create the vocabulary for the answers
answer_words=set()
for line in train_new['answer']:
    for word in line.split():
        if word not in answer_words:
            answer_words.add(word)
data_max_shapes=ppd.get_data_max_shapes([train_new['context'],
                                         train_new['question'],
                                         train_new['answer']],size,answer_words)
################################################################################
#FIX_ME: add glove download
# https://nlp.stanford.edu/projects/glove/
#get glove embeddings
print('getting the glove embeddings')
embeddings_index = {}
f = open(glove_dir)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
#######################################################
qid_to_answer_dict={}
for slice_size in range(9):
    print('training on part %s of the dataset' % slice_size)
    context=train_new['context'][size*slice_size:size*(slice_size+1)]
    question=train_new['question'][size*slice_size:size*(slice_size+1)]
    answer=train_new['answer'][size*slice_size:size*(slice_size+1)]
    data=[context,question,answer]
    input_data=ppd.process_data(data,data_max_shapes,answer_words)

    context_encoder_input=input_data['encoder_input']['context_encoder_input']
    question_encoder_input=input_data['encoder_input']['question_encoder_input']
    answer_decoder_input=input_data['decoder_input']['answer_decoder_input']
    answer_decoder_target=input_data['decoder_input']['answer_decoder_target']
    
    context_token_to_int=input_data['token_to_int']['context_token_to_int']
    question_token_to_int=input_data['token_to_int']['question_token_to_int']
    answer_token_to_int=input_data['token_to_int']['answer_token_to_int']

    answer_int_to_token=input_data['int_to_token']['answer_int_to_token']
    ############################################################################
    #extract the glove-embedding to a matrix
    context_embedding_matrix = np.zeros((data_max_shapes['len_context_vocab'], embedding_dim))
    for word, i in context_token_to_int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            context_embedding_matrix[i] = embedding_vector

    question_embedding_matrix = np.zeros((data_max_shapes['len_question_vocab'], embedding_dim))
    for word, i in question_token_to_int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            question_embedding_matrix[i] = embedding_vector

    answer_embedding_matrix = np.zeros((data_max_shapes['len_answer_vocab'], embedding_dim))
    for word, i in answer_token_to_int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            answer_embedding_matrix[i] = embedding_vector
    embedding=[context_embedding_matrix,question_embedding_matrix,answer_embedding_matrix]
    ######################################################################################
    models=mrm.models(embedding,data_max_shapes,hidden_size,embedding_dim)
    
    if os.path.isfile(path+str('train_model.h5')):
        print('load models from previous run')
        models['train_model'].load_weights(path+str('train_model.h5'))
        models['encoder_model'].load_weights(path+str('encoder_model.h5'))
        models['decoder_model'].load_weights(path+str('decoder_model.h5'))
    
    print('training model')
    models['train_model'].fit([context_encoder_input,
           question_encoder_input, 
           answer_decoder_input], 
          answer_decoder_target,
          batch_size=batch_size,
          epochs=epochs,)
    #####################################################################################
    print('save models')
    if not os.path.isdir(path):
        os.makedirs(path)
    models['train_model'].save_weights(path+str('train_model.h5')) #save weights
    models['encoder_model'].save_weights(path+str('encoder_model.h5')) #save weights
    models['decoder_model'].save_weights(path+str('decoder_model.h5')) #save weights
    
    train_model_json = models['train_model'].to_json()
    with open(path+str('train_model.json'),'w') as json_file:
        json_file.write(train_model_json)
        
    encoder_model_json = models['encoder_model'].to_json()
    with open(path+str('encoder_model.json'),'w') as json_file:
        json_file.write(encoder_model_json)
        
    decoder_model_json = models['decoder_model'].to_json()
    with open(path+str('decoder_model.json'),'w') as json_file:
        json_file.write(decoder_model_json)
    #######################################################################################
    for seq_index in tqdm(range(1)):#len(context))):
        context_input_seq = input_data['encoder_input']['context_encoder_input'][seq_index:seq_index+1]
        question_input_seq = input_data['encoder_input']['question_encoder_input'][seq_index:seq_index+1]

        decoded_sentence = ppd.decode_sequence(context_input_seq,question_input_seq,
                                               answer_token_to_int,answer_int_to_token,models)
        qid_to_answer_dict[train_new['qid'][seq_index]]=decoded_sentence
        
###############################################
with open('SQuAD/answers.json', 'w') as file:
    json.dump(qid_to_answer_dict, file)