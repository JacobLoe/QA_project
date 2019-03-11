import numpy as np
import json
import os

from keras.layers.embeddings import Embedding
from keras.layers import Concatenate
from keras.models import Model
from keras.layers import Input,Dense,LSTM
from keras.utils import plot_model

from tqdm import tqdm

def models(embeddings,data_max_shapes,hidden_size,embedding_dim):
    
    context_embedding_matrix = embeddings[0]
    question_embedding_matrix = embeddings[1]
    answer_embedding_matrix = embeddings[2]
    
    len_context_vocab = data_max_shapes['len_context_vocab']
    len_question_vocab = data_max_shapes['len_question_vocab']
    len_answer_vocab = data_max_shapes['len_answer_vocab']
    print('building models')
    ###########################################################################################################
    # https://medium.com/@dev.elect.iitd/neural-machine-translation-using-word-level-seq2seq-model-47538cba8cd7
    # train_model
    
    context_encoder_inputs = Input(shape=(None,),name='context')
    context_embedding_layer = Embedding(len_context_vocab, 
                            embedding_dim,weights=[context_embedding_matrix],trainable=False,name='context_embeddings')
    context_embedding=context_embedding_layer(context_encoder_inputs)

    context_decoder_lstm = LSTM(hidden_size,return_state=True,name='context_lstm')
    context_x, context_state_h, context_state_c = context_decoder_lstm(context_embedding)
    context_encoder_states = [context_state_h, context_state_c]


    question_encoder_inputs = Input(shape=(None,),name='question')
    question_embedding_layer = Embedding(len_question_vocab, 
                        embedding_dim,weights=[question_embedding_matrix],trainable=False,name='question_embeddings')
    question_embedding=question_embedding_layer(question_encoder_inputs)

    question_decoder_lstm = LSTM(hidden_size,return_state=True,name='question_lstm')
    question_x, question_state_h, question_state_c = question_decoder_lstm(question_embedding)
    question_encoder_states = [question_state_h, question_state_c]


    encoder_state_h=Concatenate(name='concatenate_lstm_states_h')([context_state_h,question_state_h])
    encoder_state_c=Concatenate(name='concatenate_lstm_states_c')([context_state_c,question_state_c])
    concat_encoder_states=[encoder_state_h,encoder_state_c]

    # decoder #################################
    decoder_inputs = Input(shape=(None,),name='answer')
    answer_embedding_layer = Embedding(len_answer_vocab, 
                                 embedding_dim,weights=[answer_embedding_matrix],name='answer_embeddings')
    answer_embedding = answer_embedding_layer(decoder_inputs)

    decoder_lstm = LSTM(hidden_size*2, return_sequences=True,return_state=True,name='decoder_lstm')
    decoder_lstm_output,_,_ = decoder_lstm(answer_embedding, initial_state=concat_encoder_states)

    decoder_dense = Dense(len_answer_vocab, activation='softmax',name='ouput')
    decoder_output = decoder_dense(decoder_lstm_output)

    train_model = Model([context_encoder_inputs,question_encoder_inputs, decoder_inputs], decoder_output)
    train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])
    ###########################################################################################################
    # encoder model 
    encoder_model = Model([context_encoder_inputs,question_encoder_inputs],concat_encoder_states)
    ##############################################################################################
    # decoder model
    decoder_state_input_h = Input(shape=(hidden_size*2,),name='decoder_state_h_input')
    decoder_state_input_c = Input(shape=(hidden_size*2,),name='decoder_state_c_input')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    final_dex2= answer_embedding_layer(decoder_inputs)

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]

    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2,)
    
    return_models={'train_model':train_model,'encoder_model':encoder_model,'decoder_model':decoder_model}
    
    return return_models