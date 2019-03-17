import numpy as np
import json
import os

from keras.layers.embeddings import Embedding
from keras.layers import Concatenate
from keras.models import Model
from keras.layers import Input,Dense,LSTM

from tqdm import tqdm

def models(embeddings,data_info,hidden_size,embedding_dim):
        
    print('building models')
    
    context_encoder_inputs = Input(shape=(None,),name='context')
    context_embedding_layer = Embedding(data_info['len_context_vocab'], 
                            embedding_dim,weights=[embeddings[0]],trainable=False,name='context_embeddings')
    context_embedding=context_embedding_layer(context_encoder_inputs)

    context_decoder_lstm = LSTM(hidden_size,return_state=True,name='context_lstm')
    context_x, context_state_h, context_state_c = context_decoder_lstm(context_embedding)
    context_encoder_states = [context_state_h, context_state_c]


    question_encoder_inputs = Input(shape=(None,),name='question')
    question_embedding_layer = Embedding(data_info['len_question_vocab'], 
                        embedding_dim,weights=[embeddings[1]],trainable=False,name='question_embeddings')
    question_embedding=question_embedding_layer(question_encoder_inputs)

    question_decoder_lstm = LSTM(hidden_size,return_state=True,name='question_lstm')
    question_x, question_state_h, question_state_c = question_decoder_lstm(question_embedding)
    question_encoder_states = [question_state_h, question_state_c]


    encoder_state_h=Concatenate(name='concatenate_lstm_states_h')([context_state_h,question_state_h])
    encoder_state_c=Concatenate(name='concatenate_lstm_states_c')([context_state_c,question_state_c])
    concat_encoder_states=[encoder_state_h,encoder_state_c]

    # decoder #################################
    decoder_inputs = Input(shape=(None,),name='answer')
    answer_embedding_layer = Embedding(data_info['len_answer_vocab'], 
                                 embedding_dim,weights=[embeddings[2]],name='answer_embeddings')
    answer_embedding = answer_embedding_layer(decoder_inputs)

    decoder_lstm = LSTM(hidden_size*2, return_sequences=True,return_state=True,name='decoder_lstm')
    decoder_lstm_output,_,_ = decoder_lstm(answer_embedding, initial_state=concat_encoder_states)

    decoder_dense = Dense(data_info['len_answer_vocab'], activation='softmax',name='ouput')
    decoder_output = decoder_dense(decoder_lstm_output)

    train_model = Model([context_encoder_inputs,question_encoder_inputs, decoder_inputs], decoder_output)
    train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])

    ###########################################################################################################
    #build the encoder and decoder models used during inference
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
