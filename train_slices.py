import os
import math
import json

import preprocess_data as ppd
import make_RNN_models as mrm
###################################################################
def train_slices(data,data_info,embedding,
                hidden_size,embedding_dim,batch_size,epochs,
                slice_size,size,path):
    # check whether the path is available, if not create it
    if not os.path.isdir(path):
        os.makedirs(path)
    print('training on part %s of the dataset' % slice_size)
    # save which part of the SQuAD we currently working with, to make it possible to start from there if the training aborts 
    with open(path+'/slice_size.txt','w') as file:
        file.write(str(slice_size))
    file.close()
    #prepare the input data, for the defined slice of the dataset
    input_data=ppd.process_data([data[0][size*slice_size:size*(slice_size+1)],
                                data[1][size*slice_size:size*(slice_size+1)],
                                data[2][size*slice_size:size*(slice_size+1)]],data_info)
    ######################################################################################
    #create the models based on the given parameters
    models=mrm.models(embedding,data_info,hidden_size,embedding_dim)
    
    # if there are models available from a previous run load them into the built models
    if os.path.isfile(path+str('train_model.h5')):
        print('load models from previous run')
        models['train_model'].load_weights(path+str('train_model.h5'))
        models['encoder_model'].load_weights(path+str('encoder_model.h5'))
        models['decoder_model'].load_weights(path+str('decoder_model.h5'))
    
    # train the model on the input data, with the given batch size and for the given epochs
    print('training model')
    models['train_model'].fit([input_data['encoder_input']['context_encoder_input'],
                               input_data['encoder_input']['question_encoder_input'], 
                               input_data['decoder_input']['answer_decoder_input']], 
                             input_data['decoder_input']['answer_decoder_target'],
                             batch_size=batch_size,
                             epochs=epochs)
    #####################################################################################
    # save the weights of the models to create a checkpoint
    print('save models')
    models['train_model'].save_weights(path+str('train_model.h5')) #save weights
    models['encoder_model'].save_weights(path+str('encoder_model.h5')) #save weights
    models['decoder_model'].save_weights(path+str('decoder_model.h5')) #save weights
    
    #save the architecture of the models
    train_model_json = models['train_model'].to_json()
    with open(path+str('train_model.json'),'w') as train_json_file:
        train_json_file.write(train_model_json)
    train_json_file.close()
        
    encoder_model_json = models['encoder_model'].to_json()
    with open(path+str('encoder_model.json'),'w') as encoder_json_file:
        encoder_json_file.write(encoder_model_json)
    encoder_json_file.close()
    
    decoder_model_json = models['decoder_model'].to_json()
    with open(path+str('decoder_model.json'),'w') as decoder_json_file:
        decoder_json_file.write(decoder_model_json)
    decoder_json_file.close()
    #delete the models and input data to save memory
    del models
    del input_data
