import os
import math
import json

import preprocess_data as ppd
import make_RNN_models as mrm
###################################################################
def train_slices(data,data_info,embedding,
                hidden_size,embedding_dim,batch_size,epochs,
                slice_size,size,path):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('training on part %s of the dataset' % slice_size)
    with open(path+'/slice_size.txt','w') as file:
        file.write(str(slice_size))
    file.close()
    input_data=ppd.process_data([data[0][size*slice_size:size*(slice_size+1)],
                                data[1][size*slice_size:size*(slice_size+1)],
                                data[2][size*slice_size:size*(slice_size+1)]],data_info)
    ######################################################################################
    models=mrm.models(embedding,data_info,hidden_size,embedding_dim)
    
    if os.path.isfile(path+str('train_model.h5')):
        print('load models from previous run')
        models['train_model'].load_weights(path+str('train_model.h5'))
        models['encoder_model'].load_weights(path+str('encoder_model.h5'))
        models['decoder_model'].load_weights(path+str('decoder_model.h5'))
    
    print('training model')
    models['train_model'].fit([input_data['encoder_input']['context_encoder_input'],
                               input_data['encoder_input']['question_encoder_input'], 
                               input_data['decoder_input']['answer_decoder_input']], 
                             input_data['decoder_input']['answer_decoder_target'],
                             batch_size=batch_size,
                             epochs=epochs)
    #####################################################################################
    print('save models')
    models['train_model'].save_weights(path+str('train_model.h5')) #save weights
    models['encoder_model'].save_weights(path+str('encoder_model.h5')) #save weights
    models['decoder_model'].save_weights(path+str('decoder_model.h5')) #save weights
    
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
    del models
    del input_data