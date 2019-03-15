import numpy as np

def get_data_info(data):
    # create vocabulary
    context_words=set()
    for line in data[0]:
        for word in line.split():
            if word not in context_words:
                context_words.add(word)

    question_words=set()
    for line in data[1]:
        for word in line.split():
            if word not in question_words:
                question_words.add(word)

    answer_words=set()
    for line in data[2]:
        for word in line.split():
            if word not in answer_words:
                answer_words.add(word)
    ###############################################################
    context_token_to_int = dict()

    question_token_to_int = dict()

    answer_token_to_int = dict()
    answer_int_to_token = dict()

    #Tokenize the words :
    for i,token in enumerate(sorted(list(context_words))):
        context_token_to_int[token] = i

    for i,token in enumerate(sorted(list(question_words))):
        question_token_to_int[token] = i

    for i,token in enumerate(sorted(list(answer_words))):
        answer_token_to_int[token] = i
        answer_int_to_token[i]     = token
    #############################################
    
    return_dict={'len_context_vocab':len(context_words),
                 'len_question_vocab':len(question_words),
                 'len_answer_vocab':len(answer_words),
                'max_context_len':max([len(line.split()) for line in data[0]]),
                'max_question_len':max([len(line.split()) for line in data[1]]),
                'max_answer_len':max([len(line.split()) for line in data[2]]),
                'context_words':context_words,
                'question_words':question_words,
                'answer_words':answer_words,
                'context_token_to_int':context_token_to_int,
                'question_token_to_int':question_token_to_int,
                'answer_token_to_int':answer_token_to_int,
                'answer_int_to_token':answer_int_to_token}
    return return_dict

def process_data(data,data_info):  
    # initiate numpy arrays to hold the data that our seq2seq model will use:
    encoder_input_context = np.zeros(
        (len(data[0]), data_info['max_context_len']),
        dtype='float32')
    encoder_input_question = np.zeros(
        (len(data[1]), data_info['max_question_len']),
        dtype='float32')
    decoder_input_answer = np.zeros(
        (len(data[2]), data_info['max_answer_len']),
        dtype='float32')
    decoder_target_answer = np.zeros(
        (len(data[2]), data_info['max_answer_len'], len(data_info['answer_words'])),
        dtype='float32')
    ###########################################
    # Process samples, to get input, output, target data:
    for i, (input_context, input_question,target_answer) in enumerate(zip(data[0],data[1],data[2])):
        for t, word in enumerate(input_context.split()):
            encoder_input_context[i, t] = data_info['context_token_to_int'][word]

        for t, word in enumerate(input_question.split()):
            encoder_input_question[i, t] = data_info['question_token_to_int'][word]

        for t, word in enumerate(target_answer.split()):
            # decoder_target_answer is ahead of decoder_input_answer by one timestep
            decoder_input_answer[i, t] = data_info['answer_token_to_int'][word]
            if t > 0:
                # decoder_target_answer will be ahead by one timestep
                # and will not include the start character.
                decoder_target_answer[i, t - 1, data_info['answer_token_to_int'][word]] = 1.
    #############################################
    encoder_input={'context_encoder_input':encoder_input_context,'question_encoder_input':encoder_input_question}
    decoder_input={'answer_decoder_input':decoder_input_answer,'answer_decoder_target':decoder_target_answer}
    return_dict={'encoder_input':encoder_input,'decoder_input':decoder_input}
#     print(return_dict.keys())
#     print('encoder_input keys: ',return_dict['encoder_input'].keys())
#     print('decoder_input keys: ',return_dict['decoder_input'].keys())
    return return_dict

def decode_sequence(context_input_seq,question_input_seq,answer_token_to_int,answer_int_to_token,encoder_model,decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict([context_input_seq,question_input_seq])
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = answer_token_to_int['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = answer_int_to_token[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 251):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    return decoded_sentence[:-4] #[:-4] removes the end-of-sequence token
