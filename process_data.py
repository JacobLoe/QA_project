import numpy as np
def process_data(data):
    context=data[0]
    question=data[1]
    answer=data[2]
    ######################################
    # create vocabulary
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
    #########################################
    # get lengths and sizes of vocab and data
    len_context_vocab = len(context_words)
    len_question_vocab = len(question_words)
    len_answer_vocab = len(answer_words)

    max_context_len = max([len(line.split()) for line in context])
    max_question_len = max([len(line.split()) for line in question])
    max_answer_len = max([len(line.split()) for line in answer])

    len_context = len(context)
    len_question = len(question)
    len_answer = len(answer)

    print('length of vocab: ',len_context_vocab,len_question_vocab,len_answer_vocab)
    print('max length of data:',max_context_len,max_question_len,max_answer_len)
    print('length of data: ',len_context,len_question,len_answer)
    ###############################################################
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

    #Tokenize the words :
    for i,token in enumerate(input_context_words):
        context_token_to_int[token] = i
        context_int_to_token[i]     = token

    for i,token in enumerate(input_question_words):
        question_token_to_int[token] = i
        question_int_to_token[i]     = token

    for i,token in enumerate(target_answer_words):
        answer_token_to_int[token] = i
        answer_int_to_token[i]     = token
    ######################################
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
    print('shape of the input for the model: ',
        np.shape(encoder_input_context),
          np.shape(encoder_input_question),
          np.shape(decoder_input_answer),
          np.shape(decoder_target_answer))
    ###########################################
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
    #############################################
    encoder_input={'context_encoder_input':encoder_input_context,'question_encoder_input':encoder_input_question}
    decoder_input={'answer_decoder_input':decoder_input_answer,'answer_decoder_target':decoder_target_answer}
    token_to_int={'context_token_to_int':context_token_to_int,'question_token_to_int':question_token_to_int,'answer_token_to_int':answer_token_to_int}
    int_to_token={'context_int_to_token':context_int_to_token,'question_int_to_token':question_int_to_token,'answer_int_to_token':answer_int_to_token}
    len_vocab={'context_len_vocab':len_context_vocab,'question_len_vocab':len_question_vocab,'answer_len_vocab':len_answer_vocab}
    return_dict={'encoder_input':encoder_input,'decoder_input':decoder_input,
                 'len_vocab':len_vocab,'token_to_int':token_to_int,'int_to_token':int_to_token}
    print(return_dict.keys())
    print('encoder_input keys: ',return_dict['encoder_input'].keys())
    print('decoder_input keys: ',return_dict['decoder_input'].keys())
    print('len_vocab: ',return_dict['len_vocab'].keys())
    print('token_to_int: ',return_dict['token_to_int'].keys())
    return return_dict
