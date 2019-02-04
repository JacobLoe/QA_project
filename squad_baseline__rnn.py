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

# rnn parameters
RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 100

#glove embedding parameters
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, '../glove')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

#open SQuAD-dataset and extract the relevatn data
print('open and extract SQuAD')
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
            if (qas['is_impossible']==False):
                train_new['context'].append(context)
                train_new['question'].append(qas['question'])
                train_new['answer'].append(qas['answers'][0]['text'])

#concatenate the data in on vector for preprocessing
train_all=[]
for line in train_new['context']:
    train_all.append(line)
for line in train_new['question']:
    train_all.append(line)
for line in train_new['answer']:
    train_all.append(line)

#prepare the data to use as input of the rnn
print('prepare the input data for rnn')
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_all)
context_sequences = tokenizer.texts_to_sequences(train_new['context'])
question_sequences = tokenizer.texts_to_sequences(train_new['question'])
answer_sequences = tokenizer.texts_to_sequences(train_new['answer'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

context = pad_sequences(context_sequences, maxlen=MAX_SEQUENCE_LENGTH)
question = pad_sequences(question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
answer = pad_sequences(answer_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of context tensor:', context.shape)
print('Shape of question tensor:', question.shape)
print('Shape of answer tensor:', answer.shape)

# split the data into a training set and a validation set
print('create train and test set')
indices = np.arange(context.shape[0])
np.random.shuffle(indices)
context = context[indices]
question = question[indices]
answer = answer[indices]
num_validation_samples = int(VALIDATION_SPLIT * context.shape[0])

x_train_context = context[:-num_validation_samples]
x_train_question = question[:-num_validation_samples]
y_train_answer = answer[:-num_validation_samples]

x_val_context = context[-num_validation_samples:]
x_val_question = question[-num_validation_samples:]
y_val_answer = answer[-num_validation_samples:]

#get glove embeddings
print('get the glove embeddings')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#extract the glove-embedding to a matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#create a non-trainable embedding layer
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Build model...')

context_layer = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name='Context_input')
encoded_context = embedding_layer(context_layer)
encoded_context = RNN(SENT_HIDDEN_SIZE)(encoded_context)

question_layer = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name='Question_input')
encoded_question = embedding_layer(question_layer)
encoded_question = RNN(QUERY_HIDDEN_SIZE)(encoded_question)

merged = layers.concatenate([encoded_context, encoded_question])
preds = layers.Dense(1000, activation='softmax')(merged) #dimensions of dense layer have to to the same as the answer dimensions

model = Model([context_layer, question_layer], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('successfully built the model')
print(model.summary())

print('Training')
model.fit([x_train_context, x_train_question], y_train_answer,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)

print('Evaluation')
loss, acc = model.evaluate([x_val_context, x_val_question], y_val_answer,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

print('saving weights')
with open('results/baseline_total_weights.vec','w') as f:
    print(model.get_weights(),file=f)
