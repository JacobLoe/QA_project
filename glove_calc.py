import numpy as np
import json
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
#importing the glove library
from glove import Corpus, Glove

from tqdm import tqdm

path='models/baseline_biderec'
# rnn parameters
hidden_size = 100 #100 is the standard
batch_size = 512 #for the training on the GPU this to be has to very large, otherwise the GPU is used very inefficiently
epochs = 100

#glove embedding parameters
glove_dir = '../glove/glove.6B.100d.txt'
embedding_dim = 100
eval_split = 0.2 

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
                train_new['answer'].append('\t'+c+'\n')
train_new['decoder_input']=train_new['answer']
train_new['decoder_input'][0]='\n'
print(len(train_new['context']))
print(len(train_new['question']))
print(len(train_new['answer']))
print(len(train_new['decoder_input']))

def remove_stopwords(lines):
    stop_words=set(stopwords.words('english')) 
    lines_without_stopwords=[]
    string=' '
    for line in tqdm(lines):
        temp_line=[]
        for word in line.split():
            if word not in stop_words: 
                temp_line.append(word) 
        lines_without_stopwords.append(string.join(temp_line))
    return lines_without_stopwords

def split_lines(lines):
    new_lines=[]
    for line in lines:
        new_lines.append(line.split())
    return new_lines

context=split_lines(remove_stopwords(train_new['context']))
question=split_lines(remove_stopwords(train_new['question']))
answer=split_lines(remove_stopwords(train_new['answer']))

# creating a corpus object
corpus = Corpus() 
#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(question, window=10)
#creating a Glove object which will use the matrix created in the above lines to create embeddings
#We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=100, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('question_glove.model')

# creating a corpus object
corpus = Corpus() 
#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(answer, window=10)
#creating a Glove object which will use the matrix created in the above lines to create embeddings
#We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=100, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('answer_glove.model')

# creating a corpus object
corpus = Corpus() 
#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(context, window=10)
#creating a Glove object which will use the matrix created in the above lines to create embeddings
#We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=100, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('context_glove.model')
