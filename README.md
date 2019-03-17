- the repository contains the generated answers in the folder SQuAD as the files 'answers_train.json' and 'answers_adversarial.json'

- if you want to train, extract 'glove.zip' to the same directory as the repository
- to train a new model run 'baseline.py' with python3
  - the file will also immediately run the inference after training
 - running 'adversarial.py' will run inference, if models are provided
 
 - extract 'models.zip' into the path of the repository if you want to skip training
 - to start inference for the train set without training, you have to comment lines 95-106 in 'baseline.py'
 
the project requires the following libraries:

keras

tqdm - pip3 install tqdm

numpy 

nltk

scipy - pip3 install scipy

gensim - pip3 install gensim

bert_embeddings - pip3 install bert_embedding

mxnet - pip3 install mxnet

