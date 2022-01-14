import numpy as np
import re
import os
import tensorflow.keras.utils as ku
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


for textFile in os.listdir('D:\Maynooth University\Sem 1\CS401 - Machine Learning\hw3\data'):
    with open ('data/' + textFile, 'r', encoding='latin1') as f:
        foo = f.read()
        try:
            with open("D:\Maynooth University\Sem 1\CS401 - Machine Learning\hw3\data\Output.txt", "a") as metaF:
                metaF_re = re.search("\n== RESULTS\n(.*?)\n== ISSUES\n", foo, re.I + re.M + re.S).groups()[0]
                #met.seek(0)
                print(metaF_re, textFile=metaF)
                metaF.close()

        except:
            pass


awsome_tokens = Tokenizer()
outputFile = open('D:\Maynooth University\Sem 1\CS401 - Machine Learning\hw3\data\Output.txt').read()
word_bag = outputFile.lower().split("\n")
awsome_tokens.fit_on_texts(word_bag)
total_words = len(awsome_tokens.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in word_bag:
    token_list = awsome_tokens.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)

try:
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

except:
    pass

history = model.fit(predictors, label, epochs=10, verbose=1)