#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import spacy
from time import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping

# Load SpaCy English model
nlp = spacy.load('en', disable=['ner', 'parser'])

# Load the dataset
summary = pd.read_csv("news_summary.csv", encoding='iso-8859-1')
raw = pd.read_csv("news_summary_more.csv", encoding='iso-8859-1')

# Combine text from both datasets
pre1 = raw.iloc[:, 0:2].copy()
pre2 = summary.iloc[:, 0:6].copy()
pre2['text'] = pre2['author'].str.cat(
    pre2['date'].str.cat(pre2['read_more'].str.cat(pre2['text'].str.cat(pre2['ctext'], sep=" "), sep=" "), sep=" "),
    sep=" ")
pre = pd.DataFrame()
pre['text'] = pd.concat([pre1['text'], pre2['text']], ignore_index=True)
pre['summary'] = pd.concat([pre1['headlines'], pre2['headlines']], ignore_index=True)

# Data cleaning
def text_strip(column):
    for row in column:
        row = re.sub("(\\t)", ' ', str(row)).lower()
        row = re.sub("(\\r)", ' ', str(row)).lower()
        row = re.sub("(\\n)", ' ', str(row)).lower()
        row = re.sub("(__+)", ' ', str(row)).lower()
        row = re.sub("(--+)", ' ', str(row)).lower()
        row = re.sub("(~~+)", ' ', str(row)).lower()
        row = re.sub("(\+\++)", ' ', str(row)).lower()
        row = re.sub("(\.\.+)", ' ', str(row)).lower()
        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower()
        row = re.sub("(mailto:)", ' ', str(row)).lower()
        row = re.sub(r"(\\x9\d)", ' ', str(row)).lower()
        row = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower()
        row = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower()
        row = re.sub("(\.\s+)", ' ', str(row)).lower()
        row = re.sub("(\-\s+)", ' ', str(row)).lower()
        row = re.sub("(\:\s+)", ' ', str(row)).lower()
        row = re.sub("(\s+.\s+)", ' ', str(row)).lower()
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            repl_url = url.group(3)
            row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(row))
        except:
            pass
        row = re.sub("(\s+)", ' ', str(row)).lower()
        row = re.sub("(\s+.\s+)", ' ', str(row)).lower()
        yield row

brief_cleaning1 = text_strip(pre['text'])
brief_cleaning2 = text_strip(pre['summary'])

# Tokenize and preprocess the data
t = time()
text = [str(doc) for doc in nlp.pipe(brief_cleaning1, batch_size=5000, n_threads=-1)]
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
summary = ['_START_ ' + str(doc) + ' _END_' for doc in nlp.pipe(brief_cleaning2, batch_size=5000, n_threads=-1)]
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

pre['cleaned_text'] = pd.Series(text)
pre['cleaned_summary'] = pd.Series(summary)

# Set maximum text and summary lengths
max_text_len = 100
max_summary_len = 15

# Select text and summary within the specified lengths
cleaned_text = np.array(pre['cleaned_text'])
cleaned_summary = np.array(pre['cleaned_summary'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if (len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

post_pre = pd.DataFrame({'text': short_text, 'summary': short_summary})

# Add start and end tokens to the summary
post_pre['summary'] = post_pre['summary'].apply(lambda x: 'sostok ' + x + ' eostok')

# Tokenize the text and summary
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(post_pre['text']))
x_tr_seq = x_tokenizer.texts_to_sequences(post_pre['text'])
x_tr = pad_sequences(x_tr_seq, maxlen=max_text_len, padding='post')
x_voc = x_tokenizer.num_words + 1

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(post_pre['summary']))
y_tr_seq = y_tokenizer.texts_to_sequences(post_pre['summary'])
y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_voc = y_tokenizer.num_words + 1

# Remove sequences with only start and end tokens in the summary
ind = []
for i in range(len(y_tr)):
    cnt = 0
    for j in y_tr[i]:
        if j != 0:
            cnt = cnt + 1
    if (cnt == 2):
        ind.append(i)

y_tr = np.delete(y_tr, ind, axis=0)
x_tr = np.delete(x_tr, ind, axis=0)

# Define the Seq2Seq model architecture
latent_dim = 300
embedding_dim = 200

# Encoder
encoder_inputs = Input(shape=(max_text_len,))
enc_emb = Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# Set up the decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Define early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

# Split the data into training and validation sets
x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.1, random_state=0, shuffle=True)

# Train the model
history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=50,
                    callbacks=[es], batch_size=128,
                    validation_data=([x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))

# Functions for decoding sequences
def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sostok']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        if (sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        e_h, e_c = h, c
    return decoded_sentence

def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if ((i != 0 and i != target_word_index['sostok']) and i != target_word_index['eostok']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString

def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if (i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString

# Encode and decode sequences to evaluate the model
for i in range(0, 10):
    print("Review:", seq2text(x_tr[i]))
    print("Original summary:", seq2summary(y_tr[i]))
    print("Predicted summary:", decode_sequence(x_tr[i].reshape(1, max_text_len)))
    print("\n")
