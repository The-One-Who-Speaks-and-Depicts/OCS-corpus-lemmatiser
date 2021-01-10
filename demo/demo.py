# streamlit packages
import streamlit as st 
import os

# NLP packages
import pandas as pd
import re
import numpy as np
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import argparse
import random as rnd
import Levenshtein
from fuzzywuzzy import fuzz
from pyjarowinkler import distance as jw
from pyjarowinkler.distance import JaroDistanceException
import traceback

def load_train_dataset(datafile, join, grams, lemma_split):
    join = int(join)
    with open(datafile, 'r', encoding='utf-8') as inp:
        strings = inp.readlines()
    dataset = pd.DataFrame(columns=['WORD', 'LEMMA'])
    counter = 0
    for s in strings:
        split_string = s.split(' ')
        dataset.loc[counter] = [split_string[0], split_string[1]]
        counter = counter + 1            
    return dataset

def test_split(word, pos, join, grams):
    n_grams = []
    if ((grams == 0) or (len(word) <= grams)):
        if (join == 1):
            n_grams.append([word + pos])
        else:
            n_grams.append([word])
    else:
        counter = 0
        while ((len(word) - counter) >= grams):
            resulting_word = ''
            for i in range (counter, counter + grams):
                resulting_word += word[i]
            if (join == 0):
                n_grams.append([resulting_word])
            else:
                n_grams.append([resulting_word + pos])
            counter = counter + 1
    return n_grams

def main():
    """ OCS Lemmatiser """
    # Preparation
    rnd.seed(75)
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    input_words = []
    target_words = []
    name = 'lemmas_split_batch_256_stop_3'
    join = 0
    for index, row in load_train_dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Model', 'lemmatized_' + name + '.txt'), join, 0, 0).iterrows():
      input_words.append(row['WORD'].strip())
      target_words.append(row['LEMMA'].strip())
    for index, row in load_train_dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Model', 'train_' + name + '.txt'), join, 0, lemma_split).iterrows():
      input_text = row['WORD']
      target_text = row['LEMMA']
      target_text = '\t' + target_text + '\n'
      input_texts.append(input_text)
      target_texts.append(target_text)
      for char in input_text:
        if char not in input_characters:
          input_characters.add(char)
      for char in target_text:
        if char not in target_characters:
          target_characters.add(char)
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
      for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
      for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
          # decoder_target_data will be ahead by one timestep
          # and will not include the start character.
          decoder_target_data[i, t - 1, target_token_index[char]] = 1.  
    latent_dim = 256  # latent dimensionality of the encoding space
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation=activation)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')    
    model.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Model', name + '.h5'))

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
      decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
      [decoder_inputs] + decoder_states_inputs,
      [decoder_outputs] + decoder_states)

    reverse_input_char_index = dict(
  (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
  (i, char) for char, i in target_token_index.items())   



    # Description
    st.title("OCS Hybrid Lemmatiser")
    st.subheader("Demo")
    st.markdown("""
    	#### Description
    	This is an app for lemmatisation of OCS tokens by hybrid model
    	consisiting of seq2seq NN, dictionary, and linguistic rules 
    	""")
	
    #Lemmatisation
    token = st.text_area("Enter token to analyze...", "Type Here...")
    partOfSpeech = st.text_area("Enter supposed PoS of token...", "Type Here...")
    if partOfSpeech == 'FRAG':
	    st.write('===')
    elif ((partOfSpeech == 'PUNCT') or (partOfSpeech == 'DIGIT')):
        st.write(token)
    elif ((token in input_words) or ((token + pos) in input_words)):                
        processed_token = ''
        if (join == 0):
            processed_token = token
        else:
            processed_token = token + pos
            try:
                st.write(target_words[input_words.index(token)].strip())
            except ValueError:
                st.write(target_words[input_words.index(row['TOKEN'])].strip())   
    else:
        try:
            predictions = []
            for n_gram in test_split(token, pos, 0, 0): 
                test_sentence_tokenized = np.zeros(
                            (1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
                transferred_token = n_gram
                for t, char in enumerate(transferred_token):
                    try:
                        test_sentence_tokenized[0, t, input_token_index[char]] = 1.
                    except:
                        try:
                            input_token_index.update({char:rnd.randint(min(input_token_index.values()), max(input_token_index.values()))})
                            test_sentence_tokenized[0, t, input_token_index[char]] = 1.
                        except Exception as e:
                            st.write('Error!')
                            # encode the input sequence to get the internal state vectors.
                    states_value = encoder_model.predict(test_sentence_tokenized)
                          
                    # generate empty target sequence of length 1 with only the start character
                    target_seq = np.zeros((1, 1, num_decoder_tokens))
                    target_seq[0, 0, target_token_index['\t']] = 1.
                          
                    # output sequence loop
                    stop_condition = False
                    decoded_sentence = ''
                    while not stop_condition:
                        output_tokens, h, c = decoder_model.predict(
                                [target_seq] + states_value)
                            
                    # sample a token and add the corresponding character to the 
                    # decoded sequence
                    sampled_token_index = np.argmax(output_tokens[0, -1, :])
                    sampled_char = reverse_target_char_index[sampled_token_index]
                    decoded_sentence += sampled_char
                            
                    # check for the exit condition: either hitting max length
                    # or predicting the 'stop' character
                    if (sampled_char == '\n' or 
                                  len(decoded_sentence) > max_decoder_seq_length):
                        stop_condition = True
                              
                    # update the target sequence (length 1).
                    target_seq = np.zeros((1, 1, num_decoder_tokens))
                    target_seq[0, 0, sampled_token_index] = 1.
                            
                    # update states
                    states_value = [h, c]
                predicted = decoded_sentence.strip()
                predictions.append(predicted)
            final_prediction = ''
            if (len(predictions) == 1):
                final_prediction = predictions[0]
            st.write(re.sub(r'#', "", final_prediction))
        except Exception:
            st.write('Error!')

    st.sidebar.subheader("About App")
    st.sidebar.text("OCS Lemmatiser")
    st.sidebar.info("Cudos to the Streamlit Team & Jesse E.Agbe (JCharis)")
	

if __name__ == '__main__':
    main()