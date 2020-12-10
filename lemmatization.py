import pandas as pd
import re
import numpy as np
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import argparse
import os
import random as rnd
import Levenshtein
from fuzzywuzzy import fuzz
from pyjarowinkler import distance as jw
from pyjarowinkler.distance import JaroDistanceException
import traceback


def get_destem(word, lemma):
  relevant_word = ''
  relevant_lemma = ''
  if (word == lemma):
    return word, lemma
  if (len(lemma) < len(word)):
    while (len(lemma) < len(word)):
      lemma += "#"
  if (len(word) < len(lemma)):
    while (len(word) < len(lemma)):
      word += "#"
  if ((lemma[0] == word[0]) and (lemma[1] == word[1])):
        counter = 0    
        while (lemma[counter + 1] == word[counter + 1]):
          counter = counter + 1
        for i in range(counter, len(lemma)):
          relevant_word += word[i]
          relevant_lemma += lemma[i]
        return relevant_word.strip('#'), relevant_lemma.strip('#')
  else:
    return word.strip('#'), lemma.strip('#')

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
            
            

def word_n_grams_split(word, pos, lemma, join, grams, lemma_split, stemming):
    if (stemming == 1):
      word, lemma = get_destem(word, lemma)
    n_grams = []
    if ((grams == 0) or (len(word) <= grams)):
        if (join == 0):
            n_grams.append([word, lemma])
        else:
            n_grams.append([word + pos, lemma])
    else:
        counter = 0
        while ((len(word) - counter) >= grams):
            resulting_word = ''
            if (lemma_split == 1):
              resulting_lemma = ''            
              if (len(lemma) < len(word)):
                while (len(lemma) < len(word)):
                  lemma += "#"
              if (len(word) < len(lemma)):
                while (len(word) < len(lemma)):
                  word += "#"
            for i in range (counter, counter + grams):
                resulting_word += word[i]
                if (lemma_split == 1):
                  resulting_lemma += lemma[i]
            if (join == 0):
                if (lemma_split == 1):
                  n_grams.append([resulting_word, resulting_lemma])
                else:
                  n_grams.append([resulting_word, lemma])
            else:
                if (lemma_split == 1):
                  n_grams.append([resulting_word + pos, resulting_lemma])
                else:
                  n_grams.append([resulting_word + pos, lemma])
            counter = counter + 1
    return n_grams

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

def load_json_dataset(datafile, join):
    import json
    with open(datafile, encoding='utf-8') as f:
        d = json.load(f)
    dataset = pd.DataFrame(columns=['WORD', 'POS', 'LEMMA'])
    counter = 0
    for t in d["texts"]:
        for c in t["clauses"]:
            for r in c["realizations"]:
                dataset.loc[counter] = [r["lexemeTwo"], r["realizationFields"][0]["PoS"][0]["name"], "UNK"]
                counter = counter + 1
    return dataset

def load_conllu_dataset(datafile, join, name, grams, lemma_split, modus, stemming):
    join = int(join)
    with open(datafile, encoding='utf-8') as inp:
        strings = inp.readlines()
    dataset = pd.DataFrame(columns=['ID', 'TOKEN', 'WORD', 'POS', 'LEMMA'])
    counter = 0
    word_counter = 0
    if (modus == 'training'):
        with open(args.folder + '\\lemmatized_' + name + '.txt', 'a', encoding='utf-8') as out_dictionary:
            for s in strings:
                if (s[0] != "#" and s.strip()):
                    split_string = s.split('\t')
                    try:
                        out_dictionary.write(split_string[1] + ' ' + split_string[2] + '\n')
                        for gram in word_n_grams_split(split_string[1], split_string[3], split_string[2], join, grams, lemma_split, stemming):                            
                            if (join == 1):
                                dataset.loc[counter] = [word_counter, split_string[1] + split_string[3], gram[0], split_string[3], gram[1]]
                            else:
                                dataset.loc[counter] = [word_counter, split_string[1], gram[0], split_string[3], gram[1]]
                            counter = counter + 1
                        word_counter = word_counter + 1
                    except Exception:
                        traceback.print_exc()
    else:
        for s in strings:
            if (s[0] != "#" and s.strip()):
                split_string = s.split('\t')
                dataset.loc[counter] = [counter, split_string[1], split_string[1], split_string[3], split_string[2]]
                counter = counter + 1
    return dataset

def model_training(dataset, folder, epochs, batch, dim, optim, loss, activation, model_name, early_stopping):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(args.folder + '\\train_' + model_name + '.txt', 'a', encoding='utf-8') as out:
        for index, row in dataset.iterrows():
          out.write(row['WORD'].strip() + ' ' + row['LEMMA'].strip() + '\n')
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
    batch_size = int(batch)  # batch size for training
    epochs = int(epochs)  # number of epochs to train for
    latent_dim = int(dim)  # latent dimensionality of the encoding space
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
    model = Model(inputs=[encoder_inputs, decoder_inputs], 
              outputs=decoder_outputs)
    model.compile(optimizer=optim, loss=loss)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[callback])
    model.save(args.folder + '\\' + model_name + '.h5')

def prepare_metrics(given_list, metrics):
    raw_list = []    
    for i in given_list:
        raw_list.append(i[2])
    Q1, Q3 = np.percentile(raw_list, [25,75])
    IQR=Q3-Q1
    minimum=Q1-1.5*IQR
    maximum=Q3+1.5*IQR
    result_list = []
    errors_data = pd.DataFrame(columns=['TRUE', 'PRED', 'METRICS', 'RESULT'])
    counter = 0
    for i in given_list:
        if (i[2] > minimum and i[2] < maximum):
            result_list.append(i[2])
        elif (i[2] < minimum or i[2] > maximum):
            errors_data.loc[counter] = [i[0], i[1], metrics, i[2]]
            counter = counter + 1
    return raw_list, result_list, errors_data
    
def dameraulevenshtein(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]          

def model_prediction(dataset, join, modus, dim, optim, loss, activation, name, lemma_split, priority, grams):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    input_words = []
    target_words = []
    for index, row in load_train_dataset(args.folder + '\\lemmatized_' + name + '.txt', join, 0, 0).iterrows():
      input_words.append(row['WORD'].strip())
      target_words.append(row['LEMMA'].strip())
    for index, row in load_train_dataset(args.folder + '\\train_' + name + '.txt', join, 0, lemma_split).iterrows():
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
    latent_dim = int(dim)  # latent dimensionality of the encoding space
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
    model.compile(optimizer=optim, loss=loss)    
    model.load_weights(args.folder + '\\' + name + '.h5')

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
    prediction_submitted = False
    prediction = ''
    current_id = 0
    accurate_answers = 0
    total_answers = 0
    Levenshteins = []
    damerau_Levenshteins = []
    jaro_Winklers = []
    for index, row in dataset.iterrows():
        if (row['ID'] > current_id):
            prediction = ''
            prediction_submitted = False
            current_id = row['ID']
        if (prediction_submitted):
            if (modus == 'prediction'):
                row['LEMMA'] == prediction
            else:
                if (prediction == row['LEMMA'].strip()):
                    accurate_answers = accurate_answers + 1
                Levenshteins.append([row['LEMMA'].strip(), prediction, Levenshtein.distance(row['LEMMA'].strip(), prediction)])
                try:
                    damerau_Levenshteins.append([row['LEMMA'].strip(), prediction, dameraulevenshtein(row['LEMMA'].strip(), predicted)])
                except Exception:
                    traceback.print_exc()
                try:
                    jaro_winkler = jw.get_jaro_distance(row['LEMMA'].strip(), predicted)
                    jaro_Winklers.append([row['LEMMA'].strip(), prediction, jaro_winkler])
                except JaroDistanceException:
                    jaro_Winklers.append([row['LEMMA'].strip(), pred, 0])
                total_answers = total_answers + 1
            continue        
        predicted = ''
        if ((row['LEMMA'].strip() == "UNK") or (modus == "accuracy")):
            partOfSpeech = row['POS'].strip()
            if partOfSpeech == 'FRAG':
                if (modus == 'prediction'):
                    row['LEMMA'] = '==='
                else:                    
                    predicted = '==='
                    if (predicted == row['LEMMA'].strip()):
                        accurate_answers = accurate_answers + 1
                    Levenshteins.append([row['LEMMA'].strip(), prediction, Levenshtein.distance(row['LEMMA'].strip(), prediction)])
                    try:
                        damerau_Levenshteins.append([row['LEMMA'].strip(), prediction, dameraulevenshtein(row['LEMMA'].strip(), predicted)])
                    except Exception:
                        traceback.print_exc()
                    try:
                        jaro_winkler = jw.get_jaro_distance(row['LEMMA'].strip(), predicted)
                        jaro_Winklers.append([row['LEMMA'].strip(), prediction, jaro_winkler])
                    except JaroDistanceException:
                        jaro_Winklers.append([row['LEMMA'].strip(), pred, 0])
                    total_answers = total_answers + 1
                    prediction = predicted
                    prediction_submitted = True
            elif ((partOfSpeech == 'PUNCT') or (partOfSpeech == 'DIGIT')):
                if (modus == 'prediction'):
                    row['LEMMA'] = row['TOKEN']
                else:
                    predicted = row['TOKEN']
                    if (predicted == row['LEMMA'].strip()):
                        accurate_answers = accurate_answers + 1
                    Levenshteins.append([row['LEMMA'].strip(), prediction, Levenshtein.distance(row['LEMMA'].strip(), prediction)])
                    try:
                        damerau_Levenshteins.append([row['LEMMA'].strip(), prediction, dameraulevenshtein(row['LEMMA'].strip(), predicted)])
                    except Exception:
                        traceback.print_exc()
                    try:
                        jaro_winkler = jw.get_jaro_distance(row['LEMMA'].strip(), predicted)
                        jaro_Winklers.append([row['LEMMA'].strip(), prediction, jaro_winkler])
                    except JaroDistanceException:
                        jaro_Winklers.append([row['LEMMA'].strip(), pred, 0])
                    total_answers = total_answers + 1
                    prediction = predicted
                    prediction_submitted = True
            elif ((row['TOKEN'] in input_words) or ((row['TOKEN'] + row['POS']) in input_words)):                
                token = ''
                if (join == 0):
                    token = row['TOKEN']
                else:
                    token = row['TOKEN'] + row['POS']
                if (modus == 'prediction'):
                    try:
                        row['LEMMA'] = target_words[input_words.index(token)].strip()
                    except ValueError:
                        row['LEMMA'] = target_words[input_words.index(row['TOKEN'])].strip()
                else:
                    try:
                        predicted = target_words[input_words.index(token)].strip()
                    except ValueError:
                        predicted = target_words[input_words.index(row['TOKEN'])].strip()
                    if (predicted == row['LEMMA'].strip()):
                        accurate_answers = accurate_answers + 1
                    Levenshteins.append([row['LEMMA'].strip(), prediction, Levenshtein.distance(row['LEMMA'].strip(), prediction)])
                    try:
                        damerau_Levenshteins.append([row['LEMMA'].strip(), prediction, dameraulevenshtein(row['LEMMA'].strip(), predicted)])
                    except:
                        traceback.print_exc()
                    try:
                        jaro_winkler = jw.get_jaro_distance(row['LEMMA'].strip(), predicted)
                        jaro_Winklers.append([row['LEMMA'].strip(), prediction, jaro_winkler])
                    except JaroDistanceException:
                        jaro_Winklers.append([row['LEMMA'].strip(), pred, 0])
                    total_answers = total_answers + 1
                    prediction = predicted
                    prediction_submitted = True
            else:
                try:
                    predictions = []
                    for n_gram in test_split(row['WORD'], row['POS'], join, grams): 
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
                                    traceback.print_exc()
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
                        if (modus == 'prediction'):
                            final_prediction = ''
                            if (lemma_split == 1):
                                if (len(predictions) == 1):
                                    final_prediction = predictions[0]
                                else:
                                    if (priority == 'back'):
                                        counter = 0                                 
                                        while (counter < (len(predictions) - 1)):
                                            final_prediction += predictions[counter][0]
                                            counter  = counter + 1
                                        final_prediction += predictions[len(predictions) - 1]
                                    else:
                                        final_prediction += predictions[0]
                                        for i in range(1, len(predictions)):
                                            if ((len(predictions[i]) - 1) == -1):
                                                continue
                                            else:
                                                final_prediction += predictions[i][len(predictions[i]) - 1]
                            else:
                                for pred in predictions:
                                    final_prediction += pred
                                    final_prediction += '%'
                            row['LEMMA'] = re.sub(r'#', "", final_prediction)
                        else:
                            if (lemma_split == 0):
                                for pred in predictions:
                                    pred = re.sub(r'#', "", pred)
                                    if (pred == row['LEMMA'].strip()):
                                        accurate_answers = accurate_answers + 1
                                    Levenshteins.append([row['LEMMA'].strip(), pred, Levenshtein.distance(row['LEMMA'].strip(), pred)])
                                    try:
                                        damerau_Levenshteins.append([row['LEMMA'].strip(), pred, dameraulevenshtein(row['LEMMA'].strip(), pred)])
                                    except Exception as e:
                                        traceback.print_exc()
                                    try:
                                        jaro_winkler = jw.get_jaro_distance(row['LEMMA'].strip(), pred)
                                        jaro_Winklers.append([row['LEMMA'].strip(), pred, jaro_winkler])
                                    except JaroDistanceException:
                                        jaro_Winklers.append([row['LEMMA'].strip(), pred, 0])
                                    total_answers = total_answers + 1
                            else:
                                final_prediction = ''
                                if (len(predictions) == 1):
                                    final_prediction = predictions[0]
                                else:
                                    if (priority == 'back'):
                                        counter = 0                                 
                                        while (counter < (len(predictions) - 1)):
                                            final_prediction += predictions[counter][0]
                                            counter  = counter + 1
                                        final_prediction += predictions[len(predictions) - 1]
                                    else:
                                        final_prediction += predictions[0]
                                        for i in range(1, len(predictions)):
                                            if ((len(predictions[i]) - 1) == -1):
                                                continue
                                            else:
                                                final_prediction += predictions[i][len(predictions[i]) - 1]
                                pred = re.sub(r'#', "", final_prediction)
                                if (pred == row['LEMMA'].strip()):
                                        accurate_answers = accurate_answers + 1
                                Levenshteins.append([row['LEMMA'].strip(), pred, Levenshtein.distance(row['LEMMA'].strip(), pred)])
                                try:
                                    damerau_Levenshteins.append([row['LEMMA'].strip(), pred, dameraulevenshtein(row['LEMMA'].strip(), pred)])
                                except Exception:
                                    traceback.print_exc()
                                try:
                                    jaro_winkler = jw.get_jaro_distance(row['LEMMA'].strip(), pred)
                                    jaro_Winklers.append([row['LEMMA'].strip(), pred, jaro_winkler])
                                except JaroDistanceException:
                                    jaro_Winklers.append([row['LEMMA'].strip(), pred, 0])
                                total_answers = total_answers + 1
                        prediction_submitted = True                        
                except Exception:
                    traceback.print_exc()
            if ((modus == 'prediction') and (prediction_submitted == True)):
                with open(args.folder + '\\result.txt', 'a', encoding='utf-8') as out:
                    out.write(row['TOKEN'].strip() + ' ' + row['LEMMA'] + '\n')
        else:
            if (modus == 'prediction'):
                with open(args.folder + '\\result.txt', 'a', encoding='utf-8') as out:
                    out.write(row['TOKEN'].strip() + ' ' + row['LEMMA'] + '\n')    
    if (modus == 'accuracy'):
        print('Accuracy score: ' + str(accurate_answers/total_answers*100) + '%')
        error_datasets = []
        raw_levenshteins, cleared_levenshteins, errors_levenshteins = prepare_metrics(Levenshteins, 'Levenshtein distance')
        print('Raw average Levenshtein distance: ' + str(sum(raw_levenshteins)/len(raw_levenshteins)))
        print('Normalized average Levenshtein distance: ' + str(sum(cleared_levenshteins)/len(cleared_levenshteins)))
        error_datasets.append(errors_levenshteins)        
        try:
            raw_damerau_levenshteins, cleared_damerau_levenshteins, errors_damerau_levenshteins = prepare_metrics(damerau_Levenshteins, 'Damerau-Levenshtein distance')
            print('Raw average Damerau-Levenshtein distance: ' + str(sum(raw_damerau_levenshteins)/len(raw_damerau_levenshteins)))
            print('Normalized average Damerau-Levenshtein distance: ' + str(sum(cleared_damerau_levenshteins)/len(cleared_damerau_levenshteins)))
            error_datasets.append(errors_damerau_levenshteins)
        except ZeroDivisionError:
            cleared_damerau_levenshteins = [i for i in raw_damerau_levenshteins if i > 0]
            print('Normalized average Damerau-Levenshtein distance: ' + str(sum(cleared_damerau_levenshteins)/len(cleared_damerau_levenshteins)))
        try:
            raw_jaro_winklers, cleared_jaro_winklers, errors_jaro_winklers = prepare_metrics(jaro_Winklers, 'Jaro-Winkler distance')
            print('Raw average Jaro-Winkler distance: ' + str(sum(raw_jaro_winklers)/len(raw_jaro_winklers)))
            print('Normalized average Jaro-Winkler distance: ' + str(sum(cleared_jaro_winklers)/len(cleared_jaro_winklers)))
            error_datasets.append(errors_jaro_winklers)
        except ZeroDivisionError:
            cleared_jaro_winklers = [i for i in raw_jaro_winklers if i > 0 and i < 1]
            print('Normalized average Jaro-Winkler distance: ' + str(sum(cleared_jaro_winklers)/len(cleared_jaro_winklers)))
        errors = pd.concat(error_datasets)
        errors.to_csv(args.folder + '\\errors_' + name + '.csv', index = False, encoding='utf-8')


            
                
def main(args):
    grams = int(args.grams)
    lemma_split = int(args.lemma_split)
    stemming = int(args.stemming)
    early_stopping = int(args.early_stopping)
    if (early_stopping == -1):
        early_stopping = int(args.epochs)        
    if (args.modus == 'training'):
        train_dataset = load_conllu_dataset(args.data, args.join, args.name, grams, lemma_split, args.modus, stemming)
        model_training(train_dataset, args.folder, args.epochs, args.batch, args.dim, args.optimizer, args.loss, args.activation, args.name, early_stopping)
    elif (args.modus == 'prediction'):        
        test_dataset = load_json_dataset(args.data, args.join)
        rnd.seed(75)
        model_prediction(test_dataset, args.join, args.modus, args.dim, args.optimizer, args.loss, args.activation, args.name, lemma_split, args.forming_priority, grams)
    elif (args.modus == 'accuracy'):
        validation_dataset = load_conllu_dataset(args.data, args.join, args.name, grams, lemma_split, args.modus, stemming)
        rnd.seed(75)
        model_prediction(validation_dataset, args.join, args.modus, args.dim, args.optimizer, args.loss, args.activation, args.name, lemma_split, args.forming_priority, grams)
        
            

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--folder', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--epochs', default='40')
    parser.add_argument('--batch', default='128')
    parser.add_argument('--dim', default='256')
    parser.add_argument('--join', default='0')
    parser.add_argument('--modus', default='training')
    parser.add_argument('--optimizer', default='rmsprop')
    parser.add_argument('--loss', default='categorical_crossentropy')
    parser.add_argument('--activation', default='softmax')
    parser.add_argument('--grams', default='0')
    parser.add_argument('--name', default='seq2seq')
    parser.add_argument('--lemma_split', default='0')
    parser.add_argument('--stemming', default='0')
    parser.add_argument('--forming_priority', default='forward')
    parser.add_argument('--early_stopping', default='-1')
    args = parser.parse_args()
    main(args)