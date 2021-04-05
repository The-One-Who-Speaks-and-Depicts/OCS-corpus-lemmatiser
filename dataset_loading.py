import pandas as pd
import os
import json
from data_preprocessing import word_n_grams_split
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

def load_json_dataset(datafile, join):
    with open(datafile, encoding='utf-8') as f:
        d = json.load(f)
    dataset = pd.DataFrame(columns=['ID', 'TOKEN', 'WORD', 'POS', 'LEMMA', 'DB_ID'])
    counter = 0
    for t in d["texts"]:
        for c in t["clauses"]:
            for r in c["realizations"]:
                PoS = ""
                for f in r["realizationFields"]:
                    try:
                        PoS = f["PoS"][0]["name"]
                        dataset.loc[counter] = [counter, r["lexemeTwo"], r["lexemeTwo"], PoS, "UNK", r["textID"] + "_" + r["clauseID"] + "_" + r["realizationID"]]
                        counter = counter + 1
                        break
                    except:
                        pass                                    
    return dataset

def load_conllu_dataset(datafile, join, name, grams, lemma_split, modus, stemming, folder):
    join = int(join)
    with open(datafile, encoding='utf-8') as inp:
        strings = inp.readlines()
    dataset = pd.DataFrame(columns=['ID', 'TOKEN', 'WORD', 'POS', 'LEMMA'])
    counter = 0
    word_counter = 0
    if (modus == 'training'):
        with open(folder + '\\lemmatized_' + name + '.txt', 'a', encoding='utf-8') as out_dictionary:
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