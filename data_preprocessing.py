import pandas as pd
import numpy  as np

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