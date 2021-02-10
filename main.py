import os
import argparse
from dataset_loading import load_conllu_dataset, load_json_dataset
from model import model_training, model_prediction
import random as rnd
import json

def main(args):
    grams = int(args.grams)
    lemma_split = int(args.lemma_split)
    stemming = int(args.stemming)
    early_stopping = int(args.early_stopping)
    if (early_stopping == -1):
        early_stopping = int(args.epochs)        
    if (args.modus == 'training'):
        train_dataset = load_conllu_dataset(args.data, args.join, args.name, grams, lemma_split, args.modus, stemming, args.folder)
        model_training(train_dataset, args.folder, args.epochs, args.batch, args.dim, args.optimizer, args.loss, args.activation, args.name, early_stopping)
    elif (args.modus == 'prediction'):        
        test_dataset = load_json_dataset(args.data, args.join)
        rnd.seed(75)
        result = model_prediction(test_dataset, args.join, args.modus, args.dim, args.optimizer, args.loss, args.activation, args.name, lemma_split, args.forming_priority, grams, args.folder)
        with open(args.data, encoding='utf8') as f:
            d = json.load(f)    
        for t in d["texts"]:
            for c in t["clauses"]:
                for r in c["realizations"]:
                    field_exists = False
                    for f in r["realizationFields"]:
                        if "Lemma" in f.keys():
                            field_exists = True
                            break
                    if not field_exists:
                        for index, row in result.iterrows():
                            id, lemma = row['DB_ID'], row['LEMMA']
                            textID, clauseID, realizationID = id.split('_')
                            if ((r["textID"] == textID) and (r["clauseID"] == clauseID) and (r["realizationID"] == realizationID)):                                
                                r["realizationFields"].append({"Lemma":[lemma]})        
        with open(args.data, 'w', encoding='utf8') as f:
            json.dump(d, f, ensure_ascii=False)  
    elif (args.modus == 'accuracy'):
        validation_dataset = load_conllu_dataset(args.data, args.join, args.name, grams, lemma_split, args.modus, stemming, args.folder)
        rnd.seed(75)
        model_prediction(validation_dataset, args.join, args.modus, args.dim, args.optimizer, args.loss, args.activation, args.name, lemma_split, args.forming_priority, grams, args.folder)
        
            

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--folder', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--epochs', default='40')
    parser.add_argument('--batch', default='256')
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