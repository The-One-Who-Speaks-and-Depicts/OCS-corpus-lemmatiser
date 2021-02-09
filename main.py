import os
import argparse
from dataset_loading import load_conllu_dataset, load_json_dataset
from model import model_training, model_prediction
import random as rnd

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
    #main(args)