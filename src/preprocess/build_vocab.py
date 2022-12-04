import csv
import sys
import os
sys.path.append(os.getcwd())

from src.data.tokenizer import Tokenizer

all_sentences = []

with open("./data/captions.csv", 'r') as f:
    rows = csv.DictReader(f)

    for row in rows:
        all_sentences.append(row["caption"])


t = Tokenizer()
t.build_vocab(all_sentences)
t.dump("./data/vocab.txt")


