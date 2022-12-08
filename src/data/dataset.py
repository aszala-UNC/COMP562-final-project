import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np

DATA_DIR = "./data"

class FlickrDataset(Dataset):
    def __init__(self, split, tokenizer, preprocess, max_length):
        self.split = split

        with open(f"{DATA_DIR}/{split}.json", "r") as f:
            self.data = json.load(f)
        
        self.sentence_tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length

    def __getitem__(self, idx):
        data = self.data[idx]

        image = self.preprocess(self.load_image(data["image"]))
        sent = self.sentence_tokenizer.encode(data["caption"])

        a = np.ones((self.max_length), np.int64) * self.sentence_tokenizer.pad_id
        a[0] = self.sentence_tokenizer.bos_id

        if len(sent) + 2 < self.max_length:        
            a[1: len(sent)+1] = sent
            a[len(sent)+1] = self.sentence_tokenizer.eos_id
        else:                                           
            a[1: -1] = sent[:self.max_length-2]
            a[self.max_length-1] = self.sentence_tokenizer.eos_id

        return data["id"], image, torch.from_numpy(a)

    def load_image(self, path):
        with open(f"{DATA_DIR}/Images/{path}", 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.data)