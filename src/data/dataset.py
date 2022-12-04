from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
from data.tokenizer import Tokenizer

DATA_DIR = "./data"

class FlickrDataset(Dataset):
    def __init__(self, split):
        self.split = split

        with open(f"{DATA_DIR}/{split}.json", "r") as f:
            self.data = json.load(f)
        
        self.sentence_tokenizer = Tokenizer()
        self.sentence_tokenizer.load(f"{DATA_DIR}/vocab.txt")

        self.vocab_size = self.sentence_tokenizer.vocab_size

    def __getitem__(self, idx):
        data = self.data[idx]

        return self.load_image(data["image"]), self.sentence_tokenizer.encode(data["caption"])

    def load_image(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.data)