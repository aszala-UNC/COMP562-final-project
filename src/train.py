import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

import sys
import os
sys.path.append(os.getcwd())

from data.dataset import FlickrDataset, DATA_DIR
from model.model import Encoder_Decoder
from data.tokenizer import Tokenizer

def train(tokenizer):
    dataset_train = FlickrDataset("train", tokenizer=tokenizer, preprocess=PREPROCESS, max_length=MAX_WORD_LENGTH)
    dataset_val = FlickrDataset("val", tokenizer=tokenizer, preprocess=PREPROCESS, max_length=MAX_WORD_LENGTH)

    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = Encoder_Decoder(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"Training on {len(list(model.parameters()))} parameters")

    best_val_loss = 1e9

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}")

        model.train()

        for i, (img_id, images, captions) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Step"):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(images, captions)

            loss = criterion(outputs.view(-1, VOCAB_SIZE), captions.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % VAL_STEP) == 0 or epoch == (NUM_EPOCHS-1):
            val_loss = validation_loop(val_dataloader, model, criterion)

            print(f"Epoch {epoch} | Val Loss: {val_loss}")
            print(f"Best Previous Val Loss: {best_val_loss}")

            if val_loss < best_val_loss:
                print("Updating Best Val Loss")
                best_val_loss = val_loss

                print("Saving Current Model")
                data = {
                    "model_state": model.state_dict(),
                    "embed_size": EMBED_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "vocab_size": VOCAB_SIZE
                }

                torch.save(data, f"{CHECKPOINT_PATH}/best.pth")
            else:
                print("Best Val Loss Unchanged")

            print(f"Best Current Val Loss: {best_val_loss}")


def validation_loop(val_dataloader, model, criterion):
    model.eval()

    losses = []
    with torch.no_grad():
        for i, (img_id, images, captions) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation Step"):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(images, captions)

            loss = criterion(outputs.view(-1, VOCAB_SIZE), captions.view(-1))
            losses.append(loss.item())

    return np.average(losses)


def test(model_path, tokenizer, preprocess, device, max_length=32, num_workers=4):
    print("Loading testing dataset")
    dataset_test = FlickrDataset("test", tokenizer=tokenizer, preprocess=preprocess, max_length=max_length)
    test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=num_workers)

    print("Loading Model")
    model_data = torch.load(model_path, map_location="cpu")

    model = Encoder_Decoder(model_data["embed_size"], model_data["hidden_size"], model_data["vocab_size"]).to(device)
    model.load_state_dict(model_data["model_state"], strict=False)
    model.eval()

    print("Done")

    values = { }

    with torch.no_grad():
        for img_id, images, _ in tqdm(test_dataloader, total=len(test_dataloader), desc="Testing"):
            images = images.to(device)
            images = model.cnn(images).unsqueeze(dim=1)

            out_captions = model.decoderRNN.predict(images, max_length)

            raw_output = tokenizer.decode(out_captions).split(" ")
            try:
                x = raw_output.index("<EOS>")
            except:
                x = len(raw_output)

            final_out = raw_output[1:x]

            sent = ' '.join(final_out)

            values[img_id[0]] = sent

    return values 


if __name__ == "__main__":
    CHECKPOINT_PATH = "./checkpoints/resent_attention"

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    NUM_EPOCHS = 500
    BATCH_SIZE = 50
    NUM_WORKERS = 4

    VAL_STEP = 1

    ## Hyper Parameters
    EMBED_SIZE = 256
    HIDDEN_SIZE = 100
    WEIGHT_DECAY = 0.005
    LEARNING_RATE = 1e-4

    MAX_WORD_LENGTH = 32

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PREPROCESS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tokenizer = Tokenizer()
    tokenizer.load(f"{DATA_DIR}/vocab.txt")

    VOCAB_SIZE = tokenizer.vocab_size

    train(tokenizer)
    # test(f"{CHECKPOINT_PATH}/best.pth", tokenizer)