import torch
import torch.nn as nn
from encoder import CNNEncoder
from decoder import DecoderRNN

class Encoder_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(Encoder_Decoder, self).__init__()
        self.cnn = CNNEncoder(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.cnn(images)
        outputs = self.decoderRNN(features, captions)
        
        return outputs