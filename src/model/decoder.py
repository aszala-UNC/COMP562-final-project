import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.5)

        self.hidden_size = hidden_size
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        h_0 = torch.zeros(1, features.size(dim=0), self.hidden_size).to(features.device)
        c_0 = torch.zeros(1, features.size(dim=0), self.hidden_size).to(features.device)
        hidden_init = (h_0, c_0)

        embeds = self.embed(captions)

        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        lstm_out, _ = self.lstm(inputs, hidden_init)

        outputs = self.linear(lstm_out)

        return self.dropout(outputs)