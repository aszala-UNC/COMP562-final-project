import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.5)
        # self.dropout = lambda x: x

        self.hidden_size = hidden_size

        self.attention = Attention(embed_size, hidden_size, 256)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        self.hidden = self.get_init_hidden(features.size(dim=0), features.device)

        embeds = self.dropout(self.embed(captions))

        features = features.unsqueeze(dim=1)

        ###
        # Attention
        ###
        features = self.attention(features, self.hidden[0])

        inputs = torch.cat((features, embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)

        outputs = self.linear(lstm_out)

        return outputs

    def predict(self, features, max_length):
        batch_size = features.size(dim=0)
        final_output = []
        hidden = self.get_init_hidden(batch_size, features.device)

        while True:
            ###
            # Attention
            ###
            # features = self.attention(features, hidden[0]).T

            lstm_out, hidden = self.lstm(features, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)
            
            _, max_idx = torch.max(outputs, dim=1)
            
            final_output.append(max_idx.cpu().numpy()[0].item())

            if (len(final_output) >= max_length):
                break

            features = self.dropout(self.embed(max_idx))
            features = features.unsqueeze(1)

        return final_output

    def get_init_hidden(self, batch_size, device):
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        return (h_0, c_0)


import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))
        
        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)
        
        alpha = F.softmax(attention_scores,dim=1)
        
        attention_weights = features * alpha
        attention_weights = attention_weights.sum(dim=1)
        attention_weights = attention_weights.view(attention_weights.size(dim=1), attention_weights.size(dim=0), -1)
        
        return attention_weights
