import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, output_size, infer_len, rnn_hidden_dim):
        super(TransformerDecoder, self).__init__()
        
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_ffn)
            for _ in range(num_layers)
        ])
        self.combine_net = nn.Linear(rnn_hidden_dim + output_size, d_model)
        self.infer_len = infer_len
        self.linear = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.output_size = output_size
        
    def forward(self, tgt, memory, last_actions):
        combined_input = torch.cat([tgt, last_actions], dim = -1) # hidden + last_action
        output = self.combine_net(combined_input)
        
        for layer in self.decoder_layers:
            output = layer(output, memory)

        predicted_output = self.linear(output)  # 预测输出
        
        return predicted_output
    
    def calculate_infer_loss(self, hidden, intent, last_actions, action_preference, terminated_mask, action_mask):
        loss_fn = nn.CrossEntropyLoss()
        # print(hidden.shape)
        # print(last_actions.shape)
        predicted_output = self.forward(hidden.squeeze(), intent.squeeze(), last_actions)
        action_mask = action_mask.reshape(-1, self.output_size)
        # print(action_mask.shape)
        # print(action_preference.shape)
        # print(predicted_output.shape)
        predicted_output = torch.mul(predicted_output, action_mask)
        loss = loss_fn(predicted_output.reshape(-1, self.output_size), action_preference.reshape(1, -1).squeeze())
        # terminated_mask = terminated_mask.reshape()
        # loss *= terminated_mask
        # loss *= action_mask
        return loss