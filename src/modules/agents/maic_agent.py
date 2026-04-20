import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.distributions as D
import numpy as np
from torch.distributions import kl_divergence


class MAICAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MAICAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions

        NN_HIDDEN_SIZE = args.nn_hidden_size
        activation_func = nn.LeakyReLU()

        self.embed_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2)
        )

        self.inference_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.n_actions, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2)
        )

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        self.msg_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.latent_dim, NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_actions)
        )

        self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, args.attention_dim)
        
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state, bs, t, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        latent_parameters = self.embed_net(h)
        latent_parameters[:, -self.n_agents * self.latent_dim:] = th.clamp(
            th.exp(latent_parameters[:, -self.n_agents * self.latent_dim:]),
            min=self.args.var_floor)

        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if test_mode:
            latent = latent_embed[:, :self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(latent_embed[:, :self.n_agents * self.latent_dim],
                                    (latent_embed[:, self.n_agents * self.latent_dim:]) ** (1 / 2))
            latent = gaussian_embed.rsample() # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        h_repeat = h.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        msg = self.msg_net(th.cat([h_repeat, latent], dim=-1)).view(bs, self.n_agents, self.n_agents, self.n_actions)
        
        query = self.w_query(h).unsqueeze(1)
        key = self.w_key(latent).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = th.bmm(query / (self.args.attention_dim ** (1/2)), key).view(bs, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)
        
        
        if test_mode and self.args.delay_evaluate:
            msg = self._get_delayed_message(bs, t, self.n_agents, msg)
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0 # 通信筛选             
            gated_msg = alpha * msg # 无延迟下通信内容
        elif test_mode:
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0 # 通信筛选 
            gated_msg = alpha * msg
        else:
            gated_msg = alpha * msg


        # return_q = q + th.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)
        return_q = q

        returns = {}
        if 'train_mode' in kwargs and kwargs['train_mode']:
            if hasattr(self.args, 'mi_loss_weight') and self.args.mi_loss_weight > 0:
                returns['mi_loss'] = self.calculate_action_mi_loss(h, bs, latent_embed, return_q)
            if hasattr(self.args, 'entropy_loss_weight') and self.args.entropy_loss_weight > 0:
                query = self.w_query(h.detach()).unsqueeze(1)
                key = self.w_key(latent.detach()).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
                alpha = F.softmax(th.bmm(query, key), dim=-1).reshape(bs, self.n_agents, self.n_agents)
                returns['entropy_loss'] = self.calculate_entropy_loss(alpha)

        return return_q, h, returns

    def calculate_action_mi_loss(self, h, bs, latent_embed, q):
        latent_embed = latent_embed.view(bs * self.n_agents, 2, self.n_agents, self.latent_dim)
        g1 = D.Normal(latent_embed[:, 0, :, :].reshape(-1, self.latent_dim), latent_embed[:, 1, :, :].reshape(-1, self.latent_dim) ** (1/2))
        hi = h.view(bs, self.n_agents, 1, -1).repeat(1, 1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        
        selected_action = th.max(q, dim=1)[1].unsqueeze(-1)
        one_hot_a = th.zeros(selected_action.shape[0], self.n_actions).to(self.args.device).scatter(1, selected_action, 1)
        one_hot_a = one_hot_a.view(bs, 1, self.n_agents, -1).repeat(1, self.n_agents, 1, 1)
        one_hot_a = one_hot_a.view(bs * self.n_agents * self.n_agents, -1)

        latent_infer = self.inference_net(th.cat([hi, one_hot_a], dim=-1)).view(bs * self.n_agents * self.n_agents, -1)
        latent_infer[:, self.latent_dim:] = th.clamp(th.exp(latent_infer[:, self.latent_dim:]), min=self.args.var_floor)
        g2 = D.Normal(latent_infer[:, :self.latent_dim], latent_infer[:, self.latent_dim:] ** (1/2))
        mi_loss = kl_divergence(g1, g2).sum(-1).mean()
        return mi_loss * self.args.mi_loss_weight

    def calculate_entropy_loss(self, alpha):
        alpha = th.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * th.log2(alpha)).sum(-1).mean()
        return entropy_loss * self.args.entropy_loss_weight
    
    def _get_delayed_message(self, bs, t, n_agents, messages):
        if t == 0:
            self.lastest_messages = np.zeros((bs, n_agents, n_agents, 1+messages.shape[-1])) # (t_send, message)
            
        delay_mean = self.args.delay_mean
        delay_var = self.args.delay_var
        sample_size = bs * n_agents * n_agents
        if self.args.fixed_delay: # 固定时延
            samples = np.ones(sample_size) * delay_mean
        else: # 时变时延
            samples = np.random.normal(delay_mean, delay_var, size=sample_size)
            samples = np.round(samples).astype(int)
            samples = np.clip(samples, 0, None)
        
        # 构建message存储结构（t_a, t_s, message）
        delay = samples.reshape((bs, n_agents, n_agents, 1))
        t_send = np.ones((bs, n_agents, n_agents, 1)) * t
        t_arrive = delay + t_send
        messages = messages.detach().cpu().numpy()
        stored_message = np.concatenate((t_arrive, t_send, messages), -1)
        stored_message = np.expand_dims(stored_message, axis = -2) # shape: (bs, n, n, 1, -1)
        
        if t == 0:
            self.message_queue = stored_message
        else:
            self.message_queue = np.concatenate((self.message_queue, stored_message), -2) # shape: (bs, n, n, cur_t, -1)
        
        for b in range(bs):
            for i in range(n_agents):
                    for j in range(n_agents):
                        if i == j: # 自己对自己是无延迟的
                            self.lastest_messages[b,i,j] = self.message_queue[b,i,j,-1,1:]
                            continue
                        channel_messages = self.message_queue[b,i,j]
                        indices = np.where(channel_messages[:, 0] == t)
                        first_dim_indices = indices[0]
                        if len(first_dim_indices) != 0:
                            index = first_dim_indices[-1]
                            cur_message = channel_messages[index]
                            if self.lastest_messages[b,i,j, 0] < cur_message[1]:
                                self.lastest_messages[b,i,j] = cur_message[1:]
                        # else:
                        #         self.lastest_messages[b,i,j] = np.zeros_like(self.lastest_messages[b,i,j])
            # if b == 0:
            #     print("time: ", t, "used_message_timestep: ", self.lastest_messages[b,:,:,0])
        return th.from_numpy(self.lastest_messages[:,:,:,1:]).cuda()
                            
                            
                        
                        
                            
        
        
        
        
            
        
