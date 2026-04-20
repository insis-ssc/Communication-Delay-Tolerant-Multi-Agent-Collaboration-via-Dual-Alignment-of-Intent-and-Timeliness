import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.distributions as D
from torch.distributions import kl_divergence
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Code_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(Code_Agent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.intent_dim = args.intent_dim # 意图变量 维度
        # self.content_dim = args.content_dim # 消息内容变量 维度
        self.n_actions = args.n_actions

        NN_HIDDEN_SIZE = args.nn_hidden_size
        activation_func = nn.LeakyReLU()

        self.intent_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.n_actions, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.intent_dim * 2)
        )

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.attention_dim, args.n_actions) # Net for local-Q Evaluation
        
        self.infer_net = nn.Sequential(
            nn.Linear(input_shape + args.intent_dim + args.time_dim, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_actions),
        )

        self.w_query = nn.Linear(args.intent_dim, args.attention_dim)
        self.w_key = nn.Linear(args.intent_dim, args.attention_dim)
        self.w_value = nn.Linear(args.rnn_hidden_dim+args.intent_dim, args.attention_dim)
        
        nn.init.xavier_uniform_(self.w_query.weight)
        nn.init.constant_(self.w_query.bias, 0.0)
        nn.init.xavier_uniform_(self.w_key.weight)
        nn.init.constant_(self.w_key.bias, 0.0)
        
        # self.decoder_net = nn.Sequential(
        #     nn.Linear(args.intent_dim, NN_HIDDEN_SIZE),
        #     nn.BatchNorm1d(NN_HIDDEN_SIZE),
        #     activation_func,
        #     nn.Linear(NN_HIDDEN_SIZE, args.n_actions)
        # )
        
        self.lastest_message = None
        self.current_intent = None
        
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, last_actions,hidden_state, bs, t, t_env,test_mode=False, **kwargs):
        if t == 0: # clear lastest_message and current_intent
            self.init_agent(bs)
        
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in) # hidden state
       
        last_actions = last_actions.reshape(bs * self.n_agents, self.args.n_actions)

        intent_parameters = self.intent_net(th.cat([h,last_actions], dim = -1))
        intent_parameters[:, -self.intent_dim:] = th.clamp(
            th.exp(intent_parameters[:, -self.intent_dim:]),
            min=self.args.var_floor)

        intent_embed = intent_parameters.reshape(bs * self.n_agents, self.intent_dim * 2)

        if test_mode: 
            intent = intent_embed[:, :self.intent_dim]
        else:
            # gaussian_embed = D.Normal(intent_embed[:, :self.intent_dim],
            #                         (intent_embed[:, self.intent_dim:]) ** (1 / 2))
            # intent = gaussian_embed.rsample() # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
            eps = th.randn_like(intent_embed[:, self.intent_dim:])
            intent = intent_embed[:, :self.intent_dim] + (eps*intent_embed[:, self.intent_dim:])

        received_time, received_intent, received_content = self.lastest_message[:,:,:,0], self.lastest_message[:,:,:,1:self.intent_dim+1], self.lastest_message[:,:,:,1:] # [t;intent;content] stored in the received buffer

        # Attention over real_time messages, force on most related messages
        query = self.w_query(intent).unsqueeze(1) # self intent as Query
        if test_mode and self.args.delay_evaluate:
            key = self.w_key(received_intent.cuda()).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2) # others' intents as Key
        else:
            key = self.w_key(intent.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs, self.n_agents, self.n_agents, -1)).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2) # others' intents as Key
        
        alpha = th.bmm(query / (self.args.attention_dim ** (1/2)), key).view(bs, self.n_agents, self.n_agents)
        # alpha = th.ones((bs, self.n_agents, self.n_agents)).cuda()
        
        # Mask on self-attention
        for i in range(self.n_agents): 
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)
        
        if test_mode and self.args.delay_evaluate:
            delta_t = t - received_time
            gamma_delay = th.pow(0.97, delta_t).view(bs, self.n_agents, self.n_agents, 1).cuda()
            alpha *= gamma_delay
        
        if test_mode and self.args.delay_evaluate:
            value = self.w_value(received_content.cuda())
        else:
            value = self.w_value(th.cat([intent, h], dim = -1).view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs, self.n_agents, self.n_agents, -1)) # shape: (bs, self.n_agents, self.n_agents, -1)

        combined_value = th.sum(alpha * value, dim = -2).view(bs * self.n_agents, -1) # shape: (bs, self.n_agents, -1)
        
        # Update all agents' lastest received message (Train)
        time = (th.ones((bs*self.n_agents,1)) * t).cuda()
              
        if test_mode and self.args.delay_evaluate:
            lastest_message = th.cat([time, intent, h], dim = -1).view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs, self.n_agents, self.n_agents, -1)
            self._get_delayed_message(bs, t, self.n_agents, lastest_message)
        else:
            self.lastest_message = th.cat([time, intent, h], dim = -1).view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs, self.n_agents, self.n_agents, -1)
            
        # Local-Q conditioned on (hidden_state, combined message)
        local_Q = self.fc2(th.cat([h, combined_value], dim=-1)).view(bs, self.n_agents, self.n_actions) 
                     
        
            
        
        returns = {}
        if 'train_mode' in kwargs and kwargs['train_mode']:
            
            # 分散intent分布，避免坍塌
            if t_env > 50000 and t_env % 1000 == 0 and self.args.infer_loss_weight != 0.0:
                intent_embed = self.intent_net(th.cat([h,last_actions], dim = -1)).reshape(bs * self.n_agents, self.intent_dim * 2)
                returns['embed_loss'] = self.calculate_embed_loss(intent_embed[:, :self.intent_dim], intent_embed[:, self.intent_dim:]) * self.args.embed_loss_weight
            
            self.current_intent = intent
                
            if hasattr(self.args, 'entropy_loss_weight') and self.args.entropy_loss_weight > 0: 
                query = self.w_query(intent).unsqueeze(1) # self intent as Query
                key = self.w_key(intent.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs, self.n_agents, self.n_agents, -1)).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2) # others' intents as Key
                alpha = th.bmm(query / (self.args.attention_dim ** (1/2)), key).view(bs, self.n_agents, self.n_agents)
                alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)
                returns['entropy_loss'] = self.calculate_entropy_loss(alpha)
        
        intent_embed = self.intent_net(th.cat([h,last_actions], dim = -1)).reshape(bs * self.n_agents, self.intent_dim * 2)

        return local_Q, h, returns, intent, intent_embed

    def calculate_entropy_loss(self, alpha):
        alpha = th.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * th.log2(alpha)).sum(-1).mean()
        
        loss_fn = nn.MSELoss()
        self_loss = loss_fn(th.diagonal(alpha), th.zeros_like(th.diagonal(alpha)))

        return (entropy_loss + self_loss) * self.args.entropy_loss_weight

    def calculate_embed_loss(self, latent_mean, latent_std):
        kl_divergence_loss = -0.5 * th.sum(1 + th.log(latent_std.pow(2)) - latent_mean.pow(2) - latent_std.pow(2), dim=1)
        kl_divergence_loss = th.mean(kl_divergence_loss)
        return kl_divergence_loss
    
    def init_agent(self, bs):
        self.lastest_message = th.zeros((bs, self.args.n_agents, self.args.n_agents, self.args.intent_dim+self.args.rnn_hidden_dim+1)) # (t_send, message)
        self.current_intent = None
    
    def intent_infer(self, input):
        return self.infer_net(input)
    
    def _get_delayed_message(self, bs, t, n_agents, messages):
        # print("delay......")
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
                            self.lastest_message[b,i,j] = th.from_numpy(self.message_queue[b,i,j,-1,2:]).cuda()
                            continue
                        channel_messages = self.message_queue[b,i,j]
                        indices = np.where(channel_messages[:, 0] == t)
                        first_dim_indices = indices[0]
                        if len(first_dim_indices) != 0:
                            index = first_dim_indices[-1]
                            cur_message = channel_messages[index]
                            if self.lastest_message[b,i,j, 0] < cur_message[1]:
                                self.lastest_message[b,i,j] = th.from_numpy(cur_message[2:]).cuda()
    
    def plot(self, bs, n, alpha, t):
        # if t % 5 == 0:
            for i in range(bs):
                # 获取当前批次的数据
                data = alpha[i].numpy()

                # 绘制热力图
                plt.imshow(data, cmap='coolwarm', interpolation='nearest')
                plt.gca().invert_yaxis()
                plt.colorbar(label='Value')
                for m in range(data.shape[0]):
                    for n in range(data.shape[1]):
                        plt.text(n, m, str(np.round(data[m, n], 2)), ha='center', va='center', color='black', fontsize=6)
                        

                # 保存图像
                filename = f'heatmap_batch_{i}_{t}.png'
                plt.savefig(filename)

                # 关闭图形窗口
                plt.close()

            # # 使用PIL库加载并显示保存的图像
            # image = Image.open(filename)
            # image.show()