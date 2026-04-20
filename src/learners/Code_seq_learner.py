import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn as nn
# import wandb
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from modules.time_net import Mish, SinusoidalPosEmb

class Code_seq_learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.infer_len = 3 
        self.infer_gamma = 0.9
        self.intent_dim = args.intent_dim
        
        dim = args.time_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        ).cuda()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        obs = batch["obs"][:, :-1]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # NOTE: record logging signal
        prepare_for_logging = True if t_env - self.log_stats_t >= self.args.learner_log_interval else False

        logs = []
        losses = []

        # Calculate estimated Q-Values
        mac_out = []
        hidden_states = []
        intents = []
        last_actions = []
        inputs_for_inference = []
        self.mac.init_hidden(batch.batch_size)
        old_intent = None
        criterion = nn.CrossEntropyLoss()
        continuity_loss, infer_loss = 0, 0
        calculate_num = 0

        for t in range(batch.max_seq_length):
            agent_outs, returns_, intent, intent_embed, last_action = self.mac.forward(batch, t=t, t_env=t_env,
                prepare_for_logging=prepare_for_logging,
                train_mode=True,
                mixer=self.target_mixer,
            )
            inputs_for_inference.append(self.mac._build_inputs(batch, t))
            last_actions.append(last_action)
            hidden_states.append(self.mac.hidden_states.detach())
            eps = th.randn_like(intent_embed[:, self.intent_dim:])
            intent = intent_embed[:, :self.intent_dim] + (eps*intent_embed[:, self.intent_dim:])
            intents.append(intent)
            # Calculate the continuity loss for the intents
            if t != 0:
                continuity_loss -= (F.cosine_similarity(old_intent, intent, dim=1).mean())
            old_intent = intent.detach()
            

            mac_out.append(agent_outs)
            if prepare_for_logging and 'logs' in returns_:
                logs.append(returns_['logs'])
                del returns_['logs']
            losses.append(returns_)
        
        continuity_loss = continuity_loss / t
        
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        batch_size, n_agents = mac_out.shape[0], mac_out.shape[2] 
        max_values, action_preference = th.max(mac_out.detach(), dim=-1)
        
        
        # Calculate the infer loss for the intents
        for i in range(obs.shape[1] - self.infer_len):
            inferred_preference = self.mac.agent.intent_infer(inputs_for_inference[i:i+self.infer_len], last_actions[i].reshape(batch_size * n_agents, -1), intents[i], hidden_states[i]).reshape(batch_size * n_agents * self.infer_len, -1)
            action_mask = avail_actions[:,i:i+self.infer_len,:].reshape(batch_size * n_agents * self.infer_len, -1)
            inferred_preference = th.mul(inferred_preference, action_mask)
            cur_preference = action_preference[:,i:i+self.infer_len,:].reshape(batch_size * n_agents * self.infer_len).detach()
            infer_loss += (criterion(inferred_preference, cur_preference))

            # for j in range(self.infer_len):
            #     # print(inputs_for_inference[i].shape)
            #     calculate_num += 1
            #     time = (th.ones([batch_size, n_agents, 1]) * j).cuda()
            #     time_input = self.time_mlp(time).reshape(batch_size * n_agents, -1)

            #     inputs = th.cat([intents[i], inputs_for_inference[i+j], time_input], dim=-1).cuda()
            #     inferred_preference = self.mac.agent.intent_infer(inputs).reshape(batch_size * n_agents, -1)
            #     action_mask = avail_actions[:,i,:].reshape(batch_size * n_agents, -1)
            #     inferred_preference = th.mul(inferred_preference, action_mask)
            #     max_indices = th.argmax(inferred_preference, dim=1)
            #     cur_preference = action_preference[:,i+j,:].reshape(batch_size * n_agents).detach()
            #     infer_loss += ((self.infer_gamma**j) * criterion(inferred_preference, cur_preference))
        infer_loss = infer_loss / obs.shape[1]
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _, _,_,_ = self.target_mac.forward(batch, t=t, t_env=t_env)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss += external_loss
        if t_env > 50000 and t_env % 1000 == 0:
            loss += (continuity_loss * self.args.continous_loss_weight)
            loss += (infer_loss * self.args.infer_loss_weight)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        
        loss_dict["continuity_loss"] = continuity_loss
        loss_dict["infer_loss"] = infer_loss

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self._log_for_loss(loss_dict, t_env)

            self.log_stats_t = t_env

    def _process_loss(self, losses: list, batch: EpisodeBatch):
        total_loss = 0
        loss_dict = {}
        for item in losses:
            for k, v in item.items():
                if str(k).endswith('loss'):
                    loss_dict[k] = loss_dict.get(k, 0) + v
                    total_loss += v
        for k in loss_dict.keys():
            loss_dict[k] /= batch.max_seq_length
        total_loss /= batch.max_seq_length
        return total_loss, loss_dict

    def _log_for_loss(self, losses: dict, t):
        for k, v in losses.items():
            self.logger.log_stat(k, v.item(), t)
            # wandb.log({
            #     k: v.item()
            # })

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
