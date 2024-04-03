"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from harl.runners.IGC_off_policy_base_runner import OffPolicyBaseRunner
from harl.models.base.distributions import FixedNormal
from harl.utils.models_tools import check
from harl.utils.trans_tools import _t2n


class OffPolicyMARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        if self.args["algo"] in ["hasac", "igcsac"]:
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                next_action, next_logp_action = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                    stochastic=False,
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            next_actions = torch.stack(next_actions, dim=1)
            bias_, next_action_std = self.action_attention(next_actions, torch.unsqueeze(check(sp_next_share_obs).to(self.device), 1).repeat(1, self.num_agents, 1))
            # ind_dist = FixedNormal(logits, stds)
            next_mix_dist = FixedNormal(next_actions, next_action_std)
            next_actions = next_mix_dist.rsample()
            
            next_logp_actions = next_mix_dist.log_probs(next_actions).sum(axis=-1, keepdim=True)
            next_logp_actions -= (2 * (np.log(2) - next_actions - F.softplus(-2 * next_actions))).sum(
                axis=-1, keepdim=True
            )

            next_actions = torch.tanh(next_actions)
            next_actions = self.act_limit * next_actions
            critic_loss = self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            for agent_id in range(self.num_agents):
                self.actor[agent_id].turn_on_grad()
            self.action_attention.turn_on_grad()
            # train actors
            if self.args["algo"] in ["hasac", "igcsac"]:
                actions = []
                logp_actions = []
                for agent_id in range(self.num_agents):
                    action, logp_action = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                        stochastic=False,
                    )
                    actions.append(action)
                    logp_actions.append(logp_action)
                actions = torch.stack(actions, dim=1)
                bias_, action_std = self.action_attention(actions, torch.unsqueeze(check(sp_share_obs).to(self.device), 1).repeat(1, self.num_agents, 1))
                # ind_dist = FixedNormal(logits, stds)
                mix_dist = FixedNormal(actions, action_std)
                actions = mix_dist.rsample()
                logp_actions = mix_dist.log_probs(actions).sum(axis=-1, keepdim=True)
                logp_actions -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(
                    axis=-1, keepdim=True
                )
                logp_actions = logp_actions.reshape(actions.shape[0], -1).sum(axis=-1, keepdim=True)
                
                actions = torch.tanh(actions)
                actions = self.act_limit * actions
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                
                # train agents
                if self.state_type == "EP":
                    actions_t = actions.reshape(actions.shape[0], -1)
                elif self.state_type == "FP":
                    actions_t = torch.tile(
                        torch.cat(actions, dim=-1), (self.num_agents, 1)
                    )
                value_pred = self.critic.get_values(sp_share_obs, actions_t)
                if self.algo_args["algo"]["use_policy_active_masks"]:
                    if self.state_type == "EP":
                        actor_loss = (
                            -torch.sum(
                                (value_pred - self.alpha * logp_actions).unsqueeze(0).repeat(self.num_agents, 1, 1)
                                * sp_valid_transition
                            )
                            / sp_valid_transition.sum()
                        )
                    elif self.state_type == "FP":
                        valid_transition = sp_valid_transition
                        actor_loss = (
                            -torch.sum(
                                (value_pred - self.alpha * logp_actions)
                                * valid_transition
                            )
                            / valid_transition.sum()
                        )
                else:
                    actor_loss = -torch.mean(
                        value_pred - self.alpha * logp_actions
                    )
                
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].actor_optimizer.zero_grad()
                self.action_attention_optimizer.zero_grad()
                actor_loss.backward()
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                self.action_attention_optimizer.step()
                self.action_attention.turn_off_grad()
                # train this agent's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    log_prob = (
                        logp_actions[agent_id].detach()
                        + self.target_entropy[agent_id]
                    )
                    alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                    self.alpha_optimizer[agent_id].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer[agent_id].step()
                    self.alpha[agent_id] = torch.exp(
                        self.log_alpha[agent_id].detach()
                    )
                # actions[agent_id], _ = self.actor[
                #     agent_id
                # ].get_actions_with_logprobs(
                #     sp_obs[agent_id],
                #     sp_available_actions[agent_id]
                #     if sp_available_actions is not None
                #     else None,
                # )
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            else:
                if self.args["algo"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values = self.actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)
                        actor_loss = -torch.mean(value_pred)
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                # soft update
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
            return critic_loss, actor_loss
