import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    """Small recurrent policy; score each candidate with shared weights."""
    def __init__(self, observation_size, action_size, hidden=128):
        super().__init__()
        self.spec = dict(observation_size=observation_size, action_size=action_size, hidden=hidden)
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden), nn.Tanh())
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.action_encoder = nn.Sequential(nn.Linear(action_size, hidden), nn.Tanh())
        self.value = nn.Linear(hidden, 1)

    def forward(self, observations, actions, mask, state=None):
        # [batch, time, features], [batch, time, candidates, features]
        features, state = self.memory(self.encoder(observations), state)
        candidates = self.action_encoder(actions)
        logits = (features.unsqueeze(-2) * candidates).sum(-1) / features.shape[-1] ** 0.5
        logits = logits.masked_fill(~mask, -torch.inf)
        return Categorical(logits=logits), self.value(features).squeeze(-1), state

    @torch.no_grad()
    def act(self, observation, actions, count, state=None, generator=None):
        device = next(self.parameters()).device
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)[None, None]
        acts = torch.as_tensor(actions, dtype=torch.float32, device=device)[None, None]
        mask = (torch.arange(len(actions), device=device) < count)[None, None]
        dist, value, state = self(obs, acts, mask, state)
        action = torch.multinomial(dist.probs.flatten(), 1, generator=generator).reshape(1, 1)
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item()), state


def pack(observation):
    """Engine adapter must sanitize private information before calling this."""
    keys = ("cards_", "global_", "h_actions_")
    features = np.concatenate([observation[k].reshape(-1) for k in keys]).astype(np.float32) / 255
    actions = observation["actions_"].astype(np.float32) / 255
    return features, actions
