import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

torch.manual_seed(42)


class ActorNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.classifier = nn.Softmax(dim=1)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.FloatTensor(x)
        #x = x.unsqueeze(0)
        x = self.relu(self.norm1(self.input(x)))
        x = self.relu(self.norm2(self.hidden(x)))
        x = self.classifier(self.output(x))
        return x


class CriticNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.FloatTensor(x)
        #x = x.unsqueeze(0)
        x = self.relu(self.norm1(self.input(x)))
        x = self.relu(self.norm2(self.hidden(x)))
        x = self.output_activation(self.output(x))
        return x


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    clipping_val = 0.2
    critic_discount = 0.5
    entropy_beta = 0.001
    def loss(y_pred):
        newpolicy_probs = y_pred
        ratio = torch.exp(torch.log(newpolicy_probs + 1e-10) - torch.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = torch.clip(ratio, min=1 - clipping_val, max=1 + clipping_val) * advantages
        actor_loss = -torch.mean(torch.minimum(p1, p2))
        critic_loss = torch.mean(torch.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss #- entropy_beta * torch.mean(-(newpolicy_probs * torch.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss


def train_networks(actor_net: ActorNetwork, critic_net: CriticNetwork, states, oldpolicy_probs, advantages, rewards,
                   values, returns, actions, batch_size=10, epochs=5):
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=1e-4)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=1e-4)

    actor_net.train()
    critic_net.train()

    dataset = StateDataset(states, oldpolicy_probs, advantages, rewards, values, returns, actions)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            # train actor
            actor_net.zero_grad()
            policies = actor_net(data[0])
            # evaluate on policy
            new_probs = torch.flatten(torch.gather(policies, 1, data[6].unsqueeze(1)), start_dim=0)

            loss = ppo_loss(data[1], data[2], data[3], data[4])(new_probs)
            loss.backward()
            optimizer_actor.step()

            # train critic
            # TODO actually one should train critic first and use new values (advantages) in the actor training, then critic would always be one step "smarter"
            critic_net.zero_grad()
            returns_pred = critic_net(data[0])
            critic_loss_f = nn.MSELoss()
            critic_loss = critic_loss_f(returns_pred, data[5].unsqueeze(1))
            critic_loss.backward()
            optimizer_critic.step()


class StateDataset(Dataset):

    def __init__(self, data_array, oldpolicy_probs, advantages, rewards, values, returns, actions):
        self.data_array = data_array
        self.oldpolicy_probs = oldpolicy_probs
        self.advantages = advantages
        self.rewards = rewards
        self.values = values
        self.returns = returns
        self.actions = actions

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, index) -> T_co:
        return self.data_array[index], self.oldpolicy_probs[index], self.advantages[index], self.rewards[index], \
               self.values[index], self.returns[index], self.actions[index]
