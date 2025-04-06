

import os


import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


from spikingjelly.clock_driven import neuron, functional, surrogate

from timm.models.layers import trunc_normal_


import gym
import gym_minigrid



ENV_NAME = 'MiniGrid-Empty-8x8-v0'
NUM_EPISODES = 20000
MAX_STEPS = 100

STATE_DIM = 2
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spiking-related
T_MAX = 4        # number of simulation time steps
HIDDEN_SIZE = 256
MAX_LENGTH = 100  # maximum sequence length for each trajectory
MAX_EP_LEN = 200  # used for embedding timesteps (arbitrary >= MAX_STEPS)


# OFFLINE DATASET FROM A MINIGRID ENV

def collect_minigrid_data(env_name=ENV_NAME, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS):

    env = gym.make(env_name, disable_env_checker=True)
    data = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []

        for step in range(max_steps):
            # (x, y) = agent_pos
            agent_x, agent_y = env.agent_pos
            episode_states.append([agent_x, agent_y])

            # random action
            act = env.action_space.sample()
            # Capture 5 values from env.step(...)
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            episode_actions.append(act)
            episode_rewards.append(reward)

            if done:
                break

        if episode_states:
            data.append({
                'states': np.array(episode_states, dtype=np.float32),
                'actions': np.array(episode_actions, dtype=np.int64),
                'rewards': np.array(episode_rewards, dtype=np.float32)
            })

    env.close()
    return data


class MiniGridOfflineDataset(Dataset):

    def __init__(self, data, state_mean, state_std, max_length=MAX_LENGTH):
        super().__init__()
        self.data = data
        self.state_mean = state_mean.cpu()
        self.state_std = state_std.cpu()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        episode = self.data[idx]
        states_seq = torch.tensor(episode['states'], dtype=torch.float32)   # (L,2)
        actions_seq = torch.tensor(episode['actions'], dtype=torch.long)    # (L,)
        rewards_seq = torch.tensor(episode['rewards'], dtype=torch.float32) # (L,)

        # compute returns-to-go
        returns_to_go = []
        total_r = 0
        for r in reversed(rewards_seq):
            total_r += r
            returns_to_go.insert(0, total_r)
        returns_to_go = torch.stack(returns_to_go, dim=0)

        # normalize states
        states_seq = (states_seq - self.state_mean) / self.state_std

        # timesteps
        timesteps_seq = torch.arange(len(states_seq), dtype=torch.long)

        # truncate / pad to max_length
        seq_len = len(states_seq)
        if seq_len > self.max_length:
            # truncate
            states_seq = states_seq[:self.max_length]
            actions_seq = actions_seq[:self.max_length]
            returns_to_go = returns_to_go[:self.max_length]
            timesteps_seq = timesteps_seq[:self.max_length]
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            # pad
            pad_len = self.max_length - seq_len
            states_seq = torch.cat([states_seq, torch.zeros(pad_len, 2)], dim=0)
            actions_seq = torch.cat([actions_seq, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            returns_to_go = torch.cat([returns_to_go, torch.zeros(pad_len, dtype=torch.float32)], dim=0)
            timesteps_seq = torch.cat([timesteps_seq, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long),
                                        torch.zeros(pad_len, dtype=torch.long)], dim=0)

        return {
            'states': states_seq,
            'actions': actions_seq,
            'returns_to_go': returns_to_go,
            'timesteps': timesteps_seq,
            'attention_mask': attention_mask
        }


class SpikingMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, T=4):
        super().__init__()
        self.T = T
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.lif1 = neuron.MultiStepLIFNode(
            tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True
        )

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.lif2 = neuron.MultiStepLIFNode(
            tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True
        )

    def forward(self, x):
        # x: (B, S, C)
        B, S, C = x.shape
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # (T,B,S,C)

        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)

        # average over time dimension
        x = x.mean(dim=0)  # (B,S,C)
        return x

class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, T=4):
        super().__init__()
        assert dim % num_heads == 0,
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.T = T

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.lif_q = neuron.MultiStepLIFNode(
            tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True
        )
        self.lif_k = neuron.MultiStepLIFNode(
            tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True
        )
        self.lif_v = neuron.MultiStepLIFNode(
            tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True
        )

        self.lif_out = neuron.MultiStepLIFNode(
            tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True
        )

    def forward(self, x):
        # x: (B,S,C)
        B, S, C = x.shape
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # (T,B,S,C)

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)

        # reshape for multi-head
        q = q.view(self.T,B,S,self.num_heads,self.head_dim).transpose(2,3)
        k = k.view(self.T,B,S,self.num_heads,self.head_dim).transpose(2,3)
        v = v.view(self.T,B,S,self.num_heads,self.head_dim).transpose(2,3)

        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # spiking on attention scores
        scores = self.lif_out(scores)
        # softmax
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (T,B,num_heads,S,head_dim)

        # re-combine heads
        out = out.transpose(2,3).contiguous().view(self.T,B,S,C)
        # average over time
        out = out.mean(dim=0)  # (B,S,C)
        return out


class SpikingTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, T=4):
        super().__init__()
        self.T = T
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, T=T)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLP(dim, hidden_dim, T=T)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpikingTransformerMiniGrid(nn.Module):
    def __init__(self,
                 state_dim=2,   # (x, y)
                 act_dim=7,     # e.g. standard MiniGrid has 7 discrete actions
                 embed_dim=256,
                 num_heads=8,
                 mlp_ratio=4.0,
                 num_layers=6,
                 T=4,
                 max_length=100,
                 max_ep_len=200,
                 dropout=0.1):
        super().__init__()
        self.T = T
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Embedding(act_dim, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)

        # position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, T=T)
            for _ in range(num_layers)
        ])

        self.action_head = nn.Linear(embed_dim, act_dim)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        """
        states: (B,S,2)
        actions: (B,S)
        returns_to_go: (B,S)
        timesteps: (B,S)
        """
        B, S, _ = states.shape

        # embed
        state_emb = self.embed_state(states)
        return_emb = self.embed_return(returns_to_go.unsqueeze(-1))
        time_emb = self.embed_timestep(timesteps)

        # shift actions right
        init_act = torch.zeros_like(actions[:, :1])  # shape (B,1)
        actions_input = torch.cat([init_act, actions[:, :-1]], dim=1)
        action_emb = self.embed_action(actions_input)

        x = state_emb + return_emb + time_emb + action_emb
        x = x + self.pos_embedding[:, :S, :]

        x = self.dropout(x)
        x = self.norm(x)

        # reset spiking states
        functional.reset_net(self)

        for blk in self.blocks:
            x = blk(x)

        action_preds = self.action_head(x)  # (B,S,act_dim)
        return action_preds


def plot_loss_with_variance(epochs, train_loss_means, train_loss_stds, val_loss_means, val_loss_stds):
    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_loss_means, 'b-o', label='Train Loss')
    plt.fill_between(
        epochs,
        np.array(train_loss_means) - np.array(train_loss_stds),
        np.array(train_loss_means) + np.array(train_loss_stds),
        alpha=0.2
    )
    plt.plot(epochs, val_loss_means, 'r-o', label='Val Loss')
    plt.fill_between(
        epochs,
        np.array(val_loss_means) - np.array(val_loss_stds),
        np.array(val_loss_means) + np.array(val_loss_stds),
        alpha=0.2
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Train/Val Loss')
    plt.show()

def plot_accuracy(train_accuracies, val_accuracies):
    epochs_range = range(1, len(train_accuracies)+1)
    plt.figure(figsize=(7,5))
    plt.plot(epochs_range, train_accuracies, 'b-o', label='Train Acc')
    plt.plot(epochs_range, val_accuracies, 'r-o', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title('Train/Val Accuracy')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Action')
    plt.ylabel('True Action')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate(model, dataloader, loss_fn=None):
    model.eval()
    total_correct = 0
    total_actions = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            states = batch['states'].to(DEVICE)
            actions = batch['actions'].to(DEVICE)
            returns_to_go = batch['returns_to_go'].to(DEVICE)
            timesteps = batch['timesteps'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            preds = model(states, actions, returns_to_go, timesteps, attention_mask)

            # mask out padded positions
            mask = attention_mask.bool()
            preds = preds[mask]
            true_acts = actions[mask]

            if loss_fn is not None:
                loss = loss_fn(preds, true_acts)
                total_loss += loss.item()

            pred_acts = preds.argmax(dim=-1)
            total_correct += (pred_acts == true_acts).sum().item()
            total_actions += true_acts.size(0)

    accuracy = (total_correct / total_actions) if total_actions > 0 else 0.0
    if loss_fn is not None:
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    else:
        return accuracy


def train(model, dataloader, optimizer, scheduler, epochs, writer, val_loader):
    loss_fn = nn.CrossEntropyLoss()

    train_loss_means = []
    train_loss_stds = []
    val_loss_means = []
    val_loss_stds = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader, 1):
            states = batch['states'].to(DEVICE)
            actions = batch['actions'].to(DEVICE)
            returns_to_go = batch['returns_to_go'].to(DEVICE)
            timesteps = batch['timesteps'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            optimizer.zero_grad()
            preds = model(states, actions, returns_to_go, timesteps, attention_mask)

            # mask out padding
            mask = attention_mask.bool()
            preds = preds[mask]
            true_acts = actions[mask]

            loss = loss_fn(preds, true_acts)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())
            pred_acts = preds.argmax(dim=-1)
            correct += (pred_acts == true_acts).sum().item()
            total += true_acts.size(0)

            # optional: log histograms to TensorBoard
            if batch_idx % 100 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"Weights/{name}", param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        # end of epoch
        train_loss_mean = np.mean(batch_losses)
        train_loss_std = np.std(batch_losses)
        train_loss_means.append(train_loss_mean)
        train_loss_stds.append(train_loss_std)

        train_accuracy = correct / total if total > 0 else 0
        train_accuracies.append(train_accuracy)

        # validation
        val_losses = []
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for batch_val in val_loader:
                states_val = batch_val['states'].to(DEVICE)
                actions_val = batch_val['actions'].to(DEVICE)
                returns_to_go_val = batch_val['returns_to_go'].to(DEVICE)
                timesteps_val = batch_val['timesteps'].to(DEVICE)
                mask_val = batch_val['attention_mask'].to(DEVICE)

                preds_val = model(states_val, actions_val, returns_to_go_val, timesteps_val, mask_val)

                mask_val_bool = mask_val.bool()
                preds_val = preds_val[mask_val_bool]
                true_acts_val = actions_val[mask_val_bool]

                loss_val = loss_fn(preds_val, true_acts_val)
                val_losses.append(loss_val.item())

                pred_val = preds_val.argmax(dim=-1)
                val_correct += (pred_val == true_acts_val).sum().item()
                val_total += true_acts_val.size(0)

        val_loss_mean = np.mean(val_losses)
        val_loss_std = np.std(val_losses)
        val_loss_means.append(val_loss_mean)
        val_loss_stds.append(val_loss_std)
        val_accuracy = val_correct / val_total if val_total>0 else 0
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss_mean:.4f}, Train Acc: {train_accuracy*100:.1f}% | "
              f"Val Loss: {val_loss_mean:.4f}, Val Acc: {val_accuracy*100:.1f}%")

        # TensorBoard logs
        writer.add_scalar('Loss/train', train_loss_mean, epoch)
        writer.add_scalar('Loss/val', val_loss_mean, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        scheduler.step()

    # Plot final
    epochs_range = range(1, epochs+1)
    plot_loss_with_variance(epochs_range, train_loss_means, train_loss_stds,
                            val_loss_means, val_loss_stds)
    plot_accuracy(train_accuracies, val_accuracies)

if __name__ == "__main__":
    # Offline data (with random policy)
    data = collect_minigrid_data(env_name=ENV_NAME, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
    print(f"Collected {len(data)} episodes from {ENV_NAME} with random policy.")

    # global mean/std for states
    all_states = np.concatenate([ep['states'] for ep in data], axis=0)
    state_mean = torch.tensor(all_states.mean(axis=0), dtype=torch.float32).to(DEVICE)
    state_std = torch.tensor(all_states.std(axis=0) + 1e-6, dtype=torch.float32).to(DEVICE)

    #  Split into train/val/test
    #    We'll do: 80% train, 10% val, 10% test
    num_episodes = len(data)
    num_train = int(num_episodes * 0.8)
    num_val = int(num_episodes * 0.1)
    # The remainder is test
    num_test = num_episodes - (num_train + num_val)

    train_data = data[:num_train]
    val_data = data[num_train : num_train + num_val]
    test_data = data[num_train + num_val :]

    print(f"Train set: {len(train_data)} episodes, Val set: {len(val_data)} episodes, Test set: {len(test_data)} episodes")

    train_dataset = MiniGridOfflineDataset(train_data, state_mean, state_std, max_length=MAX_LENGTH)
    val_dataset = MiniGridOfflineDataset(val_data, state_mean, state_std, max_length=MAX_LENGTH)
    test_dataset = MiniGridOfflineDataset(test_data, state_mean, state_std, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Env to figure out discrete action dimension
    #    (typical MiniGrid uses 7, e.g. left, right, forward, pickup, drop, toggle, done)
    env_tmp = gym.make(ENV_NAME, disable_env_checker=True)
    act_dim = env_tmp.action_space.n
    env_tmp.close()

    # Build the Spiking Transformer
    model = SpikingTransformerMiniGrid(
        state_dim=STATE_DIM,
        act_dim=act_dim,
        embed_dim=HIDDEN_SIZE,
        num_heads=8,
        mlp_ratio=4.0,
        num_layers=6,
        T=T_MAX,
        max_length=MAX_LENGTH,
        max_ep_len=MAX_EP_LEN,
        dropout=0.1
    ).to(DEVICE)

    # Optimizer, scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs/minigrid_spiking_transformer')

    # Train on train set, validate each epoch
    train(model, train_loader, optimizer, scheduler, EPOCHS, writer, val_loader)

    # Evaluate final on test set
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"Final Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.1f}%")

    writer.close()

