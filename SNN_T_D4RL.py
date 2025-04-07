import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import heapq
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from timm.models.layers import trunc_normal_
from spikingjelly.clock_driven import neuron, functional, surrogate

import gym
import d4rl  

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Hyperparameters
STATE_DIM = 2       
ACT_DIM = 4         
HIDDEN_SIZE = 256   
MAX_LENGTH = 100    
MAX_EP_LEN = 1000   
BATCH_SIZE = 32     
EPOCHS = 10         
LEARNING_RATE = 1e-3  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#NUM_SAMPLES = 50000   
T_MAX = 4            


ACTION_MAP = {
    (-1, 0): 0,  
    (1, 0): 1,   
    (0, -1): 2,  
    (0, 1): 3    
}


REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}


def get_discrete_action(current_pos, next_pos):
    """
    Approximate a discrete action (left, right, up, down) 
    by comparing consecutive (x, y) positions from the continuous dataset.
    """
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]


    if abs(dx) > abs(dy):
        if dx > 0:
            return 1  # Right
        else:
            return 0  # Left
    else:
        if dy > 0:
            return 3  # Down
        else:
            return 2  # Up


def load_d4rl_maze_data(env_name="maze2d-umaze-v1"):

    env = gym.make(env_name)
    dataset = env.get_dataset()

    observations = dataset['observations']   
    rewards = dataset['rewards']            
    terminals = dataset['terminals']         
    timeouts = dataset['timeouts']           


    data = []
    start_idx = 0
    N = observations.shape[0]

    for i in range(N):
        done = (terminals[i] == 1) or (timeouts[i] == 1) or (i == N - 1)
        if done:
            ep_obs = observations[start_idx:i+1, :2]  
            ep_rewards = rewards[start_idx:i+1]
            
            path = [list(pos) for pos in ep_obs]

            data.append({
                'path': path,
                'rewards': ep_rewards.tolist()
            })

            start_idx = i + 1

    return data


class MazeDataset(Dataset):
    def __init__(self, data, state_mean, state_std, max_length=MAX_LENGTH):
        """
        data[i] is expected to have:
            {
              'path': [ [x1, y1], [x2, y2], ... ],
              'rewards': [r1, r2, ...]
            }
        """
        self.data = data
        self.state_mean = state_mean.cpu()
        self.state_std = state_std.cpu()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        path = sample['path']
        rewards = sample['rewards']
        states = []
        actions = []
        timesteps = []

        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]

            state = torch.tensor(current_pos, dtype=torch.float32)
            action = get_discrete_action(current_pos, next_pos)

            states.append(state)
            actions.append(action)
            timesteps.append(i)

        returns_to_go = []
        cumulative_return = 0
        for reward in reversed(rewards):
            cumulative_return += reward
            returns_to_go.insert(0, cumulative_return)

        returns_to_go = returns_to_go[:len(states)]

        states = [((state - self.state_mean) / self.state_std).clone().detach() for state in states]

        seq_length = len(states)
        if seq_length >= self.max_length:
            states = states[:self.max_length]
            actions = actions[:self.max_length]
            returns_to_go = returns_to_go[:self.max_length]
            timesteps = timesteps[:self.max_length]
            attention_mask = [1] * self.max_length
        else:

            padding_length = self.max_length - seq_length
            states += [torch.zeros(STATE_DIM)] * padding_length
            actions += [0] * padding_length  
            returns_to_go += [0.0] * padding_length
            timesteps += [0] * padding_length
            attention_mask = [1] * seq_length + [0] * padding_length

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
        timesteps = torch.tensor(timesteps, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            'states': states,
            'actions': actions,
            'returns_to_go': returns_to_go,
            'timesteps': timesteps,
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
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.lif2 = neuron.MultiStepLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )

    def forward(self, x):
        # x shape: (B, S, C)
        B, S, C = x.shape

        # Expand input for time steps: (T, B, S, C)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)

        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)

        # Aggregate over time steps (mean)
        x = x.mean(dim=0)  # (B, S, C)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, T=4):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.T = T

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        # Spiking neurons for Q, K, V
        self.lif_q = neuron.MultiStepLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )
        self.lif_k = neuron.MultiStepLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )
        self.lif_v = neuron.MultiStepLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )

        # Spiking neuron for attention score
        self.lif_out = neuron.MultiStepLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )

    def forward(self, x):
        B, S, C = x.shape
        # (T, B, S, C)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)

        # Linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Spiking neurons
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)

        # Reshape for multi-head attention
        q = q.view(self.T, B, S, self.num_heads, self.head_dim).transpose(2, 3)  # (T, B, num_heads, S, head_dim)
        k = k.view(self.T, B, S, self.num_heads, self.head_dim).transpose(2, 3)
        v = v.view(self.T, B, S, self.num_heads, self.head_dim).transpose(2, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (T, B, num_heads, S, S)

        attn_scores = self.lif_out(attn_scores)

        attn_probs = F.softmax(attn_scores, dim=-1)


        attn_output = torch.matmul(attn_probs, v)  # (T, B, num_heads, S, head_dim)

        attn_output = attn_output.transpose(2, 3).contiguous().view(self.T, B, S, C)  # (T, B, S, C)

        output = attn_output.mean(dim=0)  # (B, S, C)
        return output


class SpikingTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., T=4):
        super().__init__()
        self.T = T
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, T=T)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLP(dim, mlp_hidden_dim, T=T)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpikingTransformerMaze(nn.Module):
    def __init__(
        self,
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        embed_dim=HIDDEN_SIZE,
        num_heads=8,
        mlp_ratio=4.,
        num_layers=6,
        T=T_MAX,
        max_length=MAX_LENGTH,
        max_ep_len=MAX_EP_LEN,
        dropout=0.1
    ):
        super().__init__()
        self.T = T  # Number of simulation time steps

        # Embedding layers
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Embedding(act_dim, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                T=T
            )
            for _ in range(num_layers)
        ])

        # Prediction head
        self.action_head = nn.Linear(embed_dim, act_dim)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        B, S, _ = states.shape

        # Encode states, returns, timesteps
        state_embeddings = self.embed_state(states)
        return_embeddings = self.embed_return(returns_to_go.unsqueeze(-1))
        time_embeddings = self.embed_timestep(timesteps)

        # Shift action embeddings right by 1
        initial_action = torch.zeros_like(actions[:, :1])
        actions_input = torch.cat([initial_action, actions[:, :-1]], dim=1)
        action_embeddings = self.embed_action(actions_input)

        # Combine embeddings
        embeddings = return_embeddings + state_embeddings + action_embeddings + time_embeddings

        # Add positional embeddings
        embeddings = embeddings + self.pos_embedding[:, :S, :]

        # Dropout + LayerNorm
        x = self.dropout(embeddings)
        x = self.norm(x)

        # Reset spiking states
        functional.reset_net(self)

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Predict action
        action_preds = self.action_head(x)
        return action_preds


# ------------------- Plot Helpers ------------------- #
def plot_loss_with_variance(epochs, train_loss_means, train_loss_stds, val_loss_means, val_loss_stds):
    plt.figure(figsize=(8, 6))

    # Plot training mean/std
    plt.plot(epochs, train_loss_means, 'b-', label='Training Loss')
    plt.fill_between(
        epochs, 
        np.array(train_loss_means) - np.array(train_loss_stds),
        np.array(train_loss_means) + np.array(train_loss_stds),
        color='blue', alpha=0.2
    )

    # Plot validation mean/std
    plt.plot(epochs, val_loss_means, 'r-', label='Validation Loss')
    plt.fill_between(
        epochs,
        np.array(val_loss_means) - np.array(val_loss_stds),
        np.array(val_loss_means) + np.array(val_loss_stds),
        color='red', alpha=0.2
    )

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (with variance shading)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(train_accuracies, val_accuracies):
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
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

            action_preds = model(
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )

            action_target = actions
            action_mask = attention_mask.bool()
            action_preds = action_preds[action_mask]
            action_target = action_target[action_mask]

            if loss_fn is not None:
                loss = loss_fn(action_preds, action_target)
                total_loss += loss.item()

            predicted_actions = action_preds.argmax(dim=-1)
            total_correct += (predicted_actions == action_target).sum().item()
            total_actions += action_target.size(0)

    if loss_fn is not None:
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_actions if total_actions > 0 else 0
        return avg_loss, accuracy
    else:
        accuracy = total_correct / total_actions if total_actions > 0 else 0
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

            action_preds = model(
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )

            action_target = actions
            action_mask = attention_mask.bool()
            action_preds = action_preds[action_mask]
            action_target = action_target[action_mask]

            loss = loss_fn(action_preds, action_target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

            predicted_actions = action_preds.argmax(dim=-1)
            correct += (predicted_actions == action_target).sum().item()
            total += action_target.size(0)

            # Log histograms every 100 batches
            if batch_idx % 100 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Weights/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        epoch_loss_mean = np.mean(batch_losses)
        epoch_loss_std = np.std(batch_losses)
        train_loss_means.append(epoch_loss_mean)
        train_loss_stds.append(epoch_loss_std)

        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        # Validation
        val_batch_losses = []
        val_correct = 0
        val_total = 0

        model.eval()
        with torch.no_grad():
            for batch_val in val_loader:
                states_val = batch_val['states'].to(DEVICE)
                actions_val = batch_val['actions'].to(DEVICE)
                returns_to_go_val = batch_val['returns_to_go'].to(DEVICE)
                timesteps_val = batch_val['timesteps'].to(DEVICE)
                attention_mask_val = batch_val['attention_mask'].to(DEVICE)

                action_preds_val = model(
                    states=states_val,
                    actions=actions_val,
                    returns_to_go=returns_to_go_val,
                    timesteps=timesteps_val,
                    attention_mask=attention_mask_val
                )

                action_target_val = actions_val
                action_mask_val = attention_mask_val.bool()
                action_preds_val = action_preds_val[action_mask_val]
                action_target_val = action_target_val[action_mask_val]

                loss_val = loss_fn(action_preds_val, action_target_val)
                val_batch_losses.append(loss_val.item())

                predicted_actions_val = action_preds_val.argmax(dim=-1)
                val_correct += (predicted_actions_val == action_target_val).sum().item()
                val_total += action_target_val.size(0)

        val_loss_mean = np.mean(val_batch_losses)
        val_loss_std = np.std(val_batch_losses)
        val_loss_means.append(val_loss_mean)
        val_loss_stds.append(val_loss_std)

        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {epoch_loss_mean:.4f}, Train Acc: {train_accuracy * 100:.2f}%, "
              f"Val Loss: {val_loss_mean:.4f}, Val Acc: {val_accuracy * 100:.2f}%")

        writer.add_scalar('Loss/train', epoch_loss_mean, epoch)
        writer.add_scalar('Loss/val', val_loss_mean, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        scheduler.step()

    # After training, plot final metrics with variance shading
    epochs_range = range(1, epochs + 1)
    plot_loss_with_variance(
        epochs=epochs_range,
        train_loss_means=train_loss_means,
        train_loss_stds=train_loss_stds,
        val_loss_means=val_loss_means,
        val_loss_stds=val_loss_stds
    )
    plot_accuracy(train_accuracies, val_accuracies)


# ------------------- Example Main Training Script ------------------- #
if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/d4rl_maze_experiment")

    # 1) Load the D4RL Maze dataset
    print("Loading D4RL Maze dataset...")
    dataset_d4rl = load_d4rl_maze_data(env_name="maze2d-umaze-v1")
    print(f"Loaded {len(dataset_d4rl)} episodes from D4RL.")

    # 2) Compute mean and std of states for normalization
    all_states = []
    for ep in dataset_d4rl:
        for s in ep['path']:
            all_states.append(s)
    all_states = torch.tensor(all_states, dtype=torch.float32)
    state_mean = all_states.mean(dim=0)
    state_std = all_states.std(dim=0) + 1e-6  # avoid division by zero

    # 3) Create train/validation split
    random.shuffle(dataset_d4rl)
    split_idx = int(len(dataset_d4rl) * 0.8)
    train_data = dataset_d4rl[:split_idx]
    val_data = dataset_d4rl[split_idx:]

    train_dataset = MazeDataset(train_data, state_mean, state_std, max_length=MAX_LENGTH)
    val_dataset = MazeDataset(val_data, state_mean, state_std, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Initialize model, optimizer, scheduler
    model = SpikingTransformerMaze(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        embed_dim=HIDDEN_SIZE,
        num_heads=8,
        mlp_ratio=4.,
        num_layers=6,
        T=T_MAX,
        max_length=MAX_LENGTH,
        max_ep_len=MAX_EP_LEN,
        dropout=0.1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # 5) Train the model
    train(model, train_loader, optimizer, scheduler, EPOCHS, writer, val_loader)

    # 6) Evaluate final performance and (optionally) confusion matrix
    print("Evaluating final model...")
    _, train_acc = evaluate(model, train_loader, nn.CrossEntropyLoss())
    _, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss())
    print(f"Final Train Accuracy: {train_acc*100:.2f}%")
    print(f"Final Val Accuracy:   {val_acc*100:.2f}%")

    # Example confusion matrix on validation set (if you want)
    # Collect all preds/targets:
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch_val in val_loader:
            states_val = batch_val['states'].to(DEVICE)
            actions_val = batch_val['actions'].to(DEVICE)
            returns_to_go_val = batch_val['returns_to_go'].to(DEVICE)
            timesteps_val = batch_val['timesteps'].to(DEVICE)
            attention_mask_val = batch_val['attention_mask'].to(DEVICE)

            action_preds_val = model(
                states=states_val,
                actions=actions_val,
                returns_to_go=returns_to_go_val,
                timesteps=timesteps_val,
                attention_mask=attention_mask_val
            )
            action_mask_val = attention_mask_val.bool()
            action_preds_val = action_preds_val[action_mask_val]
            actions_val = actions_val[action_mask_val]

            predicted_actions_val = action_preds_val.argmax(dim=-1).cpu().numpy()
            all_preds.extend(predicted_actions_val.tolist())
            all_targets.extend(actions_val.cpu().numpy().tolist())

    # Plot confusion matrix if desired
    action_labels = ["Left", "Right", "Up", "Down"]
    plot_confusion_matrix(all_targets, all_preds, classes=action_labels)
