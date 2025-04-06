
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Hyperparameters
STATE_DIM = 2       # State represented as (x, y) coordinates
ACT_DIM = 4         # Four possible actions: left, right, up, down
HIDDEN_SIZE = 256   # Hidden size of the transformer model
MAX_LENGTH = 100    # Maximum sequence length
MAX_EP_LEN = 1000   # Maximum episode length (for timestep embeddings)
BATCH_SIZE = 32     # Batch size
EPOCHS = 20          # Number of training epochs
LEARNING_RATE = 1e-3  # Learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAZE_SIZE = (21, 21)  # Maze size (width, height)
NUM_SAMPLES = 50000    # Number of samples to generate
T_MAX = 4            # Simulation time steps for spiking neurons

# Action mapping: map movements to action integers
ACTION_MAP = {
    (-1, 0): 0,  # Left
    (1, 0): 1,   # Right
    (0, -1): 2,  # Up
    (0, 1): 3    # Down
}

# Reversing action mapping: map action integers to movements
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}


class MazeDataset(Dataset):
    def __init__(self, data, state_mean, state_std, max_length=MAX_LENGTH):
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

        # path to states and actions
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            state = torch.tensor(current_pos, dtype=torch.float32)
            movement = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
            action = ACTION_MAP[movement]
            states.append(state)
            actions.append(action)
            timesteps.append(i)

        # Computing returns-to-go
        returns_to_go = []
        cumulative_return = 0
        for reward in reversed(rewards):
            cumulative_return += reward
            returns_to_go.insert(0, cumulative_return)

        # Normalizing states and detach
        states = [((state - self.state_mean) / self.state_std).clone().detach() for state in states]

        # Ensuring sequences are of length 'self.max_length'
        seq_length = len(states)
        if seq_length >= self.max_length:
            # Truncate sequences
            states = states[:self.max_length]
            actions = actions[:self.max_length]
            returns_to_go = returns_to_go[:self.max_length]
            timesteps = timesteps[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            # Pad sequences
            padding_length = self.max_length - seq_length
            states += [torch.zeros(STATE_DIM)] * padding_length
            actions += [0] * padding_length
            returns_to_go += [0.0] * padding_length
            timesteps += [0] * padding_length
            attention_mask = [1] * seq_length + [0] * padding_length

        # Converting to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
        timesteps = torch.tensor(timesteps, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        sample = {
            'states': states,
            'actions': actions,
            'returns_to_go': returns_to_go,
            'timesteps': timesteps,
            'attention_mask': attention_mask
        }

        return sample


# Maze generation and solving functions
def create_maze(width, height):

    width = width if width % 2 != 0 else width - 1
    height = height if height % 2 != 0 else height - 1

    maze = [[1 for _ in range(width)] for _ in range(height)]

    def carve_passages_from(cx, cy, maze):
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 < ny < height and 0 < nx < width and maze[ny][nx] == 1:
                maze[cy + dy // 2][cx + dx // 2] = 0  # Remove wall between cells
                maze[ny][nx] = 0
                carve_passages_from(nx, ny, maze)


    maze[1][1] = 0
    carve_passages_from(1, 1, maze)

    # Ensuring entrance and exit are open
    maze[1][0] = 0  # Entrance at (0, 1)
    maze[height - 2][width - 1] = 0  # Exit at (width - 1, height - 2)

    return maze

def solve_maze(maze):
    width = len(maze[0])
    height = len(maze)
    start = (0, 1)
    goal = (width - 1, height - 2)

    def heuristic(a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Priority queue: (f_score, g_score, position, path)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    g_score = {start: 0}
    visited = set()

    while open_set:
        f_current, g_current, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in ACTION_MAP.keys():
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0:
                tentative_g_score = g_current + 1
                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, path + [neighbor]))

    # If no path is found, return None
    return None

def generate_maze_data(num_samples, maze_size):
    data = []
    for _ in range(num_samples):
        maze = create_maze(*maze_size)
        path = solve_maze(maze)
        if path is not None:
            # Assigning rewards: -0.1 for each step, 1.0 for reaching the goal
            num_steps = len(path) - 1
            rewards = [-0.1] * num_steps
            rewards[-1] += 1.0  # Reward for reaching the goal
            data.append({'path': path, 'rewards': rewards})
    return data


# Spiking MLP
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

        # Expand input for time steps
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # Shape: (T, B, S, C)

        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)

        # Aggregate over time steps (mean)
        x = x.mean(dim=0)  # Shape: (B, S, C)

        return x


# Spiking Self-Attention
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

        # Spiking neuron for attention output
        self.lif_out = neuron.MultiStepLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )

    def forward(self, x):
        B, S, C = x.shape
        # Expand input for time steps
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # Shape: (T, B, S, C)

        # Linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Applying spiking neurons (process over time steps)
        q = self.lif_q(q)  # Shape: (T, B, S, C)
        k = self.lif_k(k)
        v = self.lif_v(v)

        # Reshaping for multi-head attention
        q = q.view(self.T, B, S, self.num_heads, self.head_dim).transpose(2, 3)  # (T, B, num_heads, S, head_dim)
        k = k.view(self.T, B, S, self.num_heads, self.head_dim).transpose(2, 3)
        v = v.view(self.T, B, S, self.num_heads, self.head_dim).transpose(2, 3)

        # Computing scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (T, B, num_heads, S, S)

        # Applyin9 spiking neuron to attention scores
        attn_scores = self.lif_out(attn_scores)

        # Appling softmax over the last dimension
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Computing attention output
        attn_output = torch.matmul(attn_probs, v)  # (T, B, num_heads, S, head_dim)

        # Concatenating heads
        attn_output = attn_output.transpose(2, 3).contiguous().view(self.T, B, S, C)  # (T, B, S, C)

        # Aggregating over time steps (mean)
        output = attn_output.mean(dim=0)  # Shape: (B, S, C)

        return output


# Spiking Transformer Block
class SpikingTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., T=4):
        super().__init__()
        self.T = T  # Number of simulation time steps
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, T=T)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLP(dim, mlp_hidden_dim, T=T)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Spiking Transformer Model for Maze Navigation
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
        self.T = T

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

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                T=T  # Pass T to each block
            ) for _ in range(num_layers)
        ])

        # Prediction head
        self.action_head = nn.Linear(embed_dim, act_dim)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        B, S, _ = states.shape

        # Encode inputs
        state_embeddings = self.embed_state(states)
        return_embeddings = self.embed_return(returns_to_go.unsqueeze(-1))
        time_embeddings = self.embed_timestep(timesteps)

        # Shifting action embeddings to the right and prepend a placeholder
        initial_action = torch.zeros_like(actions[:, :1])
        actions_input = torch.cat([initial_action, actions[:, :-1]], dim=1)
        action_embeddings = self.embed_action(actions_input)

        # Combining all embeddings
        embeddings = return_embeddings + state_embeddings + action_embeddings + time_embeddings

        # Adding positional embeddings
        embeddings = embeddings + self.pos_embedding[:, :S, :]

        # Applying dropout and normalization
        x = self.dropout(embeddings)
        x = self.norm(x)

        # Reseting neuron states
        functional.reset_net(self)

        # Passing through Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Predicting actions
        action_preds = self.action_head(x)

        return action_preds



def plot_loss_with_variance(
    epochs,
    train_loss_means, train_loss_stds,
    val_loss_means, val_loss_stds
):
    plt.figure(figsize=(8, 6))


    plt.plot(epochs, train_loss_means, 'b-o', label='Training Loss')

    plt.fill_between(
        epochs,
        np.array(train_loss_means) - np.array(train_loss_stds),
        np.array(train_loss_means) + np.array(train_loss_stds),
        color='blue', alpha=0.2
    )


    plt.plot(epochs, val_loss_means, 'r-o', label='Validation Loss')

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
    plt.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-o', label='Validation Accuracy')
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

        # For storing the batch losses each epoch
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

            # Mask padded positions
            action_mask = attention_mask.bool()
            action_preds = action_preds[action_mask]
            action_target = action_target[action_mask]

            # Compute loss
            loss = loss_fn(action_preds, action_target)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimize
            optimizer.step()

            batch_losses.append(loss.item())

            # Compute accuracy
            predicted_actions = action_preds.argmax(dim=-1)
            correct += (predicted_actions == action_target).sum().item()
            total += action_target.size(0)

            # Log histograms every 100 batches
            if batch_idx % 100 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Weights/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Compute statistics for this epoch
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

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss_mean, epoch)
        writer.add_scalar('Loss/val', val_loss_mean, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # Step the scheduler
        scheduler.step()


    epochs_range = range(1, epochs + 1)
    plot_loss_with_variance(
        epochs=epochs_range,
        train_loss_means=train_loss_means,
        train_loss_stds=train_loss_stds,
        val_loss_means=val_loss_means,
        val_loss_stds=val_loss_stds
    )
    plot_accuracy(train_accuracies, val_accuracies)



if __name__ == "__main__":
    # Creating a directory for TensorBoard logs
    log_dir = "runs/spiking_transformer_maze"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # Checking the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    if num_gpus >= 1:
        print(f"Using device: {DEVICE}")
    else:
        print("No GPU available. Exiting.")
        exit(1)

    # Generating data
    print("Generating maze data...")
    data = generate_maze_data(NUM_SAMPLES, MAZE_SIZE)
    print(f"Generated {len(data)} samples.")

    # Checking if any samples were generated
    if len(data) == 0:
        print("No data was generated. Please check the maze generation functions.")
        exit(1)

    # Shuffling and spliting the dataset
    random.shuffle(data)
    num_train = int(0.7 * len(data))
    num_val = int(0.15 * len(data))
    num_test = len(data) - num_train - num_val

    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]

    # Computing state mean and std from training data
    all_states = torch.cat([
        torch.stack([
            torch.tensor(pos, dtype=torch.float32) for pos in sample['path'][:-1]
        ]) for sample in train_data
    ], dim=0)
    state_mean = all_states.mean(dim=0)
    state_std = all_states.std(dim=0) + 1e-8  # Avoid division by zero

    # Moving state_mean and state_std to DEVICE
    state_mean = state_mean.to(DEVICE)
    state_std = state_std.to(DEVICE)

    # Creating datasets and dataloaders
    train_dataset = MazeDataset(train_data, state_mean, state_std)
    val_dataset = MazeDataset(val_data, state_mean, state_std)
    test_dataset = MazeDataset(test_data, state_mean, state_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initializing the Spiking Transformer model
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
    )
    model.to(DEVICE)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Initializing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training the model
    train(model, train_loader, optimizer, scheduler, EPOCHS, writer, val_loader)

    # Evaluating the model on test set
    print("\nFinal evaluation on test data:")
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn=nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")
    writer.add_scalar('Loss/test', test_loss)
    writer.add_scalar('Accuracy/test', test_accuracy * 100)

    # Generating confusion matrix on test set
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
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

            predicted_actions = action_preds.argmax(dim=-1).cpu().numpy()
            true_actions = action_target.cpu().numpy()

            y_true.extend(true_actions)
            y_pred.extend(predicted_actions)

    # Plotting confusion matrix
    classes = ['Left', 'Right', 'Up', 'Down']
    plot_confusion_matrix(y_true, y_pred, classes)

    # Saving the model
    model_save_path = os.path.join(log_dir, 'spiking_transformer_maze.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()
