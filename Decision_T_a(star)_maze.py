

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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools


import transformers
from transformers import GPT2Config


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Hyperparameters
STATE_DIM = 2       # State represented as (x, y) coordinates
ACT_DIM = 4         # Four possible actions: left, right, up, down
HIDDEN_SIZE = 256   # Hidden size
MAX_LENGTH = 100    # Maximum sequence length
MAX_EP_LEN = 1000   # Maximum episode length
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAZE_SIZE = (21, 21)
NUM_SAMPLES = 500

# Action mapping: map movements to action integers
ACTION_MAP = {
    (-1, 0): 0,  # Left
    (1, 0): 1,   # Right
    (0, -1): 2,  # Up
    (0, 1): 3    # Down
}
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

        # Converting path to states and actions
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

        # Normalizing states
        states = [((state - self.state_mean) / self.state_std).clone().detach() for state in states]

        # Ensuring sequences are of length 'self.max_length'
        seq_length = len(states)
        if seq_length >= self.max_length:
            # Truncate
            states = states[:self.max_length]
            actions = actions[:self.max_length]
            returns_to_go = returns_to_go[:self.max_length]
            timesteps = timesteps[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            # Pad
            padding_length = self.max_length - seq_length
            states += [torch.zeros(STATE_DIM)] * padding_length
            actions += [0] * padding_length  # Assuming '0' is a valid action and used for padding
            returns_to_go += [0.0] * padding_length
            timesteps += [0] * padding_length
            attention_mask = [1] * seq_length + [0] * padding_length

        # Converting to tensors
        states = torch.stack(states)
        actions_indices = torch.tensor(actions, dtype=torch.long)  # (T,)
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
        timesteps = torch.tensor(timesteps, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # One-hot encode actions and convert to float
        actions_one_hot = F.one_hot(actions_indices, num_classes=ACT_DIM).float()  # (T, act_dim)

        # If actions are less than max_length due to padding, ensure they are zeroed
        if len(actions_one_hot) < self.max_length:
            padding = torch.zeros(self.max_length - len(actions_one_hot), ACT_DIM)
            actions_one_hot = torch.cat([actions_one_hot, padding], dim=0)

        sample = {
            'states': states,                       # (T, STATE_DIM)
            'actions': actions_one_hot,             # (T, ACT_DIM)
            'actions_indices': actions_indices,     # (T,)
            'returns_to_go': returns_to_go,          # (T,)
            'timesteps': timesteps,                 # (T,)
            'attention_mask': attention_mask        # (T,)
        }
        return sample


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
                maze[cy + dy // 2][cx + dx // 2] = 0
                maze[ny][nx] = 0
                carve_passages_from(nx, ny, maze)

    maze[1][1] = 0
    carve_passages_from(1, 1, maze)
    # entrance and exit
    maze[1][0] = 0
    maze[height - 2][width - 1] = 0
    return maze

def solve_maze(maze):
    width = len(maze[0])
    height = len(maze)
    start = (0, 1)
    goal = (width - 1, height - 2)

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
    return None

def generate_maze_data(num_samples, maze_size):
    data = []
    for _ in range(num_samples):
        maze = create_maze(*maze_size)
        path = solve_maze(maze)
        if path is not None:
            num_steps = len(path) - 1
            rewards = [-0.1] * num_steps
            rewards[-1] += 1.0
            data.append({'path': path, 'rewards': rewards})
    return data



def plot_loss_with_variance(epochs,
                            train_loss_means, train_loss_stds,
                            val_loss_means, val_loss_stds):
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


# Decision Transformer


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        raise NotImplementedError

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        raise NotImplementedError


class GPT2ModelCustom(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transformer = transformers.GPT2Model(config)

    def forward(self, inputs_embeds, attention_mask=None):
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state


class DecisionTransformer(TrajectoryModel):


    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=False,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=8,
            n_head=8,

            **kwargs
        )


        self.transformer = GPT2ModelCustom(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)


        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention_mask for GPT: 1 => attend, 0 => ignore
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        #  Embedding each modality
        state_embeddings = self.embed_state(states)                 # (B, T, hidden)
        action_embeddings = self.embed_action(actions)              # (B, T, hidden)
        returns_embeddings = self.embed_return(returns_to_go.unsqueeze(-1))  # (B, T, hidden)
        time_embeddings = self.embed_timestep(timesteps)            # (B, T, hidden)

        # Adding time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        #  Re-arrange into sequence: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # shape => (B, 3*T, hidden_size)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        )  # (B, 3, T, hidden_size)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(
            batch_size, 3 * seq_length, self.hidden_size
        )

        #  LayerNorm
        stacked_inputs = self.embed_ln(stacked_inputs)

        #  Similarly, repeating attention mask
        # shape => (B, 3*T)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        )  # (B, 3, T)
        stacked_attention_mask = stacked_attention_mask.permute(0, 2, 1).reshape(
            batch_size, 3 * seq_length
        )

        #  Passing into GPT2
        x = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
        )
        # shapeing (B, 3*T, hidden_size)

        # Reshaping back to (B, T, 3, hidden) => (B, 3, T, hidden)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])  # shape: (B, T, 1)
        state_preds = self.predict_state(x[:, 2])    # shape: (B, T, state_dim)
        # We want to predict the *next* action from the state token => index 1
        action_preds = self.predict_action(x[:, 1])  # shape: (B, T, act_dim)

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):


        _, action_preds, _ = self.forward(states, actions, rewards, returns_to_go, timesteps, **kwargs)
        return action_preds[:, -1]  # shape (B, act_dim)



def evaluate(model, dataloader, loss_fn=None):
    model.eval()
    total_correct = 0
    total_actions = 0
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            states = batch['states'].to(DEVICE)
            actions = batch['actions'].to(DEVICE)
            actions_indices = batch['actions_indices'].to(DEVICE)
            returns_to_go = batch['returns_to_go'].to(DEVICE)
            timesteps = batch['timesteps'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            # DecisionTransformer returns: (state_preds, action_preds, return_preds)
            _, action_preds, _ = model(
                states=states,
                actions=actions,
                rewards=None,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )  # action_preds: (B, T, ACT_DIM)

            # Flattening the tensors for loss computation
            action_preds = action_preds.view(-1, ACT_DIM)
            actions_target = actions_indices.view(-1)

            # Computing loss only on non-padded actions
            mask = attention_mask.view(-1) > 0
            action_preds = action_preds[mask]
            actions_target = actions_target[mask]

            if loss_fn is not None:
                loss = loss_fn(action_preds, actions_target)
                total_loss += loss.item()

            predicted_actions = action_preds.argmax(dim=-1)
            total_correct += (predicted_actions == actions_target).sum().item()
            total_actions += actions_target.size(0)

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
            actions_indices = batch['actions_indices'].to(DEVICE)
            returns_to_go = batch['returns_to_go'].to(DEVICE)
            timesteps = batch['timesteps'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            optimizer.zero_grad()

            # forward pass
            _, action_preds, _ = model(
                states=states,
                actions=actions,
                rewards=None,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )  # action_preds: (B, T, ACT_DIM)

            # Flattening the tensors for loss computation
            action_preds = action_preds.view(-1, ACT_DIM)
            actions_target = actions_indices.view(-1)


            mask = attention_mask.view(-1) > 0
            action_preds = action_preds[mask]
            actions_target = actions_target[mask]

            loss = loss_fn(action_preds, actions_target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())

            # accuracy
            predicted_actions = action_preds.argmax(dim=-1)
            correct += (predicted_actions == actions_target).sum().item()
            total += actions_target.size(0)

            # Logging histograms every 100 batches
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
                actions_indices_val = batch_val['actions_indices'].to(DEVICE)
                returns_to_go_val = batch_val['returns_to_go'].to(DEVICE)
                timesteps_val = batch_val['timesteps'].to(DEVICE)
                attention_mask_val = batch_val['attention_mask'].to(DEVICE)

                _, action_preds_val, _ = model(
                    states=states_val,
                    actions=actions_val,
                    rewards=None,
                    returns_to_go=returns_to_go_val,
                    timesteps=timesteps_val,
                    attention_mask=attention_mask_val
                )  # action_preds_val: (B, T, ACT_DIM)


                action_preds_val = action_preds_val.view(-1, ACT_DIM)
                actions_target_val = actions_indices_val.view(-1)

                mask_val = attention_mask_val.view(-1) > 0                         # (B*T,)
                action_preds_val = action_preds_val[mask_val]
                actions_target_val = actions_target_val[mask_val]

                loss_val = loss_fn(action_preds_val, actions_target_val)
                val_batch_losses.append(loss_val.item())

                # accuracy
                predicted_actions_val = action_preds_val.argmax(dim=-1)            # (B*T,)
                val_correct += (predicted_actions_val == actions_target_val).sum().item()
                val_total += actions_target_val.size(0)

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
    log_dir = "runs/decision_transformer_maze"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus < 1:
        print("No GPU available.")
        exit(1)

    print("Generating maze data...")
    data = generate_maze_data(NUM_SAMPLES, MAZE_SIZE)
    print(f"Generated {len(data)} samples.")
    if len(data) == 0:
        print("No data was generated. Please check the maze generation functions.")
        exit(1)

    random.shuffle(data)
    num_train = int(0.7 * len(data))
    num_val = int(0.15 * len(data))
    num_test = len(data) - num_train - num_val

    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]


    all_states = torch.cat([
        torch.stack([
            torch.tensor(pos, dtype=torch.float32) for pos in sample['path'][:-1]
        ]) for sample in train_data
    ], dim=0)
    state_mean = all_states.mean(dim=0)
    state_std = all_states.std(dim=0) + 1e-8

    state_mean = state_mean.to(DEVICE)
    state_std = state_std.to(DEVICE)

    train_dataset = MazeDataset(train_data, state_mean, state_std)
    val_dataset = MazeDataset(val_data, state_mean, state_std)
    test_dataset = MazeDataset(test_data, state_mean, state_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


    # Instantiating Decision Transformer

    model = DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        hidden_size=HIDDEN_SIZE,
        max_length=MAX_LENGTH,
        max_ep_len=MAX_EP_LEN,
        action_tanh=False
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


    train(model, train_loader, optimizer, scheduler, EPOCHS, writer, val_loader)

    # Final test evaluation
    print("\nFinal evaluation on test data:")
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn=nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")
    writer.add_scalar('Loss/test', test_loss)
    writer.add_scalar('Accuracy/test', test_accuracy * 100)

    # Confusion matrix
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            states = batch['states'].to(DEVICE)
            actions = batch['actions'].to(DEVICE)
            actions_indices = batch['actions_indices'].to(DEVICE)
            returns_to_go = batch['returns_to_go'].to(DEVICE)
            timesteps = batch['timesteps'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            _, action_preds, _ = model(
                states=states,
                actions=actions,
                rewards=None,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )  # action_preds: (B, T, ACT_DIM)


            action_preds = action_preds.view(-1, ACT_DIM)
            actions_target = actions_indices.view(-1)

            mask = attention_mask.view(-1) > 0
            action_preds = action_preds[mask]
            actions_target = actions_target[mask]

            predicted_actions = action_preds.argmax(dim=-1).cpu().numpy()
            true_actions = actions_target.cpu().numpy()

            y_true.extend(true_actions)
            y_pred.extend(predicted_actions)

    classes = ['Left', 'Right', 'Up', 'Down']
    plot_confusion_matrix(y_true, y_pred, classes)

    model_save_path = os.path.join(log_dir, 'decision_transformer_maze.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()
