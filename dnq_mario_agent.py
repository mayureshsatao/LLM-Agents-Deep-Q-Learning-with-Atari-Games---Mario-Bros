"""
Deep Q-Learning Agent for Mario Bros - Complete Solution
This single file contains everything needed for training and demonstration
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import sys
from datetime import datetime

# For ale-py 0.11+, ROMs should be auto-discovered from AutoROM
# No manual registration needed - just import
try:
    import ale_py
    from ale_py import roms
    print("✓ ale_py imported successfully")
except ImportError as e:
    print(f"⚠ Warning: Could not import ale_py: {e}")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==================== HYPERPARAMETERS ====================
LEARNING_RATE = 0.00025
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1  # Keep more exploration (was 0.01)
EPSILON_DECAY = 0.9995  # Slower decay
MEMORY_SIZE = 100000
BATCH_SIZE = 32
TARGET_UPDATE = 1000
NUM_EPISODES = 1000
MAX_STEPS = 10000

# ==================== PROGRESS BAR ====================
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar to console"""
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

# ==================== PREPROCESSING ====================
class FramePreprocessor:
    """Preprocesses frames for efficient learning"""
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height

    def process(self, frame):
        """Convert frame to grayscale and resize"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized

class FrameStack:
    """Stack frames for temporal information"""
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        """Initialize with the same frame repeated"""
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return np.array(self.frames)

    def append(self, frame):
        """Add new frame and return stacked frames"""
        self.frames.append(frame)
        return np.array(self.frames)

# ==================== REPLAY MEMORY ====================
class ReplayMemory:
    """Experience Replay Buffer for storing transitions"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ==================== DQN NETWORK ====================
class DQN(nn.Module):
    """Deep Q-Network with Convolutional Neural Network architecture"""
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()

        # Convolutional layers for feature extraction from frames
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==================== DQN AGENT ====================
class DQNAgent:
    """Deep Q-Learning Agent"""
    def __init__(self, num_actions, device):
        self.num_actions = num_actions
        self.device = device
        self.epsilon = EPSILON_START

        # Policy network and target network
        self.policy_net = DQN(4, num_actions).to(device)
        self.target_net = DQN(4, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, use_epsilon=True):
        """Epsilon-greedy action selection"""
        if use_epsilon and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def get_q_values(self, state):
        """Get Q-values for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy()[0]

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < BATCH_SIZE:
            return None

        # Sample batch from memory
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        # Compute Q(s_t, a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Compute expected Q values
        expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

# ==================== TRAINING ====================
def train_dqn_agent(num_episodes=NUM_EPISODES, save_path='mario_dqn_model.pth', game_name=None):
    """Main training loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to create environment with fallback options
    if game_name is None:
        game_options = [
            "ALE/Breakout-v5",  # Easier to learn - moved to first
            "ALE/Pong-v5",      # Also easier
            "ALE/MarioBros-v5",
            "ALE/SpaceInvaders-v5"
        ]
    else:
        game_options = [game_name]

    env = None
    game_name = None
    for game in game_options:
        try:
            env = gym.make(game)
            game_name = game
            print(f"✓ Successfully created environment: {game}")
            break
        except Exception as e:
            print(f"✗ Could not create {game}: {e}")
            continue

    if env is None:
        print(f"\n{'='*70}")
        print("ERROR: Could not create ANY Atari environment!")
        print("="*70)
        print("\nPlease install Atari ROMs with ONE of these methods:")
        print("\nMethod 1 (Recommended):")
        print("  pip3 install ale-py")
        print("  pip3 install 'gymnasium[atari,accept-rom-license]'")
        print("\nMethod 2:")
        print("  pip3 install AutoROM")
        print("  AutoROM --accept-license")
        print("\nMethod 3:")
        print("  pip3 install gymnasium[atari]")
        print("  python3 -m atari_py.import_roms <path_to_roms>")
        print("="*70)
        return None, None, None

    num_actions = env.action_space.n
    print(f"Number of actions: {num_actions}")

    # Initialize components
    agent = DQNAgent(num_actions, device)
    preprocessor = FramePreprocessor()
    frame_stack = FrameStack()

    # Training metrics
    episode_rewards = []
    episode_losses = []
    avg_rewards = []

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for episode in range(num_episodes):
        state, info = env.reset()
        processed_frame = preprocessor.process(state)
        stacked_state = frame_stack.reset(processed_frame)

        episode_reward = 0
        episode_loss = []

        for step in range(MAX_STEPS):
            # Select and perform action
            action = agent.select_action(stacked_state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Process next state
            processed_next_frame = preprocessor.process(next_state)
            stacked_next_state = frame_stack.append(processed_next_frame)

            # Reward shaping to encourage movement and exploration
            shaped_reward = reward

            # Penalty for repeated NOOP actions (encourage movement)
            if action == 0:  # NOOP
                shaped_reward -= 0.01

            # Small reward for taking any action (encourage exploration)
            if action != 0:
                shaped_reward += 0.001

            # Clip rewards for stability
            clipped_reward = np.clip(shaped_reward, -1, 1)

            # Store transition
            agent.memory.push(stacked_state, action, clipped_reward, stacked_next_state, float(done))

            # Move to next state
            stacked_state = stacked_next_state
            episode_reward += reward

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            # Update target network
            agent.steps_done += 1
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target_network()

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # Calculate moving average
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
        else:
            avg_reward = np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        # Progress bar with stats
        progress_suffix = f"Ep {episode + 1}/{num_episodes} | Reward: {episode_reward:.1f} | Avg: {avg_reward:.1f} | ε: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}"
        print_progress_bar(episode + 1, num_episodes, prefix='Training Progress:', suffix=progress_suffix, length=40)

        # Detailed progress every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{num_episodes} - Detailed Stats:")
            print(f"  Current Reward: {episode_reward:.2f}")
            print(f"  Average Reward (last 100): {avg_reward:.2f}")
            print(f"  Best Reward (so far): {max(episode_rewards):.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Memory Size: {len(agent.memory)}/{MEMORY_SIZE}")
            print(f"  Total Steps: {agent.steps_done}")
            print(f"{'='*70}\n")

    env.close()

    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Plot results
    plot_training_results(episode_rewards, avg_rewards, episode_losses)

    # Save model
    agent.save(save_path)
    print(f"\n✓ Model saved as '{save_path}'")

    return agent, episode_rewards, avg_rewards

def plot_training_results(episode_rewards, avg_rewards, losses):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot rewards
    axes[0].plot(episode_rewards, alpha=0.6, label='Episode Reward', color='steelblue')
    axes[0].plot(avg_rewards, label='Average Reward (100 episodes)', linewidth=2, color='darkred')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Reward', fontsize=12)
    axes[0].set_title('Training Rewards over Episodes', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot losses
    axes[1].plot(losses, label='Loss', color='red', alpha=0.7)
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training Loss over Episodes', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("✓ Training results saved as 'training_results.png'")
    plt.close()

# ==================== DEMONSTRATION ====================
def create_info_overlay(frame, episode_reward, step, action, q_values):
    """Add information overlay to frame"""
    frame_copy = frame.copy()

    # Add black background for text
    overlay_height = 60
    overlay = np.zeros((overlay_height, frame_copy.shape[1], 3), dtype=np.uint8)
    frame_with_overlay = np.vstack([frame_copy, overlay])

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    # Display metrics
    text_y = frame_copy.shape[0] + 20
    cv2.putText(frame_with_overlay, f"Step: {step}", (10, text_y),
                font, font_scale, color, thickness)
    cv2.putText(frame_with_overlay, f"Reward: {episode_reward:.1f}", (150, text_y),
                font, font_scale, color, thickness)

    # Action names
    action_names = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
                    'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
                    'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
                    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

    cv2.putText(frame_with_overlay, f"Action: {action_names[action]}", (10, text_y + 25),
                font, font_scale, (0, 255, 0), thickness)

    max_q = np.max(q_values)
    cv2.putText(frame_with_overlay, f"Max Q: {max_q:.2f}", (300, text_y + 25),
                font, font_scale, (255, 255, 0), thickness)

    return frame_with_overlay

def demonstrate_agent(model_path='mario_dqn_model.pth', num_episodes=5, record_video=True, game_name=None):
    """Demonstrate trained agent"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to create environment with fallback
    if game_name is None:
        game_options = [
            "ALE/MarioBros-v5",
            "ALE/Breakout-v5",
            "ALE/Pong-v5",
            "ALE/SpaceInvaders-v5"
        ]

        env = None
        for game in game_options:
            try:
                env = gym.make(game, render_mode="rgb_array")
                game_name = game
                print(f"✓ Using environment: {game}")
                break
            except Exception:
                continue

        if env is None:
            print(f"\nERROR: Could not create ANY Atari environment!")
            return None, None
    else:
        try:
            env = gym.make(game_name, render_mode="rgb_array")
        except Exception as e:
            print(f"\nERROR: Could not create environment!")
            print(f"Error message: {e}")
            return None, None

    num_actions = env.action_space.n

    # Load model
    print(f"Loading model from {model_path}...")
    agent = DQNAgent(num_actions, device)
    agent.load(model_path)
    agent.policy_net.eval()
    print("✓ Model loaded successfully!")

    preprocessor = FramePreprocessor()

    # Video setup
    frames_to_save = []
    all_rewards = []
    all_steps = []

    print(f"\nStarting demonstration for {num_episodes} episodes...")
    print("=" * 70)

    for episode in range(num_episodes):
        state, info = env.reset()
        frame_stack = FrameStack()
        processed_frame = preprocessor.process(state)
        stacked_state = frame_stack.reset(processed_frame)

        episode_reward = 0
        step = 0
        done = False

        while not done and step < MAX_STEPS:
            # Get Q-values and select action
            q_values = agent.get_q_values(stacked_state)
            action = agent.select_action(stacked_state, use_epsilon=False)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record frame
            if record_video:
                frame = env.render()
                frame_with_info = create_info_overlay(frame, episode_reward, step, action, q_values)
                frames_to_save.append(frame_with_info)

            # Process next state
            processed_next_frame = preprocessor.process(next_state)
            stacked_state = frame_stack.append(processed_next_frame)

            episode_reward += reward
            step += 1

        all_rewards.append(episode_reward)
        all_steps.append(step)

        # Progress bar
        progress_suffix = f"Ep {episode + 1}/{num_episodes} | Reward: {episode_reward:.1f} | Steps: {step}"
        print_progress_bar(episode + 1, num_episodes, prefix='Demo Progress:', suffix=progress_suffix, length=40)

    env.close()
    print()

    # Save video
    if record_video and frames_to_save:
        print("\nSaving video...")
        height, width = frames_to_save[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('mario_gameplay.mp4', fourcc, 30, (width, height))

        for i, frame in enumerate(frames_to_save):
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            if (i + 1) % 100 == 0:
                print_progress_bar(i + 1, len(frames_to_save), prefix='Saving Video:', suffix=f'{i+1}/{len(frames_to_save)} frames', length=40)

        out.release()
        print(f"\n✓ Video saved as 'mario_gameplay.mp4'")
        print(f"  Total frames: {len(frames_to_save)}")
        print(f"  Duration: {len(frames_to_save)/30:.1f} seconds")

    # Summary
    print("\n" + "=" * 70)
    print("DEMONSTRATION SUMMARY")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Best reward: {np.max(all_rewards):.2f}")
    print(f"Worst reward: {np.min(all_rewards):.2f}")
    print(f"Average steps: {np.mean(all_steps):.2f}")
    print("=" * 70)

    # Plot
    plot_demo_results(all_rewards, all_steps)

    return all_rewards, all_steps

def plot_demo_results(rewards, steps):
    """Plot demonstration results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(rewards)), rewards, color='skyblue', edgecolor='navy')
    axes[0].axhline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}', linewidth=2)
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Total Reward', fontsize=12)
    axes[0].set_title('Demonstration Rewards', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(steps)), steps, color='lightgreen', edgecolor='darkgreen')
    axes[1].axhline(np.mean(steps), color='red', linestyle='--', label=f'Mean: {np.mean(steps):.2f}', linewidth=2)
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Steps', fontsize=12)
    axes[1].set_title('Steps per Episode', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demonstration_results.png', dpi=150, bbox_inches='tight')
    print("✓ Demonstration results saved as 'demonstration_results.png'")
    plt.close()

# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Mario Bros Agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'demo', 'both'],
                       help='Mode: train, demo, or both')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                       help='Number of training episodes')
    parser.add_argument('--model', type=str, default='mario_dqn_model.pth',
                       help='Model save/load path')
    parser.add_argument('--demo-episodes', type=int, default=5,
                       help='Number of demonstration episodes')
    parser.add_argument('--game', type=str, default=None,
                       help='Specific game (e.g., ALE/Pong-v5, ALE/Breakout-v5)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  DEEP Q-LEARNING AGENT FOR MARIO BROS")
    print("=" * 70)

    if args.mode == 'train' or args.mode == 'both':
        print("\n" + "=" * 70)
        print("TRAINING MODE")
        print("=" * 70)
        result = train_dqn_agent(
            num_episodes=args.episodes,
            save_path=args.model,
            game_name=args.game
        )
        if result[0] is None:
            sys.exit(1)

    if args.mode == 'demo' or args.mode == 'both':
        print("\n" + "=" * 70)
        print("DEMONSTRATION MODE")
        print("=" * 70)

        if not os.path.exists(args.model):
            print(f"\n✗ ERROR: Model file '{args.model}' not found!")
            print("Please train the agent first using --mode train")
            sys.exit(1)
        else:
            demonstrate_agent(
                model_path=args.model,
                num_episodes=args.demo_episodes,
                record_video=True
            )

    print("\n" + "=" * 70)
    print("✓ ALL DONE!")
    print("=" * 70 + "\n")
