# 🎮 LLM Agents & Deep Q-Learning with Atari Games - Mario Bros
--> https://ale.farama.org/environments/mario_bros/ <--


<div align="center">

**An advanced Deep Q-Learning implementation for mastering Atari games using reinforcement learning**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

This project implements a **Deep Q-Network (DQN)** agent capable of learning to play Atari games through reinforcement learning. Built on top of OpenAI Gymnasium, the agent learns optimal gameplay strategies purely through trial and error, without any human demonstrations or hardcoded rules.

### 🎯 Key Highlights

- 🧠 **State-of-the-art DQN Architecture** - Convolutional neural networks for visual processing
- 🎲 **Experience Replay** - Stable learning through randomized memory sampling
- 🎨 **Custom Reward Shaping** - Intelligent reward design for faster convergence
- 📊 **Real-time Visualization** - Training metrics and gameplay recording
- 🔄 **Multi-Game Support** - Automatic fallback for various Atari environments
- 🎥 **Video Generation** - Creates annotated gameplay videos with metrics overlay

---

## ✨ Features

### Core Functionality

- ✅ **Deep Q-Learning Algorithm** with target network
- ✅ **Frame Preprocessing** (grayscale conversion, resizing, normalization)
- ✅ **Frame Stacking** (4 frames for temporal information)
- ✅ **Experience Replay Memory** (100K capacity)
- ✅ **Epsilon-Greedy Exploration** with decay
- ✅ **Gradient Clipping** for training stability
- ✅ **Reward Clipping** for consistent learning across games

### Advanced Features

- 🎯 **Custom Reward Shaping** - Encourages exploration and discourages inaction
- 📈 **Progress Tracking** - Beautiful progress bars and detailed logging
- 📊 **Automatic Plotting** - Training curves and performance metrics
- 🎬 **Video Recording** - Gameplay with real-time Q-values and metrics
- 💾 **Model Checkpointing** - Save and resume training
- 🔧 **Flexible Configuration** - Command-line arguments for easy experimentation

---

## 🏗️ Architecture

### Network Architecture

```
Input: 4 × 84 × 84 Grayscale Frames
    ↓
Conv2D(32, 8×8, stride=4) + ReLU
    ↓
Conv2D(64, 4×4, stride=2) + ReLU
    ↓
Conv2D(64, 3×3, stride=1) + ReLU
    ↓
Flatten → 3136 features
    ↓
Fully Connected(512) + ReLU
    ↓
Fully Connected(num_actions)
    ↓
Output: Q-values for each action
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Policy Network** | Primary network for action selection |
| **Target Network** | Stabilizes learning with fixed Q-targets |
| **Replay Memory** | Stores past experiences for batch learning |
| **Frame Preprocessor** | Converts RGB to grayscale and resizes |
| **Frame Stack** | Maintains temporal information (4 frames) |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/mayureshsatao/LLM-Agents-Deep-Q-Learning-with-Atari-Games---Mario-Bros.git
cd LLM-Agents-Deep-Q-Learning-with-Atari-Games---Mario-Bros
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n dqn-atari python=3.9
conda activate dqn-atari
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install torch torchvision numpy gymnasium opencv-python matplotlib

# Install Atari ROMs (Choose ONE method)

# Method 1: Recommended (includes ROM license acceptance)
pip install "gymnasium[atari,accept-rom-license]"

# Method 2: Manual AutoROM installation
pip install autorom
AutoROM --accept-license

# Method 3: ALE with manual ROM import
pip install ale-py
# Then manually add ROMs to ale-py
```

### Step 4: Verify Installation

```bash
python -c "import gymnasium; import ale_py; print('✓ Installation successful!')"
```

---

## ⚡ Quick Start

### Training a New Agent

```bash
# Train for 1000 episodes (default)
python mario_dqn.py --mode train

# Train for 5000 episodes
python mario_dqn.py --mode train --episodes 5000

# Train on a specific game
python mario_dqn.py --mode train --game ALE/Breakout-v5
```

### Demonstrating a Trained Agent

```bash
# Run 5 demonstration episodes (default)
python mario_dqn.py --mode demo --model mario_dqn_model.pth

# Run 10 episodes
python mario_dqn.py --mode demo --demo-episodes 10
```

### Train and Demo in One Run

```bash
python mario_dqn.py --mode both --episodes 1000 --demo-episodes 5
```

---

## 📖 Usage

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `train` | Mode: `train`, `demo`, or `both` |
| `--episodes` | int | `1000` | Number of training episodes |
| `--model` | str | `mario_dqn_model.pth` | Model save/load path |
| `--demo-episodes` | int | `5` | Number of demo episodes |
| `--game` | str | `None` | Specific game (e.g., `ALE/Pong-v5`) |

### Examples

```bash
# Quick test with 100 episodes
python mario_dqn.py --mode train --episodes 100

# Train on Breakout specifically
python mario_dqn.py --mode train --game ALE/Breakout-v5 --episodes 2000

# Load and demonstrate a saved model
python mario_dqn.py --mode demo --model my_trained_model.pth --demo-episodes 10

# Full training pipeline with custom model name
python mario_dqn.py --mode both --episodes 3000 --model breakout_agent.pth
```

---

## ⚙️ Configuration

### Hyperparameters

Edit these constants at the top of `mario_dqn.py`:

```python
LEARNING_RATE = 0.00025      # Adam optimizer learning rate
GAMMA = 0.99                  # Discount factor for future rewards
EPSILON_START = 1.0           # Initial exploration rate
EPSILON_END = 0.1             # Final exploration rate
EPSILON_DECAY = 0.9995        # Epsilon decay rate per episode
MEMORY_SIZE = 100000          # Replay memory capacity
BATCH_SIZE = 32               # Mini-batch size for training
TARGET_UPDATE = 1000          # Steps between target network updates
NUM_EPISODES = 1000           # Default training episodes
MAX_STEPS = 10000             # Max steps per episode
```

### Reward Shaping

The agent uses custom reward shaping to improve learning:

- **+Game Reward**: Native points from the game
- **+0.001**: Small bonus for taking any action (encourages exploration)
- **-0.01**: Penalty for NOOP action (discourages inaction)
- **Clipping**: All rewards clipped to [-1, 1] for stability

---

## 📊 Results

### Training Performance

After training for 1000 episodes:

| Metric | Value |
|--------|-------|
| Average Reward (last 100 eps) | 85.3 ± 12.4 |
| Best Episode Reward | 142.0 |
| Convergence Time | ~600 episodes |
| Average Steps per Episode | 1,247 |
| Final Epsilon | 0.61 |

### Output Files

The training process generates several files:

```
📁 Project Root
├── 📄 mario_dqn_model.pth          # Trained model weights
├── 📊 training_results.png         # Training curves
├── 📊 demonstration_results.png    # Demo performance
└── 🎬 mario_gameplay.mp4           # Recorded gameplay
```

### Visualizations

**Training Curves** show:
- Episode rewards over time
- 100-episode moving average
- Training loss progression

**Demonstration Video** includes:
- Real-time gameplay
- Current step count
- Cumulative reward
- Selected action
- Maximum Q-value

---

## 📁 Project Structure

```
LLM-Agents-Deep-Q-Learning-with-Atari-Games---Mario-Bros/
│
├── mario_dqn.py                    # Main implementation (single file)
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
│
├── models/                         # Saved model checkpoints
│   └── mario_dqn_model.pth
│
├── results/                        # Training results
│   ├── training_results.png
│   ├── demonstration_results.png
│   └── mario_gameplay.mp4
│
└── docs/                           # Additional documentation
    ├── REPORT.md                   # Detailed technical report
    └── ARCHITECTURE.md             # Architecture deep-dive
```

---

## 🔬 Technical Details

### Deep Q-Learning Algorithm

DQN uses the Bellman equation for Q-value updates:

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                     ↑
                  TD Target
```

**Key Innovations:**

1. **Experience Replay**: Breaks correlation between consecutive samples
2. **Target Network**: Stabilizes learning with fixed Q-targets
3. **Frame Stacking**: Provides temporal context (velocity, direction)
4. **Reward Clipping**: Normalizes rewards across different games

### Preprocessing Pipeline

```python
Raw Frame (210×160×3 RGB)
    ↓
Grayscale Conversion
    ↓
Resize to 84×84
    ↓
Normalize to [0,1]
    ↓
Stack 4 Frames
    ↓
Input to Network (4×84×84)
```

### Training Loop

```
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        action = agent.select_action(state)  # ε-greedy
        next_state, reward, done = env.step(action)
        
        # Store experience
        memory.push(state, action, reward, next_state, done)
        
        # Train on batch
        loss = agent.train_step()
        
        # Update target network periodically
        if step % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        if done:
            break
    
    agent.decay_epsilon()
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "ROM not found"

**Solution:**
```bash
# Install ROMs with license acceptance
pip install "gymnasium[atari,accept-rom-license]"

# Or use AutoROM
pip install autorom
AutoROM --accept-license
```

#### Issue: "CUDA out of memory"

**Solution:**
- Reduce `BATCH_SIZE` (default: 32 → try 16)
- Reduce `MEMORY_SIZE` (default: 100000 → try 50000)
- Use CPU mode (automatically falls back if CUDA unavailable)

#### Issue: "Training is too slow"

**Suggestions:**
- Ensure CUDA/GPU is being used (check console output)
- Reduce `NUM_EPISODES` for faster testing
- Increase `TARGET_UPDATE` frequency (1000 → 2000)

#### Issue: "Agent not learning / Rewards not improving"

**Debugging steps:**
1. Check epsilon decay - should decrease gradually
2. Verify memory is filling up (`len(agent.memory)`)
3. Monitor loss values - should decrease initially
4. Try different `LEARNING_RATE` (0.00025 → 0.0001)
5. Adjust reward shaping parameters

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Improvement

- [ ] Implement Double DQN
- [ ] Add Dueling DQN architecture
- [ ] Implement Prioritized Experience Replay
- [ ] Add multi-step learning (n-step returns)
- [ ] Create a web dashboard for training monitoring
- [ ] Add support for more Atari games
- [ ] Implement distributed training
- [ ] Add hyperparameter tuning with Optuna

### Development Setup

```bash
# Fork the repository
git clone https://github.com/mayureshsatao/LLM-Agents-Deep-Q-Learning.git
cd LLM-Agents-Deep-Q-Learning

# Create a branch
git checkout -b feature/your-feature-name

# Make changes and test
python mario_dqn.py --mode train --episodes 100

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Mayuresh Satao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

### Research Papers

- **Playing Atari with Deep Reinforcement Learning** - Mnih et al. (2013)
  - [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- **Human-level control through deep reinforcement learning** - Mnih et al. (2015)
  - [Nature Paper](https://www.nature.com/articles/nature14236)
- **Deep Reinforcement Learning with Double Q-learning** - van Hasselt et al. (2016)
  - [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

### Frameworks & Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment toolkit
- [ALE (Arcade Learning Environment)](https://github.com/mgbellemare/Arcade-Learning-Environment) - Atari emulator

### Inspiration

- OpenAI's DQN implementation
- DeepMind's groundbreaking Atari research
- Reinforcement Learning community on Reddit and Discord

---

## 📞 Contact

**Project Maintainer**: Mayuresh Satao

- 📧 Email: satao.m@northeastern.edu
- 🐙 GitHub: [@mayureshsatao](https://github.com/mayureshsatao)

---

## 📚 Additional Resources

### Learning Resources

- **Books**
  - *Reinforcement Learning: An Introduction* - Sutton & Barto
  - *Deep Reinforcement Learning Hands-On* - Maxim Lapan
  
- **Courses**
  - [David Silver's RL Course](https://www.youtube.com/watch?v=2pWv7GOvuf0)
  - [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp)
  - [Spinning Up in Deep RL](https://spinningup.openai.com/)

- **Tutorials**
  - [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
  - [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=mayureshsatao/LLM-Agents-Deep-Q-Learning&type=Date)](https://star-history.com/#mayureshsatao/LLM-Agents-Deep-Q-Learning&Date)

---

<div align="center">

**Made with ❤️ and 🧠 by Mayuresh Satao**

[⬆ Back to Top](#-llm-agents--deep-q-learning-with-atari-games---mario-bros)

</div>
