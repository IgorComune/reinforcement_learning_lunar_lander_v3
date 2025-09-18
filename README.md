# Gymnasium - Lunar Lander v3

## This project implements both **discrete (DQN)** and **continuous (TD3)** reinforcement learning agents for the LunarLander-v3 environment (Gymnasium).  
It includes modular training and testing scripts, pretrained models, and automated video recording of agent performance.  
Experiments are organized into separate folders for discrete and continuous agents, with outputs saved under `videos/` for easy review.

# Setup
* WSL Ubuntu 22.04.5 LTS
* Anaconda
* Python 3.10.12
* Git clone
* pip install requirements

# Structure
```
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── src/
│   ├── continuous/
│   │   ├── __pycache__/
│   │   │   └── continuous_agent.cpython-310.pyc
│   │   ├── continuous_agent.py
│   │   ├── td3_actor.pth
│   │   ├── td3_critic1.pth
│   │   ├── td3_critic2.pth
│   │   ├── test.py
│   │   └── train.py
│   └── discrete/
│       ├── __pycache__/
│       │   └── discrete_agent.cpython-310.pyc
│       ├── discrete_agent.pth
│       ├── discrete_agent.py
│       ├── test.py
│       └── train.py
├── tests/
│   └── notebook.ipynb
└── videos/
    ├── continuous/
    │   ├── test/
    │   │   └── continuous_test.mp4
    │   └── train/
    │       └── continuous_train.mp4
    └── discrete/
        ├── test/
        │   └── discrete_test.mp4
        └── train/
            └── discrete_train.mp4
```

# Execution
* Inside `SRC` folder you'll find both types of problem: `continuous` and `discrete`
* You'll just need to run the files, they are meant to be executed independently.

# Youtube videos
* Reinforcement Learning - Gymnasium - Lunar Lander v3 - Continuous Agent Test
* https://youtu.be/yPhitQ7yjhg

* Reinforcement Learning - Gymnasium - Lunar Lander v3 - Continuous Agent Train
* https://youtu.be/xoewPJNn-40

* Reinforcement Learning - Gymnasium - Lunar Lander v3 - Discrete Agent Train
* https://youtu.be/OJ4G2X7Wv78

* Reinforcement Learning - Gymnasium - Lunar Lander v3 - Discrete Agent Test
* https://youtu.be/eQAXo_WNRn8
