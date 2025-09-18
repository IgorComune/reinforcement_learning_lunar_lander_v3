import gymnasium as gym
import torch
import numpy as np
from collections import deque
from gymnasium.wrappers import RecordVideo

from continuous_agent import TD3Agent


def train_lunarlander(env_name: str,
                      continuous: bool = True,
                      gravity: float = -10.0,
                      enable_wind: bool = False,
                      wind_power: float = 15.0,
                      turbulence_power: float = 1.5,
                      n_episodes: int = 1000,
                      max_t: int = 1000,
                      record_video: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # criar ambiente com os parâmetros
    env = gym.make(
        env_name,
        continuous=continuous,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        render_mode="rgb_array" if record_video else "human"
    )

    if record_video:
        env = RecordVideo(
            env,
            video_folder="videos/continuous/train",
            name_prefix="lunarlander",
            episode_trigger=lambda x: x % 10 == 0,
        )

    # observação
    state, _ = env.reset()
    state_size = env.observation_space.shape[0]
    if continuous:
        action_size = env.action_space.shape[0]  # normalmente 2
        action_high = env.action_space.high
        action_low = env.action_space.low
    else:
        action_size = env.action_space.n


    agent = TD3Agent(
        state_size=state_size,
        action_size=action_size,
    )

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0

        for t in range(max_t):
            if continuous:
                action = agent.act(state)
                # escalar ação para espaço real do ambiente
                # seu agente já deve emitir algo no [-1,1]
                scaled_action = np.clip(action, action_low, action_high)
            else:
                action = agent.act(state)
                scaled_action = action

            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        scores_window.append(score)

        print(
            f"Episode {i_episode} | "
            f"Score: {score:.2f} | "
            f"Average100: {np.mean(scores_window):.2f}"
        )

    env.close()

    if continuous:
        torch.save(agent.actor.state_dict(), "src/continuous/td3_actor.pth")
        torch.save(agent.critic_1.state_dict(), "src/continuous/td3_critic1.pth")
        torch.save(agent.critic_2.state_dict(), "src/continuous/td3_critic2.pth")
    else:
        agent.save("src/continuous/discrete_agent.pth")

    print("Training finished.")
    return scores

if __name__ == "__main__":
    scores = train_lunarlander(
        env_name="LunarLander-v3",
        continuous=True,
        gravity=-9.8,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.3,
        n_episodes=1500,
        max_t=1000,
        record_video=True
    )
