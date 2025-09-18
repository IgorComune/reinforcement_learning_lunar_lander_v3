import gymnasium as gym
import torch
from collections import deque
from gymnasium.wrappers import RecordVideo

from discrete_agent import DQNAgent


def train_dqn(env_name: str, n_episodes: int, max_t: int, record_video: bool):
    """
    Train a DQN agent in the given Gymnasium environment.

    Args:
        env_name (str): Name of the Gymnasium environment.
        n_episodes (int): Number of training episodes.
        max_t (int): Maximum number of timesteps per episode.
        record_video (bool): Whether to record videos of the training.

    Returns:
        list: Episode scores during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(env_name, render_mode="rgb_array" if record_video else "human")
    env = RecordVideo(
        env,
        video_folder="videos/discrete/train",
        name_prefix="eval",
        episode_trigger=lambda x: x % 10 == 0,
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        lr=1e-3,
        epsilon_decay=0.996,
    )

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0

        for _ in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        scores_window.append(score)
        agent.decay_epsilon()

        print(
            f"Episode {i_episode} | "
            f"Score: {score:.1f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    env.close()
    agent.save("src/discrete/discrete_agent.pth")
    print("Training finished. Model saved.")

    return scores


if __name__ == "__main__":
    print("Starting training...")
    train_dqn(
        env_name="LunarLander-v3",
        n_episodes=2000,
        max_t=1000,
        record_video=True
    )
