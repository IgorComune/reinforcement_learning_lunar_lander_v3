import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

from continuous_agent import TD3Agent


class TD3Tester:
    def __init__(self,
                 env_name: str,
                 model_path: str,
                 device: str = None,
                 record_video: bool = False,
                 gravity: float = -10.0,
                 enable_wind: bool = False,
                 wind_power: float = 15.0,
                 turbulence_power: float = 1.5):
        """
        Classe para testar agentes TD3.
        """
        self.env_name = env_name
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.render_mode = "rgb_array" if record_video else "human"

        # Cria ambiente
        self.env = gym.make(
            env_name,
            continuous=True,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            render_mode=self.render_mode
        )

        if record_video:
            self.env = RecordVideo(
                self.env,
                video_folder="videos/continuous/test",
                name_prefix="lunarlander_test",
                episode_trigger=lambda ep: True,
            )

        # Inicializa agente
        self.agent = self._load_agent()

    def _load_agent(self):
        """
        Cria agente TD3 e carrega pesos do ator.
        """
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]
        agent = TD3Agent(state_size=state_size, action_size=action_size)

        agent.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
        agent.actor.eval()
        return agent

    def run_episode(self):
        """
        Executa um episódio de teste com política determinística.
        Retorna a recompensa total.
        """
        state, _ = self.env.reset()
        total_reward = 0.0
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low

        while True:
            action = self.agent.act(state, noise=0.0)
            action = np.clip(action, action_low, action_high)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        return total_reward

    def test(self, num_episodes: int = 10):
        """
        Executa múltiplos episódios de teste e imprime resultados.
        """
        print(f"[INFO] Usando dispositivo: {self.device}")
        rewards = []

        for episode in range(1, num_episodes + 1):
            score = self.run_episode()
            rewards.append(score)
            print(f"[TESTE] Episódio {episode} - Recompensa total: {score:.2f}")

        avg_score = np.mean(rewards)
        print(f"[TESTE] Recompensa média após {num_episodes} episódios: {avg_score:.2f}")
        return rewards

    def close(self):
        """Fecha o ambiente."""
        self.env.close()


if __name__ == "__main__":
    MODEL_PATH = "src/continuous/td3_actor.pth"
    ENV_NAME = "LunarLander-v3"

    tester = TD3Tester(
        env_name=ENV_NAME,
        model_path=MODEL_PATH,
        record_video=True,
        gravity=-9.8,
        enable_wind=True,
        wind_power=10.0,
        turbulence_power=1.3
    )
    tester.test(num_episodes=10)
    tester.close()
