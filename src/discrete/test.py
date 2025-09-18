import torch
import gymnasium as gym
from discrete_agent import DQNAgent
from gymnasium.wrappers import RecordVideo

class DQNTester:
    def __init__(self, env_name: str, model_path: str, device: str = None, record_video: str = 'rgb_array'):
        """
        Classe para testar agentes DQN.
        """
        self.env_name = env_name
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.render_mode = "rgb_array" if record_video else "human"

        # Cria o ambiente
        self.env = gym.make(env_name, render_mode=self.render_mode)
        self.env = RecordVideo(
            self.env,
            video_folder="videos/discrete/test",
            name_prefix="lunarlander_test",
            episode_trigger=lambda ep: True,
        )

        # Inicializa o agente
        self.agent = self._load_agent()

    def _load_agent(self):
        """
        Cria agente e carrega pesos treinados.
        """
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        agent = DQNAgent(state_size, action_size, device=self.device)
        agent.load(self.model_path)

        # Para teste, força epsilon baixo para comportamento greedy
        agent.epsilon = 0.01
        return agent

    def run_episode(self):
        """
        Executa um episódio de teste usando política greedy.
        Retorna a recompensa total.
        """
        state, _ = self.env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Aqui usamos greedy=True para ação determinística
            action = self.agent.act(state, greedy=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            total_reward += reward
            state = next_state
            done = terminated or truncated

            if self.render_mode == "human":
                self.env.render()

        return total_reward

    def test(self, num_episodes: int = 10):
        """
        Executa múltiplos episódios de teste e imprime resultados.
        """
        print(f"[INFO] Usando dispositivo: {self.device}")
        rewards = []

        for episode in range(1, num_episodes + 1):
            total_reward = self.run_episode()
            rewards.append(total_reward)
            print(f"[TESTE] Episódio {episode} - Recompensa total: {total_reward:.2f}")

        avg_reward = sum(rewards) / len(rewards)
        print(f"[TESTE] Recompensa média após {num_episodes} episódios: {avg_reward:.2f}")
        return rewards

    def close(self):
        """
        Fecha o ambiente.
        """
        self.env.close()


if __name__ == "__main__":
    MODEL_PATH = "src/discrete/discrete_agent.pth"
    ENV_NAME = "LunarLander-v3"

    tester = DQNTester(env_name=ENV_NAME, model_path=MODEL_PATH, record_video="rgb_array")
    tester.test(num_episodes=10)
    tester.close()
