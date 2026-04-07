import torch
from Mirage_RL.training.agent import Agent
from Mirage_RL.client import QueryClient
from Mirage_RL.models import QueryAction

agent = Agent(num_tables=3)
episodes = 100

with QueryClient(base_url="http://localhost:8000").sync() as env:
    for episode in range(episodes):
        result = env.reset()
        obs = result.observation

        state = agent.encode_state(obs)
        total_reward = 0
        done = False

        while not done:
            # select action
            (table, join, index), action_id = agent.select_action(obs)

            action = QueryAction(
                next_table=table,
                join_type=join,
                use_index=index
            )

            result = env.step(action)

            next_obs = result.observation
            reward = result.reward
            done = result.done

            next_state = agent.encode_state(next_obs)

            # train
            agent.train_step(state, action_id, reward, next_state, done)

            state = next_state
            obs = next_obs
            total_reward += reward

        print(f"Episode {episode:>3} | Total Reward: {total_reward:>8.2f} | Epsilon: {agent.epsilon:.3f}")