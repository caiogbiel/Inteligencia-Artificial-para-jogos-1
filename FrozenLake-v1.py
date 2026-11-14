import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)


n_episodes = 5000       
max_steps = 100         
learning_rate = 0.1     
gamma = 0.99            
epsilon = 1.0           
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005      

q_table = np.zeros((env.observation_space.n, env.action_space.n))

rewards_per_episode = []

for episode in range(n_episodes):
    state, info = env.reset()
    total_reward = 0
    
    for step in range(max_steps):        
        exp_thresh = np.random.uniform(0, 1)
        if exp_thresh < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(q_table[state, :]) 
        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_per_episode.append(total_reward)

env.close()
print("Treinamento concluído!")

final_rewards = rewards_per_episode[-500:] # Últimos 500 episódios
media_recompensa = np.mean(final_rewards)
desvio_padrao = np.std(final_rewards)

print(f"Média de Recompensa Final: {media_recompensa:.4f}")
print(f"Desvio Padrão Final: {desvio_padrao:.4f}")

plt.figure(figsize=(10, 5))
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid')
plt.plot(smoothed_rewards, color='blue', label='Recompensa Média Móvel (100 episódios)')
plt.title('Desempenho do Q-learning no FrozenLake-v1')
plt.xlabel('Episódios')
plt.ylabel('Recompensa Média')
plt.grid(True)
plt.legend()
plt.savefig('q_learning_desempenho.png')
plt.show()
