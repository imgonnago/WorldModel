import torch.nn as nn
import torch
import gymnasium as gym
import gym_pusht
import numpy as np
import os
import config

def collect_random_rollouts(
        num_episodes: int = None,
        max_steps: int = config.MAX_STEPS,
        save_dir: str = None,
        smoothing_std: float = 15.0,   # action이 이전 값에서 얼마나 벗어날지 (범위 0~512 기준)
    ):
        os.makedirs(save_dir, exist_ok=True)
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")

        all_obs = []
        all_actions = []
        all_next_obs = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            prev_action = env.action_space.sample()   # 첫 액션은 랜덤 초기화

            for step in range(max_steps):
                noise = np.random.normal(0, smoothing_std, size=prev_action.shape)
                action = prev_action + noise
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_obs, reward, terminated, truncated, info = env.step(action)

                all_obs.append(obs)
                all_actions.append(action)
                all_next_obs.append(next_obs)

                obs = next_obs
                prev_action = action

                if terminated or truncated:
                    break

            if (episode + 1) % 20 == 0:
                print(f"Episode {episode+1}/{num_episodes} 완료 (누적 {len(all_obs)} 스텝)")

        env.close()

        # numpy array로 변환 후 저장
        obs_array = np.array(all_obs, dtype=np.uint8)          # (N, 96, 96, 3)
        action_array = np.array(all_actions, dtype=np.float32)  # (N, 2)
        next_obs_array = np.array(all_next_obs, dtype=np.uint8) # (N, 96, 96, 3)

        np.save(os.path.join(save_dir, "obs.npy"), obs_array)
        np.save(os.path.join(save_dir, "actions.npy"), action_array)
        np.save(os.path.join(save_dir, "next_obs.npy"), next_obs_array)

        if save_dir == "./save_data/train":    
            print(f"train set 저장 완료: {obs_array.shape[0]}개 스텩")
            print(f"train  obs shape: {obs_array.shape}")
            print(f"train  action shape: {action_array.shape}")
            
        if save_dir == "./save_data/val":    
            print(f"val set 저장 완료: {obs_array.shape[0]}개 스텩")
            print(f"val  obs shape: {obs_array.shape}")
            print(f"val  action shape: {action_array.shape}")

        return obs_array, action_array, next_obs_array

if __name__ == "__main__":
    collect_random_rollouts(num_episodes=1000, save_dir = "./save_data/train")
    collect_random_rollouts(num_episodes=100, save_dir = "./save_data/val")