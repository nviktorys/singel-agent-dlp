from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv

import time

import single_agent_dlp.env.simple_pricing as simple_pricing

# pettingzoo 1.12.0
# https://github.com/Farama-Foundation/PettingZoo/tree/master/pettingzoo/mpe
REGISTRY = {}
REGISTRY["simple_pricing"] = simple_pricing

policy_mapping_dict = {
    "simple_pricing": {
        "description": "one agent & customers with random decisioning",
        "team_prefix": ("adversary_", "agent_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    }
}


class RLlibDLP(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        env = REGISTRY[map](**env_config)
        # For environments where all agents have simultaneous actions and observations
        # use ParallelPettingZooEnv. This API is based around the paradigm of Partially
        # Observable Stochastic Games - https://pettingzoo.farama.org/api/parallel/
        # self.env = env
        self.env = ParallelPettingZooEnv(env)
        # assume all agent same action/obs space
        # self.action_space = self.env.action_space[0]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {"obs": o[key]}
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    # def render(self, mode=None):
    #     self.env.render()
    #     time.sleep(0.05)
    #     return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 25,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info
