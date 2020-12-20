import os

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env import AegisEnv
import json

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

obs_url = os.getenv("OBS_URL")
obs_shape = os.getenv("OBS_SHAPE", "[]")
obs_shape = json.loads(obs_shape)
action_shape = os.getenv("ACTION_SHAPE", "[]")
action_shape = json.loads(action_shape)
port = int(os.getenv("PORT", 80))
policy = os.getenv("POLICY", "MlpPolicy")
nsteps = int(os.getenv("STEPS", 10000))
RESET = os.getenv("RESET", "").lower() in (True, 'true') #default to false
MODEL_PATH = os.getenv("MODEL_PATH", "models/model")

# Create environment
env = AegisEnv(obs_url, obs_shape, action_shape, port=port)
env = DummyVecEnv([lambda: env])
#TODO: support LSTM policy

model = None
#load model if not RESET
if RESET:
    model = PPO2(policy, env)
else:
    try:
        print("Loading", MODEL_PATH)
        model = PPO2.load(MODEL_PATH, env=env)
        print(MODEL_PATH, "loaded")
    except ValueError as e:
        print(e)
        print('"{}" not found, or other ValueError occurred when loading model. Creating new model.'.format(MODEL_PATH))
        model = PPO2(policy, env)
        model.save(MODEL_PATH)

#train
while True:
  model.learn(total_timesteps=nsteps)
  model.save(MODEL_PATH)
