import os

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env import AegisEnv
import json

import sys
import signal
signal.signal(signal.SIGTERM, lambda: sys.exit(0))

obs_shape = os.getenv("OBS_SHAPE", "[]")
obs_shape = json.loads(obs_shape)
action_shape = os.getenv("ACTION_SHAPE", "[]")
action_shape = json.loads(action_shape)
port = int(os.getenv("PORT", 80))
policy = os.getenv("POLICY", "MlpPolicy")
nsteps = int(os.getenv("STEPS", 10000))
RESET = os.getenv("RESET", "").lower() in (True, 'true') #default to false
MODEL_PATH = os.getenv("MODEL_PATH", "models/model")
VERBOSE = int(os.getenv("VERBOSE", 0))

#ppo params
#according to defaults here https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
#TODO: make these default to none so they arent filled in maybe idk
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.00025))
GAMMA = float(os.getenv("GAMMA", 0.99))
LAMBDA = float(os.getenv("LAMBDA", 0.95))

# Create environment
env = AegisEnv(obs_shape, action_shape, port=port)
env = DummyVecEnv([lambda: env])
#TODO: support LSTM/CNN policies

model = None
#load model if not RESET
if RESET:
    model = PPO2(policy, env)
else:
    try:
        print("Loading", MODEL_PATH)
        model = PPO2.load(MODEL_PATH, env=env, learning_rate=LEARNING_RATE, gamma=GAMMA, lam=LAMBDA, verbose=VERBOSE)
        print(MODEL_PATH, "loaded")
    except ValueError as e:
        print(e)
        print('"{}" not found, or other ValueError occurred when loading model. Creating new model.'.format(MODEL_PATH))
        model = PPO2(policy, env, learning_rate=LEARNING_RATE, gamma=GAMMA, lam=LAMBDA, verbose=VERBOSE)
        model.save(MODEL_PATH)

#train
while True:
  model.learn(total_timesteps=nsteps)
  model.save(MODEL_PATH)
