import gym
from gym import spaces
import numpy as np
import threading

from protopost import ProtoPost
from protopost import protopost_client as ppcl
from nd_to_json import nd_to_json, json_to_nd

class AegisEnv(gym.Env):
    def __init__(self, obs_url, obs_shape, action_shape, port=80, nsteps=None, action_low=-1, action_high=1):
        self.obs_url = obs_url
        print(obs_shape, action_shape)

        if type(action_shape) is int:
            self.action_space = spaces.Discrete(action_shape)
        else:
            #self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=action_shape)
            self.action_space = spaces.Box(low=action_low, high=action_high, shape=action_shape) #TODO: testing this

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)

        self.step_reward = 0 #received rewards
        self.action = None #action to serve

        self.nsteps = nsteps
        self.step_counter = 0

        self.update_event = threading.Event()
        self.flask_wait = threading.Event()

        self.start_server(port)

    def start_server(self, port):
        #build protopost routes
        def pp_action(data):
            return nd_to_json(self.action)

        def pp_reward(data):
            self.step_reward += data

        def pp_step(data):
            #set update event
            self.update_event.set()
            #wait for step to be ready
            self.flask_wait.wait()
            self.flask_wait.clear()

        routes = {
            "action": pp_action,
            "reward": pp_reward,
            "step": pp_step
        }

        def start_app():
            ProtoPost(routes).start(port)

        #run flask in separate thread
        thread = threading.Thread(target=start_app)
        thread.daemon = True
        thread.start()

    def get_observation(self):
        #catch None/errors and keep trying until we get a response
        obs = None
        while obs is None:
            try:
                obs = json_to_nd(ppcl(self.obs_url))
            except ValueError as e:
                print("Error when getting observation")
                print(e)

        return obs

    def step(self, action):
        #set action
        self.action = action

        #wait for update call
        self.flask_wait.set()
        self.update_event.wait()
        self.update_event.clear()

        #grab obs
        obs = self.get_observation()
        #flip reward
        r = self.step_reward
        self.step_reward = 0

        done = False
        if self.nsteps is not None and self.step_counter >= self.nsteps:
            done = True

        return obs, r, done, {}

    def reset(self):
        self.step_counter = 0

        return self.get_observation()
