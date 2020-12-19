# Aegis PPO node

## Environment variables
* `OBS_URL` - observation url
* `OBS_SHAPE` - observation shape as int/json array
* `ACTION_SHAPE` - action shape as int/json array
* `PORT` - port to listen for action requests on, defaults to 80
* `POLICY` - Stable Baselines PPO policy to use, defaults to `MlpPolicy`
* `STEPS` - Steps to run for before looping, defaults to 10000

## TODO
* more documentation
* support LSTM/CNN policies
