import meiro

env = meiro.meiro(grid_size=9)
obs = env.reset()
env.render()
n_steps = 10

for step in range(n_steps):
  print("Step {}".format(step + 1))
  myaction = 0 # L, R, U, D
  obs, reward, done, info = env.step(myaction)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render()
  if done:
    print("Goal !!", "reward=", reward)
    break
