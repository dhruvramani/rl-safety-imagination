env = GridworldEnv("conveyor_belt")

ob_space = envs.observation_space.shape
nc, nw, nh = ob_space

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

done = False
state = env.reset()
while(done != True):
    print(state)
    action = env.action_space.sample()
    state, reward, done, _ = envs.step(action)
