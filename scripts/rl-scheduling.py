import gymnasium
import saga.experiment.ml


def main():
    env = gymnasium.make('saga/Scheduling-v0')
    env.reset()
    
    done = False
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        # print(action)
        observation, reward, done, truncated, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(_+1))
            break

    if not done:
        print("Episode finished after 1000 timesteps")
    env.close()


if __name__ == "__main__":
    main()