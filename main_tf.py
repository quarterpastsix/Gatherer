import gym
#from gatherer.dqn_tf import Agent
from dqn_tf import Agent
# from gatherer.ReinforcementLearning.DeepQLearning.utils import plotLearning
import numpy as np
#from gatherer.GatherEnvShare import Simulation
from GatherEnvShare import Simulation
import matplotlib.pyplot as plt

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def preprocess(observation):
    return observation
    # return np.mean(observation[30:,:], axis=2).reshape(180,160,1)

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
        stacked_frames[buffer_size-1, :] = frame

    stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)

    return stacked_frames


if __name__ == '__main__':
    env = Simulation()
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=(64,64,1),
                  n_actions=10, mem_size=25000, batch_size=32)
    if load_checkpoint:
        agent.load_models()
    filename = 'gatherer-alpha0p000025-gamma0p99-only-one-fc.png'
    scores = []
    eps_history = []
    numGames = 1000
    stack_size = 1 # motion doesnt really matter
    score = 0
    # uncomment the line below to record every episode.
    #env = wrappers.Monitor(env, "tmp/breakout-0",
    #                         video_callable=lambda episode_id: True, force=True)

    # Not always necessary to load up agent's memory with random gameplay
    # in the lunar lander, I found it to be counter productive.

    """
    print("Loading up the agent's memory with random gameplay")

    while agent.mem_cntr < 25000:
        done = False
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        while not done:
            action = env.action_space.sample()
            # action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(stacked_frames,
                                        preprocess(observation_), stack_size)
            action -= 1
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
    print("Done with random gameplay. Game on.")
    """
    steps = 0
    for i in range(numGames):
        done = False
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % agent.epsilon, 'steps', steps)
            #agent.save_models()
        else:
            print('episode: ', i,'score: ', score, 'steps', steps)

        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            steps += 1
            # doesn't look like we need to add 1 to the action, because
            # the action_space is a discrete(10), so should be 0, ..., 10 yeah?
            #action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(stacked_frames,
                                        preprocess(observation_), stack_size)
            score += reward
            #action -= 1
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)
    x = [i+1 for i in range(numGames)]
    plotLearning(x, scores, eps_history, filename)
