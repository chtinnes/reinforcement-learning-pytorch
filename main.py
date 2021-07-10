import os

import gym
import numpy as np
import torch
from models import ActorNetwork, CriticNetwork, train_networks
from os import path, makedirs

gamma = 0.99
lmbda = 0.95
shaped_reward_alpha = 0.9


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    returns = torch.FloatTensor(returns)
    adv = returns - values[:-1]
    return returns, (adv - torch.mean(adv)) / (torch.std(adv) + 1e-10)


def test_reward(render=False):
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    model_actor.eval()
    limit = 0
    with torch.no_grad():
        while not done:
            # predict
            action_dist = model_actor([state])
            if render:
                env.render()
            # choose max action
            action = torch.argmax(action_dist[0, :]).item()
            observation, reward, done, info = env.step(action)

            state = observation
            total_reward += reward
            limit += 1
#            if limit > 50:
#                print("Eval hit limit!")
#                break'
    return total_reward


ENVIRONMENT = "MountainCar-v0"
ACTOR_PATH = "models/best_actor%s.tf" % ENVIRONMENT
CRITIC_PATH = "models/best_critic%s.tf" % ENVIRONMENT
env = gym.make(ENVIRONMENT)
#env._max_episode_steps = 5000
state_dims = env.observation_space.shape
print(state_dims)
n_actions = env.action_space.n
print(n_actions)
ppo_steps = 800
max_steps = env._max_episode_steps

target_reached = False
best_reward = -200
iters = 0
max_iters = 1000


if path.exists(ACTOR_PATH):
    model_actor = torch.load(ACTOR_PATH)
else:
    model_actor = ActorNetwork(state_dims[0], n_actions, 5)

if path.exists(CRITIC_PATH):
    model_critic = torch.load(CRITIC_PATH)
else:
    model_critic = CriticNetwork(state_dims[0], 5)

observation = env.reset()


def shape_reward(observation, reward, agent_reached_goal, max_steps, current_steps):
    # Maximize total average energy ;-)
    energy_reward = np.sin(3 * observation[0]) * .45 + .55 + 200 * observation[1] * observation[1]

    if agent_reached_goal:
        # the anti-suicide reward
        # add discounted rest livetime average award in order to not learn it's not good to reach to goal. (no one wants to die if one feels good)
        N = max_steps-current_steps
        reward = energy_reward * (1-np.power(gamma, N))/(1-gamma)
        print("Reached goal. Rest livetime reward: " + str(reward))
    else:
        reward = energy_reward
    #return (1-shaped_reward_alpha) * reward + shaped_reward_alpha * (np.sin(3 * observation[0]) * .45 + .55 + 200*observation[1]*observation[1])

    return reward

def goal_reached(observation):
    return observation[0] > 0.5

while not target_reached and iters < max_iters:
    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None

    # Get batches (memory)
    with torch.no_grad():
        # Set to eval only
        model_actor.eval()
        model_critic.eval()
        current_steps = 0
        for i_episode in range(ppo_steps):
            state = observation
            if iters % 100 == 0:
                env.render()
            # print(observation)

            # predict
            action_dist = model_actor([state])
            value = model_critic([state])

            # choose action
            action = torch.multinomial(action_dist[0, :], 1)[0].item()
            action_prob = action_dist[0][action]
            observation, reward, done, info = env.step(action)
            current_steps += 1
            agent_reached_goal = goal_reached(observation)

            reward = shape_reward(observation, reward, agent_reached_goal, max_steps, current_steps)
            mask = not done

            states.append(state)
            actions.append(action)
            values.append(value[0])
            masks.append(mask)
            rewards.append(reward)
            actions_probs.append(action_prob)

            if done:
                #print("Episode finished after {} timesteps".format(t + 1))
                env.reset()
                current_steps = 0

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        masks = torch.FloatTensor(masks)
        rewards = torch.FloatTensor(rewards)
        actions_probs = torch.stack(actions_probs)

        # now, we always use one more (n+1) q values to compute the returns and advantages
        values.append(model_critic([observation])[0])
        values = torch.FloatTensor(values)

        returns, advantages = get_advantages(values, masks, rewards)

    # train networks
    train_networks(model_actor, model_critic, states, actions_probs, advantages, rewards, values[:-1], returns, actions, batch_size=64, epochs=32)

    # eval actor
    avg_reward = np.mean([test_reward() for _ in range(2)])
    print('total test reward=' + str(avg_reward))
    if avg_reward > best_reward:
        print('best reward=' + str(avg_reward))

        # Save models
        if not path.exists('models'):
            makedirs('models')
        torch.save(model_actor, ACTOR_PATH)
        torch.save(model_critic, CRITIC_PATH)

        best_reward = avg_reward
        test_reward(render=True)

    if best_reward > 1000 or iters > max_iters:
        target_reached = True
        # reset env and got into next iteration
    env.reset()
    iters += 1

env.close()

