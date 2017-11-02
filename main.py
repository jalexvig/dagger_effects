import os
import pickle

import gym
import numpy as np
import pandas as pd
import tensorflow as tf

import load_policy


def _get_env_policy(gym_envname):

    fp_policy = os.path.join('experts', gym_envname + '.pkl')
    policy_fn = load_policy.load_policy(fp_policy)

    env = gym.make(gym_envname)

    return env, policy_fn


def build_supervised_model(optimizer_name, env):

    dim_obs = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    inputs = tf.placeholder(tf.float32, shape=[None, dim_obs], name='observation')

    fc1 = tf.contrib.layers.fully_connected(inputs, 10, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    fc2 = tf.contrib.layers.fully_connected(fc1, dim_action, activation_fn=tf.tanh, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    outputs = tf.identity(fc2, name='outputs')

    targets = tf.placeholder(tf.float32, shape=[None, dim_action], name='actions')

    unreg_loss = tf.nn.l2_loss(targets - outputs, name='unreg_loss')

    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
    loss = tf.add(unreg_loss, reg_losses, name='loss')

    tf.summary.scalar('l2_reg_loss', loss)

    if optimizer_name == 'adadelta':
        opt = tf.train.AdadeltaOptimizer()
    elif optimizer_name == 'adam':
        opt = tf.train.AdamOptimizer()
    else:
        raise ValueError('Unknown optimizer name {}'.format(optimizer_name))

    update = opt.minimize(loss, name='update')

    return inputs, targets, outputs, loss, update


def load_supervised_model():

    graph = tf.get_default_graph()

    inputs = graph.get_tensor_by_name("observation:0")
    outputs = graph.get_tensor_by_name("outputs:0")
    targets = graph.get_tensor_by_name("actions:0")
    loss = graph.get_tensor_by_name("loss:0")
    update = graph.get_operation_by_name("update")

    return inputs, outputs, targets, loss, update


def train(num_epochs=10000,
          load=False,
          optimizer='adam',
          gym_envname='Hopper-v1',
          dagger_iters=0,
          num_rollouts=20,
          max_timesteps=500,
          num_observations_cap=None,
          rewards_freq=None):

    env, policy_fn = _get_env_policy(gym_envname)

    # TODO(jalex): Batch this

    with tf.Session() as sess:

        data = _get_data(policy_fn, env, num_rollouts=num_rollouts, max_timesteps=max_timesteps)

        if load:
            saver = _load_saver(gym_envname)
            inputs, targets, outputs, loss, update, *extra = load_supervised_model()
        else:
            inputs, targets, outputs, loss, update, *extra = build_supervised_model(optimizer, env)
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

        def bc_fn(obs):
            return outputs.eval(feed_dict={inputs: obs[None, :]})

        writer = tf.summary.FileWriter('tensorboard', sess.graph)

        merged_summary = tf.summary.merge_all()

        stats = []

        i = 0

        num_inputs = 0

        old_eval_idx = -1

        while i <= dagger_iters:

            observations, actions = map(np.array, zip(*data))

            for j in range(num_epochs):

                if num_observations_cap:
                    ub = num_observations_cap - num_inputs
                    if ub < observations.shape[0]:
                        if ub == 0:
                            print('Hit datapoint cap', num_observations_cap)
                            break
                        observations = observations[:ub]
                        actions = actions[:ub]

                l, _, summary = sess.run([loss, update, merged_summary], feed_dict={inputs: observations, targets: actions})

                num_inputs += observations.shape[0]

                writer.add_summary(summary, i * j + j)

                if j % 100 == 0:
                    print(i, j, l / observations.shape[0])

                if rewards_freq:
                    new_eval_idx = num_inputs // rewards_freq
                    if new_eval_idx > old_eval_idx:
                        stats.append(_get_reward_stats(env, outputs, inputs, max_timesteps, num_rollouts, name=num_inputs))
                        old_eval_idx = new_eval_idx

            if num_inputs >= num_observations_cap:
                break

            if i < dagger_iters:
                data += _get_data(policy_fn, env, action_fn=bc_fn, num_rollouts=num_rollouts, max_timesteps=max_timesteps)

            i += 1

        checkpoint_name = gym_envname + ('_dagger{}'.format(i) if dagger_iters > 0 else '')
        saver.save(sess, "./models/{}.ckpt".format(checkpoint_name))

        writer.close()

        res = [pd.concat(l, axis=1).T for l in zip(*stats)]

        return res


def _load_saver(sess, checkpoint_name):

    saver = tf.train.import_meta_graph('./models/{}.ckpt.meta'.format(checkpoint_name))
    saver.restore(sess, './models/{}.ckpt'.format(checkpoint_name))

    return saver


def _get_data(policy_fn, env, action_fn=None, num_rollouts=20, max_timesteps=1000, render=False):

    mc_states = []

    for i in range(num_rollouts):

        obs = env.reset()

        for j in range(max_timesteps):
            action = policy_fn(obs[None, :])
            mc_states.append((obs, action[0]))

            if action_fn:
                action = action_fn(obs)
            obs, r, done, debug = env.step(action)

            if done:
                break

            if render:
                env.render()

    return mc_states


def run_supervised_model(num_rollouts=1,
                         max_timesteps=1000,
                         gym_envname='Hopper-v1',
                         checkpoint=None,
                         render=False,
                         record=None):

    env, policy_fn = _get_env_policy(gym_envname)

    if record:
        env = gym.wrappers.Monitor(env, './videos', record)

    with tf.Session() as sess:

        _load_saver(sess, checkpoint or gym_envname)

        graph = tf.get_default_graph()

        inputs = graph.get_tensor_by_name("observation:0")
        outputs = graph.get_tensor_by_name("outputs:0")

        return _get_reward_stats(env, outputs, inputs, max_timesteps, num_rollouts, render)


def _get_reward_stats(env, outputs, inputs, max_timesteps, num_rollouts, render=False, name=None):

    steps, rewards = [], []

    for i in range(num_rollouts):

        obs = env.reset()

        reward_rollout = 0

        for j in range(max_timesteps):
            action_bc = outputs.eval(feed_dict={inputs: obs[None, :]})
            obs, r, done, debug = env.step(action_bc)

            reward_rollout += r

            if done:
                # print(j, done)
                break

            if render:
                env.render()

        steps.append(j)
        rewards.append(reward_rollout)

    s = pd.Series(steps, name=name)
    r = pd.Series(rewards, name=name)

    return s, r


if __name__ == '__main__':

    env = 'Hopper-v1'

    kwargs = {
        'num_epochs': 10000,
        'num_rollouts': 100,
        'max_timesteps': 1000,
        'gym_envname': env,
        'num_observations_cap': int(1e8),
        'rewards_freq': 500000,
    }

    # Train with original data
    stats = train(**kwargs)

    with open('models/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    kwargs = kwargs.copy()
    kwargs['num_epochs'] = 100
    kwargs['num_rollouts'] = 100
    kwargs['dagger_iters'] = np.inf

    # Train with dagger augmented data
    stats = train(**kwargs)

    with open('models/stats_dagger.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # Video record performance
    run_supervised_model(1, checkpoint='Hopper-v1_dagger24', render=True, record=lambda episode_id: True)
    run_supervised_model(1, checkpoint='Hopper-v1', render=True, record=lambda episode_id: True)
