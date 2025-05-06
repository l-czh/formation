import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.9'

import jax
import wandb
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import optax
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, NamedTuple, Tuple, Optional, Union, Any, Dict
from flax.training.train_state import TrainState
import distrax
import tensorboardX
import jax.experimental
from envs.wrappers import LogWrapper
from envs.aeroplanax_heading_pitch_V import AeroPlanaxHeading_Pitch_V_Env, Heading_Pitch_V_TaskParams
import orbax.checkpoint as ocp


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and not f.endswith('.orbax-checkpoint-tmp-0')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1]))
    return os.path.abspath(os.path.join(checkpoint_dir, latest_checkpoint))


def save_checkpoint(train_state, epoch, save_dir):
    """保存checkpoint"""
    # 将参数转移到CPU
    params = jax.device_put(train_state.params, jax.devices('cpu')[0])
    opt_state = jax.device_put(train_state.opt_state, jax.devices('cpu')[0])
    
    checkpoint = {
        "params": params,
        "opt_state": opt_state,
        "epoch": jnp.array(epoch)
    }
    checkpoint_path = os.path.abspath(os.path.join(save_dir, f"checkpoint_epoch_{epoch}"))
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint))
    ckptr.wait_until_finished()
    print(f"Checkpoint saved at epoch {epoch}")


def make_train(config):
    env_params = Heading_Pitch_V_TaskParams()
    env = AeroPlanaxHeading_Pitch_V_Env(env_params)
    env = LogWrapper(env)
    config["NUM_ACTORS"] = env.num_agents
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(4, config=config)  # 4 actions: roll, pitch, yaw, throttle

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"] * config["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # 尝试加载最新的checkpoint
        latest_checkpoint = find_latest_checkpoint(config["SAVEDIR"])
        if latest_checkpoint:
            print(f"Loading latest checkpoint from {latest_checkpoint}")
            ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
            checkpoint = ckptr.restore(latest_checkpoint)
            
            # 将参数转移到当前设备
            params = jax.device_put(checkpoint["params"], jax.devices()[0])
            opt_state = jax.device_put(checkpoint["opt_state"], jax.devices()[0])
            
            train_state = train_state.replace(
                params=params,
                opt_state=opt_state
            )
            start_epoch = int(checkpoint["epoch"])
            print(f"Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0
            print("No checkpoint found, starting new training")

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        # INIT Tensorboard
        if config.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(config["LOGDIR"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (
                    last_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, 
                  unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))
                reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
                done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (
                last_obs[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(0),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                batch = (
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                )
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            # adding an additional "fake" dimensionality to perform minibatching correctly
            initial_hstate = initial_hstate[None, :]
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }

            rng = update_state[-1]
            metric["update_steps"] = update_steps
            if config.get("DEBUG"):
                def callback(metric):
                    env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                    for k, v in metric["loss"].items():
                        writer.add_scalar('loss/{}'.format(k), v, env_steps)
                    writer.add_scalar('eval/episodic_return', metric["returned_episode_returns"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/episodic_length', metric["returned_episode_lengths"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/success_times', metric["heading_turn_counts"][metric["returned_episode"].squeeze()].mean(), env_steps)
                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessTimes={:.3f}".format(
                        metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
                        metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                        metric["heading_turn_counts"][metric["returned_episode"].squeeze()].mean(),
                    ))
                jax.experimental.io_callback(callback, None, metric)

            # 保存checkpoint
            def save_checkpoint_callback(update_steps, train_state):
                if update_steps % config["CHECKPOINT_FREQ"] == 0:
                    save_checkpoint(train_state, update_steps, config["SAVEDIR"])
                return None

            jax.experimental.io_callback(save_checkpoint_callback, None, update_steps, train_state)

            update_steps = update_steps + 1    
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
            jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )

        # 计算总更新次数
        total_updates = config["NUM_UPDATES"] - start_epoch
        print(f"Starting training with {total_updates} updates from epoch {start_epoch}")
        
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, start_epoch), None, total_updates
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def main():
    # 训练配置
    config = {
        "GROUP": "heading_pitch_V",
        "SEED": 42,
        "LR": 3e-4,
        "NUM_ENVS": 20000,
        "NUM_ACTORS": 1,
        "NUM_STEPS": 200,
        "TOTAL_TIMESTEPS": 1.2e7,
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "UPDATE_EPOCHS": 16,
        "NUM_MINIBATCHES": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 1e-3,
        "VF_COEF": 1,
        "MAX_GRAD_NORM": 2,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,
        "DEBUG": True,
        "CHECKPOINT_FREQ": 100,  # 每100个epoch保存一次checkpoint
        "OUTPUTDIR": os.path.abspath("results/heading_pitch_V"),  # 使用绝对路径
        "LOGDIR": os.path.abspath("results/heading_pitch_V/logs"),
        "SAVEDIR": os.path.abspath("results/heading_pitch_V/checkpoints"),
        "NUM_TRAINING_ITERATIONS": 5,  # 训练迭代次数
    }

    # 创建必要的目录
    Path(config["OUTPUTDIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["LOGDIR"]).mkdir(parents=True, exist_ok=True)

    # 初始化wandb
    seed = config['SEED']
    wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
    wandb.init(
        project="AeroPlanax",
        config=config,
        name=config['GROUP'] + f'_agent{config["NUM_ACTORS"]}_seed_{seed}',
        group=config['GROUP'],
        notes='continuous training with automatic checkpoint',
        reinit=False,
    )

    # 训练循环
    for iteration in range(config["NUM_TRAINING_ITERATIONS"]):
        print(f"\nStarting training iteration {iteration + 1}/{config['NUM_TRAINING_ITERATIONS']}")
        
        # 查找最新的checkpoint
        latest_checkpoint = find_latest_checkpoint(config["SAVEDIR"])
        if latest_checkpoint:
            print(f"Loading latest checkpoint from {latest_checkpoint}")
            config["LOADDIR"] = latest_checkpoint
        else:
            print("No checkpoint found, starting new training")
            config["LOADDIR"] = None

        try:
            # 运行训练
            rng = jax.random.PRNGKey(seed)
            train_jit = jax.jit(make_train(config))
            out = train_jit(rng)
            
            # 保存训练结果图表，使用迭代次数作为文件名的一部分
            plt.plot(out["metric"]["returned_episode_returns"].mean(-1).reshape(-1))
            plt.xlabel("Update Step")
            plt.ylabel("Return")
            plt.savefig(os.path.join(config["OUTPUTDIR"], f'returned_episode_returns_iter_{iteration}.png'))
            plt.cla()
            
            plt.plot(out["metric"]["returned_episode_lengths"].mean(-1).reshape(-1))
            plt.xlabel("Update Step")
            plt.ylabel("Episode Length")
            plt.savefig(os.path.join(config["OUTPUTDIR"], f'returned_episode_lengths_iter_{iteration}.png'))
            plt.cla()
            
            # 更新随机种子
            config["SEED"] = config["SEED"] + 1
            print(f"Training iteration {iteration + 1} completed. Starting next iteration with seed {config['SEED']}")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            break
        except Exception as e:
            print(f"Training failed with error: {e}")
            print("Continuing to next iteration...")
            continue

    wandb.finish()


if __name__ == "__main__":
    main() 