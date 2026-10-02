"""Train a Soft Actor-Critic precoder for energy efficiency.
 Two reward modes:
  - 'energy_efficiency_dinkelbach_adaptive': rate - lambda * power, with lambda
    tracking the achieved rate/power ratio (Dinkelbach) via a running EMA.
  - 'sum_rate_only': rate-only baseline (no power term).
"""

import os
import gzip
import pickle
from pathlib import Path
from sys import path as sys_path
from datetime import datetime
from shutil import rmtree

project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

import numpy as np
import tensorflow as tf
from matplotlib.pyplot import show as plt_show

import src
from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.models.algorithms.soft_actor_critic import SoftActorCritic
from src.models.helpers.get_state_norm_factors import get_state_norm_factors
from src.data.calc_sum_rate import calc_sum_rate
from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
)
from src.utils.plot_sweep import plot_sweep
from src.utils.progress_printer import progress_printer
from src.utils.update_sim import update_sim

def train_sac_energy_effiency(config: 'src.config.config.Config') -> Path:
    """Train a Soft Actor-Critic precoder according to the config."""

    def progress_print(to_log: bool = False) -> None:
        progress = (
            (training_episode_id * training_steps_per_episode + training_step_id + 1)
            / (config.config_learner.training_episodes * training_steps_per_episode)
        )
        logger_arg = {'logger': logger} if to_log else {}
        progress_printer(progress=progress, real_time_start=real_time_start, **logger_arg)

    def compute_mmse_action_and_reward():
        """MMSE precoder and its reward on the current (erroneous-CSI) channel."""
        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mmse_args
        )
        reward_mmse = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=config.noise_power_watt,
        )
        return w_mmse, reward_mmse

    def save_model_checkpoint(extra):
        name = f'full_snap_energy_effiency_{extra:.3f}' if extra is not None else ''
        checkpoint_path = Path(
            config.trained_models_path, config.config_learner.training_name, 'base', name,
        )
        logger.info(f'Saved model checkpoint at mean reward {extra:.3f}')

        model_path = Path(checkpoint_path, 'model')
        rmtree(path=model_path, ignore_errors=True)
        model_path.mkdir(parents=True, exist_ok=True)
        sac.networks['policy'][0]['primary'].save_weights(Path(model_path, 'weights.weights.h5'))

        config.save(Path(checkpoint_path, 'config'))
        with gzip.open(Path(checkpoint_path, 'config', 'norm_dict.gzip'), 'wb') as file:
            pickle.dump(norm_dict, file)

        with open(Path(checkpoint_path, 'config', 'dinkelbach_lambda_ee.txt'), 'w') as file:
            file.write(f'{lambda_ee}\n')

        for high_score_prior_id, high_score_prior in enumerate(reversed(high_scores)):
            if high_score > 1.05 * high_score_prior or high_score_prior_id > 3:
                prior_checkpoint_path = Path(
                    config.trained_models_path, config.config_learner.training_name, 'base',
                    f'full_snap_energy_effiency_{high_score_prior:.3f}',
                )
                rmtree(path=prior_checkpoint_path, ignore_errors=True)
                high_scores.remove(high_score_prior)

        return checkpoint_path

    def save_results():
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'base')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, 'training_error_learned_full.gzip'), 'wb') as file:
            pickle.dump(metrics, file=file)

    logger = config.logger.getChild(__name__)
    config.config_learner.algorithm_args['network_args']['num_actions'] = (
        2 * config.sat_nr * config.sat_ant_nr * config.user_nr
    )

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)
    sac = SoftActorCritic(rng=config.rng, **config.config_learner.algorithm_args)

    norm_dict = get_state_norm_factors(config=config, satellite_manager=satellite_manager, user_manager=user_manager)
    logger.info('State normalization factors found')
    logger.info(norm_dict)

    training_steps_per_episode = config.config_learner.training_steps_per_episode

    metrics: dict = {
        'mean_reward_per_episode': -np.inf * np.ones(config.config_learner.training_episodes),
        'lambda_ee_per_episode': np.nan * np.ones(config.config_learner.training_episodes),
        'episode_mean_rate_dinkelbach': np.nan * np.ones(config.config_learner.training_episodes),
        'episode_mean_power_dinkelbach': np.nan * np.ones(config.config_learner.training_episodes),
    }
    high_score = -np.inf
    high_scores = []
    best_model_path = None

    dinkelbach_rate_ema = None
    dinkelbach_power_ema = None
    lambda_ee = 0.0
    dinkelbach_lambda_window_steps = getattr(config, 'dinkelbach_lambda_window_steps', training_steps_per_episode)
    dinkelbach_lambda_ema_alpha = 2.0 / (dinkelbach_lambda_window_steps + 1)

    dinkelbach_high_score_warmup_episodes = 30

    valid_reward_keys = ['energy_efficiency_dinkelbach_adaptive', 'sum_rate_only']

    real_time_start = datetime.now()
    step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}

    for training_episode_id in range(config.config_learner.training_episodes):

        episode_metrics: dict = {
            'reward_per_step': np.nan * np.ones(training_steps_per_episode),
            'mean_log_prob_density': np.nan * np.ones(training_steps_per_episode),
            'value_loss': np.nan * np.ones(training_steps_per_episode),
            'dinkelbach_rate_per_step': np.nan * np.ones(training_steps_per_episode),
            'dinkelbach_power_per_step': np.nan * np.ones(training_steps_per_episode),
        }

        update_sim(config, satellite_manager, user_manager)
        state_next = config.config_learner.get_state(
            config=config, user_manager=user_manager, satellite_manager=satellite_manager,
            norm_factors=norm_dict['norm_factors'], **config.config_learner.get_state_args
        )

        for training_step_id in range(training_steps_per_episode):
            simulation_step = training_episode_id * training_steps_per_episode + training_step_id

            state_current = state_next
            step_experience['state'] = state_current

            action = sac.get_action(state=state_current)
            step_experience['action'] = action

            w_precoder_vector = real_vector_to_half_complex_vector(action)
            w_precoder = w_precoder_vector.reshape((config.sat_nr * config.sat_ant_nr, config.user_nr))
            power_precoder = np.real(np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
            raw_power_precoder = power_precoder

            if power_precoder > config.power_constraint_watt:
                norm_factor = np.sqrt(config.power_constraint_watt / power_precoder)
                w_precoder = norm_factor * w_precoder

            reward = 0

            if 'energy_efficiency_dinkelbach_adaptive' in config.config_learner.reward:
                sum_rate_reward_dinkelbach = calc_sum_rate(
                    channel_state=satellite_manager.channel_state_information,
                    w_precoder=w_precoder,
                    noise_power_watt=config.noise_power_watt,
                )

                transmit_power_dinkelbach = raw_power_precoder / config.pa_efficiency
                circuit_power_dinkelbach = config.sat_nr * config.sat_ant_nr * config.circuit_power_watt
                total_power_dinkelbach = (transmit_power_dinkelbach + circuit_power_dinkelbach) / config.power_constraint_watt

                energy_efficiency = sum_rate_reward_dinkelbach - lambda_ee * total_power_dinkelbach
                reward += config.config_learner.reward['energy_efficiency_dinkelbach_adaptive'] * energy_efficiency

                episode_metrics['dinkelbach_rate_per_step'][training_step_id] = sum_rate_reward_dinkelbach
                episode_metrics['dinkelbach_power_per_step'][training_step_id] = total_power_dinkelbach

                if dinkelbach_rate_ema is None:
                    dinkelbach_rate_ema = sum_rate_reward_dinkelbach
                    dinkelbach_power_ema = total_power_dinkelbach
                else:
                    dinkelbach_rate_ema += dinkelbach_lambda_ema_alpha * (sum_rate_reward_dinkelbach - dinkelbach_rate_ema)
                    dinkelbach_power_ema += dinkelbach_lambda_ema_alpha * (total_power_dinkelbach - dinkelbach_power_ema)
                if dinkelbach_power_ema > 1e-9:
                    lambda_ee = dinkelbach_rate_ema / dinkelbach_power_ema

            if 'sum_rate_only' in config.config_learner.reward:
                sum_rate_reward_only = calc_sum_rate(
                    channel_state=satellite_manager.channel_state_information,
                    w_precoder=w_precoder,
                    noise_power_watt=config.noise_power_watt,
                )
                reward += config.config_learner.reward['sum_rate_only'] * sum_rate_reward_only

            if any(key not in valid_reward_keys for key in config.config_learner.reward.keys()):
                raise ValueError("No valid reward provided")

            if not np.isfinite(reward):
                raise RuntimeError(
                    f'Non-finite reward ({reward}) at episode {training_episode_id}, '
                    f'step {training_step_id} -- training diverged (lambda_ee={lambda_ee}).'
                )

            step_experience['reward'] = reward

            add_mmse_sample = config.rng.random() < config.config_learner.percentage_mmse_samples_added_to_exp_buffer
            if add_mmse_sample:
                w_mmse, reward_mmse = compute_mmse_action_and_reward()

            update_sim(config, satellite_manager, user_manager)
            state_next = config.config_learner.get_state(
                config=config, user_manager=user_manager, satellite_manager=satellite_manager,
                norm_factors=norm_dict['norm_factors'], **config.config_learner.get_state_args
            )
            step_experience['next_state'] = state_next

            sac.add_experience(experience=step_experience)

            if add_mmse_sample and (
                reward_mmse > reward or not config.config_learner.only_add_mmse_samples_with_greater_reward
            ):
                sac.add_experience(experience={
                    'state': state_current,
                    'action': complex_vector_to_double_real_vector(w_mmse.flatten()),
                    'reward': reward_mmse,
                    'next_state': state_next,
                })

            train_policy = config.config_learner.policy_training_criterion(simulation_step=simulation_step)
            train_value = config.config_learner.value_training_criterion(simulation_step=simulation_step)
            if train_value or train_policy:
                mean_log_prob_density, value_loss = sac.train(
                    toggle_train_value_networks=train_value,
                    toggle_train_policy_network=train_policy,
                    toggle_train_entropy_scale_alpha=True,
                )
            else:
                mean_log_prob_density = np.nan
                value_loss = np.nan

            episode_metrics['reward_per_step'][training_step_id] = reward
            episode_metrics['mean_log_prob_density'][training_step_id] = mean_log_prob_density
            episode_metrics['value_loss'][training_step_id] = value_loss

            if config.verbosity > 0 and training_step_id % 50 == 0:
                progress_print()

        episode_mean_reward = np.nanmean(episode_metrics['reward_per_step'])
        metrics['mean_reward_per_episode'][training_episode_id] = episode_mean_reward

        if config.verbosity > 0:
            print('\r', end='')
        progress_print(to_log=True)
        current_entropy_scale_alpha = float(tf.exp(sac.log_entropy_scale_alpha).numpy())
        logger.info(
            f'Episode {training_episode_id}:'
            f' Episode mean reward: {episode_mean_reward:.4f}'
            f' std {np.nanstd(episode_metrics["reward_per_step"]):.2f},'
            f' current exploration: {np.nanmean(episode_metrics["mean_log_prob_density"]):.2f},'
            f' value loss: {np.nanmean(episode_metrics["value_loss"]):.5f},'
            f' entropy_scale_alpha: {current_entropy_scale_alpha:.4f}'
        )

        checkpoint_score = episode_mean_reward
        if 'energy_efficiency_dinkelbach_adaptive' in config.config_learner.reward:
            episode_mean_rate = np.nanmean(episode_metrics['dinkelbach_rate_per_step'])
            episode_mean_power = np.nanmean(episode_metrics['dinkelbach_power_per_step'])
            logger.info(
                f'Dinkelbach lambda updated to {lambda_ee:.4f} '
                f'(episode mean rate {episode_mean_rate:.4f}, '
                f'mean power {episode_mean_power:.4f} x budget '
                f'= {episode_mean_power * config.power_constraint_watt:.2f} W DC draw)'
            )
            checkpoint_score = episode_mean_rate / episode_mean_power if episode_mean_power > 1e-9 else -np.inf
            metrics['lambda_ee_per_episode'][training_episode_id] = lambda_ee
            metrics['episode_mean_rate_dinkelbach'][training_episode_id] = episode_mean_rate
            metrics['episode_mean_power_dinkelbach'][training_episode_id] = episode_mean_power

        is_past_dinkelbach_warmup = (
            'energy_efficiency_dinkelbach_adaptive' not in config.config_learner.reward
            or training_episode_id >= dinkelbach_high_score_warmup_episodes
        )
        if is_past_dinkelbach_warmup and checkpoint_score > high_score:
            high_score = checkpoint_score
            high_scores.append(high_score)
            best_model_path = save_model_checkpoint(checkpoint_score)

    save_results()

    if config.show_plots:
        plot_sweep(range(config.config_learner.training_episodes), metrics['mean_reward_per_episode'],
                   'Training Episode', 'Reward')
        plt_show()

    return best_model_path, metrics

if __name__ == '__main__':
    cfg = Config()

    reward_mode = os.environ.get('EE_REWARD_MODE', 'energy_efficiency_dinkelbach_adaptive')

    error_bound = float(os.environ.get('EE_TRAIN_ERROR_BOUND', 0.0))
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -error_bound
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = error_bound
    error_suffix = f'_aod{error_bound}' if error_bound != 0.0 else ''

    cfg.dinkelbach_lambda_window_steps = int(os.environ.get('EE_DINKELBACH_LAMBDA_WINDOW_STEPS', 5000))

    if reward_mode == 'energy_efficiency_dinkelbach_adaptive':
        cfg.config_learner.reward = {'energy_efficiency_dinkelbach_adaptive': 1.0}
        cfg.config_learner.training_name = f'EE_dinkelbach_adaptive{error_suffix}'
    elif reward_mode == 'sum_rate_only':
        cfg.config_learner.reward = {'sum_rate_only': 1.0}
        cfg.config_learner.training_name = f'SAC_rateonly{error_suffix}'
    else:
        raise ValueError(f'Unknown EE_REWARD_MODE: {reward_mode!r}')

    train_sac_energy_effiency(config=cfg)
