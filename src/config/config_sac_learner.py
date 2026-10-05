import tensorflow as tf

from src.models.helpers.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)

class ConfigSACLearner:
    """Defines parameters to use when learning with Soft Actor Critic."""

    def __init__(
            self,
            sat_nr,
            sat_ant_nr,
            user_nr,
    ) -> None:

        self.training_name: str = 'EE_dinkelbach_adaptive'

        self.reward: dict = {
            'energy_efficiency_dinkelbach_adaptive': 1.0,
        }

        self.get_state = get_state_erroneous_channel_state_information
        self.get_state_args = {
            'csi_format': 'rad_phase',
            'norm_state': True,
        }
        self.get_state_norm_factors_iterations: int = 100_000

        self.percentage_mmse_samples_added_to_exp_buffer: float = 0.0
        self.only_add_mmse_samples_with_greater_reward: bool = True

        self.action_bound_mode: str or None = None
        self.training_args: dict = {
            'future_reward_discount_gamma': 0.0,
            'entropy_scale_alpha_initial': 1.0,
            'target_entropy': 1.0,
            'entropy_scale_optimizer': tf.keras.optimizers.SGD,
            'entropy_scale_optimizer_args': {
                'learning_rate': 1e-3,
            },
            'training_minimum_experiences': 1_000,
            'training_batch_size': 1024,
            'training_target_update_momentum_tau': 0,
            'training_l2_norm_scale_value': 0.01,
            'training_l2_norm_scale_policy': 0.01,
        }
        self.experience_buffer_args: dict = {
            'buffer_size': 100_000,
            'priority_scale_alpha': 0.0,
            'importance_sampling_correction_beta': 1.0
        }
        self.train_policy_every_k_steps: int = 10
        self.train_policy_after_j_steps: int = 0
        self.train_value_every_k_steps: int = 10
        self.train_value_after_j_steps: int = 0
        self.network_args: dict = {
            'value_network_args': {
                'hidden_layer_units': [512, 512, 512, 512, ],
                'activation_hidden': 'leaky_relu',
                'kernel_initializer_hidden': 'glorot_uniform',
                'batch_norm_input': False,
                'batch_norm': True,
            },
            'value_network_optimizer': tf.keras.optimizers.Adam,
            'value_network_optimizer_args': {

                'learning_rate': 8.8e-6,

                'amsgrad': False,
            },
            'policy_network_args': {
                'hidden_layer_units': [512, 512, 512, 512, ],
                'activation_hidden': 'penalized_tanh',
                'kernel_initializer_hidden': 'glorot_uniform',
                'batch_norm_input': False,
                'batch_norm': True,
            },
            'policy_network_optimizer': tf.keras.optimizers.Adam,
            'policy_network_optimizer_args': {

                 'learning_rate': 4.2e-5,

                'amsgrad': True,
            },
        }

        self.training_episodes: int = 13_000
        self.training_steps_per_episode: int = 1_000

        self.num_parallel_envs: int = 64

        self._post_init(sat_nr=sat_nr, sat_ant_nr=sat_ant_nr, user_nr=user_nr)

    def _post_init(
            self,
            sat_nr,
            sat_ant_nr,
            user_nr,
    ) -> None:

        self.training_args['training_minimum_experiences'] = max(self.training_args['training_minimum_experiences'],
                                                                 self.training_args['training_batch_size'])

        self.update(sat_nr=sat_nr, sat_ant_nr=sat_ant_nr, user_nr=user_nr)

        self.algorithm_args = {
            **self.training_args,
            'network_args': self.network_args,
            'experience_buffer_args': self.experience_buffer_args,
            'action_bound_mode': self.action_bound_mode,
        }

    def update(
            self,
            sat_nr: int,
            sat_ant_nr: int,
            user_nr: int,
    ) -> None:
        """Update those config parameters that are calculated from others."""

        if self.get_state == get_state_aods:
            self.network_args['size_state'] = sat_nr * user_nr
        elif self.get_state == get_state_erroneous_channel_state_information:
            if self.get_state_args['csi_format'] in ['rad_phase', 'real_imag']:
                self.network_args['size_state'] = 2 * sat_nr * sat_ant_nr * user_nr
            elif self.get_state_args['csi_format'] == 'rad_phase_reduced':
                self.network_args['size_state'] = sat_nr * sat_ant_nr * user_nr + sat_nr * user_nr

    def policy_training_criterion(
            self,
            simulation_step
    ) -> bool:
        """Train policy networks only every k steps and/or only after j total steps to ensure a good value function"""
        if (
                simulation_step > self.train_policy_after_j_steps
                and
                (simulation_step % self.train_policy_every_k_steps) == 0
        ):
            return True
        return False

    def value_training_criterion(
            self,
            simulation_step
    ) -> bool:
        """Train value networks only every k steps and/or only after j total steps"""
        if (
                simulation_step > self.train_value_after_j_steps
                and
                (simulation_step % self.train_value_every_k_steps) == 0
        ):
            return True
        return False
