import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from sys import stdout
from datetime import datetime

import numpy as np
from scipy import constants
from tensorflow import get_logger as tf_get_logger
from tensorflow.config import list_physical_devices

from src.config.config_error_model import (
    ConfigErrorModel,
)
from src.config.config_sac_learner import (
    ConfigSACLearner,
)
from src.data.channel.los_channel_model import (
    los_channel_model,
)
from src.utils.get_wavelength import (
    get_wavelength,
)
from src.utils.compare_configs import (
    compare_configs,
)
from src.utils.format_value import (
    format_value,
)

class Config:
    """The config sets up all global parameters."""

    def __init__(
            self,
    ) -> None:

        self._pre_init()

        self.profile: bool = False
        self.show_plots: bool = True

        self.verbosity: int = 1
        self._logging_level_stdio = logging.INFO
        self._logging_level_file = logging.DEBUG
        self._logging_level_tensorflow = logging.WARNING
        self._logging_level_matplotlib = logging.INFO

        self.logfile_max_bytes: int = 10_000_000

        self.freq: float = 2 * 10**9
        self.noise_power_watt: float = 10**(7 / 10) * 290 * constants.value('Boltzmann constant') * 30 * 10**6

        self.power_constraint_watt = float(os.environ.get('EE_POWER_BUDGET_WATT', 75))
        self.circuit_power_watt = 1.0

        self.pa_efficiency = 0.60

        self.wavelength: float = get_wavelength(self.freq)

        self.altitude_orbit: float = 600 * 10**3
        self.radius_earth: float = 6378.1 * 10**3

        self.radius_orbit: float = self.altitude_orbit + self.radius_earth

        self.user_nr: int = int(os.environ.get('EE_USER_NR', 3))
        self.user_gain_dBi: float = 0
        self.user_dist_average: float = 100_000
        self.user_dist_bound: float = 0.5
        self.user_center_aod_earth_deg: float = 90

        _target_elevation_deg = os.environ.get('EE_TARGET_ELEVATION_DEG')
        if _target_elevation_deg is not None:
            _elevation_rad = np.deg2rad(float(_target_elevation_deg))
            _earth_central_angle_rad = (
                np.pi / 2 - _elevation_rad
                - np.arcsin(self.radius_earth / self.radius_orbit * np.cos(_elevation_rad))
            )
            self.user_center_aod_earth_deg -= float(np.rad2deg(_earth_central_angle_rad))
        self.user_area = self.user_dist_average
        self.user_activity_selection: str = 'all_active_area_flexible'

        self.user_distribution_mode: str = 'uniform'
        self.beta_a: float = 0.5
        self.beta_b: float = 0.5
        self.weighting_factor_beta_distribution: float = 0.5

        self.user_gain_linear: float = 10**(self.user_gain_dBi / 10)

        self.sat_nr: int = 1

        self.sat_tot_ant_nr: int = int(os.environ.get('EE_SAT_TOT_ANT_NR', 16))

        self.sat_gain_dBi: float = float(os.environ.get('EE_SAT_GAIN_DBI', 30))
        self.sat_dist_average: float = 100_000
        self.sat_dist_bound: float = 0
        self.sat_center_aod_earth_deg: float = 90

        self.sat_gain_linear: float = 10**(self.sat_gain_dBi / 10)
        self.sat_ant_nr: int = int(self.sat_tot_ant_nr / self.sat_nr)
        self.sat_ant_gain_linear: float = self.sat_gain_linear / self.sat_tot_ant_nr
        self.sat_ant_dist: float = 3 * self.wavelength / 2

        self.channel_model = los_channel_model

        self.csi_error_scale = 2
        self.local_csi_own_quality = 'error_free'
        self.local_csi_others_quality = 'erroneous'

        self.common_part_precoding_style='basic'

        self._post_init()

    def _pre_init(
            self,
    ) -> None:

        if self.__class__.__module__ == 'src.config.config':
            self._inert = False
        else:
            self._inert = True

        self.rng = np.random.default_rng()
        self.logger = logging.getLogger(datetime.now().strftime('%H-%M-%S'))

        self.project_root_path = Path(__file__).parent.parent.parent
        self.performance_profile_path = Path(self.project_root_path, 'outputs', 'performance_profiles')
        self.output_metrics_path = Path(self.project_root_path, 'outputs', 'metrics')
        self.trained_models_path = Path(self.project_root_path, 'models')
        self.logfile_path = Path(self.project_root_path, 'outputs', 'logs', 'log.txt')

        if not self._inert:
            self.performance_profile_path.mkdir(parents=True, exist_ok=True)
            self.output_metrics_path.mkdir(parents=True, exist_ok=True)
            self.trained_models_path.mkdir(parents=True, exist_ok=True)
            self.logfile_path.parent.mkdir(parents=True, exist_ok=True)

    def _post_init(
            self,
    ) -> None:

        if not self._inert:
            self.__logging_setup()

        self.config_error_model = ConfigErrorModel(
            channel_model=self.channel_model,
            rng=self.rng,
            wavelength=self.wavelength,
            user_nr=self.user_nr,
        )

        self.config_learner = ConfigSACLearner(
            sat_nr=self.sat_nr,
            sat_ant_nr=self.sat_ant_nr,
            user_nr=self.user_nr,
        )

        self.satellite_args: dict = {
            'rng': self.rng,
            'antenna_nr': self.sat_ant_nr,
            'antenna_distance': self.sat_ant_dist,
            'antenna_gain_linear': self.sat_ant_gain_linear,
            'user_nr': self.user_nr,
            'freq': self.freq,
            'center_aod_earth_deg': self.sat_center_aod_earth_deg,
            'error_functions': self.config_error_model.error_rngs
        }

        self.user_args: dict = {
            'gain_linear': self.user_gain_linear,
        }

        self.mmse_args: dict = {
            'power_constraint_watt': self.power_constraint_watt,
            'noise_power_watt': self.noise_power_watt,
            'sat_nr': self.sat_nr,
            'sat_ant_nr': self.sat_ant_nr,
        }

        self.mrc_args: dict = {
            'power_constraint_watt': self.power_constraint_watt,
        }

        self.learned_precoder_args: dict = {
            'sat_nr': self.sat_nr,
            'sat_ant_nr': self.sat_ant_nr,
            'user_nr': self.user_nr,
            'power_constraint_watt': self.power_constraint_watt,

            'action_format': 'real_imag',
        }

    def generate_name_from_config(
            self,
    ) -> str:
        """
        Generates a path for a specific config
        """

        config_name = None

        default_configs_path = Path(self.project_root_path, 'src', 'config', 'default_configs')
        for default_config_path in [subitem for subitem in default_configs_path.iterdir() if subitem.is_dir()]:
            if compare_configs(self, default_config_path, log_differences=False):
                self.logger.info(f'current config matches default config {default_config_path.stem}')
                config_name = default_config_path.stem
                break

        if config_name is None:
            config_name = (
                f'{self.sat_nr}sat_'
                f'{self.sat_tot_ant_nr}ant_'
                f'{format_value(self.sat_dist_average)}~'
                f'{format_value(self.sat_dist_bound)}_'
                f'{self.user_nr}usr_'
                f'{format_value(self.user_dist_average)}~'
                f'{format_value(self.user_dist_bound)}'
            )

        return config_name

    def save(
            self,
            path: Path,
    ) -> None:
        """
        Serialize config to json
        """

        path.mkdir(parents=True, exist_ok=True)
        with open(Path(path, 'config.json'), 'w') as file:
            json.dump(vars(self), file, indent=4, default=lambda o: f'{str(o)}')
        with open(Path(path, 'config_sac_learner.json'), 'w') as file:
            json.dump(vars(self.config_learner), file, indent=4, default=lambda o: f'{str(o)}')
        with open(Path(path, 'config_error_model.json'), 'w') as file:
            json.dump(vars(self.config_error_model), file, indent=4, default=lambda o: f'{str(o)}')

    def __logging_setup(
            self,
    ) -> None:

        logging_formatter = logging.Formatter(
            '{asctime} : {levelname:8s} : {name:30} : {funcName:25} :: {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{',
        )

        logging_file_handler = RotatingFileHandler(self.logfile_path, maxBytes=self.logfile_max_bytes, backupCount=1)
        logging_stdio_handler = logging.StreamHandler(stdout)

        logging_file_handler.setLevel(self._logging_level_file)

        if self.verbosity == 0:
            logging_stdio_handler.setLevel(logging.CRITICAL + 1)
        else:
            logging_stdio_handler.setLevel(self._logging_level_stdio)

        tensorflow_logger = tf_get_logger()
        tensorflow_logger.setLevel(self._logging_level_tensorflow)
        if len(tensorflow_logger.handlers) > 0:
            tensorflow_logger.handlers.pop(0)

        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(self._logging_level_matplotlib)

        self.logger.setLevel(logging.DEBUG-1)

        logging_file_handler.setFormatter(logging_formatter)
        logging_stdio_handler.setFormatter(logging_formatter)

        self.logger.addHandler(logging_file_handler)
        self.logger.addHandler(logging_stdio_handler)

        self.logger.propagate = False

        self.logger.info(f'GPUs detected: {list_physical_devices("GPU")}')
