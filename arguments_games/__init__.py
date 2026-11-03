#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

from arguments import ParamGroup


class OptimizationParamsMesh(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.vertices_lr = 0.0  # 0.00016
        self.alpha_lr = 0.001
        self.feature_lr = 0.005  # Increased from 0.0025 for better color fitting
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.random_background = False
        self.use_mesh = True
        self.lambda_dssim = 0.3  # Increased from 0.2 for better perceptual quality
        # densification params - optimized for mesh-guided
        self.percent_dense = 0.01
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 1000  # Later start for mesh-guided (was 500)
        self.densify_until_iter = 25000  # Extended for more densification (was 15000)
        self.densify_grad_threshold = 0.0005  # Higher threshold to prevent over-densification (was 0.0002)
        super().__init__(parser, "Optimization Parameters")
