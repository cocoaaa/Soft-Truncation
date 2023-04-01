# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

from configs.default_lsun_configs import get_default_configs
import ml_collections
import os
import torch

def get_config():
  config = get_default_configs(  )
  # see: ^ this for default configs.
  
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.importance_sampling = False
  training.st = True
  training.k = 2.0
  training.likelihood_weighting = False
  training.truncation_time = 1e-5

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.probability_flow = False
  # sampling.probability_flow = True #--here: #[[Qs]]  will this make sampling stochastic? if not, need to set batch_size=1
  sampling.batch_size = 16
  sampling.sample_more = True # --???

  # defaults for sampling config -- in `default_lsun_configs.py -- > get_default_configs`
  # sampling.n_steps_each = 1
  # sampling.noise_removal = True
  # sampling.probability_flow = False
  # sampling.snr = 0.075
  # sampling.batch_size = 16
  # sampling.truncation_time = 1e-3

  # data
  data = config.data
  data.dataset = 'CelebAHQ'
  data.image_size = 256

  data.tfrecords_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))),
    'data/CelebAHQ_256/celeba_r08.tfrecords')

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.ckpt_dir = 'ckpts' #  e.g.: $exp_dir/"checkpoints"
  # model.ckpt_meta_fp = 'checkpoints-meta/checkpoint.pth' # e.g.: $exp_dir/"checkpoints-meta/checkpoint.pth"
  model.ckpt_fp = '/data/hayley-old/Github/Scores/Soft-Truncation/ckpts/CelebAHQ256/uncsnpp_rve_st/checkpoint.pth'
  
  model.sigma_max = 348
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.fourier_feature = False

  # evaluation -- from "default_celeba_configs.py"
  config.eval = evaluate = ml_collections.ConfigDict()
  # evaluate.begin_ckpt = 1
  # evaluate.end_ckpt = 26
  # evaluate.batch_size = 4 #--> use sampling.batch_size instead
  evaluate.enable_sampling = True # -- this calls my sample_save pipeline
  
  evaluate.num_samples = 100_000 # 50_000 # this is for fid eval;
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  
  
  
  
  # evaluate.bpd_dataset = 'test'
  # evaluate.num_test_data = 19962
  # evaluate.residual = True
  # evaluate.lambda_ = 0.0
  # evaluate.probability_flow = True
  # evaluate.nelbo_iter = 0
  # evaluate.nll_iter = 0
  
  
  # config.seed = 42
  # config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  # print('device: ', config.device)
  return config
