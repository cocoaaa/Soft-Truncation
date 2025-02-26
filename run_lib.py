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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import sampling_lib
import datasets
# import evaluation
import sde_lib
from absl import flags
import torch
import utils
import losses

FLAGS = flags.FLAGS


def train(config, workdir, assetdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)

  # Setup SDEs
  sde = sde_lib.get_sde(config, None)

  # Initialize model.
  state = utils.load_model(config, workdir, sde=sde)
  score_model = state['score_model']
  ema = state['ema']
  initial_step = int(state['step'])

  # Build data iterators
  logging.info(f'loading {config.data.dataset}...')
  train_ds, eval_ds = datasets.get_dataset(config)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Build one-step loss functions
  train_step_fn, nll_fn, nelbo_fn, sampling_fn = utils.get_loss_fns(config, sde, inverse_scaler)

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  for step in range(initial_step, config.training.n_iters + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch, train_iter = datasets.get_batch(config, train_iter, train_ds)
    if config.data.dequantization == 'uniform':
      batch = (255. * batch + torch.rand_like(batch)) / 256.
    batch = scaler(batch)
    # Execute one training step
    losses_ = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training loss mean: %.5e, training loss std: %.5e" % (step, torch.mean(losses_).item(), torch.std(losses_).item()))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      utils.save_checkpoint(config, checkpoint_meta_dir, state)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == config.training.n_iters:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      utils.save_checkpoint(config, os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

    if step != 0 and step % config.training.snapshot_freq == 0:
      if config.eval.enable_bpd:
        torch.cuda.empty_cache()
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        evaluation.compute_bpd(config, eval_ds, scaler, inverse_scaler, nelbo_fn, nll_fn, score_model, state, step=step)
        ema.restore(score_model.parameters())
        torch.cuda.empty_cache()

    if step != 0 and step % config.training.snapshot_freq == 0 or step == config.training.n_iters or config.training.whatever_sampling:
      # Generate and save samples
      if config.training.snapshot_sampling:
        logging.info('sampling start ...')
        torch.cuda.empty_cache()
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        for _ in range((config.eval.num_samples - 1) // config.sampling.batch_size + 1):
          sampling_lib.get_samples(config, score_model, state, sampling_fn, step, np.random.randint(1000000), sample_dir)
        ema.restore(score_model.parameters())
        torch.cuda.empty_cache()
        logging.info('sampling end ... computing FID ...')
        evaluation.compute_fid_and_is(config, score_model, state, sampling_fn, step, sample_dir, assetdir, config.eval.num_samples)
        torch.cuda.empty_cache()

def evaluate(config,
             workdir,
             assetdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  sde = sde_lib.get_sde(config, None)

  # Initialize model.
  state = utils.load_model(config, workdir, sde=sde)
  score_model = state['score_model']
  ema = state['ema']
  logging.info(f'score model step: {int(state["step"])}')
  ema.copy_to(score_model.parameters())

  # Build one-step loss functions
  _, nll_fn, nelbo_fn, sampling_fn = utils.get_loss_fns(config, sde, inverse_scaler)

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds, eval_ds = datasets.get_dataset(config)

  if config.eval.enable_bpd:
    evaluation.compute_bpd(config, eval_ds, scaler, inverse_scaler, nelbo_fn, nll_fn, score_model, step=int(state['step']), eval=True)

  if config.eval.enable_sampling:
    sample_dir = os.path.join(workdir, "eval")
    step = int(state['step'])
    logging.info('sampling start ...')
    torch.cuda.empty_cache()
    ema.copy_to(score_model.parameters())
    if config.sampling.sample_more:
      for _ in range((config.eval.num_samples - 1) // config.sampling.batch_size + 1):
        sampling_lib.get_samples(config, score_model, state, sampling_fn, step, np.random.randint(1000000), sample_dir)
    ema.restore(score_model.parameters())
    torch.cuda.empty_cache()
    logging.info('sampling end ... computing FID ...')
    evaluation.compute_fid_and_is(config, score_model, state, sampling_fn, step, sample_dir, assetdir, config.eval.num_samples)

    
# cocoaaa: 20230314-19510
# @torch.no_grad
def sample_and_save(
    config: str, #contains eval.num_samples and eval.batch_size
    workdir: str,
    outdir_root: str #'./Generated-Images'
    ):
  """Sample from the model using the checkpoint in ckpt_fp = config.model.ckpt_fp,
  and save each generated image to a image file in `out_dir`

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    out_dir: Full dirpath to save generated images
  """
  run_id = utils.now2str('')
  
  out_dir = Path(outdir_root) / run_id
  if not out_dir.exists():
    out_dir.mkdir(parents=True)
    print("Created: ", out_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config) # here, identity function (ie no scaling)
  inverse_scaler = datasets.get_data_inverse_scaler(config) # inverse of the data scaling function; here also identity function (ie no scaling)

  # Setup SDEs
  sde = sde_lib.get_sde(config, None)  #vesde
  

  # Initialize model.
  state = utils.load_model(config, workdir, sde=sde)
  
  # debug:
  print('state keys: ', state.keys())
  
  
  score_model = state['model']
  ema = state['ema']
  logging.info(f'score model step: {int(state["step"])}')
  ema.copy_to(score_model.parameters())
  # Build one-step loss functions
  _, _, _, sampling_fn = utils.get_loss_fns(config, sde, inverse_scaler)

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  # train_ds, eval_ds = datasets.get_dataset(config)

  # if config.eval.enable_bpd:
  #   evaluation.compute_bpd(config, eval_ds, scaler, inverse_scaler, nelbo_fn, nll_fn, score_model, step=int(state['step']), eval=True)

  if config.eval.enable_sampling:
    step = int(state['step'])
    logging.info("Sampling from ckpt at step: ", step)
    torch.cuda.empty_cache()
    # ema.copy_to(score_model.parameters())
    
    n_samples = config.eval.num_samples
    batch_size = config.sampling.batch_size
    n_iters = int(np.ceil(n_samples / batch_size))
    
    # debug
    print('nsamples: ', n_samples)
    print('bs: ', batch_size)
    print('niters: ', n_iters)
     
    n_done = 0
    for iter in range(n_iters):
        np_samples = sampling_lib.my_get_samples(config, score_model, sampling_fn)
        
        utils.save_each_npimg(
            np_samples, 
            out_dir=out_dir,
            prefix=run_id,
            suffix_start_idx=n_done,
            is_pilimg=False,
          )
        n_done += len(np_samples)
        if n_done >= n_samples:
          break
          
    print('Done sampling and saving: ', n_done, " images to: ", out_dir)
    
    # ema.restore(score_model.parameters())
    torch.cuda.empty_cache()
    
    
    
    
# ---
# save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
# img_fp = output_dir / f'{img_id}.png'
# npimg = timg.cpu().numpy()
# npimg = np.transpose(npimg, (1, 2, 0))
# plt.imsave(img_fp, npimg)
# img_id += 1