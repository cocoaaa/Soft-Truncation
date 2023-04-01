from PIL import Image
from pathlib import Path
from typing import Dict, Iterable, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import tensorflow as tf
import os
import logging
import numpy as np
import torch.distributed as dist

from models import utils as mutils
from models.ema import ExponentialMovingAverage
import losses
import likelihood, sampling


def restore_checkpoint(ckpt_dir: str, 
                       state: Dict, 
                       device:Union[str, torch.device]):
  
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    logging.info(ckpt_dir + ' loaded ...')
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(config, ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)

def create_name(prefix, name, ext):
  try:
    name = f'{prefix}_{int(name)}.{ext}'
  except:
    if len(name.split('.')) == 1:
      name = f'{prefix}_{name}.{ext}'
    else:
      name = name.split('/')[-1]
      name = f'{prefix}_{name.split(".")[0]}.{ext}'
  return name

def load_model(config, workdir, print_=True, sde=None):
  # Initialize model.
  score_model = mutils.create_model(config, sde)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, 
               model=score_model,
               ema=ema, step=0)

  if print_:
    # print(score_model)
    model_parameters = filter(lambda p: p.requires_grad, score_model.parameters())
    model_params = sum([np.prod(p.size()) for p in model_parameters])
    total_num_params = sum([np.prod(p.size()) for p in score_model.parameters()])
    logging.info(f"model parameters: {model_params}")
    logging.info(f"total number of parameters: {total_num_params}")

  # Create checkpoints directory
  checkpoint_dir = config.model.ckpt_dir or os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_fp = config.model.ckpt_fp or os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_fp))
  
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_fp, state, config.device)

  return state#, score_model, ema, checkpoint_dir, checkpoint_meta_fp

def get_loss_fns(config, sde, inverse_scaler, train=True):
  optimize_fn = losses.optimization_manager(config)
  train_step_fn = losses.get_step_fn(config, sde, train=train, optimize_fn=optimize_fn)
  nll_fn = likelihood.get_likelihood_fn(config, sde, inverse_scaler)
  nelbo_fn = likelihood.get_elbo_fn(config, sde, inverse_scaler=inverse_scaler)
  sampling_shape = (config.sampling.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, config.sampling.truncation_time)
  return train_step_fn, nll_fn, nelbo_fn, sampling_fn


def now2str(delimiter: Optional[str]='-'):
    now = datetime.now()
    now_str = now.strftime(f"%Y%m%d{delimiter}%H%M%S")
    return now_str

def save_each_npimg(npimgs: Iterable[Union[np.ndarray, Image.Image]],
                   out_dir: Path,
                    prefix: Union[str, int]='',
                   suffix_start_idx: int=0,
                   is_pilimg:bool=False,
                   **plot_kwargs) -> None:
    """Save each npimg in `npimgs` as png file using plt.imsave:
    File naming convention: out_dir / {prefix}_{start_idx + i}.png for ith image 
    in the given list.
    Save npimg using `plt.imsave' if not is_pilimg, else the input is actually a pilimage,
    and we save each pilimage using `PIL.Image.Image.save(fp)`.
    
    Note:
    - When the input images are np.arrays: 
        if vmin and vmax are not given, then the min/max of each nparr is mapped 
        to the min/max of the colormap (default, unless given as kwarg). 
        So, not specifying the vmin/vmax in kwargs has essentially the same effect
        as normalizing each nparr to [0.0., 1.0] and then converting each float value 
        to a colorvalue in the colormap by linear-map (0.0 -> colormap.min, 1.0 -> colormap.max)
        
    Resources: 
    - [plt.image.save](https://tinyurl.com/2jqcemdo) 
    - [matplotlib.cm](https://matplotlib.org/stable/api/cm_api.html)

    """    
    bs = len(npimgs)
    for i in range(bs):
        idx = suffix_start_idx + i
        fp = out_dir / f'{prefix}_{idx:07d}.png'
        
        if not is_pilimg:
            plt.imsave(fp, npimgs[i], **plot_kwargs)   
        else: #npimgs are actually an array of pil_img's (rgb)
            npimgs[i].save(fp, **plot_kwargs)
#         print('saved: ', fp)