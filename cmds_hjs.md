## Training code in their readme for celebahq 256:
### Arguments:
#### Required:
- `--workdir`: path to working dir; 
- `--config`: path to training config file 
- `--mode`: "train" or "eval" 

#### Optional:
- `--eval_folder`: default "eval"
  - folder to store evaluation results

### Eg. training cmd
```bash
python main.py --config configs/ve/celebahq/uncsnpp_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```


### My sampling cmd
- eval_dir will be set to $workdir / $eval_folder
  - export workdir='.'
  - export eval_folder='./Exp-Outputs
    - 

From repo: To train,
```bash
python main.py --config configs/ve/celebahq/uncsnpp_st.py 
--workdir YOUR_SAVING_DIRECTORY --mode train
```
- so i need to modify this config for sampling from celebahq256 pretrained model
  --> did it: see `./configs/ve/celebahq/sampling_unsnpp_st.py`


To sample: 

  --> todo -- #here
  - [ ] i need to probabily look at my modification of the sampling script for 
  ncsn++ (yang song's repo) and make similar modifications on this main.py
    - > this one: /data/hayley-old/Github/Scores/LSGM/evaluate_vada.py
    - [ ] prob. need to add a new eval_mode "sample"
      - [ ] load the pretrained model
      - [ ] look at lsgm sampling loop and copy paste the sampling iterations part
      - 
```bash
export config=/data/hayley-old/Github/Scores/Soft-Truncation/configs/ve/celebahq/sampling_uncsnpp_st.py
export mode=sample
export workdir='.'
export outdir_root=./Generated-Images

#these should be specified in the config file: e.g. evaluate.batch_size = 4, evaluate.num_samples = 8 
export nsamples=8
export bs=4

export CUDA_VISIBLE_DEVICES=3

nohup python main.py --mode sample --config $config \
--workdir $workdir --outdir_root $outdir_root  \
&
# --n_samples $nsamples --batch_size $bs \ 
```

## ~~Run 1~~:
### commands

export config=/data/hayley-old/Github/Scores/Soft-Truncation/configs/ve/celebahq/sampling_uncsnpp_st.py
export mode=sample
export workdir='.'
export outdir_root=./Generated-Images

- these should be specified in the config file: 
export nsamples=100000  # --> `sampling.batch_size = 4`, 
export bs=16 # --> `evaluate.num_samples = 8` 

export CUDA_VISIBLE_DEVICES=3

nohup \ 
CUDA_VISIBLE_DEVICES=3 python main.py --mode sample --config $config \
--workdir $workdir --outdir_root $outdir_root &

- this is running with NCSN++

### Progress:  -- killed; ignore the result
- run_id: 20230316164342/ -- need to check
- out_dir: Generated-Images/20230316164342/
- started: around 2023-0316-1616
- gpu: 3
- pid: 14654 

- 20230317-164228: i noticed every 16 images are repeated 
- so I killed this process. need to fix it:
- [ ] Fix why the sampling from rve is generating same set of 16 images?
    - Options:
      - 1. maybe batchsize = 16; or some kind of random seed is being seated at each iter?
        - [ ] test by removing config.seed from the config file
          - [ ] first, figure out where the seed is used
          - [ ] 

ls  Generated-Images/20230316164342/ | wc -l


## ~~Run 2~~: -- #killed by a crash of arya - 3/22/2023
- fixed the duplicated savings at every batch_size - it was my bug (in sample_and_save,  i was saving the entire batch (size16) at batch-size times (so, 16*16, and then the first new images samples will be actually saved, so most of the time, it was just saving that first batch (16images), by saving 16times of that batch..ikkkkkkkkkkssss!!! but i fixed it now. )
- [x] so just need to test that there is no duplication every 16th images, 
- [x] check this at least 16 batch times )

- started: 20230317-183736
- pid: 8771 
- gpu: 3
- outdir: /data/hayley-old/Github/Scores/Soft-Truncation/Generated-Images/20230317183313
- check with:
  - ls /data/hayley-old/Github/Scores/Soft-Truncation/Generated-Images/20230317183313 | wc -l
- klled by a crash in arya - 3/22/2023
- started Run 3 on Generated-Images-Run2 folder (see below)


## Run 3: #killed to release for weiwei (after generating 48 imgs)

todo:
- [ ] check where random seed is set; add argparser if not existing
- [ ] sample 32 images with bs=4 and check if these generated imgs are diff. from the ones from the first run (Ie run2)
      in ./Generated-Images folder

```bash
export config=/data/hayley-old/Github/Scores/Soft-Truncation/configs/ve/celebahq/sampling_uncsnpp_st.py
export mode=sample
export workdir='.'
export outdir_root=./Generated-Images-test

#!! -- these should be specified in the config file: e.g. evaluate.batch_size = 4, evaluate.num_samples = 8 
# export nsamples=8
# export bs=4

export CUDA_VISIBLE_DEVICES=2

nohup python main.py --mode sample --config $config \
--workdir $workdir --outdir_root $outdir_root  \
&
# --n_samples $nsamples --batch_size $bs \ --> should be specfied in the config file
```

### Full command I ran:
```bash
python main.py --mode sample --config /data/hayley-old/Github/Scores/Soft-Truncation/configs/ve/celebahq/sampling_uncsnpp_st.py --workdir . --outdir_root ./Generated-Images-test
```
### Note:
- [x] Check sde.N and sampling timesteps in `sampling.py: pc_sampler` 
  :: they are essentially the same : `sde.N = 2000, len(timesteps) = 2000`,
     b.c: `timesteps = torch.linspace(sde.T, eps, sde.N, device=device)`

### Progress:
- started:
- pid: 15757
- gpu: 2
- outdir: /data/hayley-old/Github/Scores/Soft-Truncation/Generated-Images-test/20230323171745/
- check n_imgs: 48
  - ls $outdir | wc -l
- status: 
  - **killed**: to release it for weiwei -- 20230324-134339  

## Run 4: -- #waitingFor

```bash
export config=/data/hayley-old/Github/Scores/Soft-Truncation/configs/ve/celebahq/sampling_uncsnpp_st.py
export mode=sample
export workdir='.'
export outdir_root=./Generated-Images-test

#!! -- these should be specified in the config file: e.g. evaluate.batch_size = 4, evaluate.num_samples = 8 
# export nsamples=8
# export bs=4

export CUDA_VISIBLE_DEVICES=1

nohup python main.py --mode sample --config $config \
--workdir $workdir --outdir_root $outdir_root  \
&
# --n_samples $nsamples --batch_size $bs \ --> should be specfied in the config file
```

### Full command I ran:
```bash
python main.py --mode sample --config /data/hayley-old/Github/Scores/Soft-Truncation/configs/ve/celebahq/sampling_uncsnpp_st.py --workdir . --outdir_root $outdir_root
```
### Note:

### Progress:
- started: 20230328-173337
- pid: 7596
- gpu: 1
- outdir: /data/hayley-old/Github/Scores/Soft-Truncation/Generated-Images-test/20230328173335
- check n_imgs: 
  - ls $outdir | wc -l
- status: running