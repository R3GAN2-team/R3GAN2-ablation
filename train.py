# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import click
import re
import json
import tempfile
import signal
import psutil
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def _parse_resume_arg(resume):
    """Return ('none'|'auto'|'pkl', value) for --resume."""
    if resume is None:
        return 'none', None

    s = str(resume).strip()
    if s == '':
        return 'none', None

    sl = s.lower()
    if sl in ['0', 'false', 'no', 'off', 'none']:
        return 'none', None
    if sl in ['1', 'true', 'yes', 'on']:
        return 'auto', None

    return 'pkl', s


def _collect_numbered_run_dirs(outdir):
    if not os.path.isdir(outdir):
        return []

    run_dirs = []
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        if not os.path.isdir(path):
            continue
        match = re.match(r'^(\d+)', name)
        if match is None:
            continue
        run_dirs.append((int(match.group(1)), name, path))

    run_dirs.sort(key=lambda x: (x[0], x[1]))
    return run_dirs


def _find_latest_run_dir(outdir):
    run_dirs = _collect_numbered_run_dirs(outdir)
    if len(run_dirs) == 0:
        return None
    return run_dirs[-1][2]



def _set_parent_death_signal():
    """Ask Linux to SIGTERM this process if its parent launcher dies."""
    if os.name != 'posix':
        return
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


def _collect_process_tree(extra_pids=None):
    """Collect launcher children and optional extra PID subtrees."""
    procs = []
    seen = set()

    def add_proc(proc):
        try:
            pid = proc.pid
        except psutil.NoSuchProcess:
            return
        if pid == os.getpid() or pid in seen:
            return
        seen.add(pid)
        procs.append(proc)
        try:
            children = proc.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in children:
            add_proc(child)

    try:
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            add_proc(child)
    except psutil.NoSuchProcess:
        pass

    for pid in extra_pids or []:
        try:
            add_proc(psutil.Process(pid))
        except psutil.NoSuchProcess:
            pass

    return procs


def _terminate_process_tree(extra_pids=None, term_timeout=10, kill_timeout=5, hard=False):
    """Terminate all child processes, or SIGKILL them immediately if hard=True."""
    procs = _collect_process_tree(extra_pids=extra_pids)
    if len(procs) == 0:
        return

    if hard:
        print(f'Killing {len(procs)} child processes...', flush=True)
        for proc in procs:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(procs, timeout=kill_timeout)
        return

    print(f'Terminating {len(procs)} child processes...', flush=True)
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

    _gone, alive = psutil.wait_procs(procs, timeout=term_timeout)
    if len(alive) == 0:
        return

    print(f'{len(alive)} child processes did not terminate; killing...', flush=True)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=kill_timeout)

def subprocess_fn(rank, c, temp_dir):
    if c.num_gpus > 1:
        _set_parent_death_signal()
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    try:
        training_loop.training_loop(rank=rank, **c)
    finally:
        if c.num_gpus > 1 and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    resume_auto = getattr(c, 'resume_dir', None) == 'auto'

    if resume_auto:
        latest_run_dir = _find_latest_run_dir(outdir)
        if latest_run_dir is not None:
            c.run_dir = latest_run_dir
            c.resume_dir = c.run_dir
            assert os.path.isdir(c.run_dir)
        else:
            print(f'WARNING: --resume=1 was specified, but no previous run directories were found in "{outdir}". Starting a new run instead.')
            resume_auto = False
            if 'resume_dir' in c:
                del c.resume_dir

    if not resume_auto:
        prev_run_ids = [run_id for run_id, _name, _path in _collect_numbered_run_dirs(outdir)]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    if resume_auto:
        print(f'Resume directory:    {c.resume_dir}')
    elif getattr(c, 'resume_pkl', None) is not None:
        print(f'Resume checkpoint:   {c.resume_pkl}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory for new runs. In-place resume reuses the existing
    # run directory and lets training_loop append/truncate logs as needed.
    if resume_auto:
        print('Reusing output directory...')
    else:
        print('Creating output directory...')
        os.makedirs(c.run_dir)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            def hard_shutdown_single(signum, frame):
                print(f'\nReceived signal {signum}; killing child processes now...', flush=True)
                _terminate_process_tree(kill_timeout=3, hard=True)
                os._exit(128 + signum)

            old_sigint = signal.signal(signal.SIGINT, hard_shutdown_single)
            old_sigterm = signal.signal(signal.SIGTERM, hard_shutdown_single)
            try:
                subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
            finally:
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
        else:
            ctx = None
            shutdown_started = False

            def worker_pids():
                if ctx is None:
                    return []
                return [proc.pid for proc in ctx.processes if proc.pid is not None]

            def hard_shutdown(signum, frame):
                print(f'\nReceived signal {signum} again; killing workers immediately...', flush=True)
                _terminate_process_tree(extra_pids=worker_pids(), kill_timeout=3, hard=True)
                os._exit(128 + signum)

            def request_shutdown(signum, frame):
                nonlocal shutdown_started
                if shutdown_started:
                    hard_shutdown(signum, frame)

                shutdown_started = True
                signal.signal(signal.SIGINT, hard_shutdown)
                signal.signal(signal.SIGTERM, hard_shutdown)

                print(f'\nReceived signal {signum}; killing workers now...', flush=True)
                _terminate_process_tree(extra_pids=worker_pids(), kill_timeout=3, hard=True)
                os._exit(128 + signum)

            old_sigint = signal.signal(signal.SIGINT, request_shutdown)
            old_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
            try:
                ctx = torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus, join=False)
                while not ctx.join(timeout=1):
                    pass
            finally:
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
                if shutdown_started:
                    _terminate_process_tree(extra_pids=worker_pids(), kill_timeout=3, hard=True)


#----------------------------------------------------------------------------

        
def init_dataset_kwargs(data):
    try:
        # Fast path: contiguous mmap latent dataset created by latent_zip_to_mmap.py.
        meta_path = os.path.join(data, 'metadata.json') if os.path.isdir(data) else None
        if meta_path is not None and os.path.isfile(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if meta.get('format') == 'r3gan_latent_mmap_v1':
                dataset_kwargs = dnnlib.EasyDict(
                    class_name='training.dataset.MMapLatentDataset',
                    path=data,
                    use_labels=True,
                    max_size=None,
                    xflip=False,
                )
                dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
                dataset_kwargs.resolution = dataset_obj.resolution
                dataset_kwargs.use_labels = dataset_obj.has_labels
                dataset_kwargs.max_size = len(dataset_obj)
                return dataset_kwargs, dataset_obj.name

        # Original StyleGAN/NVIDIA image-folder/zip path.
        dataset_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=data,
            use_labels=True,
            max_size=None,
            xflip=False,
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution
        dataset_kwargs.use_labels = dataset_obj.has_labels
        dataset_kwargs.max_size = len(dataset_obj)
        return dataset_kwargs, dataset_obj.name

    except IOError as err:
        raise click.ClickException(f'--data: {err}')        
        

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--eval',         help='Evaluation data', metavar='[ZIP|DIR]',                    type=str, default='none', show_default=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--preset',       help='Preset configs', metavar='STR',                           type=str, required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Enable Augmentation', metavar='BOOL',                     type=bool, default=True, show_default=True)
@click.option('--resume',       help='Resume from network pickle, or --resume=1 to resume latest run in --outdir', metavar='[PATH|URL|BOOL]', type=str)

# Misc hyperparameters.
@click.option('--g-batch-gpu',  help='Limit batch size per GPU for G', metavar='INT',           type=click.IntRange(min=1))
@click.option('--d-batch-gpu',  help='Limit batch size per GPU for D', metavar='INT',           type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=10000000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--ema-snap',     help='How often to save ema snapshots', metavar='TICKS',        type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

def main(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    
    c.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator')
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator')
    
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.,0.], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.,0.], eps=1e-8)
    
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.R3GANLoss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    
    if opts.eval == 'none':
        opts.eval = opts.data
    c.eval_set_kwargs, _ = init_dataset_kwargs(data=opts.eval)

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.g_batch_gpu = opts.g_batch_gpu or opts.batch // opts.gpus
    c.d_batch_gpu = opts.d_batch_gpu or opts.batch // opts.gpus
    
    if opts.preset == 'CIFAR10':
        WidthPerStage = [x // 2 for x in [1024, 1024, 1024]]
        BlocksPerStage = [['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN']]
        NoiseDimension = 64
        aug_config = dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=0.5, contrast=0.5, lumaflip=0.5, hue=0.5, saturation=0.5, cutout=1)
        ema_stds = [0.010, 0.050, 0.100]
       
        c.G_kwargs.ClassEmbeddingDimension = NoiseDimension
        c.D_kwargs.ClassEmbeddingDimension = WidthPerStage[0]
       
        decay_nimg = 2e7
       
        c.aug_scheduler = { 'base_value': 0, 'final_value': 0.55, 'total_nimg': decay_nimg, 'rampup_Mimg': 1 }
        c.lr_scheduler = { 'batch_size': 512, 'max_lr': 1e-2, 'ref_lr': 2e-3, 'ref_batches': 23000.0, 'early_decay_mult': 8.0, 'total_nimg': decay_nimg, 'rampup_Mimg': 1 }
        c.gamma_scheduler = { 'base_value': 0.01 / 4, 'final_value': 0.01, 'total_nimg': decay_nimg, 'rampup_Mimg': 1 }
        c.beta2_scheduler = { 'start_beta2': 0.9, 'final_beta2': 0.99, 'total_nimg': decay_nimg, 'rampup_Mimg': 1 }
    
    if opts.preset == 'ImageNet-1x':
        WidthPerStage = [x // 2 for x in [1024, 1024, 1024]]
        BlocksPerStage = [['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN']]
        NoiseDimension = 64
        aug_config = dict(rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, cutout=1)
        ema_stds = [0.050, 0.100, 0.150, 0.200]
       
        c.G_kwargs.ClassEmbeddingDimension = NoiseDimension
        c.D_kwargs.ClassEmbeddingDimension = WidthPerStage[0]
       
        decay_nimg = 2e8
       
        c.aug_scheduler = { 'base_value': 0, 'final_value': 0.3, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }
        c.lr_scheduler = { 'batch_size': 4096, 'max_lr': 1e-2, 'ref_lr': 4e-3, 'ref_batches': 251116.071429, 'early_decay_mult': 5, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }
        c.gamma_scheduler = { 'base_value': 2/4, 'final_value': 2, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }
        c.beta2_scheduler = { 'start_beta2': 0.9, 'final_beta2': 0.95, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }  
    
    if opts.preset == 'ImageNet-2x':
        WidthPerStage = [x for x in [1024, 1024, 1024]]
        BlocksPerStage = [['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN']]
        NoiseDimension = 64
        aug_config = dict(rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, cutout=1)
        ema_stds = [0.050, 0.100, 0.150, 0.200]
       
        c.G_kwargs.ClassEmbeddingDimension = NoiseDimension
        c.D_kwargs.ClassEmbeddingDimension = WidthPerStage[0]
       
        decay_nimg = 2e8
       
        c.aug_scheduler = { 'base_value': 0, 'final_value': 0.3, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }
        c.lr_scheduler = { 'batch_size': 4096, 'max_lr': 1e-2, 'ref_lr': 2.5e-3, 'ref_batches': 251116.071429, 'early_decay_mult': 8, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }
        c.gamma_scheduler = { 'base_value': 1, 'final_value': 4, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }
        c.beta2_scheduler = { 'start_beta2': 0.9, 'final_beta2': 0.95, 'total_nimg': decay_nimg, 'rampup_Mimg': 10 }

    c.G_kwargs.NoiseDimension = NoiseDimension
    c.G_kwargs.WidthPerStage = WidthPerStage
    c.G_kwargs.BlocksPerStage = BlocksPerStage
    c.G_kwargs.FFNWidthRatio = 2
    c.G_kwargs.ChannelsPerConvolutionGroup = 32
    
    c.D_kwargs.WidthPerStage = [*reversed(WidthPerStage)]
    c.D_kwargs.BlocksPerStage = [*reversed(BlocksPerStage)]
    c.D_kwargs.FFNWidthRatio = 2
    c.D_kwargs.ChannelsPerConvolutionGroup = 32
    
    
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.ema_snapshot_ticks = opts.ema_snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.g_batch_gpu) != 0 or c.batch_size % (c.num_gpus * c.d_batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

            
    # Augmentation.
    if opts.aug:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **aug_config)
        
    c.ema_kwargs = dnnlib.EasyDict(class_name='training.phema.PowerFunctionEMA', stds=ema_stds)

    # Resume.
    resume_mode, resume_value = _parse_resume_arg(opts.resume)
    if resume_mode == 'auto':
        c.resume_dir = 'auto'
    elif resume_mode == 'pkl':
        c.resume_pkl = resume_value

    # Performance-related toggles.
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
