# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import glob
import re
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from .feat_collector import CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude

def linear_schedule(cur_nimg, base_value, total_nimg, final_value, rampup_Mimg=0):
    rampup_nimg = rampup_Mimg * 1e6

    if cur_nimg >= total_nimg:
        return final_value
    if cur_nimg <= rampup_nimg:
        return base_value

    t = (cur_nimg - rampup_nimg) / (total_nimg - rampup_nimg)
    return base_value + t * (final_value - base_value)


def log_linear_schedule(cur_nimg, base_value, total_nimg, final_value, rampup_Mimg=0):
    rampup_nimg = rampup_Mimg * 1e6

    if cur_nimg >= total_nimg:
        return final_value
    if cur_nimg <= rampup_nimg:
        return base_value

    t = (cur_nimg - rampup_nimg) / (total_nimg - rampup_nimg)
    return base_value * (final_value / base_value) ** t


def beta2_from_exact_horizon_schedule(
    cur_nimg,
    total_nimg,
    start_beta2=0.9,
    final_beta2=0.99,
    rampup_Mimg=0,
):
    rampup_nimg = rampup_Mimg * 1e6

    if cur_nimg >= total_nimg:
        return final_beta2
    if cur_nimg <= rampup_nimg:
        return start_beta2

    start_tau = -1.0 / np.log(start_beta2)
    final_tau = -1.0 / np.log(final_beta2)

    tau = log_linear_schedule(
        cur_nimg=cur_nimg,
        base_value=start_tau,
        total_nimg=total_nimg,
        final_value=final_tau,
        rampup_Mimg=rampup_Mimg,
    )

    return float(np.exp(-1.0 / tau))

#----------------------------------------------------------------------------

def edm2_power_bridge_learning_rate_schedule(
    cur_nimg,
    batch_size,
    max_lr,
    ref_lr,
    ref_batches,
    rampup_Mimg,
    total_nimg,
    early_decay_mult=5.0,
):
    warmup_nimg = rampup_Mimg * 1e6
    decay_ref_nimg = ref_batches * batch_size

    # Stage 1: linear warmup, 0 -> max_lr.
    if cur_nimg < warmup_nimg:
        return float(max_lr * (cur_nimg / warmup_nimg))

    # Stage 3: shifted 1/sqrt tail, starting exactly at total_nimg.
    if cur_nimg >= total_nimg:
        x = cur_nimg - total_nimg
        return float(ref_lr / np.sqrt(1.0 + x / decay_ref_nimg))

    # Stage 2: monotone power bridge, max_lr -> ref_lr.
    bridge_nimg = total_nimg - warmup_nimg
    delta = max_lr - ref_lr

    u = (cur_nimg - warmup_nimg) / bridge_nimg
    v = 1.0 - u

    # Match the derivative of the sqrt tail at total_nimg.
    s = ref_lr * bridge_nimg / (2.0 * decay_ref_nimg * delta)

    # Set the initial bridge decay speed.
    # early_decay_mult = 5 means initial bridge slope is 5x the average slope.
    a = (early_decay_mult - 1.0) / (1.0 - s)

    lr = ref_lr + delta * v * (s + (1.0 - s) * v**a)
    return float(lr)

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, grid_size):
    lo, hi = [0,255]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def remap_optimizer_state_dict(state_dict, device):
    state_dict = copy.deepcopy(state_dict)
    for param in state_dict['state'].values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return state_dict


#----------------------------------------------------------------------------

def _snapshot_kimg(path):
    m = re.search(r'network-snapshot-(\d+)\.pkl$', os.path.basename(path))
    return int(m.group(1)) if m is not None else -1

#----------------------------------------------------------------------------

def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass

#----------------------------------------------------------------------------

def atomic_pickle_dump(obj, final_path):
    tmp_path = f'{final_path}.tmp.{os.getpid()}'
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        _fsync_dir(os.path.dirname(final_path))
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise

#----------------------------------------------------------------------------

def _remove_file(path, reason):
    try:
        os.remove(path)
        print(f'WARNING: deleted {reason}: {path}', flush=True)
    except FileNotFoundError:
        pass

#----------------------------------------------------------------------------

def _load_network_snapshot_for_cleanup(path):
    try:
        with open(path, 'rb') as f:
            data = legacy.load_network_pkl(f)
        for key in ['G', 'D', 'cur_nimg', 'G_opt_state', 'D_opt_state']:
            if key not in data:
                raise KeyError(f'missing key {key}')
        return data
    except Exception as err:
        print(f'WARNING: corrupted network snapshot: {path} ({type(err).__name__}: {err})', flush=True)
        return None

#----------------------------------------------------------------------------

def _validate_ema_snapshot(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not hasattr(data, 'ema') and not (isinstance(data, dict) and 'ema' in data):
            raise KeyError('missing key ema')
        return True
    except Exception as err:
        print(f'WARNING: corrupted EMA snapshot: {path} ({type(err).__name__}: {err})', flush=True)
        return False

#----------------------------------------------------------------------------

def _expected_ema_suffixes(ema_kwargs):
    stds = [] if ema_kwargs is None else list(ema_kwargs.get('stds', []))
    return {f'-{float(std):.5f}' for std in stds}

#----------------------------------------------------------------------------

def _is_url_path(path):
    return re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', str(path)) is not None

#----------------------------------------------------------------------------

def _cleanup_atomic_tmp_files(run_dir):
    for subdir in ['snapshots', 'ema']:
        root = os.path.join(run_dir, subdir)
        for path in glob.glob(os.path.join(root, '*.tmp.*')):
            _remove_file(path, 'stale atomic-write temp file')

#----------------------------------------------------------------------------

def _cleanup_latest_network_snapshot(run_dir):
    # Check only the current tail of the network-snapshot stream. If the newest
    # snapshot is corrupt, delete it and try the next newest candidate until the
    # latest remaining snapshot is loadable. This avoids scanning the whole run
    # history on every resume while still recovering from interrupted writes.
    snapshot_dir = os.path.join(run_dir, 'snapshots')
    while True:
        paths = sorted(glob.glob(os.path.join(snapshot_dir, 'network-snapshot-*.pkl')), key=_snapshot_kimg, reverse=True)
        if len(paths) == 0:
            return None, None
        path = paths[0]
        data = _load_network_snapshot_for_cleanup(path)
        if data is not None:
            return path, data
        _remove_file(path, 'corrupted latest network snapshot')

#----------------------------------------------------------------------------

def _parse_ema_snapshot_path(path):
    m = re.match(r'^ema-snapshot-(\d+)(.*)\.pkl$', os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1)), m.group(1), m.group(2)

#----------------------------------------------------------------------------

def _list_ema_snapshot_groups(run_dir):
    ema_dir = os.path.join(run_dir, 'ema')
    groups = dict()
    for path in glob.glob(os.path.join(ema_dir, 'ema-snapshot-*.pkl')):
        parsed = _parse_ema_snapshot_path(path)
        if parsed is None:
            continue
        kimg_int, kimg_str, suffix = parsed
        groups.setdefault(kimg_int, (kimg_str, dict()))[1][suffix] = path
    return groups

#----------------------------------------------------------------------------

def _remove_ema_snapshot_group(kimg, suffix_to_path, reason):
    for path in sorted(suffix_to_path.values()):
        _remove_file(path, f'{reason} EMA snapshot group {kimg}')

#----------------------------------------------------------------------------

def _validate_ema_snapshot_group(kimg, suffix_to_path, ema_kwargs):
    # EMA groups are atomic for EDM2 compatibility: all expected std files for a
    # kimg must exist and be loadable. If the expected suffix set is known, any
    # missing or unexpected file invalidates the whole group.
    expected_suffixes = _expected_ema_suffixes(ema_kwargs)
    if len(expected_suffixes) > 0:
        found_suffixes = set(suffix_to_path.keys())
        if found_suffixes != expected_suffixes:
            missing = sorted(expected_suffixes - found_suffixes)
            unexpected = sorted(found_suffixes - expected_suffixes)
            print(
                f'WARNING: incomplete EMA snapshot group {kimg}: '
                f'missing={missing}, unexpected={unexpected}',
                flush=True,
            )
            return False

    for path in sorted(suffix_to_path.values()):
        if not _validate_ema_snapshot(path):
            return False
    return True

#----------------------------------------------------------------------------

def _cleanup_ema_groups_after_kimg(run_dir, max_kimg):
    # EMA groups newer than the selected network checkpoint belong to a future
    # branch that will be rolled back on resume, even if each EMA file is valid.
    groups = _list_ema_snapshot_groups(run_dir)
    for kimg_int, (kimg, suffix_to_path) in sorted(groups.items()):
        if kimg_int > max_kimg:
            print(
                f'WARNING: EMA snapshot group {kimg} is newer than latest valid network snapshot {max_kimg:09d}; deleting it.',
                flush=True,
            )
            _remove_ema_snapshot_group(kimg, suffix_to_path, 'future')

#----------------------------------------------------------------------------

def _cleanup_tail_ema_snapshot_groups(run_dir, ema_kwargs, max_kimg):
    # Check only the current EMA tail at or before max_kimg. If it is invalid,
    # delete that tail group and repeat, so stale/corrupted tail groups are
    # peeled off without scanning the entire history.
    while True:
        groups = {k: v for k, v in _list_ema_snapshot_groups(run_dir).items() if k <= max_kimg}
        if len(groups) == 0:
            return None

        kimg_int, (kimg, suffix_to_path) = max(groups.items(), key=lambda item: item[0])
        if _validate_ema_snapshot_group(kimg, suffix_to_path, ema_kwargs):
            return kimg_int, kimg, suffix_to_path

        _remove_ema_snapshot_group(kimg, suffix_to_path, 'corrupted latest')

#----------------------------------------------------------------------------

def _network_snapshot_requires_ema_group(snapshot_data, ema_snapshot_ticks, total_kimg):
    if ema_snapshot_ticks is None:
        return False

    saved_ema_snapshot_ticks = snapshot_data.get('ema_snapshot_ticks', ema_snapshot_ticks)
    if saved_ema_snapshot_ticks is None:
        return False

    requires = False
    if 'cur_tick' in snapshot_data:
        requires = (int(snapshot_data['cur_tick']) % int(saved_ema_snapshot_ticks)) == 0

    # New snapshots explicitly record whether they were saved by the terminal
    # `done` path. Older snapshots did not, so fall back to the current total_kimg
    # as a conservative best-effort detector.
    if bool(snapshot_data.get('done', False)):
        requires = True
    elif total_kimg is not None and 'cur_nimg' in snapshot_data:
        try:
            requires = requires or (int(snapshot_data['cur_nimg']) >= int(total_kimg * 1000))
        except Exception:
            pass

    return requires

#----------------------------------------------------------------------------

def cleanup_corrupt_resume_files(run_dir, ema_kwargs, ema_snapshot_ticks=None, total_kimg=None):
    _cleanup_atomic_tmp_files(run_dir)

    while True:
        network_path, network_data = _cleanup_latest_network_snapshot(run_dir)
        if network_path is None:
            return

        network_kimg = _snapshot_kimg(network_path)
        _cleanup_ema_groups_after_kimg(run_dir, network_kimg)
        latest_ema_group = _cleanup_tail_ema_snapshot_groups(run_dir, ema_kwargs, network_kimg)

        if _network_snapshot_requires_ema_group(network_data, ema_snapshot_ticks, total_kimg):
            if latest_ema_group is None or latest_ema_group[0] != network_kimg:
                found = 'none' if latest_ema_group is None else latest_ema_group[1]
                print(
                    f'WARNING: network snapshot {network_kimg:09d} requires a complete EMA group at the same kimg, '
                    f'but latest valid EMA group is {found}; deleting network snapshot and falling back.',
                    flush=True,
                )
                _remove_file(network_path, 'network snapshot with missing required EMA group')
                continue

        return

#----------------------------------------------------------------------------

def find_latest_network_snapshot(run_dir):
    snapshot_dir = os.path.join(run_dir, 'snapshots')
    paths = glob.glob(os.path.join(snapshot_dir, 'network-snapshot-*.pkl'))
    if len(paths) == 0:
        raise FileNotFoundError(f'No network snapshots found in {snapshot_dir}')

    return max(paths, key=_snapshot_kimg)

#----------------------------------------------------------------------------

def derive_next_tick_from_nimg(cur_nimg, batch_size, kimg_per_tick):
    if cur_nimg <= 0:
        return 0
    tick_nimg = int(np.ceil((kimg_per_tick * 1000) / batch_size)) * batch_size
    return int(1 + max(0, (cur_nimg - batch_size) // tick_nimg))

#----------------------------------------------------------------------------

def _stat_value(x):
    if isinstance(x, dict):
        if 'mean' in x:
            return x['mean']
        if 'value' in x:
            return x['value']
    if isinstance(x, (int, float)):
        return x
    return None

#----------------------------------------------------------------------------

def _read_last_total_sec(stats_path, max_kimg=None):
    if not os.path.isfile(stats_path):
        return 0.0

    last = 0.0
    with open(stats_path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if max_kimg is not None:
                kimg = _stat_value(obj.get('Progress/kimg'))
                if kimg is not None and kimg > max_kimg + 1e-9:
                    continue

            total_sec = _stat_value(obj.get('Timing/total_sec'))
            if total_sec is not None:
                last = float(total_sec)
    return last

#----------------------------------------------------------------------------

def _truncate_stats_jsonl_at_kimg(stats_path, max_kimg):
    if not os.path.isfile(stats_path):
        return

    keep = []
    with open(stats_path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
                kimg = _stat_value(obj.get('Progress/kimg'))
            except Exception:
                keep.append(line)
                continue

            if kimg is None or kimg <= max_kimg + 1e-9:
                keep.append(line)

    with open(stats_path, 'wt') as f:
        f.writelines(keep)

#----------------------------------------------------------------------------

def _parse_snapshot_kimg_from_path(path):
    m = re.search(r'network-snapshot-(\d+)\.pkl', str(path))
    return int(m.group(1)) if m is not None else None

#----------------------------------------------------------------------------

def _truncate_metric_jsonl_at_kimg(run_dir, max_kimg):
    for path in glob.glob(os.path.join(run_dir, 'metric-*.jsonl')):
        keep = []
        with open(path, 'rt') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    kimg = _parse_snapshot_kimg_from_path(obj.get('snapshot_pkl', ''))
                except Exception:
                    keep.append(line)
                    continue

                if kimg is None or kimg <= max_kimg:
                    keep.append(line)

        with open(path, 'wt') as f:
            f.writelines(keep)

#----------------------------------------------------------------------------

def load_and_increment_resume_count(run_dir, resume_run, rank, device, num_gpus):
    if not resume_run:
        resume_count = 0
    elif rank == 0:
        state_path = os.path.join(run_dir, 'resume_state.json')
        if os.path.isfile(state_path):
            with open(state_path, 'rt') as f:
                state = json.load(f)
            resume_count = int(state.get('resume_count', 0)) + 1
        else:
            resume_count = 1

        tmp_path = state_path + '.tmp'
        with open(tmp_path, 'wt') as f:
            json.dump(dict(resume_count=resume_count), f, indent=2)
            f.write('\n')
        os.replace(tmp_path, state_path)
    else:
        resume_count = 0

    if resume_run and num_gpus > 1:
        value = torch.tensor([resume_count], device=device, dtype=torch.int64)
        torch.distributed.broadcast(value, src=0)
        resume_count = int(value.item())

    return resume_count

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    eval_set_kwargs         = {},       # Options for eval set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    lr_scheduler            = None,
    beta2_scheduler         = None,
    ema_kwargs              = None,
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    gamma_scheduler         = None,
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    g_batch_gpu             = 4,        # Number of samples processed at a time by one GPU.
    d_batch_gpu             = 4,        # Number of samples processed at a time by one GPU.
    aug_scheduler           = None,
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    ema_snapshot_ticks      = 50,
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_dir              = None,     # Existing run dir to resume in-place. If set, auto-load latest snapshot.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    compile_main_layers = False, # compile is broken
    compile_mode = 'default',
    compile_fullgraph = False,
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    # Resolve in-place resume before seeding or constructing the sampler.
    resume_run = resume_dir is not None
    if resume_run:
        run_dir = resume_dir
        if rank == 0:
            cleanup_corrupt_resume_files(run_dir, ema_kwargs, ema_snapshot_ticks=ema_snapshot_ticks, total_kimg=total_kimg)
        if num_gpus > 1:
            torch.distributed.barrier()
        if resume_pkl is not None and (not _is_url_path(resume_pkl)) and (not os.path.exists(resume_pkl)):
            resume_pkl = None
        if resume_pkl is None:
            resume_pkl = find_latest_network_snapshot(run_dir)
        if rank == 0:
            print(f'Resuming run directory: {run_dir}')
            print(f'Auto-selected checkpoint: {resume_pkl}')

    # Fresh runs use the original seed exactly. In-place resumed runs advance the
    # stochastic streams with a persistent sidecar counter instead of replaying
    # the same sampler/latent/augmentation prefix after every restart.
    resume_count = load_and_increment_resume_count(
        run_dir=run_dir,
        resume_run=resume_run,
        rank=rank,
        device=device,
        num_gpus=num_gpus,
    )
    effective_seed = random_seed + 1000003 * resume_count
    np.random.seed(effective_seed * num_gpus + rank)
    torch.manual_seed(effective_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
          
    if rank == 0:
        if resume_run:
            os.makedirs(os.path.join(run_dir, 'ema'), exist_ok=True)
            os.makedirs(os.path.join(run_dir, 'snapshots'), exist_ok=True)
        else:
            os.mkdir(os.path.join(run_dir, 'ema'))
            os.mkdir(os.path.join(run_dir, 'snapshots'))

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs) # subclass of training.dataset.Dataset
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=effective_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

    if rank == 0:
        print('Setting up encoder...')
    if training_set.num_channels == 3:
        encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif training_set.num_channels == 8:
        encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.Flux2VAEEncoder')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    
    ref_image, _ = training_set[0]
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', list(ref_image[0].shape))
        print('Label shape:', training_set.label_shape)
        print()
        

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=ref_image.shape[1])
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    ema = dnnlib.util.construct_class_by_name(net=G, **ema_kwargs)
    ema_preview = ema.emas[1]

    from R3GAN.Networks import FeedForwardNetwork
    from R3GAN.kernels.fused_ffn import install_fused_ffn, kernel_status
    install_fused_ffn(FeedForwardNetwork, enable=True, vjp=True)

    resume_data = None
    cur_nimg = 0

    # Resume from existing pickle.
    if resume_pkl is not None:
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        cur_nimg = int(resume_data['cur_nimg'])
        assert cur_nimg % batch_size == 0, (
            f'Checkpoint cur_nimg={cur_nimg} is not divisible by current batch_size={batch_size}'
        )
        if 'batch_size' in resume_data:
            assert int(resume_data['batch_size']) == int(batch_size), (
                f'Checkpoint batch_size={resume_data["batch_size"]} != current batch_size={batch_size}'
            )
        if rank == 0:
            print(f'Resuming from "{resume_pkl}" at {cur_nimg / 1e3:.1f} kimg')
            for name, module in [('G', G), ('D', D)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=resume_run)
            ema.load_state_dict(resume_data)

    # Print network summary tables.
    if rank == 0 and not resume_run:
        z = torch.empty([min(g_batch_gpu, d_batch_gpu), G.z_dim], device=device)
        c = torch.empty([min(g_batch_gpu, d_batch_gpu), G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None

    if (augment_kwargs is not None) and (aug_scheduler is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        
    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, ema.net, *ema.emas]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)



    # Compile only FFN-heavy residual groups. Do not compile the full model,
    # R1 function, or StyleGAN upfirdn2d transition layers.
    if compile_main_layers:
        if rank == 0:
            print(f'Compiling G/D main residual groups with torch.compile(mode={compile_mode})...')

        try:
            import importlib
            dynamo = importlib.import_module("torch._dynamo")
            dynamo.config.recompile_limit = 128
            dynamo.config.cache_size_limit = 128
        except Exception:
            pass

        # Compile G main layers per stage with static shapes.
        target = G.Model if hasattr(G, 'Model') else G
        if hasattr(target, 'MainLayers'):
            for stage_idx, layer in enumerate(target.MainLayers):
                if hasattr(layer, 'CompileForward'):
                    if rank == 0:
                        print(f'  Compiling G.MainLayers[{stage_idx}] static...')
                    layer.CompileForward(
                        mode=compile_mode,
                        fullgraph=compile_fullgraph,
                        dynamic=False,
                    )
                elif rank == 0:
                    print(f'Warning: G.MainLayers[{stage_idx}] does not expose CompileForward().')
        elif hasattr(target, 'CompileMainLayers'):
            if rank == 0:
                print(f'Warning: {type(target).__name__} has no MainLayers; falling back to CompileMainLayers(dynamic=False).')
            target.CompileMainLayers(
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=False,
            )
        else:
            if rank == 0:
                print(f'Warning: {type(target).__name__} does not expose MainLayers or CompileMainLayers().')

        # Compile D main layers per stage with static shapes.
        target = D.Model if hasattr(D, 'Model') else D
        if hasattr(target, 'MainLayers'):
            for stage_idx, layer in enumerate(target.MainLayers):
                if hasattr(layer, 'CompileForward'):
                    if rank == 0:
                        print(f'  Compiling D.MainLayers[{stage_idx}] static...')
                    layer.CompileForward(
                        mode=compile_mode,
                        fullgraph=compile_fullgraph,
                        dynamic=False,
                    )
                elif rank == 0:
                    print(f'Warning: D.MainLayers[{stage_idx}] does not expose CompileForward().')
        elif hasattr(target, 'CompileMainLayers'):
            if rank == 0:
                print(f'Warning: {type(target).__name__} has no MainLayers; falling back to CompileMainLayers(dynamic=False).')
            target.CompileMainLayers(
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=False,
            )
        else:
            if rank == 0:
                print(f'Warning: {type(target).__name__} does not expose MainLayers or CompileMainLayers().')




    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    
    opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **D_opt_kwargs)
    if resume_pkl is not None:
        opt.load_state_dict(remap_optimizer_state_dict(resume_data['D_opt_state'], device))
    phases += [dnnlib.EasyDict(name='D', module=D, opt=opt, batch_gpu=d_batch_gpu)]
    
    opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
    if resume_pkl is not None:
        opt.load_state_dict(remap_optimizer_state_dict(resume_data['G_opt_state'], device))
    phases += [dnnlib.EasyDict(name='G', module=G, opt=opt, batch_gpu=g_batch_gpu)]
    
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0 and False:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=eval_set)
        reals_path = os.path.join(run_dir, 'reals.png')
        if (not resume_run) or (not os.path.isfile(reals_path)):
            save_image_grid(images, reals_path, grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(g_batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(g_batch_gpu)
        if not resume_run:
            images = torch.cat([encoder.decode(ema_preview(z, c)).cpu() for z, c in zip(grid_z, grid_c)]).to(torch.float).numpy()
            save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    resume_time_offset = 0.0
    if rank == 0:
        stats_path = os.path.join(run_dir, 'stats.jsonl')
        if resume_run:
            resume_kimg = cur_nimg / 1e3
            _truncate_stats_jsonl_at_kimg(stats_path, resume_kimg)
            _truncate_metric_jsonl_at_kimg(run_dir, int(cur_nimg // 1000))
            resume_time_offset = _read_last_total_sec(stats_path, max_kimg=resume_kimg)
            stats_jsonl = open(stats_path, 'a')
        else:
            stats_jsonl = open(stats_path, 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            if resume_run:
                stats_tfevents = tensorboard.SummaryWriter(run_dir, purge_step=int(cur_nimg // 1000) + 1)
            else:
                stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    if (resume_pkl is not None) and resume_run:
        cur_tick = int(resume_data.get('cur_tick_next', derive_next_tick_from_nimg(cur_nimg, batch_size, kimg_per_tick)))
        batch_idx = cur_nimg // batch_size
    else:
        cur_tick = 0
        batch_idx = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
        
    # Dummy Timing, required to fix phase shift
    for phase in phases:
        if phase.start_event is not None:
            phase.start_event.record(torch.cuda.current_stream(device))
        if phase.end_event is not None:
            phase.end_event.record(torch.cuda.current_stream(device))
        
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            D_img, D_img_c = next(training_set_iterator)
            D_img = encoder.encode_latents(D_img.to(device))
            D_z = torch.randn([batch_size, G.z_dim], device=device)
            
            G_img, G_img_c = next(training_set_iterator)
            G_img = encoder.encode_latents(G_img.to(device))
            G_z = torch.randn([batch_size, G.z_dim], device=device)
            
            all_real_img = []
            all_real_c = []
            all_gen_z = []
            
            # D
            all_real_img += [D_img.detach().clone().split(d_batch_gpu)]
            all_real_c += [D_img_c.detach().clone().to(device).split(d_batch_gpu)]
            all_gen_z += [D_z.detach().clone().split(d_batch_gpu)]
            
            # G
            all_real_img += [G_img.detach().clone().split(g_batch_gpu)]
            all_real_c += [G_img_c.detach().clone().to(device).split(g_batch_gpu)]
            all_gen_z += [G_z.detach().clone().split(g_batch_gpu)]
        
        cur_lr = edm2_power_bridge_learning_rate_schedule(cur_nimg, **lr_scheduler)
        cur_beta2 = beta2_from_exact_horizon_schedule(cur_nimg, **beta2_scheduler)
        cur_gamma = log_linear_schedule(cur_nimg, **gamma_scheduler)
        cur_aug_p = linear_schedule(cur_nimg, **aug_scheduler)
        
        if augment_pipe is not None:
            augment_pipe.p.copy_(misc.constant(cur_aug_p, device=device))
        
        # Execute training phases.
        for phase, phase_gen_z, phase_real_img, phase_real_c in zip(phases, all_gen_z, all_real_img, all_real_c):
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z in zip(phase_real_img, phase_real_c, phase_gen_z):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gamma=cur_gamma, gain=num_gpus * phase.batch_gpu / batch_size)
            phase.module.requires_grad_(False)
        
            # Update weights.  
            for g in phase.opt.param_groups:
                g['lr'] = cur_lr
                g['betas'] = (0, cur_beta2)
                      
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()
                
                # Forced weight norm
                for x in phase.module.Model.modules():
                    if hasattr(x, 'NormalizeWeight'):
                        x.NormalizeWeight()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Update EMA
        ema.update(cur_nimg=cur_nimg, batch_size=batch_size)
        
        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        elapsed_total_sec = resume_time_offset + (tick_end_time - start_time)
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', elapsed_total_sec)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Progress/lr', cur_lr)
        training_stats.report0('Progress/beta2', cur_beta2)
        training_stats.report0('Progress/gamma', cur_gamma)
        training_stats.report0('Timing/total_hours', elapsed_total_sec / (60 * 60))
        training_stats.report0('Timing/total_days', elapsed_total_sec / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if False and (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([encoder.decode(ema_preview(z, c)).cpu() for z, c in zip(grid_z, grid_c)]).to(torch.float).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:09d}.png'), grid_size=grid_size)

        if False and (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            for real_img, real_c, gen_z in zip(phase_real_img, phase_real_c, phase_gen_z):
                with torch.no_grad():
                    for x in CollectGeneratorFeatures(G.Model, gen_z, real_c):
                        training_stats.report('MagnitudeG/avg/' + str(x.shape[2]), CollectMagnitude(x, 'avg'))
                        training_stats.report('MagnitudeG/max/' + str(x.shape[2]), CollectMagnitude(x, 'max'))
                    for x in CollectDiscriminatorFeatures(D.Model, real_img, real_c):
                        training_stats.report('MagnitudeD/avg/' + str(x.shape[2]), CollectMagnitude(x, 'avg'))
                        training_stats.report('MagnitudeD/max/' + str(x.shape[2]), CollectMagnitude(x, 'max'))
        
        if (ema_snapshot_ticks is not None) and (done or cur_tick % ema_snapshot_ticks == 0):
            ema_list = ema.get()
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(training_set_kwargs=dict(training_set_kwargs), eval_set_kwargs=dict(eval_set_kwargs), encoder_kwargs=dict(encoder_kwargs))
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False)
                fname = f'ema-snapshot-{cur_nimg//1000:09d}{ema_suffix}.pkl'
                if rank == 0:
                    print(f'Saving {fname} ... ', end='', flush=True)
                    atomic_pickle_dump(data, os.path.join(run_dir, 'ema', fname))
                    print('done')
                del data # conserve memory
        
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(
                G=G,
                D=D,
                training_set_kwargs=dict(training_set_kwargs),
                eval_set_kwargs=dict(eval_set_kwargs),
                encoder_kwargs=dict(encoder_kwargs),
                cur_nimg=cur_nimg,
                cur_tick=cur_tick,
                cur_tick_next=cur_tick + 1,
                done=done,
                batch_size=batch_size,
                kimg_per_tick=kimg_per_tick,
                network_snapshot_ticks=network_snapshot_ticks,
                image_snapshot_ticks=image_snapshot_ticks,
                ema_snapshot_ticks=ema_snapshot_ticks,
                **ema.state_dict(),
            )
            for phase in phases:
                snapshot_data[phase.name + '_opt_state'] = remap_optimizer_state_dict(phase.opt.state_dict(), 'cpu')
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(run_dir, 'snapshots', f'network-snapshot-{cur_nimg//1000:09d}.pkl')
            if rank == 0:
                atomic_pickle_dump(snapshot_data, snapshot_pkl)

        # Evaluate metrics.
        already_evaluated_resume_snapshot = (
            resume_run
            and resume_pkl is not None
            and snapshot_pkl is not None
            and os.path.abspath(snapshot_pkl) == os.path.abspath(resume_pkl)
        )
        if False and (snapshot_data is not None) and (len(metrics) > 0) and (not already_evaluated_resume_snapshot):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=ema_preview, encoder_kwargs=encoder_kwargs,
                    dataset_kwargs=eval_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = resume_time_offset + (timestamp - start_time)
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
