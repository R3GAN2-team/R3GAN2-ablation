#!/usr/bin/env python3
"""Asynchronous FID + magnitude evaluator for training run directories.

Scans run_dir/snapshots/network-snapshot-*.pkl, evaluates fid50k_full for the
requested EMA stds using exactly 50,000 generated images and the full real dataset, evaluates magnitude statistics exactly under the same tags
used by training_loop.py, and regenerates one compact TensorBoard event file
from JSONL source-of-truth files.

Outputs:
  fid50k_full_ema_<std>.jsonl   one file per requested EMA std
  magnitude.jsonl               one line per evaluated checkpoint
  samples/samples_ema_<std>.jsonl one file per evaluated sample EMA
  events.out.tfevents.*.async_eval  compact TensorBoard view
"""

import argparse
import copy
import glob
import json
import os
import re
import tempfile
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import dnnlib
import legacy
from metrics import metric_utils
from torch_utils import misc

#----------------------------------------------------------------------------

def _import_feat_collector():
    try:
        from training.feat_collector import CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude
        return CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude
    except Exception:
        pass
    try:
        from R3GAN.training.feat_collector import CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude
        return CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude
    except Exception:
        pass
    from feat_collector import CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude
    return CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude

#----------------------------------------------------------------------------

def _snapshot_kimg(path):
    match = re.search(r'network-snapshot-(\d+)\.pkl$', os.path.basename(str(path)))
    if match is not None:
        return int(match.group(1))
    match = re.fullmatch(r'(\d+)', os.path.basename(str(path)))
    if match is not None:
        return int(match.group(1))
    return None

#----------------------------------------------------------------------------

def _snapshot_relpath(run_dir, snapshot_path):
    return os.path.relpath(snapshot_path, run_dir).replace(os.sep, '/')

#----------------------------------------------------------------------------
def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

#----------------------------------------------------------------------------

def _append_jsonl_record(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'at') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(os.path.dirname(path))

#----------------------------------------------------------------------------

def _list_snapshots(run_dir):
    snapshot_dir = os.path.join(run_dir, 'snapshots')
    paths = []
    for path in sorted(glob.glob(os.path.join(snapshot_dir, 'network-snapshot-*.pkl'))):
        kimg = _snapshot_kimg(path)
        if kimg is not None:
            paths.append((kimg, path))
    paths.sort(key=lambda item: item[0])
    return paths

#----------------------------------------------------------------------------

def _resolve_start_kimg(checkpoint):
    if checkpoint is None or str(checkpoint).lower() in ['none', '']:
        return None
    kimg = _snapshot_kimg(checkpoint)
    if kimg is None:
        raise RuntimeError(f'Could not parse checkpoint kimg from: {checkpoint}')
    return kimg

#----------------------------------------------------------------------------

def _format_ema_std(ema_std):
    return f'{float(ema_std):.5f}'

#----------------------------------------------------------------------------

def _fid_jsonl_path(run_dir, ema_std):
    return os.path.join(run_dir, f'fid50k_full_ema_{_format_ema_std(ema_std)}.jsonl')

#----------------------------------------------------------------------------

def _magnitude_jsonl_path(run_dir):
    return os.path.join(run_dir, 'magnitude.jsonl')

#----------------------------------------------------------------------------

def _samples_dir(run_dir):
    return os.path.join(run_dir, 'samples')

#----------------------------------------------------------------------------

def _samples_jsonl_path(run_dir, ema_std):
    return os.path.join(_samples_dir(run_dir), f'samples_ema_{_format_ema_std(ema_std)}.jsonl')

#----------------------------------------------------------------------------

def _sample_png_path(run_dir, kimg, ema_std):
    return os.path.join(_samples_dir(run_dir), f'fakes{kimg:09d}_ema_{_format_ema_std(ema_std)}.png')

#----------------------------------------------------------------------------

def _reals_png_path(run_dir):
    return os.path.join(_samples_dir(run_dir), 'reals.png')

#----------------------------------------------------------------------------

def _sample_metric_name(ema_std):
    return f'samples_ema_{_format_ema_std(ema_std)}'

#----------------------------------------------------------------------------

def _latest_sample_kimg(run_dir, ema_std):
    return _latest_evaluated_kimg(_samples_jsonl_path(run_dir, ema_std), metric_name=_sample_metric_name(ema_std))

#----------------------------------------------------------------------------

def _fid_metric_name(ema_std):
    return f'fid50k_full_ema_{_format_ema_std(ema_std)}'

#----------------------------------------------------------------------------

def _fid_metric_tag(ema_std):
    return f'Metrics/fid50k_full_ema_{_format_ema_std(ema_std)}'

#----------------------------------------------------------------------------

def _async_event_globs(run_dir):
    # Delete old async_fid files too, so switching from the previous script does
    # not leave duplicate async FID curves around.
    return [
        os.path.join(run_dir, 'events.out.tfevents.*.async_eval'),
        os.path.join(run_dir, 'events.out.tfevents.*.async_fid'),
    ]

#----------------------------------------------------------------------------

def _latest_evaluated_kimg(jsonl_path, metric_name=None):
    if not os.path.isfile(jsonl_path):
        return None
    latest = None
    with open(jsonl_path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if metric_name is not None and obj.get('metric') != metric_name:
                continue
            kimg = _snapshot_kimg(obj.get('snapshot_pkl', ''))
            if kimg is None:
                continue
            latest = kimg if latest is None else max(latest, kimg)
    return latest

#----------------------------------------------------------------------------

def _read_fid_points_from_jsonl(run_dir, ema_std):
    metric_name = _fid_metric_name(ema_std)
    path = _fid_jsonl_path(run_dir, ema_std)
    points = {}
    if not os.path.isfile(path):
        return []
    with open(path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('metric') != metric_name:
                continue
            kimg = _snapshot_kimg(obj.get('snapshot_pkl', ''))
            if kimg is None:
                continue
            try:
                fid = float(obj['results']['fid50k_full'])
            except Exception:
                continue
            points[kimg] = fid # Last valid duplicate wins.
    return sorted(points.items())

#----------------------------------------------------------------------------

def _read_magnitude_points_from_jsonl(run_dir):
    path = _magnitude_jsonl_path(run_dir)
    points = {}
    if not os.path.isfile(path):
        return []
    with open(path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            kimg = _snapshot_kimg(obj.get('snapshot_pkl', ''))
            results = obj.get('results', None)
            if kimg is None or not isinstance(results, dict):
                continue
            clean = {}
            for key, value in results.items():
                try:
                    clean[str(key)] = float(value)
                except Exception:
                    pass
            if clean:
                points[kimg] = clean # Last valid duplicate wins.
    return sorted(points.items())

#----------------------------------------------------------------------------

def _rewrite_async_tensorboard(run_dir, ema_stds):
    # TensorBoard event files are append-only and SummaryWriter creates a new
    # file every process. Keep JSONLs as the source of truth, and regenerate a
    # single async event file from all JSONLs each time new data arrives.
    for pattern in _async_event_globs(run_dir):
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    writer = SummaryWriter(run_dir, filename_suffix='.async_eval')
    try:
        for ema_std in ema_stds:
            tag = _fid_metric_tag(ema_std)
            for kimg, fid in _read_fid_points_from_jsonl(run_dir, ema_std):
                writer.add_scalar(tag, fid, global_step=kimg)
        for kimg, results in _read_magnitude_points_from_jsonl(run_dir):
            for tag, value in sorted(results.items()):
                writer.add_scalar(tag, value, global_step=kimg)
        writer.flush()
    finally:
        writer.close()

#----------------------------------------------------------------------------


def _append_fid_result(run_dir, ema_std, result_dict, snapshot_pkl):
    rel_snapshot_pkl = _snapshot_relpath(run_dir, snapshot_pkl)
    path = _fid_jsonl_path(run_dir, ema_std)
    _append_jsonl_record(path, dict(result_dict, snapshot_pkl=rel_snapshot_pkl, timestamp=time.time()))
#----------------------------------------------------------------------------


def _append_magnitude_result(run_dir, results, snapshot_pkl, cur_nimg):
    rel_snapshot_pkl = _snapshot_relpath(run_dir, snapshot_pkl)
    record = dict(
        results=results,
        metric='magnitude',
        num_gpus=None,
        snapshot_pkl=rel_snapshot_pkl,
        cur_nimg=int(cur_nimg),
        timestamp=time.time(),
    )
    _append_jsonl_record(_magnitude_jsonl_path(run_dir), record)
#----------------------------------------------------------------------------


def _append_sample_result(run_dir, ema_std, snapshot_pkl, cur_nimg):
    rel_snapshot_pkl = _snapshot_relpath(run_dir, snapshot_pkl)
    rel_sample_png = os.path.relpath(_sample_png_path(run_dir, cur_nimg // 1000, ema_std), run_dir).replace(os.sep, '/')
    record = dict(
        results=dict(sample_png=rel_sample_png),
        metric=_sample_metric_name(ema_std),
        num_gpus=None,
        snapshot_pkl=rel_snapshot_pkl,
        cur_nimg=int(cur_nimg),
        timestamp=time.time(),
    )
    _append_jsonl_record(_samples_jsonl_path(run_dir, ema_std), record)
#----------------------------------------------------------------------------

def _load_snapshot_payload(snapshot_pkl):
    with dnnlib.util.open_url(snapshot_pkl) as f:
        data = legacy.load_network_pkl(f)
    if 'emas' not in data:
        raise KeyError(f'{snapshot_pkl} does not contain PowerFunctionEMA state key "emas"')
    stds = [float(x) for x in data.get('stds', [])]
    if len(stds) != len(data['emas']):
        raise RuntimeError(f'{snapshot_pkl} has mismatched stds ({len(stds)}) and emas ({len(data["emas"])}).')
    base_G = copy.deepcopy(data['G']).eval().requires_grad_(False)
    D = copy.deepcopy(data['D']).eval().requires_grad_(False) if 'D' in data else None
    eval_set_kwargs = dnnlib.EasyDict(data['eval_set_kwargs'])
    training_set_kwargs = dnnlib.EasyDict(data.get('training_set_kwargs', data['eval_set_kwargs']))
    encoder_kwargs = dnnlib.EasyDict(data['encoder_kwargs'])
    cur_nimg = int(data.get('cur_nimg', _snapshot_kimg(snapshot_pkl) * 1000))
    ema_state_by_std = {_format_ema_std(std): state for std, state in zip(stds, data['emas'])}
    return base_G, D, ema_state_by_std, stds, training_set_kwargs, eval_set_kwargs, encoder_kwargs, cur_nimg

#----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        label_groups = dict()
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, grid_size):
    lo, hi = [0, 255]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    import PIL.Image
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    tmp = f'{fname}.tmp.{os.getpid()}'
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(tmp, format='PNG')
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(tmp, format='PNG')
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, fname)
    _fsync_dir(os.path.dirname(fname))

#----------------------------------------------------------------------------

def _ensure_reals_png(run_dir, eval_set_kwargs):
    os.makedirs(_samples_dir(run_dir), exist_ok=True)
    reals_path = _reals_png_path(run_dir)
    if os.path.isfile(reals_path):
        return
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs)
    grid_size, images, _labels = setup_snapshot_image_grid(training_set=eval_set)
    save_image_grid(images, reals_path, grid_size=grid_size)

#----------------------------------------------------------------------------

def _generate_sample_png(base_G, ema_state_by_std, ema_std, eval_set_kwargs, encoder_kwargs, run_dir, cur_nimg, device, batch_size, seed):
    sample_std_key = _format_ema_std(ema_std)
    G = copy.deepcopy(base_G).eval().requires_grad_(False).to(device)
    G.load_state_dict(ema_state_by_std[sample_std_key], strict=True)
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs)
    grid_size, _images, labels = setup_snapshot_image_grid(training_set=eval_set)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    grid_z_all = torch.randn([labels.shape[0], G.z_dim], device=device, generator=gen)
    grid_c_all = torch.from_numpy(labels).to(device)
    chunk_size = _auto_eval_chunk_size(batch_size)
    images = []
    with torch.no_grad():
        for z, c in zip(grid_z_all.split(chunk_size), grid_c_all.split(chunk_size)):
            images.append(encoder.decode(G(z, c)).cpu())
    images = torch.cat(images).to(torch.float32).numpy()
    os.makedirs(_samples_dir(run_dir), exist_ok=True)
    save_image_grid(images, _sample_png_path(run_dir, cur_nimg // 1000, ema_std), grid_size=grid_size)
    del G, encoder, eval_set, grid_z_all, grid_c_all

#----------------------------------------------------------------------------

def _load_training_options(run_dir):
    path = os.path.join(run_dir, 'training_options.json')
    if not os.path.isfile(path):
        return dnnlib.EasyDict()
    with open(path, 'rt') as f:
        return dnnlib.EasyDict(json.load(f))

#----------------------------------------------------------------------------
def _derive_eval_runtime_options(run_dir, eval_num_gpus):
    opts = _load_training_options(run_dir)
    if 'batch_size' not in opts:
        raise RuntimeError(f'Could not find batch_size in {os.path.join(run_dir, "training_options.json")}')

    batch_size = int(opts.batch_size)
    if batch_size < 1:
        raise RuntimeError(f'Invalid batch_size in training_options.json: {batch_size}')
    if batch_size % int(eval_num_gpus) != 0:
        raise RuntimeError(f'Training batch_size={batch_size} is not divisible by eval --gpus={eval_num_gpus}')

    data_loader_kwargs = dnnlib.EasyDict(opts.get('data_loader_kwargs', {}))
    data_loader_kwargs.setdefault('pin_memory', True)
    # Keep DataLoader worker behavior tied to training_options.json. If an old
    # run lacks the field, use PyTorch's single-process loading rather than
    # silently inventing a large worker count.
    data_loader_kwargs.setdefault('num_workers', 0)
    if int(data_loader_kwargs.get('num_workers', 0)) > 0:
        data_loader_kwargs.setdefault('prefetch_factor', 2)
    else:
        data_loader_kwargs.pop('prefetch_factor', None)

    return dnnlib.EasyDict(
        batch_size=batch_size,
        local_batch_size=batch_size // int(eval_num_gpus),
        data_loader_kwargs=data_loader_kwargs,
    )

#----------------------------------------------------------------------------
# Fast FID: GPU covariance accumulation + torch.linalg.eigh.
# This is the same path validated against the reference CPU-cov/SciPy FID.

_DETECTOR_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
_DETECTOR_KWARGS = dict(return_features=True)

def _format_time(seconds):
    return dnnlib.util.format_time(float(seconds))

#----------------------------------------------------------------------------

def _fid_data_loader_kwargs(num_workers):
    kwargs = dict(pin_memory=True, num_workers=int(num_workers))
    if int(num_workers) > 0:
        kwargs.update(prefetch_factor=2)
    return kwargs

#----------------------------------------------------------------------------

def _fid_barrier(fid_opts):
    if fid_opts.num_gpus > 1:
        torch.distributed.barrier()

#----------------------------------------------------------------------------

def _sync_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

#----------------------------------------------------------------------------

def _local_take_for_global_append(global_done, local_batch, global_max_items, num_gpus, rank):
    remaining = int(global_max_items) - int(global_done)
    if remaining <= 0:
        return 0, 0
    global_batch = int(local_batch) * int(num_gpus)
    global_take = min(remaining, global_batch)
    full_rows = global_take // int(num_gpus)
    remainder = global_take % int(num_gpus)
    local_take = full_rows + (1 if int(rank) < remainder else 0)
    local_take = min(int(local_batch), int(local_take))
    return local_take, global_take

#----------------------------------------------------------------------------

class GpuCovStats:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device
        self.num_items = 0
        self.raw_sum = None
        self.raw_cov = None

    def _init(self, num_features):
        self.raw_sum = torch.zeros([num_features], dtype=self.dtype, device=self.device)
        self.raw_cov = torch.zeros([num_features, num_features], dtype=self.dtype, device=self.device)

    def append_local(self, features, take):
        if take <= 0:
            return
        x = features[:take]
        if self.raw_sum is None:
            self._init(x.shape[1])
        x = x.to(self.dtype)
        self.num_items += int(x.shape[0])
        self.raw_sum.add_(x.sum(dim=0))
        self.raw_cov.add_(x.t().matmul(x))

    def finalize(self, fid_opts, device):
        n = torch.tensor([self.num_items], dtype=torch.float64, device=device)
        if fid_opts.num_gpus > 1:
            torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.raw_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.raw_cov, op=torch.distributed.ReduceOp.SUM)
        mean = self.raw_sum / n
        cov = self.raw_cov / n - torch.outer(mean, mean)
        cov = (cov + cov.t()) * 0.5
        return mean, cov, int(round(float(n.cpu()[0])))

#----------------------------------------------------------------------------

def _get_real_stats_reference_cache(dataset_kwargs, fid_opts, rank, device):
    dataset_kwargs = dnnlib.EasyDict(copy.deepcopy(dict(dataset_kwargs)))
    dataset_kwargs.update(max_size=None, xflip=False)
    opts = metric_utils.MetricOptions(
        G=None,
        dataset_kwargs=dataset_kwargs,
        num_gpus=fid_opts.num_gpus,
        rank=rank,
        device=device,
        cache=not fid_opts.no_cache,
    )
    stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=_DETECTOR_URL,
        detector_kwargs=_DETECTOR_KWARGS,
        rel_lo=0,
        rel_hi=0,
        capture_mean_cov=True,
        max_items=None,
        batch_size=fid_opts.batch_gpu,
        data_loader_kwargs=_fid_data_loader_kwargs(fid_opts.data_workers),
    )
    mu, cov = stats.get_mean_cov()
    return (
        torch.from_numpy(mu).to(device=device, dtype=torch.float64),
        torch.from_numpy(cov).to(device=device, dtype=torch.float64),
    )

#----------------------------------------------------------------------------

def _compute_gen_stats_gpu_cov(G_in, dataset_kwargs, encoder_kwargs, fid_opts, rank, device):
    dataset_kwargs = dnnlib.EasyDict(copy.deepcopy(dict(dataset_kwargs)))
    dataset_kwargs.update(max_size=None, xflip=False)
    opts = metric_utils.MetricOptions(
        G=G_in,
        encoder_kwargs=encoder_kwargs,
        dataset_kwargs=dataset_kwargs,
        num_gpus=fid_opts.num_gpus,
        rank=rank,
        device=device,
        cache=not fid_opts.no_cache,
    )
    G = copy.deepcopy(G_in).eval().requires_grad_(False).to(device)
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    detector = metric_utils.get_feature_detector(
        url=_DETECTOR_URL,
        device=device,
        num_gpus=fid_opts.num_gpus,
        rank=rank,
        verbose=(rank == 0),
    )
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=fid_opts.batch_gen)
    dtype = torch.float64 if fid_opts.gpu_accum_dtype == 'float64' else torch.float32
    stats = GpuCovStats(dtype=dtype, device=device)
    global_done = 0
    with torch.no_grad():
        while global_done < fid_opts.max_items:
            images = []
            for _i in range(fid_opts.batch_gpu // fid_opts.batch_gen):
                z = torch.randn([fid_opts.batch_gen, G.z_dim], device=device)
                img = encoder.decode(G(z, next(c_iter)))
                images.append(img)
            images = torch.cat(images)
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            features = detector(images, **_DETECTOR_KWARGS)
            local_take, global_take = _local_take_for_global_append(
                global_done,
                features.shape[0],
                fid_opts.max_items,
                fid_opts.num_gpus,
                rank,
            )
            stats.append_local(features, local_take)
            global_done += global_take
    mean, cov, n = stats.finalize(fid_opts=fid_opts, device=device)
    if rank == 0:
        print(f'    generated stats items={n}', flush=True)
    del G, encoder
    return mean.to(torch.float64), cov.to(torch.float64)

#----------------------------------------------------------------------------

def _fid_torch_eigh(mu_gen, cov_gen, mu_real, cov_real, eps=0.0):
    mu_gen = mu_gen.to(torch.float64)
    mu_real = mu_real.to(torch.float64)
    cov_gen = ((cov_gen + cov_gen.t()) * 0.5).to(torch.float64)
    cov_real = ((cov_real + cov_real.t()) * 0.5).to(torch.float64)
    eig_r, vec_r = torch.linalg.eigh(cov_real)
    eig_r = eig_r.clamp_min(float(eps))
    sqrt_r = (vec_r * eig_r.sqrt().unsqueeze(0)) @ vec_r.t()
    middle = sqrt_r @ cov_gen @ sqrt_r
    middle = (middle + middle.t()) * 0.5
    eig_m = torch.linalg.eigvalsh(middle).clamp_min(0)
    trace_sqrt = eig_m.sqrt().sum()
    fid = (
        (mu_gen - mu_real).square().sum()
        + torch.trace(cov_gen)
        + torch.trace(cov_real)
        - 2 * trace_sqrt
    )
    return float(fid.detach().cpu())

#----------------------------------------------------------------------------

def _time_fid_phase(label, fn, fid_opts, rank, device):
    _fid_barrier(fid_opts)
    _sync_cuda(device)
    t0 = time.time()
    out = fn()
    _sync_cuda(device)
    elapsed = time.time() - t0
    if fid_opts.num_gpus > 1:
        t = torch.tensor([elapsed], dtype=torch.float64, device=device)
        gathered = [torch.zeros_like(t) for _ in range(fid_opts.num_gpus)]
        torch.distributed.all_gather(gathered, t)
        elapsed_all = [float(x.cpu()[0]) for x in gathered]
    else:
        elapsed_all = [elapsed]
    if rank == 0:
        print(
            f'    {label}: min={_format_time(min(elapsed_all))} '
            f'mean={_format_time(sum(elapsed_all) / len(elapsed_all))} '
            f'max={_format_time(max(elapsed_all))}',
            flush=True,
        )
    return out

#----------------------------------------------------------------------------

def _compute_fast_fid(G, dataset_kwargs, encoder_kwargs, fid_opts, rank, device):
    mu_real, cov_real = _time_fid_phase(
        'real stats cache/load',
        lambda: _get_real_stats_reference_cache(dataset_kwargs, fid_opts, rank, device),
        fid_opts,
        rank,
        device,
    )
    mu_gen, cov_gen = _time_fid_phase(
        f'generated stats GPU-cov ({fid_opts.gpu_accum_dtype})',
        lambda: _compute_gen_stats_gpu_cov(G, dataset_kwargs, encoder_kwargs, fid_opts, rank, device),
        fid_opts,
        rank,
        device,
    )

    def eig_fn():
        if rank == 0:
            return _fid_torch_eigh(mu_gen, cov_gen, mu_real, cov_real, eps=fid_opts.eigh_eps)
        return float('nan')

    fid = _time_fid_phase('torch.linalg.eigh FID', eig_fn, fid_opts, rank, device)
    if fid_opts.num_gpus > 1:
        t = torch.tensor([fid], dtype=torch.float64, device=device)
        torch.distributed.broadcast(tensor=t, src=0)
        fid = float(t.cpu()[0])
    return float(fid)

#----------------------------------------------------------------------------

def _build_eval_plan(run_dir, checkpoint, ema_stds, eval_magnitude, eval_samples):
    snapshots = _list_snapshots(run_dir)
    if len(snapshots) == 0:
        raise RuntimeError(f'No network snapshots found in {os.path.join(run_dir, "snapshots")}')

    requested_start = _resolve_start_kimg(checkpoint)
    latest_fid_done = {}
    for ema_std in ema_stds:
        latest_fid_done[_format_ema_std(ema_std)] = _latest_evaluated_kimg(
            _fid_jsonl_path(run_dir, ema_std), metric_name=_fid_metric_name(ema_std))
    latest_magnitude_done = _latest_evaluated_kimg(_magnitude_jsonl_path(run_dir), metric_name='magnitude') if eval_magnitude else None
    latest_sample_done = {}
    if eval_samples:
        for ema_std in ema_stds:
            latest_sample_done[_format_ema_std(ema_std)] = _latest_sample_kimg(run_dir, ema_std)

    plan = []
    for kimg, path in snapshots:
        if requested_start is not None and kimg < requested_start:
            continue
        pending_ema_stds = []
        for ema_std in ema_stds:
            std_key = _format_ema_std(ema_std)
            done_kimg = latest_fid_done[std_key]
            if done_kimg is None or kimg > done_kimg:
                pending_ema_stds.append(float(ema_std))
        pending_magnitude = bool(eval_magnitude and (latest_magnitude_done is None or kimg > latest_magnitude_done))
        pending_sample_stds = []
        if eval_samples:
            for ema_std in ema_stds:
                std_key = _format_ema_std(ema_std)
                done_kimg = latest_sample_done[std_key]
                if done_kimg is None or kimg > done_kimg:
                    pending_sample_stds.append(float(ema_std))
        if pending_ema_stds or pending_magnitude or pending_sample_stds:
            plan.append((path, pending_ema_stds, pending_magnitude, pending_sample_stds))

    return plan, requested_start, latest_fid_done, latest_magnitude_done, latest_sample_done

#----------------------------------------------------------------------------

def _auto_eval_chunk_size(local_batch_size):
    # Internal memory-only chunking. This must not change the sampled latents,
    # labels, or magnitude semantics; it only avoids requiring the eval GPU to
    # hold the entire training global/local batch in one forward pass.
    return max(1, min(int(local_batch_size), 16))

#----------------------------------------------------------------------------

def _init_distributed(rank, num_gpus, temp_dir):
    if num_gpus <= 1:
        return
    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
    if os.name == 'nt':
        init_method = 'file:///' + init_file.replace('\\', '/')
        backend = 'gloo'
    else:
        init_method = f'file://{init_file}'
        backend = 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=num_gpus)

#----------------------------------------------------------------------------

def _reduce_scalar(value, device, num_gpus):
    if isinstance(value, torch.Tensor):
        value = float(value.detach().to(torch.float32).mean().cpu())
    else:
        value = float(value)
    if num_gpus > 1:
        t = torch.tensor([value], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        value = float((t / num_gpus).cpu()[0])
    return value

#----------------------------------------------------------------------------

def _compute_magnitude(base_G, D, training_set_kwargs, encoder_kwargs, rank, num_gpus, device, local_batch_size, seed, data_loader_kwargs):
    if D is None:
        raise RuntimeError('Checkpoint does not contain D; cannot evaluate discriminator magnitudes.')
    CollectGeneratorFeatures, CollectDiscriminatorFeatures, CollectMagnitude = _import_feat_collector()

    G = copy.deepcopy(base_G).eval().requires_grad_(False).to(device)
    D = copy.deepcopy(D).eval().requires_grad_(False).to(device)
    G_model = G.Model if hasattr(G, 'Model') else G
    D_model = D.Model if hasattr(D, 'Model') else D

    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=seed)
    loader = torch.utils.data.DataLoader(dataset=training_set, sampler=sampler, batch_size=local_batch_size, **data_loader_kwargs)
    images, labels = next(iter(loader))

    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    gen_z_all = torch.randn([local_batch_size, G.z_dim], device=device)
    chunk_size = _auto_eval_chunk_size(local_batch_size)

    local_sums = {}
    local_counts = {}
    with torch.no_grad():
        for img_i, real_c_i, gen_z_i in zip(images.split(chunk_size), labels.split(chunk_size), gen_z_all.split(chunk_size)):
            real_img_i = encoder.encode_latents(img_i.to(device))
            real_c_i = real_c_i.to(device)
            chunk_n = int(real_c_i.shape[0])

            for x in CollectGeneratorFeatures(G_model, gen_z_i, real_c_i):
                key = 'MagnitudeG/avg/' + str(x.shape[2])
                local_sums[key] = local_sums.get(key, 0.0) + float(CollectMagnitude(x, 'avg')) * chunk_n
                local_counts[key] = local_counts.get(key, 0) + chunk_n

                key = 'MagnitudeG/max/' + str(x.shape[2])
                local_sums[key] = local_sums.get(key, 0.0) + float(CollectMagnitude(x, 'max')) * chunk_n
                local_counts[key] = local_counts.get(key, 0) + chunk_n

            for x in CollectDiscriminatorFeatures(D_model, real_img_i, real_c_i):
                key = 'MagnitudeD/avg/' + str(x.shape[2])
                local_sums[key] = local_sums.get(key, 0.0) + float(CollectMagnitude(x, 'avg')) * chunk_n
                local_counts[key] = local_counts.get(key, 0) + chunk_n

                key = 'MagnitudeD/max/' + str(x.shape[2])
                local_sums[key] = local_sums.get(key, 0.0) + float(CollectMagnitude(x, 'max')) * chunk_n
                local_counts[key] = local_counts.get(key, 0) + chunk_n

            del real_img_i, real_c_i

    local_results = {key: local_sums[key] / local_counts[key] for key in sorted(local_sums)}
    results = {key: _reduce_scalar(value, device=device, num_gpus=num_gpus) for key, value in sorted(local_results.items())}
    del G, D, training_set, loader, images, labels, gen_z_all
    return results

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir, eval_plan):
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    _init_distributed(rank=rank, num_gpus=args.num_gpus, temp_dir=temp_dir)

    torch.backends.cudnn.benchmark = not args.nobench
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        torch.backends.cudnn.rnn.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
    except Exception:
        pass

    runtime_opts = _derive_eval_runtime_options(args.run_dir, args.num_gpus)
    data_loader_kwargs = runtime_opts.data_loader_kwargs
    fid_opts = dnnlib.EasyDict(
        num_gpus=args.num_gpus,
        no_cache=args.no_cache,
        batch_gpu=args.fid_batch_gpu,
        batch_gen=args.fid_batch_gen,
        data_workers=args.fid_data_workers,
        max_items=50000,  # Hard-coded: fid50k_full always uses exactly 50,000 generated images.
        gpu_accum_dtype=args.fid_gpu_accum_dtype,
        eigh_eps=args.fid_eigh_eps,
    )
    if rank == 0:
        print(f'Using training batch_size={runtime_opts.batch_size}, local_batch_size={runtime_opts.local_batch_size}', flush=True)
        print(f'Using DataLoader num_workers={int(data_loader_kwargs.get("num_workers", 0))}', flush=True)
        print(f'Using fid50k_full: 50000 generated images, full real dataset; batch_gpu={fid_opts.batch_gpu}, batch_gen={fid_opts.batch_gen}, data_workers={fid_opts.data_workers}, accum={fid_opts.gpu_accum_dtype}', flush=True)

    try:
        for snapshot_pkl, pending_ema_stds, pending_magnitude, pending_sample_stds in eval_plan:
            if rank == 0:
                fid_msg = ', '.join(_format_ema_std(x) for x in pending_ema_stds) if pending_ema_stds else 'none'
                mag_msg = 'yes' if pending_magnitude else 'no'
                samp_msg = ', '.join(_format_ema_std(x) for x in pending_sample_stds) if pending_sample_stds else 'none'
                print(f'Evaluating {os.path.basename(snapshot_pkl)} [FID EMA stds: {fid_msg}; magnitude: {mag_msg}; sample EMA stds: {samp_msg}]...', flush=True)

            np.random.seed(args.seed * args.num_gpus + rank)
            torch.manual_seed(args.seed * args.num_gpus + rank)

            base_G, D, ema_state_by_std, stds, training_set_kwargs, eval_set_kwargs, encoder_kwargs, cur_nimg = _load_snapshot_payload(snapshot_pkl)
            step = cur_nimg // 1000

            if rank == 0 and pending_sample_stds:
                print('  Ensuring real image grid...', flush=True)
                _ensure_reals_png(args.run_dir, eval_set_kwargs)
                print(f'  Real image grid ready: {os.path.relpath(_reals_png_path(args.run_dir), args.run_dir)}', flush=True)
                for ema_std in pending_sample_stds:
                    print(f'  Generating samples using EMA std={_format_ema_std(ema_std)}...', flush=True)
                    _generate_sample_png(
                        base_G=base_G,
                        ema_state_by_std=ema_state_by_std,
                        ema_std=ema_std,
                        eval_set_kwargs=eval_set_kwargs,
                        encoder_kwargs=encoder_kwargs,
                        run_dir=args.run_dir,
                        cur_nimg=cur_nimg,
                        device=device,
                        batch_size=runtime_opts.local_batch_size,
                        seed=args.seed,
                    )
                    _append_sample_result(args.run_dir, ema_std, snapshot_pkl, cur_nimg)
                    print(f'  Sample image written: {os.path.relpath(_sample_png_path(args.run_dir, cur_nimg // 1000, ema_std), args.run_dir)}', flush=True)
                    print(f'  Wrote {os.path.relpath(_samples_jsonl_path(args.run_dir, ema_std), args.run_dir)}', flush=True)
            if args.num_gpus > 1:
                torch.distributed.barrier()

            if pending_magnitude:
                if rank == 0:
                    print(f'  Magnitude using checkpoint G/D; checkpoint kimg={cur_nimg / 1000:.1f}; local_batch_size={runtime_opts.local_batch_size}; chunk_size={_auto_eval_chunk_size(runtime_opts.local_batch_size)}', flush=True)
                np.random.seed((args.seed + 1) * args.num_gpus + rank)
                torch.manual_seed((args.seed + 1) * args.num_gpus + rank)
                magnitude_results = _compute_magnitude(
                    base_G=base_G,
                    D=D,
                    training_set_kwargs=training_set_kwargs,
                    encoder_kwargs=encoder_kwargs,
                    rank=rank,
                    num_gpus=args.num_gpus,
                    device=device,
                    local_batch_size=runtime_opts.local_batch_size,
                    seed=args.seed,
                    data_loader_kwargs=data_loader_kwargs,
                )
                if rank == 0:
                    _append_magnitude_result(args.run_dir, magnitude_results, snapshot_pkl, cur_nimg)
                    print(f'  Wrote {os.path.relpath(_magnitude_jsonl_path(args.run_dir), args.run_dir)}', flush=True)
                    print(f'  Magnitude tags written: {len(magnitude_results)}', flush=True)
                if args.num_gpus > 1:
                    torch.distributed.barrier()

            for ema_std in pending_ema_stds:
                std_key = _format_ema_std(ema_std)
                if std_key not in ema_state_by_std:
                    available = ', '.join(sorted(ema_state_by_std.keys()))
                    raise KeyError(f'{snapshot_pkl} does not contain requested EMA std {std_key}. Available stds: {available}')

                G = copy.deepcopy(base_G).eval().requires_grad_(False)
                G.load_state_dict(ema_state_by_std[std_key], strict=True)

                if rank == 0:
                    print(f'  FID using EMA std={std_key}; checkpoint kimg={cur_nimg / 1000:.1f}', flush=True)

                metric_name = _fid_metric_name(ema_std)
                t0 = time.time()
                fid_value = _compute_fast_fid(
                    G=G,
                    dataset_kwargs=eval_set_kwargs,
                    encoder_kwargs=encoder_kwargs,
                    fid_opts=fid_opts,
                    rank=rank,
                    device=device,
                )
                total_time = time.time() - t0
                result_dict = dnnlib.EasyDict(
                    results=dnnlib.EasyDict(fid50k_full=float(fid_value)),
                    metric=metric_name,
                    total_time=float(total_time),
                    total_time_str=_format_time(total_time),
                    num_gpus=args.num_gpus,
                    method=f'gpu_cov_{fid_opts.gpu_accum_dtype}_torch_eigh',
                )

                if rank == 0:
                    _append_fid_result(args.run_dir, ema_std, result_dict, snapshot_pkl)
                    print(f'  Wrote {os.path.relpath(_fid_jsonl_path(args.run_dir, ema_std), args.run_dir)}', flush=True)
                    print(f'  {metric_name} = {result_dict.results.fid50k_full:.6g}  time = {result_dict.total_time_str}', flush=True)

                del G
                torch.cuda.empty_cache()
                if args.num_gpus > 1:
                    torch.distributed.barrier()

            del base_G, D
            torch.cuda.empty_cache()
            if args.num_gpus > 1:
                torch.distributed.barrier()

    finally:
        if args.num_gpus > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

#----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Asynchronously evaluate fid50k_full, magnitude statistics, and sample images for checkpoints in a training run directory.')
    parser.add_argument('--run-dir', required=True, help='Training run directory containing snapshots/.')
    parser.add_argument('--gpus', dest='num_gpus', type=int, required=True, help='Number of GPUs to use.')
    parser.add_argument('--checkpoint', default=None, help='First checkpoint filename/path/kimg to consider. Default: all checkpoints, continuing after each source JSONL if present.')
    parser.add_argument('--ema-stds', type=float, nargs='+', required=True, help='List of EMA stds to evaluate for FID, e.g. --ema-stds 0.05 0.1 0.2 0.3')
    parser.add_argument('--no-magnitude', action='store_true', help='Disable magnitude evaluation.')
    parser.add_argument('--no-samples', action='store_true', help='Disable sample image generation.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for generated FID samples, magnitude batches, and sample image grids; reset for each checkpoint.')
    parser.add_argument('--fid-batch-gpu', type=int, default=128, help='FID detector-feature batch size per GPU. Default: 128.')
    parser.add_argument('--fid-batch-gen', type=int, default=None, help='FID generator microbatch per GPU. Default: same as --fid-batch-gpu.')
    parser.add_argument('--fid-data-workers', type=int, default=1, help='DataLoader workers per rank for uncached real FID stats. Default: 1.')
    parser.add_argument('--fid-gpu-accum-dtype', choices=['float64', 'float32'], default='float64', help='GPU covariance accumulation dtype. float64 matches the reference FID closely.')
    parser.add_argument('--fid-eigh-eps', type=float, default=0.0, help='Eigenvalue floor for the GPU eigensolver. Default: 0.')
    parser.add_argument('--no-cache', action='store_true', help='Disable real feature stats cache.')
    parser.add_argument('--nobench', action='store_true', help='Disable cuDNN benchmarking.')
    args = parser.parse_args()

    args.run_dir = os.path.abspath(args.run_dir)
    if args.fid_batch_gen is None:
        args.fid_batch_gen = args.fid_batch_gpu
    if args.fid_batch_gpu < 1 or args.fid_batch_gen < 1:
        raise RuntimeError('--fid-batch-gpu and --fid-batch-gen must be positive')
    if args.fid_batch_gpu % args.fid_batch_gen != 0:
        raise RuntimeError('--fid-batch-gpu must be divisible by --fid-batch-gen')
    if args.fid_data_workers < 0:
        raise RuntimeError('--fid-data-workers must be nonnegative')
    if args.num_gpus < 1:
        raise RuntimeError('--gpus must be at least 1')
    if len(args.ema_stds) == 0:
        raise RuntimeError('--ema-stds must contain at least one value')
    seen = set()
    deduped = []
    for ema_std in args.ema_stds:
        key = _format_ema_std(ema_std)
        if key not in seen:
            seen.add(key)
            deduped.append(float(ema_std))
    args.ema_stds = deduped
    return args

#----------------------------------------------------------------------------

def main():
    args = parse_args()
    eval_plan, requested_start, latest_fid_done, latest_magnitude_done, latest_sample_done = _build_eval_plan(
        args.run_dir, args.checkpoint, args.ema_stds, eval_magnitude=(not args.no_magnitude), eval_samples=(not args.no_samples))

    # Compact TensorBoard from already completed JSONLs at startup.
    _rewrite_async_tensorboard(args.run_dir, args.ema_stds)

    print(f'Run directory:             {args.run_dir}')
    print(f'Number of GPUs:            {args.num_gpus}')
    print(f'Requested checkpoint:      {args.checkpoint if args.checkpoint is not None else "none"}')
    print(f'Requested start kimg:      {requested_start if requested_start is not None else "none"}')
    print(f'Requested EMA stds:        {", ".join(_format_ema_std(x) for x in args.ema_stds)}')
    for ema_std in args.ema_stds:
        std_key = _format_ema_std(ema_std)
        latest = latest_fid_done[std_key]
        print(f'Latest FID done [{std_key}]: {latest if latest is not None else "none"}')
    if not args.no_magnitude:
        print(f'Latest magnitude done:     {latest_magnitude_done if latest_magnitude_done is not None else "none"}')
    if not args.no_samples:
        for ema_std in args.ema_stds:
            std_key = _format_ema_std(ema_std)
            latest = latest_sample_done[std_key]
            print(f'Latest sample done [{std_key}]: {latest if latest is not None else "none"}')
    pending_fids = sum(len(stds) for _, stds, _, _ in eval_plan)
    pending_mags = sum(int(flag) for _, _, flag, _ in eval_plan)
    pending_samps = sum(len(stds) for _, _, _, stds in eval_plan)
    print(f'Pending checkpoints:       {len(eval_plan)}')
    print(f'Pending FID jobs:          {pending_fids}')
    if not args.no_magnitude:
        print(f'Pending magnitude jobs:    {pending_mags}')
    if not args.no_samples:
        print(f'Pending sample jobs:       {pending_samps}')
    if len(eval_plan) > 0:
        print(f'First pending:             {os.path.basename(eval_plan[0][0])}')
        print(f'Last pending:              {os.path.basename(eval_plan[-1][0])}')
    print()

    if len(eval_plan) == 0:
        print('Nothing to evaluate.')
        _rewrite_async_tensorboard(args.run_dir, args.ema_stds)
        return

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:
                subprocess_fn(rank=0, args=args, temp_dir=temp_dir, eval_plan=eval_plan)
            else:
                torch.multiprocessing.spawn(
                    fn=subprocess_fn,
                    args=(args, temp_dir, eval_plan),
                    nprocs=args.num_gpus,
                )
    finally:
        # JSONLs and PNGs are written immediately after each completed result.
        # TensorBoard is compacted/rebuilt in one pass from those JSONLs.
        _rewrite_async_tensorboard(args.run_dir, args.ema_stds)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------
