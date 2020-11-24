import time
from run_nerf import raw2outputs
import torch
import os.path
import imageio


from tqdm import tqdm
from typing import Any, Callable, Tuple
from run_nerf_helpers import NeRF, get_embedder, to8b
from load_blender import load_blender_data

from run_nerf import render, run_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nerf(ckpt_path: str,
              netdepth: int,
              netwidth: int,
              netdepth_fine: int,
              netwidth_fine: int,
              use_viewdirs,
              i_embed,
              multires,
              multires_views,
              N_importance
              ) -> Tuple[NeRF, NeRF, Any]:
    # TODO(mackopes): add type hints
    netchunk = 65536  # TODO(mackopes): remove
    assert os.path.isfile(ckpt_path), "checkpoint path file does not exist"

    embed_fn, input_ch = get_embedder(multires, i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(multires_views, i_embed)
    output_ch = 5 if N_importance > 0 else 4

    model = NeRF(D=netdepth, W=netwidth,
                 input_ch=input_ch, output_ch=output_ch,
                 input_ch_views=input_ch_views, use_viewdirs=use_viewdirs).to(device)

    model_fine = None
    if N_importance > 0:
        model_fine = NeRF(D=netdepth_fine, W=netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch,
                          input_ch_views=input_ch_views, use_viewdirs=use_viewdirs).to(device)

    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt['network_fn_state_dict'])
    if model_fine is not None:
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn, embed_fn=embed_fn,
                           embeddirs_fn=embeddirs_fn, netchunk=netchunk)

    return model, model_fine, network_query_fn


def get_z_vals(N_rays, N_samples, near, far):
    # TODO(mackopes): add lindisp arg

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    return z_vals

    # TODO(mackopes): add perturb


def get_depth_map(rays_o,
                  rays_d,
                  near: float,
                  far: float,
                  viewdirs,
                  N_samples,
                  network_fn: NeRF,
                  network_query_fn: Callable[..., Any],
                  raw_noise_std: float,
                  white_bkgd: bool
                  ):
    assert rays_o.shape[0] == rays_d.shape[0]

    N_rays = rays_o.shape[0]
    z_vals = get_z_vals(N_rays, N_samples, near, far)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    return depth_map


def train_depth_pred(
    datadir,
    dataset_type="blender"
):
    raw_noise_std = 0.0
    white_bkgd = True
    half_res = True
    testskip = 8

    if dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            datadir, half_res, testskip)
        print('Loaded blender', images.shape,
              poses.shape, hwf, datadir)

        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if white_bkgd:
            images = images[..., :3] * \
                images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        print('Unknown dataset type', dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    ckpt_path = "logs/blender_paper_lego/200000.tar"
    netdepth = 8
    netdepth_fine = 8
    netwidth = 256
    netwidth_fine = 256
    i_embed = 0
    use_viewdirs = True
    multires = 10
    multires_views = 4
    N_importance = 128
    lindisp = False
    N_samples = 64

    render_factor = 0
    chunk = 32768

    model, model_fine, network_query_fn = load_nerf(
        ckpt_path, netdepth, netwidth, netdepth_fine, netwidth_fine,
        use_viewdirs, i_embed, multires, multires_views, N_importance)

    render_kwargs = {
        'network_query_fn': network_query_fn,
        'perturb': False,
        'N_importance': N_importance,
        'network_fine': model_fine,
        'N_samples': N_samples,
        'network_fn': model,
        'use_viewdirs': use_viewdirs,
        'white_bkgd': white_bkgd,
        # 'raw_noise_std': raw_noise_std,
        'lindisp': lindisp,
        'ndc': False,
        'raw_noise_std': 0.,
        'near': near,
        'far': far
    }

    poses = torch.Tensor(poses).to(device)
    poses = poses[:1]
    print(f"moved render_poses to {device}")

    print('RENDER ONLY')
    with torch.no_grad():
        testsavedir = "depth_pred_test"
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', poses.shape)

        rgbs, disps, depths = get_depths(hwf, poses, chunk, render_kwargs)

        print('Done rendering', testsavedir)
        for i, (rgb, disp, depth) in enumerate(zip(rgbs, disps, depths)):
            disp = to8b(disp)
            depth = to8b(depth)
            rgb = to8b(rgb)
            filename = os.path.join(testsavedir, f'rgb_{i:03d}.png')
            imageio.imwrite(filename, rgb)
            filename = os.path.join(testsavedir, f'disp_{i:03d}.png')
            imageio.imwrite(filename, disp)
            filename = os.path.join(testsavedir, f'depth_{i:03d}.png')
            imageio.imwrite(filename, depth)
        # imageio.mimwrite(os.path.join(
        #     testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)


def build_depth_dataset(dataset_dir):
    pass


def get_depths(hwf, poses, chunk, render_kwargs):
    rgbs = []
    disps = []
    depths = []

    H, W, focal = hwf

    t = time.time()
    for i, c2w in enumerate(tqdm(poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, all_ret = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(all_ret["depth_map"].cpu().numpy())

    return rgbs, disps, depths


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train_depth_pred("data/nerf_synthetic/lego")
