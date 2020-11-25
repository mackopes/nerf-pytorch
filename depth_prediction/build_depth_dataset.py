# add ../ to the path using virtualenv
from run_nerf import render, run_network
from load_blender import load_blender_data
from run_nerf_helpers import NeRF, get_embedder, get_rays, to8b
from typing import Any, Tuple
from tqdm import tqdm

import time
import torch
import os.path
import imageio

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


# def get_depth_map(rays_o,
#                   rays_d,
#                   near: float,
#                   far: float,
#                   viewdirs,
#                   N_samples,
#                   network_fn: NeRF,
#                   network_query_fn: Callable[..., Any],
#                   raw_noise_std: float,
#                   white_bkgd: bool
#                   ):
#     assert rays_o.shape[0] == rays_d.shape[0]

#     N_rays = rays_o.shape[0]
#     z_vals = get_z_vals(N_rays, N_samples, near, far)

#     pts = rays_o[..., None, :] + rays_d[..., None, :] * \
#         z_vals[..., :, None]  # [N_rays, N_samples, 3]
#     raw = network_query_fn(pts, viewdirs, network_fn)
#     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
#         raw, z_vals, rays_d, raw_noise_std, white_bkgd)

#     return depth_map


def load_data(datadir):
    white_bkgd = True
    half_res = True
    testskip = 1

    dataset_type = 'blender'

    if dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            datadir, half_res, testskip)
        print('Loaded blender', images.shape,
              poses.shape, hwf, datadir)

        i_train, i_val, i_test = i_split

        # near = 2.
        # far = 6.

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

    return images, poses, render_poses, hwf, i_split


def get_render_kwargs():
    raw_noise_std = 0.0
    use_viewdirs = True
    N_importance = 128
    lindisp = False
    N_samples = 64

    # same as in load_data
    white_bkgd = True
    half_res = True
    testskip = 8

    # depends on data
    near = 2.
    far = 6.
    render_kwargs = {
        # 'network_query_fn': network_query_fn,
        'perturb': False,
        'N_importance': N_importance,
        # 'network_fine': model_fine,
        'N_samples': N_samples,
        # 'network_fn': model,
        'use_viewdirs': use_viewdirs,
        'white_bkgd': white_bkgd,
        # 'raw_noise_std': raw_noise_std,
        'lindisp': lindisp,
        'ndc': False,
        'raw_noise_std': 0.,
        'near': near,
        'far': far
    }

    return render_kwargs


def save_depth_dataset(rays_l, depths_l, target_file):
    dataset_dir = os.path.dirname(target_file)
    os.makedirs(dataset_dir, exist_ok=True)

    rays_d_stack = []
    rays_o_stack = []
    depths_stack = []

    for rays, depths in zip(rays_l, depths_l):  # an iteration per image
        rays_d, rays_o = rays

        assert rays_d.shape[0] == rays_o.shape[0] == depths.shape[0]
        assert rays_d.shape[1] == rays_o.shape[1] == depths.shape[1]
        assert rays_d.shape[2] == rays_o.shape[2] == 3

        rays_d = torch.reshape(rays_d, (-1, 3)).float()
        rays_o = torch.reshape(rays_o, (-1, 3)).float()
        depths = torch.reshape(depths, (-1, 1))

        rays_d_stack.append(rays_d)
        rays_o_stack.append(rays_o)
        depths_stack.append(depths)

    rays_d_stack = torch.cat(rays_d_stack, dim=0)
    rays_o_stack = torch.cat(rays_o_stack, dim=0)
    depths_stack = torch.cat(depths_stack, dim=0)

    rays_depths_stack = torch.cat(
        [rays_d_stack, rays_o_stack, depths_stack], dim=1)

    torch.save(rays_depths_stack, target_file)


def build_depth_dataset(target_dataset_dir, datadir):
    """
    builds a dataset and stores it in the target_dataset_dir.
    Warning: the final dataset might be too big.
    the size of the final dataset is N * H * W * 7 * sizeof(float)
    """
    chunk = 32768
    dirname = os.path.dirname(__file__)
    ckpt_path = os.path.join(dirname, "../logs/blender_paper_lego/200000.tar")
    netdepth = 8
    netdepth_fine = 8
    netwidth = 256
    netwidth_fine = 256
    i_embed = 0
    multires = 10
    multires_views = 4

    render_kwargs = get_render_kwargs()
    images, poses, render_poses, hwf, i_split = load_data(datadir)
    model, model_fine, network_query_fn = load_nerf(ckpt_path, netdepth, netwidth, netdepth_fine, netwidth_fine,
                                                    render_kwargs["use_viewdirs"], i_embed, multires, multires_views, render_kwargs['N_importance'])

    render_kwargs.update({
        "network_query_fn": network_query_fn,
        "network_fine": model_fine,
        "network_fn": model,
    })

    poses = torch.Tensor(poses).to(device)
    print(f"moved poses to {device}")

    for split_name, i in zip(["train", "val", "test"], i_split):
        print("rendering split", split_name)
        with torch.no_grad():
            poses_i = poses[i]
            print(f'{split_name} poses shape', poses_i.shape)

            rays_l, rgbs_l, disps_l, depths_l = get_depths(
                hwf, poses_i, chunk, render_kwargs)
        print("saving split", split_name)
        target_file = os.path.join(target_dataset_dir, f"{split_name}.pt")
        save_depth_dataset(rays_l, depths_l, target_file)


def load_depth_dataset(dataset_loc):
    rds = torch.load(dataset_loc)

    rays_d = rds[:, :3]
    rays_o = rds[:, 3:6]
    depths = rds[:, 6].reshape((-1, 1))

    return rays_d, rays_o, depths


def get_depths(hwf, poses, chunk, render_kwargs):
    rays_l = []
    rgbs = []
    disps = []
    depths = []

    H, W, focal = hwf

    t = time.time()
    for i, c2w in enumerate(tqdm(poses)):
        print(i, time.time() - t)
        t = time.time()
        rays = get_rays(H, W, focal, c2w)
        rgb, disp, acc, all_ret = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], rays=rays, **render_kwargs)
        rays_l.append(rays)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(all_ret["depth_map"])

    return rays_l, rgbs, disps, depths


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    build_depth_dataset(os.path.join(dirname, 'depth_datasets/lego/'),
                        os.path.join(dirname, '../data/nerf_synthetic/lego'))
