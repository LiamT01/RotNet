import numpy as np
import torch
import pdb


def get_invariant_pos(pos):
    # Convert to numpy array
    if isinstance(pos, torch.Tensor):
        pos = pos.numpy()

    # Shift pos to be centered
    pos_centered = pos - pos.mean(0).reshape((1, -1))

    # SVD decomposition
    _, _, vt = np.linalg.svd(pos_centered)

    # Each row of vt is a principle vector
    # vt is orthonormal
    # The rotation matrix is simply the transposed vt
    # First, get a tentative transformed result
    rot_mat = vt.T
    trans = pos_centered @ rot_mat

    # Take the farthest node from the origin as the reference node
    ref_node = np.argmax(np.linalg.norm(trans, axis=1))
    ref_coord = trans[ref_node]

    # vt from SVD can have vectors of arbitrary directions
    # Invert axes to make ref_node lie in the first quadrant/octant using a mask
    mask = np.ones(pos.shape[1])
    mask[ref_coord < 0] = -1
    mask = mask.reshape((1, pos.shape[1]))

    # The final rot_mat and trans
    rot_mat = rot_mat * mask
    trans = trans * mask

    return {
        'trans': trans,
        'rot_mat': rot_mat,
    }


def gaussian_expand(dist, num_steps):
    mu = torch.linspace(0, 1, num_steps).to(dist.device)
    sigma = 1 / (num_steps - 1)
    return torch.exp(-(dist[..., None] - mu) ** 2 / (2 * sigma ** 2)).flatten(start_dim=1)


if __name__ == '__main__':
    pos = np.random.rand(10, 3)

    new_x = np.random.rand(3)
    new_x = new_x / np.linalg.norm(new_x)
    new_z = np.cross(new_x, np.random.rand(3))
    new_z = new_z / np.linalg.norm(new_z)
    new_y = np.cross(new_z, new_x)
    new_y = new_y / np.linalg.norm(new_y)
    rotation = np.linalg.inv(np.stack([new_x, new_y, new_z]))

    _pos = pos @ rotation + np.random.rand(3)

    invariant = get_invariant_pos(pos)
    _invariant = get_invariant_pos(_pos)

    assert np.allclose(invariant['trans'], _invariant['trans'])
    print('Passed.')
