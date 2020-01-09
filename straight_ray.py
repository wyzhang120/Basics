import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import inv, spsolve, lsqr
import matplotlib.pyplot as plt
import h5py

"""
    Straing ray tomo in crosswell geometry. src on the left and rec on the right.
"""
par = {'length': 300., 'height': 105., 'dx': 5., 'dz': 5.,
       'xsrc': 0., 'osV': 0., 'dsrc': 1., 'bsV': 105.5,
       'xrec': 300, 'drec': 1., 'orV': 0., 'brV': 105.5,
       'vp_max': 5800, 'vp_min': 2500, 'damping': 0.5}

par['nx'] = int(par['length'] / par['dx'] + 1.5)
par['nz'] = int(par['height'] / par['dz'] + 1.5)
zsrc = np.arange(par['osV'], par['bsV'], par['dsrc'])
zrec = np.arange(par['orV'], par['brV'], par['drec'])
nsrc = len(zsrc)
nrec = len(zrec)

# Read time table shape = (106, 106)
tt = np.fromfile('reco_rV2.rsf@', dtype=np.float32).reshape([-1, 1])


def ray_seg_len(xzsrc, xzrec, nx, nz, dx=1., dz=1.,
                zmin=0., xmin=0.,):
    """
    Given src and rec coords, compute length ray of segments in cells traveled through.
    :param xzsrc: tuple, float
    :param xzrec: tuple, float
    :param nx: int, model size
    :param nz: int,
    :param dx: float
    :param dz: float
    :param zmin: float, model z bound
    :param xmin: float,
    :return:
        vec: 1d array, shape = nx * nz. Assuming fast dim = z
    """
    xmax = (nx - 1) * dx + xmin
    zmax = (nz - 1) * dz + zmin
    xgrid = np.arange(xmin - dx/2, xmax + dx, dx)
    zgrid = np.arange(zmin-dz/2, zmax + dz, dz)
    xgrid = np.clip(xgrid, xmin, xmax)
    zgrid = np.clip(zgrid, zmin, zmax)
    xz0 = np.array((xmin - dx/2, zmin-dz/2))
    xzsrc = np.array(xzsrc)
    xzrec = np.array(xzrec)
    vec = np.zeros(nx * nz, np.float32)
    xgrid = xgrid.reshape([-1, 1])
    zgrid = zgrid.reshape([-1, 1])
    k = (xzrec[1] - xzsrc[1]) / (xzrec[0] - xzsrc[0])
    xgrid_z = k * (xgrid - xzsrc[0]) + xzsrc[1]
    if k != 0.:
        zgrid_x = 1./k * (zgrid - xzsrc[1]) + xzsrc[0]
        xz = np.vstack((np.hstack((xgrid, xgrid_z)), np.hstack((zgrid_x, zgrid))))
    else:
        xz = np.hstack((xgrid, xgrid_z))
    xz = np.unique(xz, axis=0)
    idx = np.logical_and(xz[:, 0] >= xmin, xz[:, 1] >= zmin-dz/2)
    xz = xz[idx]
    ray_len = np.sqrt(np.sum((xz - xzsrc)**2, axis=1))
    max_len = np.sqrt(np.sum((xzsrc - xzrec)**2))
    idx = np.argsort(ray_len)
    xz = xz[idx]
    id_tmp = np.argmin(np.abs(ray_len - max_len))
    if np.abs(ray_len[id_tmp] - max_len) <= 1e-10 or ray_len[id_tmp] > max_len:
        id_max = id_tmp
    else:
        ray_len = np.vstack((ray_len.reshape([-1, 1]), np.array([max_len]).reshape([1, 1])))
        xz = np.vstack((xz, xzrec))
        id_max = len(ray_len) - 1
    ray_len[id_max] = max_len
    xz[id_max] = xzrec
    ray_len = ray_len[:id_max+1]
    xz = xz[:id_max+1]
    dxz = np.array((dx, dz))
    ids = np.int32((xz - xz0) / dxz)
    ids[:, 0] = np.clip(ids[:, 0], 0, nx-1)
    ids[:, 1] = np.clip(ids[:, 1], 0, nz-1)
    diff = np.diff(ray_len.flatten())
    ids_out = np.ravel_multi_index(ids.T, (nx, nz))
    vec[ids_out[:-1]] = diff
    return vec


npair = nsrc * nrec
nnode = par['nx']*par['nz']
raymat = np.zeros([npair, nnode])
for isrc in range(nsrc):
    for irec in range(nrec):
        ipair = isrc * nrec + irec
        xzsrc = (par['xsrc'], zsrc[isrc])
        xzrec = (par['xrec'], zrec[irec])
        raymat[ipair] = ray_seg_len(xzsrc, xzrec, par['nx'], par['nz'],
                                    par['dx'], par['dz'])


G = sparse.csr_matrix(raymat)
GTG = G.T.dot(G)
R = sparse.diags(1./GTG.diagonal())
rhs = R.dot(G.T).dot(tt)
op = R.dot(GTG) + par['damping'] * sparse.eye(G.shape[1])
slowness = spsolve(op, rhs)
# slowness = lsqr(R.dot(GTG), rhs, damp=par['damping'])[0]
vp_inv = 1./slowness.reshape([par['nx'], par['nz']])
vp_inv = np.clip(vp_inv, par['vp_min'], par['vp_max'])
fig, ax = plt.subplots()
im = ax.imshow(vp_inv.T, extent=(
    -par['dx']/2., par['length']+par['dx']/2,
    par['height']+par['dz']/2., -par['dz']/2.))
plt.colorbar(im)
plt.show()


# v = op.diagonal().reshape([par['nx'], par['nz']])
# clip = 0.01
# fig, ax = plt.subplots()
# im = ax.imshow(v.T, extent=(
#     -par['dx']/2., par['length']+par['dx']/2,
#     par['height']+par['dz']/2., -par['dz']/2.), vmax=clip*v.max())
# plt.colorbar(im)
# plt.show()

# with h5py.File('raymat.h', 'w') as f:
#     f.create_dataset('raymat', data=raymat)
# xzsrc = (0, 105)
# xzrec = (300, 70)
# vec = ray_seg_len(xzsrc, xzrec, par['nx'], par['nz'],
#                   par['dx'], par['dz'])
# vec = vec.reshape((par['nx'], par['nz']))
# xline = np.linspace(xzsrc[0], xzrec[0], 100)
# zline = (xzrec[1] - xzsrc[1]) / (xzrec[0] - xzsrc[0]) * (xline - xzsrc[0]) + xzsrc[1]
# fig, ax = plt.subplots()
# ax.imshow(vec.T, extent=(-par['dx']/2., par['length']+par['dx']/2,
#                          par['height']+par['dz']/2., -par['dz']/2.))
# ax.plot(xline, zline, 'w-', linewidth=0.5)
# plt.grid(color='grey', which='both', linestyle='--', linewidth=1)
# plt.show()
