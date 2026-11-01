"""Make the waypoints in mppi/waypoints/ from the simulator's map and centerline.

Smooths the centerline toward minimum curvature while keeping `margin` m from the
walls of the map, resamples it every 0.1 m (utils.py assumes uniform spacing) and
gives it a speed profile: sqrt(a_lat / curvature), clipped to [v_min, v_max], then
limited by a_lon when accelerating and braking.

usage: gen_waypoints.py <maps_dir> <levine|spielberg> <margin> <a_lat> <a_lon> <v_min> <v_max> [iters] [out.csv]
e.g.   gen_waypoints.py ~/sim_ws/src/f1tenth_gym_ros/maps levine 0.55 5.0 2.5 2.0 6.0
writes x_m,y_m,yaw_rad,vel_mps"""
import os, sys, numpy as np, yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
M = os.path.expanduser(sys.argv[1]) + '/'
track = sys.argv[2]; MARGIN, A_LAT, A_LON, V_MIN, V_MAX = map(float, sys.argv[3:8])
ITERS = int(sys.argv[8]) if len(sys.argv) > 8 else 400
OUT = sys.argv[9] if len(sys.argv) > 9 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'mppi', 'waypoints', f'{track}.csv')
DS = 0.1
mapname, cl = {'levine': ('levine_blocked', 'levine_centerline.csv'), 'spielberg': ('Spielberg', 'Spielberg_centerline.csv')}[track]
meta = yaml.safe_load(open(M + mapname + '.yaml')); res = meta['resolution']; ox, oy = meta['origin'][:2]
img = np.array(Image.open(M + meta['image']).convert('L')).astype(float)
dist = distance_transform_edt(img >= (1 - 0.65) * 255) * res  # metres to the nearest wall
Hh = img.shape[0]
def clearance(p):
    return map_coordinates(dist, [Hh - 1 - (p[:, 1] - oy) / res, (p[:, 0] - ox) / res], order=1, mode='nearest')

def resample(xy, ds=DS):
    seg = np.hypot(*np.diff(np.r_[xy, xy[:1]], axis=0).T); s = np.r_[0, np.cumsum(seg)]
    n = int(round(s[-1] / ds)); t = np.linspace(0, s[-1], n, endpoint=False)
    loop = np.r_[xy, xy[:1]]
    return np.column_stack((np.interp(t, s, loop[:, 0]), np.interp(t, s, loop[:, 1])))

def normals(xy):
    t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0); t /= np.hypot(*t.T)[:, None]
    return np.column_stack((-t[:, 1], t[:, 0]))

xy = resample(np.loadtxt(M + cl, delimiter=',', comments='#')[:, :2])
start = xy[0].copy()
for it in range(ITERS):
    # Laplacian step: lowers the curvature, i.e. cuts the corners
    xy = xy + 0.3 * (0.5 * (np.roll(xy, 1, 0) + np.roll(xy, -1, 0)) - xy)
    # push points closer than MARGIN to a wall back, along the normal, toward the free side
    n = normals(xy)
    for _ in range(3):
        c = clearance(xy); low = c < MARGIN
        if not low.any(): break
        side = np.sign(clearance(xy + 0.05 * n) - clearance(xy - 0.05 * n))
        xy[low] += (n * (side * (MARGIN - c))[:, None])[low]
    if it % 20 == 19: xy = resample(xy)
xy = resample(xy)
xy = np.roll(xy, -np.argmin(np.hypot(*(xy - start).T)), 0)  # keep the centerline's first point first

# curvature (Menger, through points +-4 apart = +-0.4 m), lightly smoothed
k_ = 4; a, b, c = np.roll(xy, k_, 0), xy, np.roll(xy, -k_, 0)
cross = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
kap = 2 * np.abs(cross) / (np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(a - c).T))
kap = np.convolve(np.r_[kap[-3:], kap, kap[:3]], np.ones(7) / 7, 'valid')
v = np.clip(np.sqrt(A_LAT / np.maximum(kap, 1e-6)), V_MIN, V_MAX)
seg = np.hypot(*np.diff(np.r_[xy, xy[:1]], axis=0).T); N = len(v)
for _ in range(2):
    for i in range(2 * N):  # acceleration limit
        p, q = i % N, (i + 1) % N; v[q] = min(v[q], np.sqrt(v[p] ** 2 + 2 * A_LON * seg[p]))
    for i in range(2 * N, 0, -1):  # braking limit
        p, q = (i - 1) % N, i % N; v[p] = min(v[p], np.sqrt(v[q] ** 2 + 2 * A_LON * seg[p]))
d1 = np.roll(xy, -1, 0) - np.roll(xy, 1, 0); yaw = np.arctan2(d1[:, 1], d1[:, 0])
np.savetxt(OUT, np.column_stack((xy, yaw, v)), delimiter=',', fmt='%.4f', header='x_m,y_m,yaw_rad,vel_mps', comments='')
cl_ = clearance(xy)
print(f'{track}: {N} pts, spacing {seg.min():.3f}-{seg.max():.3f} m, length {seg.sum():.1f} m, min wall clearance {cl_.min():.2f} m, '
      f'max curvature {kap.max():.2f}, speed {v.min():.2f}-{v.max():.2f}, est lap {np.sum(seg / v):.2f} s')
