import os
import yaml
import math
import jax.numpy as jnp
import torch
import jax
a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2-f)
pi = jnp.pi


def parse_config(filename):
    """Parse F16Sim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)

def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')

def _t2n(x):
    return x.detach().cpu().numpy()

def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in degrees
    # h in meters
    lamb = jnp.radians(lat)
    phi = jnp.radians(lon)
    s = jnp.sin(lamb)
    N = a / jnp.sqrt(1 - e_sq * s * s)

    sin_lambda = jnp.sin(lamb)
    cos_lambda = jnp.cos(lamb)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = jnp.radians(lat0)
    phi = jnp.radians(lon0)
    s = jnp.sin(lamb)
    N = a / jnp.sqrt(1 - e_sq * s * s)
    sin_lambda = jnp.sin(lamb)
    cos_lambda = jnp.cos(lamb)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda
    xd = x - x0
    yd = y - y0
    zd = z - z0
    t = -cos_phi * xd -  sin_phi * yd
    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = t * sin_lambda  + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
    return xEast, yNorth, zUp

def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):
    lamb = jnp.radians(lat0)
    phi = jnp.radians(lon0)
    s = jnp.sin(lamb)
    N = a / jnp.sqrt(1 - e_sq * s * s)
    sin_lambda = jnp.sin(lamb)
    cos_lambda = jnp.cos(lamb)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda
    t = cos_lambda * zUp - sin_lambda * yNorth
    zd = sin_lambda * zUp + cos_lambda * yNorth
    xd = cos_phi * t - sin_phi * xEast 
    yd = sin_phi * t + cos_phi * xEast
    x = xd + x0 
    y = yd + y0 
    z = zd + z0 
    return x, y, z

def ecef_to_geodetic(x, y, z):
   # Convert from ECEF cartesian coordinates to 
   # latitude, longitude and height.  WGS-84
    x2 = x ** 2 
    y2 = y ** 2 
    z2 = z ** 2 
    a = 6378137.0000    # earth radius in meters
    b = 6356752.3142    # earth semiminor in meters 
    e = jnp.sqrt (1 - (b / a) ** 2) 
    b2 = b * b 
    e2 = e ** 2 
    ep = e * (a / b) 
    r = jnp.sqrt(x2 + y2) 
    r2 = r * r 
    E2 = a ** 2 - b ** 2 
    F = 54 * b2 * z2 
    G = r2 + (1 - e2) * z2 - e2 * E2 
    c = (e2 * e2 * F * r2) / (G * G * G) 
    s = (1 + c + jnp.sqrt(c * c + 2 * c)) ** (1 / 3) 
    P = F / (3 * (s + 1 / s + 1) ** 2 * G * G) 
    Q = jnp.sqrt(1 + 2 * e2 * e2 * P) 
    ro = -(P * e2 * r) / (1 + Q) + jnp.sqrt((a * a / 2) * (1 + 1 / Q) - (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2) 
    tmp = (r - e2 * ro) ** 2 
    U = jnp.sqrt(tmp + z2) 
    V = jnp.sqrt(tmp + (1 - e2) * z2) 
    zo = (b2 * z) / (a * V) 
    height = U * (1 - b2 / (a * V)) 
    lat = jnp.atan((z + ep * ep *zo) / r) 
    temp = jnp.atan(y / x) 

    # 逐元素判断 x >= 0 / (x < 0 & y >= 0) / 其它
    mask1 = (x >= 0)
    mask2 = (x < 0) & (y >= 0)

    long_ = jnp.where(
        mask1,
        temp,                   # x >= 0 时
        jnp.where(
            mask2,
            jnp.pi + temp,      # x < 0 & y >= 0
            temp - jnp.pi       # 剩余情况
        )
    )

    lat0 = lat * (180.0 / jnp.pi)
    lon0 = long_ * (180.0 / jnp.pi)
    h0 = height
    return lat0, lon0, h0

def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = geodetic_to_ecef(lat, lon, h)
    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

def enu_to_geodetic(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref):
    x,y,z = enu_to_ecef(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref)
    return ecef_to_geodetic(x,y,z)

def wrap_2PI(angle):
    res = angle % (2 * jnp.pi)
    mask1 = res < 0
    res += 2 * jnp.pi * mask1
    return res

def wrap_PI(angle):
    res = wrap_2PI(angle)
    mask1 = res > jnp.pi
    res -= 2 * jnp.pi * mask1
    return res

def get_AO_TA_R(ego_pos, enm_pos, ego_vel, enm_vel, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, altitude, vn, ve, vu)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_v = torch.linalg.norm(ego_vel, dim=1)
    enm_v = torch.linalg.norm(enm_vel, dim=1)
    delta_pos = enm_pos - ego_pos
    distance = torch.linalg.norm(delta_pos, dim=1)

    proj_dist = torch.sum(delta_pos * ego_vel, dim=1)
    ego_AO = torch.arccos(torch.clamp(proj_dist / (distance * ego_v + 1e-8), -1, 1))
    proj_dist = torch.sum(delta_pos * enm_vel, dim=1)
    ego_TA = torch.arccos(torch.clamp(proj_dist / (distance * enm_v + 1e-8), -1, 1))
    if not return_side:
        return ego_AO, ego_TA, distance
    else:
        temp_ego_vel = torch.hstack((ego_vel[:, :-1], torch.zeros_like(ego_vel[:, -1].reshape(-1, 1))))
        temp_delta_pos = torch.hstack((delta_pos[:, :-1], torch.zeros_like(delta_pos[:, -1].reshape(-1, 1))))
        cross = torch.cross(temp_ego_vel, temp_delta_pos)
        side_flag = torch.sign(cross[:, -1])
        return ego_AO, ego_TA, distance, side_flag


def get2d_AO_TA_R(ego_pos, enm_pos, ego_vel, enm_vel, return_side=False):
    ego_vel = ego_vel[:, :-1]
    enm_vel = enm_vel[:, :-1]
    ego_pos = ego_pos[:, :-1]
    enm_pos = enm_pos[:, :-1]
    ego_v = torch.linalg.norm(ego_vel, dim=1)
    enm_v = torch.linalg.norm(enm_vel, dim=1)
    delta_pos = enm_pos - ego_pos
    distance = torch.linalg.norm(delta_pos, dim=1)

    proj_dist = torch.sum(delta_pos * ego_vel, dim=1)
    ego_AO = torch.arccos(torch.clamp(proj_dist / (distance * ego_v + 1e-8), -1, 1))
    proj_dist = torch.sum(delta_pos * enm_vel, dim=1)
    ego_TA = torch.arccos(torch.clamp(proj_dist / (distance * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, distance
    else:
        temp_ego_vel = torch.hstack((ego_vel, torch.zeros_like(ego_vel[:, -1].reshape(-1, 1))))
        temp_delta_pos = torch.hstack((delta_pos, torch.zeros_like(delta_pos[:, -1].reshape(-1, 1))))
        cross = torch.cross(temp_ego_vel, temp_delta_pos)
        side_flag = torch.sign(cross[:, -1])
        return ego_AO, ego_TA, distance, side_flag

def orientation_reward(AO, TA, version='v2'):
    if version == 'v0':
        return (1 - torch.tanh(9 * (AO - torch.pi / 9))) / 3 + 1 / 3 \
            + torch.min((torch.arctanh(1 - torch.max(2 * TA / torch.pi, 1e-4 * torch.ones_like(TA)))) / (2 * torch.pi), torch.zeros_like(TA)) + 0.5
    elif version == 'v1':
        return (1 - torch.tanh(2 * (AO - torch.pi / 2))) / 2 \
            * (torch.arctanh(1 - torch.max(2 * TA / torch.pi, 1e-4 * torch.ones_like(TA)))) / (2 * torch.pi) + 0.5
    elif version == 'v2':
        return 1 / (50 * AO / torch.pi + 2) + 1 / 2 \
            + torch.min((torch.arctanh(1 - torch.max(1.9 * TA / torch.pi, 1e-4 * torch.ones_like(TA)))) / (2 * torch.pi), torch.zeros_like(TA)) + 0.5
    else:
        raise NotImplementedError(f"Unknown orientation function version: {version}")

def range_reward(target_dist, R, version='v3'):
    if version == 'v0':
        return torch.exp(-(R - target_dist) ** 2 * 0.004) / (1 + torch.exp(-(R - target_dist + 2) * 2))
    elif version == 'v1':
        return torch.clamp(1.2 * torch.min(torch.exp(-(R - target_dist) * 0.21), torch.ones_like(R)) /
                                    (1 + torch.exp(-(R - target_dist + 1) * 0.8)), 0.3, 1)
    elif version == 'v2':
        return torch.max(torch.clamp(1.2 * torch.min(torch.exp(-(R - target_dist) * 0.21), torch.ones_like(R)) /
                                        (1 + torch.exp(-(R - target_dist + 1) * 0.8)), 0.3, 1), torch.sign(7 - R))
    elif version == 'v3':
        return 1 * (R < 5) + (R >= 5) * torch.clamp(-0.032 * R ** 2 + 0.284 * R + 0.38, 0, 1) + torch.clamp(torch.exp(-0.16 * R), 0, 0.2)
    else:
        raise NotImplementedError(f"Unknown range function version: {version}")

def orientation_fn(AO):
    mask1 = AO >= 0
    mask2 = AO <= torch.pi / 6
    mask3 = mask1 & mask2
    mask1 = AO <= 0
    mask2 = AO >= -torch.pi / 6
    mask4 = mask1 & mask2
    result = (1 - 6 * AO / torch.pi) * mask3 + (1 + 6 * AO / torch.pi) * mask4
    return result

def distance_fn(R):
    mask1 = R <= 1
    mask2 = (R > 1) & (R <= 3)
    result = 1 * mask1 + (3 - R) / 2 * mask2
    return result

def wedge_formation(num_agents, spacing):
    max_layers = num_agents  # 最大层数
    positions = jnp.zeros((num_agents, 3))  # 预分配空间

    def layer_loop(i, carry):
        layers, positions, count = carry
        layer_capacity = 2 * layers
        current_layer = jnp.minimum(num_agents - count, layer_capacity)
        layer_spacing = spacing * (1.0 / layers)

        def agent_loop(j, carry):
            positions, count = carry
            dx = jax.lax.cond(
                j % 2 == 0,
                lambda: -((j // 2) + 1) * layer_spacing,
                lambda: ((j // 2) + 1) * layer_spacing
            )
            dy = layers * spacing
            new_position = jnp.array([dx, dy, 0.0])
            return positions.at[count].set(new_position), count + 1

        positions, count = jax.lax.fori_loop(0, current_layer, agent_loop, (positions, count))
        return layers + 1, positions, count

    # 使用 fori_loop 替代 while_loop
    _, positions, _ = jax.lax.fori_loop(0, max_layers, layer_loop, (1, positions, 0))
    return positions

def line_formation(num_agents, spacing):
    positions = jnp.zeros((num_agents, 3))  # 预分配空间
    start_x = -(num_agents - 1) * spacing / 2
    positions = positions.at[:, 0].set(start_x + spacing * jnp.arange(num_agents))
    return positions

def diamond_formation(num_agents, spacing):
    max_layers = num_agents  # 最大层数
    positions = jnp.zeros((num_agents, 3))  # 预分配空间
    positions = positions.at[0].set(jnp.array([0.0, 0.0, 0.0]))  # 添加长机位置

    def layer_loop(i, carry):
        layer, positions, count = carry
        directions = jnp.array([[-1, 1], [1, 1], [0, 2]])

        def direction_loop(j, carry):
            positions, count = carry
            dx, dy = directions[j]
            new_position = jnp.array([dx * layer * spacing, dy * layer * spacing, 0.0])
            return positions.at[count].set(new_position), count + 1

        positions, count = jax.lax.fori_loop(0, len(directions), direction_loop, (positions, count))
        return layer + 1, positions, count

    # 使用 fori_loop 替代 while_loop
    _, positions, _ = jax.lax.fori_loop(0, max_layers, layer_loop, (1, positions, 1))
    return positions

def enforce_safe_distance(positions, center, safe_distance):
    """
    确保所有位置在指定中心周围的距离不超过安全距离。

    Args:
        positions (jnp.ndarray): 初始位置数组，形状为 (n, 3)，n 表示位置数量。
        center (jnp.ndarray): 中心位置，形状为 (3,)。
        safe_distance (float): 安全距离。

    Returns:
        jnp.ndarray: 调整后的位置数组，形状为 (n, 3)。

    """
    def agent_loop(i, carry):
        formation_positions, positions = carry
        pos = positions[i] + center

        def distance_loop(j, pos):
            existing = formation_positions[j]
            dist = jnp.linalg.norm(pos - existing)
            return jnp.where(
                dist < safe_distance,
                existing + (pos - existing) * safe_distance / dist,
                pos
            )

        pos = jax.lax.fori_loop(0, i, distance_loop, pos)
        formation_positions = formation_positions.at[i].set(pos)
        return formation_positions, positions

    formation_positions = jnp.zeros_like(positions)
    formation_positions, _ = jax.lax.fori_loop(0, len(positions), agent_loop, (formation_positions, positions))
    return formation_positions