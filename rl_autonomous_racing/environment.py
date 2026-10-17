"""
Vice Coast Circuit — Authentic Gymnasium Reinforcement Learning Environment
Designed for Autonomous Vehicle Racing with Deep Q-Networks (DQN).

Conforms strictly to the Gymnasium (gym.Env) interface.
Derives track geometry, physics, and kinematics directly from the authoritative
KLUSTOR 3D Vice Coast Circuit (3.38 km Catmull-Rom spline).
"""

import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False


# ============================================================
# Authoritative Track Geometry Constants
# ============================================================
ROAD_WIDTH = 12.0          # meters
CAR_HALF_WIDTH = 1.1       # meters
MAX_LEGAL_OFFSET = ROAD_WIDTH / 2.0 - CAR_HALF_WIDTH  # 4.9m
TRACK_SAMPLES = 500

WAYPOINTS = [
    [0.0, 0.0, 0.0],
    [150.0, 0.0, 0.0],
    [250.0, 0.0, 40.0],
    [300.0, 0.0, -20.0],
    [400.0, 0.0, 20.0],
    [550.0, 0.0, 50.0],
    [650.0, 0.0, 150.0],
    [450.0, 0.0, 200.0],
    [550.0, 0.0, 300.0],
    [650.0, 0.0, 450.0],
    [600.0, 0.0, 600.0],
    [450.0, 0.0, 750.0],
    [350.0, 0.0, 850.0],
    [250.0, 0.0, 750.0],
    [150.0, 0.0, 850.0],
    [0.0, 0.0, 800.0],
    [-100.0, 0.0, 750.0],
    [-150.0, 0.0, 600.0],
    [-250.0, 0.0, 650.0],
    [-300.0, 0.0, 500.0],
    [-200.0, 0.0, 450.0],
    [-250.0, 0.0, 300.0],
    [-450.0, 0.0, 350.0],
    [-550.0, 0.0, 250.0],
    [-400.0, 0.0, 150.0],
    [-500.0, 0.0, 50.0],
    [-300.0, 0.0, -50.0],
    [-150.0, 0.0, 30.0],
    [-50.0, 0.0, 0.0],
]


def catmull_rom_spline_closed(points, num_samples=500):
    """Generates closed centripetal Catmull-Rom spline interpolation with exactly num_samples points."""
    pts = np.array(points, dtype=np.float64)
    m = len(pts)
    samples = []

    for i in range(num_samples):
        # Continuous parameter along the closed circuit [0, m)
        u = (float(i) / float(num_samples)) * float(m)
        seg = int(u) % m
        t = u - float(int(u))

        p0 = pts[(seg - 1 + m) % m]
        p1 = pts[seg]
        p2 = pts[(seg + 1) % m]
        p3 = pts[(seg + 2) % m]

        t2 = t * t
        t3 = t2 * t

        # Standard Catmull-Rom matrix
        q = 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        samples.append(q)

    return np.array(samples, dtype=np.float64)


class ViceCoastTrack:
    """Precomputed Frenet track coordinates, tangents, normals, and curvatures."""

    def __init__(self, num_samples=TRACK_SAMPLES):
        self.points = catmull_rom_spline_closed(WAYPOINTS, num_samples)
        self.num_samples = len(self.points)
        self.tangents = np.zeros_like(self.points)
        self.normals = np.zeros_like(self.points)
        self.curvatures = np.zeros(self.num_samples, dtype=np.float64)
        self.arc_lengths = np.zeros(self.num_samples, dtype=np.float64)

        total_len = 0.0
        for i in range(self.num_samples):
            curr_p = self.points[i]
            next_p = self.points[(i + 1) % self.num_samples]
            prev_p = self.points[(i - 1 + self.num_samples) % self.num_samples]

            ds = np.linalg.norm(next_p - curr_p)
            total_len += ds
            self.arc_lengths[i] = total_len

            # Tangent (forward direction)
            t_vec = next_p - curr_p
            t_len = np.linalg.norm(t_vec)
            if t_len > 1e-6:
                t_vec = t_vec / t_len
            self.tangents[i] = t_vec

            # Normal (rightward in XZ plane: (-tz, 0, tx))
            self.normals[i] = np.array([-t_vec[2], 0.0, t_vec[0]], dtype=np.float64)

            # Curvature calculation: d(theta) / ds
            t_prev = curr_p - prev_p
            t_prev_len = np.linalg.norm(t_prev)
            if t_prev_len > 1e-6:
                t_prev = t_prev / t_prev_len

            cross_y = t_prev[0] * t_vec[2] - t_prev[2] * t_vec[0]
            dot = np.clip(np.dot(t_prev, t_vec), -1.0, 1.0)
            angle = math.atan2(cross_y, dot)
            seg_len = 0.5 * (t_prev_len + t_len)
            self.curvatures[i] = angle / max(seg_len, 1e-4)

        self.total_length = total_len

    def find_nearest_index(self, pos_x, pos_z):
        """Finds closest track centerline index to world position (X, Z)."""
        dx = self.points[:, 0] - pos_x
        dz = self.points[:, 2] - pos_z
        dist_sq = dx * dx + dz * dz
        return int(np.argmin(dist_sq))


# ============================================================
# Vice Coast Racing Gymnasium Environment
# ============================================================
class ViceCoastRacingEnv:
    """
    OpenAI Gymnasium-compatible environment for RL Racing.
    
    Observation Space (Box(13,)):
      0: Normalized speed (v / 66.67 m/s)
      1: Normalized lateral offset (d / 4.9m)
      2: Heading error sin(theta_car - theta_track)
      3: Heading error cos(theta_car - theta_track)
      4: Curvature at current track position
      5: Curvature lookahead (+15m)
      6: Curvature lookahead (+35m)
      7-11: 5-ray simulated LIDAR rangefinders (-60°, -30°, 0°, +30°, +60°)
      12: Boost reserve ratio (0.0 to 1.0)

    Action Space (Discrete(10)):
      0: Hard Left  + Throttle (steer = -1.0, throttle = 1.0)
      1: Mild Left  + Throttle (steer = -0.45, throttle = 1.0)
      2: Straight   + Throttle (steer =  0.0, throttle = 1.0)
      3: Straight   + Nitro Boost (steer = 0.0, throttle = 1.0, boost = 1.0)
      4: Mild Right + Throttle (steer = +0.45, throttle = 1.0)
      5: Hard Right + Throttle (steer = +1.0, throttle = 1.0)
      6: Heavy Trail Brake (steer = 0.0, throttle = 0.0, brake = 1.0, KERS Active)
      7: Coast / Idle (steer = 0.0, throttle = 0.25, brake = 0.0)
      8: Mild Left  + Nitro Boost (steer = -0.35, throttle = 1.0, boost = 1.0)
      9: Mild Right + Nitro Boost (steer = +0.35, throttle = 1.0, boost = 1.0)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Action Mapping Table: (steering, throttle, brake, boost)
    ACTION_MAP = {
        0: (-1.0, 1.0, 0.0, False),   # Hard Left + Accel
        1: (-0.45, 1.0, 0.0, False),  # Mild Left + Accel
        2: (0.0, 1.0, 0.0, False),    # Straight + Full Accel
        3: (0.0, 1.0, 0.0, True),     # Straight + Boost
        4: (0.45, 1.0, 0.0, False),   # Mild Right + Accel
        5: (1.0, 1.0, 0.0, False),    # Hard Right + Accel
        6: (0.0, 0.0, 1.0, False),    # Heavy Trail Brake (KERS Rapid Regen)
        7: (0.0, 0.25, 0.0, False),   # Coast
        8: (-0.35, 1.0, 0.0, True),   # Mild Left + Boost (Sweepers & Corner Exit)
        9: (0.35, 1.0, 0.0, True),    # Mild Right + Boost (Sweepers & Corner Exit)
    }

    def __init__(self, max_steps_per_episode=2000, dt=0.05):
        self.dt = dt
        self.max_steps = max_steps_per_episode
        self.track = ViceCoastTrack()

        # Vehicle Kinematics (Authoritative 10/10 Prototype)
        self.top_speed_unboosted = 150.0 / 3.6  # 41.67 m/s (~150 km/h)
        self.top_speed_boosted   = 240.0 / 3.6  # 66.67 m/s (~240 km/h)
        self.accel = 11.2                       # m/s^2
        self.brake_strength = 28.0              # m/s^2
        self.steer_strength = 3.2               # rad/s
        self.friction = 5.0                     # m/s^2

        # State dimensions
        self.obs_dim = 13
        self.act_dim = 10

        if HAS_GYMNASIUM:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(self.act_dim)

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Start vehicle at track start line
        self.pos_x = float(self.track.points[0, 0])
        self.pos_z = float(self.track.points[0, 2])
        start_tangent = self.track.tangents[0]
        self.rot_y = math.atan2(start_tangent[0], start_tangent[2])  # Heading angle

        self.speed = 10.0            # Initial roll (m/s)
        self.boost_reserve = 1.0     # 100% boost capacity
        self.step_count = 0
        self.total_distance = 0.0
        self.last_track_idx = 0
        self.collision_count = 0
        self.lap_completed = False
        self.laps_done = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_count += 1
        action = int(action)
        steer_input, throttle, brake, wants_boost = self.ACTION_MAP[action]

        # 1. Smart Boost & KERS (Kinetic Energy Recovery System) Dynamics
        is_boosting = False
        kers_charging = False

        if wants_boost and self.boost_reserve > 0.03:
            # Consume boost at 0.20/sec (~5.0s continuous burn)
            self.boost_reserve = max(0.0, self.boost_reserve - 0.20 * self.dt)
            is_boosting = True
        else:
            # Smart Regeneration:
            if brake > 0.1:
                # KERS: braking recovers kinetic energy at 3.5x rate into capacitor
                speed_ratio = self.speed / max(1.0, self.top_speed_unboosted)
                kers_regen = 0.42 * brake * (0.6 + 0.4 * speed_ratio)
                self.boost_reserve = min(1.0, self.boost_reserve + kers_regen * self.dt)
                kers_charging = True
            else:
                # Continuous alternator charge while cruising
                speed_ratio = self.speed / max(1.0, self.top_speed_unboosted)
                base_regen = 0.10 + 0.06 * speed_ratio
                self.boost_reserve = min(1.0, self.boost_reserve + base_regen * self.dt)

        # 2. Forward speed update
        b_mult = 1.6 if is_boosting else 1.0
        cur_max_speed = self.top_speed_unboosted * b_mult
        cur_accel = self.accel * (1.2 if is_boosting else 1.0)

        if throttle > 0:
            self.speed = min(cur_max_speed, self.speed + cur_accel * throttle * self.dt)
        elif brake > 0:
            self.speed = max(0.0, self.speed - self.brake_strength * brake * self.dt)
        else:
            f = self.friction * self.dt
            if self.speed > f:
                self.speed -= f
            else:
                self.speed = 0.0

        # 3. Steering update
        speed_factor = min(1.0, self.speed / 5.0)
        if abs(self.speed) > 0.2 and steer_input != 0:
            self.rot_y += steer_input * self.steer_strength * speed_factor * self.dt

        # 4. Position integration
        prev_x, prev_z = self.pos_x, self.pos_z
        self.pos_x += math.sin(self.rot_y) * self.speed * self.dt
        self.pos_z += math.cos(self.rot_y) * self.speed * self.dt
        step_dist = math.hypot(self.pos_x - prev_x, self.pos_z - prev_z)
        self.total_distance += step_dist

        # 5. Track coordinates (Frenet frame projection)
        nearest_idx = self.track.find_nearest_index(self.pos_x, self.pos_z)
        track_pt = self.track.points[nearest_idx]
        tangent = self.track.tangents[nearest_idx]
        normal = self.track.normals[nearest_idx]

        to_car_x = self.pos_x - track_pt[0]
        to_car_z = self.pos_z - track_pt[2]
        lateral_offset = to_car_x * normal[0] + to_car_z * normal[2]

        # Heading error relative to track tangent
        track_angle = math.atan2(tangent[0], tangent[2])
        heading_err = (self.rot_y - track_angle + math.pi) % (2.0 * math.pi) - math.pi

        # Check track boundary collision
        is_collision = False
        if abs(lateral_offset) > MAX_LEGAL_OFFSET:
            is_collision = True
            self.collision_count += 1
            # Elastic bounce back inside boundary
            clamped_offset = math.copysign(MAX_LEGAL_OFFSET, lateral_offset)
            longitudinal = to_car_x * tangent[0] + to_car_z * tangent[2]
            self.pos_x = track_pt[0] + tangent[0] * longitudinal + normal[0] * clamped_offset
            self.pos_z = track_pt[2] + tangent[2] * longitudinal + normal[2] * clamped_offset
            self.speed *= 0.75  # Kinetic energy dissipation from barrier slam
            # Wall impact capacitor disruption
            self.boost_reserve = max(0.0, self.boost_reserve - 0.08)

        # 6. Lap progress & checkpointing
        idx_diff = (nearest_idx - self.last_track_idx) % self.track.num_samples
        if idx_diff > self.track.num_samples // 2:
            idx_diff -= self.track.num_samples  # Reverse movement penalty

        # Lap completion detection
        if self.last_track_idx > (self.track.num_samples * 0.9) and nearest_idx < (self.track.num_samples * 0.1):
            self.lap_completed = True
            self.laps_done += 1

        self.last_track_idx = nearest_idx

        # 7. Authentic Racing Reward Function Formulation
        # Progress reward: +1.6 per track waypoint increment
        r_progress = float(idx_diff) * 1.6

        # Aggressive Speed Incentive: convex curve heavily rewarding 180-240 km/h
        norm_v = self.speed / self.top_speed_boosted
        r_speed = 1.25 * (norm_v ** 1.6)

        # Smart Boost Utilization Bonus: rewarded for boosting when aligned with the track
        r_boost = 0.0
        if is_boosting:
            if abs(heading_err) < 0.28:
                r_boost = 0.65  # Clean boost down straight or gentle sweeper
            else:
                r_boost = -0.20  # Wasting nitro sideways into wall

        # Apex clipping & Racing Line reward:
        norm_lat = abs(lateral_offset) / MAX_LEGAL_OFFSET
        curve_k = self.track.curvatures[nearest_idx]
        if abs(curve_k) > 0.008:
            # In a corner: reward clipping the inside curb apex
            inside_offset = -np.sign(curve_k) * (lateral_offset / MAX_LEGAL_OFFSET)
            r_racing_line = 0.35 * np.clip(inside_offset, -0.4, 0.8)
            # Only penalize if drifting dangerously near outer barrier
            r_deviation = -0.8 * (max(0.0, norm_lat - 0.75) ** 2)
        else:
            # Straightaway: stay near centerline for cleanest line
            r_racing_line = 0.0
            r_deviation = -0.25 * (norm_lat ** 2)

        # Collision penalty
        r_collision = -20.0 if is_collision else 0.0

        # Finish bonus
        r_finish = 150.0 if self.lap_completed else 0.0

        reward = r_progress + r_speed + r_boost + r_racing_line + r_deviation + r_collision + r_finish

        # 8. Termination conditions
        terminated = bool(self.lap_completed or self.collision_count >= 5)
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "collision": is_collision,
            "lateral_offset": lateral_offset,
            "speed_kmh": self.speed * 3.6,
            "lap_completed": self.lap_completed,
            "track_progress": nearest_idx / float(self.track.num_samples),
            "is_boosting": is_boosting,
            "kers_charging": kers_charging,
            "boost_reserve": self.boost_reserve,
        })

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Constructs the continuous 13-dimensional state representation."""
        idx = self.track.find_nearest_index(self.pos_x, self.pos_z)
        track_pt = self.track.points[idx]
        tangent = self.track.tangents[idx]
        normal = self.track.normals[idx]

        # 1. Normalized speed
        norm_speed = self.speed / self.top_speed_boosted

        # 2. Normalized lateral offset
        to_car_x = self.pos_x - track_pt[0]
        to_car_z = self.pos_z - track_pt[2]
        lat_offset = to_car_x * normal[0] + to_car_z * normal[2]
        norm_lat = np.clip(lat_offset / MAX_LEGAL_OFFSET, -2.0, 2.0)

        # 3-4. Heading error components
        track_angle = math.atan2(tangent[0], tangent[2])
        heading_err = (self.rot_y - track_angle + math.pi) % (2.0 * math.pi) - math.pi
        sin_err = math.sin(heading_err)
        cos_err = math.cos(heading_err)

        # 5-7. Curvature preview
        curv_curr = float(self.track.curvatures[idx]) * 30.0
        idx_ahead_1 = (idx + 12) % self.track.num_samples
        idx_ahead_2 = (idx + 28) % self.track.num_samples
        curv_ahead_1 = float(self.track.curvatures[idx_ahead_1]) * 30.0
        curv_ahead_2 = float(self.track.curvatures[idx_ahead_2]) * 30.0

        # 8-12. 5-Ray Simulated LIDAR Rangefinders (-60, -30, 0, 30, 60 deg)
        lidar_angles = [-math.pi / 3.0, -math.pi / 6.0, 0.0, math.pi / 6.0, math.pi / 3.0]
        lidar_readings = []
        max_ray_len = 25.0

        for ang in lidar_angles:
            ray_dir = self.rot_y + ang
            ray_dx = math.sin(ray_dir)
            ray_dz = math.cos(ray_dir)

            # Approximate ray intersection with track corridor boundary
            dist = max_ray_len
            for step_dist in np.linspace(1.0, max_ray_len, 10):
                probe_x = self.pos_x + ray_dx * step_dist
                probe_z = self.pos_z + ray_dz * step_dist
                probe_idx = self.track.find_nearest_index(probe_x, probe_z)
                p_pt = self.track.points[probe_idx]
                p_norm = self.track.normals[probe_idx]
                p_lat = abs((probe_x - p_pt[0]) * p_norm[0] + (probe_z - p_pt[2]) * p_norm[2])
                if p_lat >= MAX_LEGAL_OFFSET:
                    dist = float(step_dist)
                    break
            lidar_readings.append(dist / max_ray_len)

        # 13. Boost reserve
        boost_norm = self.boost_reserve

        obs = np.array([
            norm_speed,
            norm_lat,
            sin_err,
            cos_err,
            curv_curr,
            curv_ahead_1,
            curv_ahead_2,
            lidar_readings[0],
            lidar_readings[1],
            lidar_readings[2],
            lidar_readings[3],
            lidar_readings[4],
            boost_norm,
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "step": self.step_count,
            "speed": self.speed,
            "boost": self.boost_reserve,
            "collisions": self.collision_count,
        }
