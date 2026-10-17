# ZED-mini wrist camera: sim config that matches the real camera

Source of truth: `source/pi_tasks/pi_tasks/tasks/direct/fr3_irl_sort/fr3_sort_cfg.py`
(constants `ZED_*`, function `resolve_wrist_camera`). The balance and hidden-color tasks use the same camera config.

## 1. Intrinsics (pinhole = the ZED SDK's *rectified* left image)

| value | number | note |
| --- | --- | --- |
| focal length | **3.06 mm** | ZED-mini |
| horizontal aperture | **5.367 mm** | = 2·f·tan(HFOV/2). This is the physical sensor width (2688 px × 2.0 µm) |
| HFOV / VFOV / DFOV | **82.504° / 52.517° / 90.358°** | measured, see below |
| render resolution | **96 × 54** (W × H) | 16:9, the same aspect as HD720, so there's no stretch |
| fx = fy (at 96×54) | 54.73 px, cx 48, cy 27 | square pixels |
| clipping range | (0.002, 1e5) m | a 2 mm near clip, so the lens never "sees through" close objects |
| focus / f-stop | 400 / 0.0 | no depth of field |

Isaac Lab spawn:
```python
sim_utils.PinholeCameraCfg(focal_length=3.06, horizontal_aperture=5.3675,
                           focus_distance=400.0, f_stop=0.0, clipping_range=(0.002, 1e5))
TiledCameraCfg(prim_path="/World/envs/env_.*/Robot/fr3_hand_tcp/wrist_cam",
               width=96, height=54, data_types=["rgb"], spawn=..., offset=...)
```

**How the FOV was measured.** It comes from the SDK calibration of our unit (serial 11284537,
VIEW.LEFT, HD720 1280×720): `left_cam.fx = 729.734`, which gives HFOV = 2·atan(1280 / (2·729.734)) = 82.504°.
**Don't use the datasheet FOV (102×57×118°).** That's the raw field before rectification and includes the
lens distortion. Using it makes the sim image 41% too wide, which scales every lateral distance by 1.41×.
`fx` is only valid together with the width of the stream it was read from. Recompute the FOV for your own unit:
`hfov_deg_from_fx(fx, width)`.

## 2. Extrinsics (camera in the `fr3_hand_tcp` frame, which is the real `O_T_EE`)

| value | number |
| --- | --- |
| position | **(0.0642, -0.0315, -0.0640) m** |
| rotation RPY | **(0, -20, 90)°**, composed as R = Ry(pitch)·Rz(yaw)·Rx(roll) |
| quaternion (w,x,y,z) | (0.69636, -0.12279, -0.12279, 0.69636) |
| convention | `"ros"` (+z looks forward, +y points down) |
| optical axis in TCP | (-0.342, 0, 0.940) |

The position is the CAD lens point (0.07, -0.0315, -0.08) moved 17 mm forward along the optical axis.
The camera is parented to `fr3_hand_tcp`, so it moves exactly with the TCP.
The robot asset `fr3_zed_instanceable.usd` puts an invisible 8 mm collision sphere around the lens, so the policy can't push the camera into objects.
The ZED mount bracket is hidden from rendering.

## 3. Real-side preprocessing (needed for the match)

In `maple_interface/clients/fr3_control_client_balance_belief.py` (`WristCamera`):
1. ZED **HD720 @ 30 fps, LEFT eye, rectified**.
2. **Crop** around (cx, cy) to exactly the sim HFOV of 82.504°. The VFOV follows from the 96:54 aspect.
   With our unit the crop is close to the full frame.
3. `cv2.resize(..., (96, 54), interpolation=cv2.INTER_AREA)`. The policy gets uint8 RGB in 0..255
   (the network divides by 255).
4. Optional: fix the camera settings with `--zed-exposure`, `--zed-gain` and `--zed-white-balance`.
   Auto-exposure is a domain shift the DR below doesn't model.

## 4. Domain randomization applied on top of the render (sim only)

| what | default | when it's drawn |
| --- | --- | --- |
| exposure gain (log-uniform) | 0.55 – 1.8 | per episode |
| gamma, x → x^γ (applied before the gain) | 0.7 – 1.45 | per episode |
| per-channel gain (white balance) | 0.85 – 1.18 | per episode |
| gain flicker | ±6% | per step |
| additive pixel noise, std | 0.02 (≈ 5/255) | per step |
| camera position jitter | ±2 mm per axis (the measured mount tolerance) | once per env, fixed |
| camera rotation jitter | ±0.5° per axis | once per env, fixed |
| background | DROID images composited where depth > 1.2 m, p = 0.95 | per episode |

Config fields: `photo_*`, `cam_pos_jitter_m`, `cam_rpy_jitter_deg`, `background_*`. All of them can be overridden with Hydra,
e.g. `env.zed_cam_hfov_deg=...` or `env.wrist_cam_res=54`.
