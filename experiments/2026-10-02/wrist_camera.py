"""Select the wrist D405 explicitly. With a second RealSense on the Jetson (a side D455/D435, 2026-10-09), a bare
rs.config() lets librealsense open whichever device first satisfies 640x480 rgb8 + z16, which both models do, so
wrist frames could silently come from the side camera. Every wrist pipeline calls enable_wrist(rs, cfg)."""
import os

WRIST_SERIAL='260522271635'  # the D405 on the A1Z wrist


def pick_wrist(devices,known=WRIST_SERIAL,override=None):
    """devices: [(name, serial)]. The override (env A1Z_WRIST_SERIAL) wins; otherwise the single D405, preferring the
    known wrist serial if several D405s are attached. Raises instead of falling back to another camera."""
    if override:
        if any(s==override for _,s in devices):return override
        raise RuntimeError(f'A1Z_WRIST_SERIAL={override} is not connected (found {devices})')
    d405=[s for n,s in devices if 'D405' in n]
    if known in d405:return known
    if len(d405)==1:return d405[0]
    raise RuntimeError(f'wrist D405 not found or ambiguous among RealSense devices {devices}; set A1Z_WRIST_SERIAL')


def enable_wrist(rs,cfg):
    """Bind cfg to the wrist D405 and return its serial."""
    devs=[(d.get_info(rs.camera_info.name),d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()]
    serial=pick_wrist(devs,override=os.environ.get('A1Z_WRIST_SERIAL'))
    cfg.enable_device(serial);return serial
