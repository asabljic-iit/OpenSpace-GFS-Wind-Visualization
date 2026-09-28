import struct
import os
import glob
import urllib.request
import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
import torch
import torch.nn.functional as F

# Automatically detect GPU if available, fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(torch.get_num_threads())
print(f"PyTorch using device: {DEVICE}")


R_EARTH = 6371000.0  # Earth radius in meters
ALTITUDE =  10000.0  # Altitude offset above surface in meters for fieldlines
# Set integration step time (e.g., 1800 seconds = 30 minutes of transport per step)
DT_SECONDS = 3600.0
NUM_STEPS = 40
NUM_FLOWLINES = 9000


def iso_to_j2000_seconds(iso_str):
    """Converts ISO timestamp string to J2000 seconds for OpenSpace time conversion."""
    # Format: YYYY-MM-DDThh:mm:ss.000
    cleaned_iso = iso_str.replace("-", "").replace(":", "").replace("T", " ")
    dt = datetime.strptime(cleaned_iso[:15], "%Y%m%d %H%M%S").replace(tzinfo=timezone.utc)

    # J2000 Epoch: 2000-01-01 12:00:00 UTC
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return (dt - j2000_epoch).total_seconds()

def trace_wind_fieldlines(ds, alt_m=ALTITUDE, device=DEVICE):
    # Extract U and V variables dynamically regardless of shortName
    u_var = [v for v in ds.data_vars if v.lower() in ['u10', 'u', 'u_component_of_wind']][0]
    v_var = [v for v in ds.data_vars if v.lower() in ['v10', 'v', 'v_component_of_wind']][0]

    lat_grid = ds.latitude.values
    lon_grid = ds.longitude.values
    u_data = ds[u_var].values
    v_data = ds[v_var].values

    # Handle descending latitude array if needed
    if lat_grid[0] > lat_grid[-1]:
        lat_grid = lat_grid[::-1]
        u_data = np.flip(u_data, axis=0)
        v_data = np.flip(v_data, axis=0)

    lat_min, lat_max = float(lat_grid[0]), float(lat_grid[-1])
    lon_min, lon_max = float(lon_grid[0]), float(lon_grid[-1])

    uv_tensor = torch.from_numpy(
        np.stack([u_data, v_data], axis=0)
    ).unsqueeze(0).float().to(device)

    np.random.seed(42)
    init_lats = np.degrees(np.arcsin(np.random.uniform(-0.995, 0.995, NUM_FLOWLINES)))
    init_lons = np.random.uniform(0, 360, NUM_FLOWLINES)

    curr_lats = torch.tensor(init_lats, dtype=torch.float32, device=device)
    curr_lons = torch.tensor(init_lons, dtype=torch.float32, device=device)

    all_steps_pts = []

    # Calculate physical radius for this specific altitude layer
    r_sphere = R_EARTH + alt_m

    for step in range(NUM_STEPS):
        lat_rad = torch.deg2rad(curr_lats)
        lon_rad = torch.deg2rad(curr_lons)

        cos_lat = torch.cos(lat_rad)
        x = r_sphere * cos_lat * torch.cos(lon_rad)
        y = r_sphere * cos_lat * torch.sin(lon_rad)
        z = r_sphere * torch.sin(lat_rad)

        norm_lon = 2.0 * (curr_lons - lon_min) / (lon_max - lon_min) - 1.0
        norm_lat = 2.0 * (curr_lats - lat_min) / (lat_max - lat_min) - 1.0

        grid_coords = torch.stack([norm_lon, norm_lat], dim=-1).unsqueeze(0).unsqueeze(0)
        uv_sampled = F.grid_sample(uv_tensor, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=True)

        u_vals = uv_sampled[0, 0, 0, :]
        v_vals = uv_sampled[0, 1, 0, :]
        speeds = torch.sqrt(u_vals**2 + v_vals**2)

        step_data = torch.stack([x, y, z, speeds], dim=-1)
        all_steps_pts.append(step_data)

        d_lat = (v_vals * DT_SECONDS) / 111000.0
        safe_cos = torch.clamp(torch.abs(cos_lat), min=0.01)
        d_lon = (u_vals * DT_SECONDS) / (111000.0 * safe_cos)

        curr_lats = torch.clamp(curr_lats + d_lat, min=-85.0, max=85.0)
        curr_lons = torch.remainder(curr_lons + d_lon, 360.0)

    flowlines_tensor = torch.stack(all_steps_pts, dim=0).permute(1, 0, 2).cpu().numpy()
    fieldlines = [line.tolist() for line in flowlines_tensor]
    return fieldlines

def export_to_osfls(osfls_path, fieldlines, trigger_time_j2000, scalar_name="grid_value"):
    """Writes fieldlines directly to native OpenSpace OSFLS binary format."""
    n_lines = len(fieldlines)
    extra_names_bytes = scalar_name.encode('utf-8') + b'\0'
    byte_size_all_names = len(extra_names_bytes)

    line_start = []
    line_count = []
    vertex_positions = []
    extra_quantities = []

    curr_index = 0
    for line in fieldlines:
        pts_count = len(line)
        line_start.append(curr_index)
        line_count.append(pts_count)
        curr_index += pts_count

        for pt in line:
            # pt structure: [x, y, z, speed]
            vertex_positions.extend(pt[:3])
            extra_quantities.append(pt[3])

    n_points = curr_index

    # Convert arrays to standard C data types matching OpenSpace C++ structs
    line_start_arr = np.array(line_start, dtype=np.int32)
    line_count_arr = np.array(line_count, dtype=np.uint32)
    vertex_positions_arr = np.array(vertex_positions, dtype=np.float32)
    extra_quantities_arr = np.array(extra_quantities, dtype=np.float32)

    with open(osfls_path, 'wb') as ofs:
        # Header Fields
        ofs.write(struct.pack('i', 0))                    # CurrentVersion = 0
        ofs.write(struct.pack('d', trigger_time_j2000))    # _triggerTime
        ofs.write(struct.pack('i', 0))                    # _model = Invalid/Generic (0)
        ofs.write(struct.pack('B', 0))                    # _isMorphable = 0 (false)
        ofs.write(struct.pack('Q', n_lines))              # nLines
        ofs.write(struct.pack('Q', n_points))             # nPoints
        ofs.write(struct.pack('Q', 1))                    # nExtras (1 quantity: grid_value)
        ofs.write(struct.pack('Q', byte_size_all_names))  # byteSizeAllNames

        # Array Data Payload
        line_start_arr.tofile(ofs)
        line_count_arr.tofile(ofs)
        vertex_positions_arr.tofile(ofs)
        extra_quantities_arr.tofile(ofs)

        # String Data Payload
        ofs.write(extra_names_bytes)

LOOKBACK_HOURS = 72
FORECAST_HOURS = 48

now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start_time = now_utc - timedelta(hours=LOOKBACK_HOURS)
end_time = now_utc + timedelta(hours=FORECAST_HOURS)

print(f"Generating 5-day OSFLS sequence from {start_time} to {end_time}\n")

current_time = start_time

LEVEL_CONFIGS = {
    "1001": ({'shortName': '10u'}, {'shortName': '10v'}, 10.0),            # 10m Surface
    "700": ({'shortName': 'u', 'typeOfLevel': 'isobaricInhPa', 'level': 700},
            {'shortName': 'v', 'typeOfLevel': 'isobaricInhPa', 'level': 700}, 3000.0), # 700 hPa
    "250": ({'shortName': 'u', 'typeOfLevel': 'isobaricInhPa', 'level': 250},
            {'shortName': 'v', 'typeOfLevel': 'isobaricInhPa', 'level': 250}, 10500.0), # 250 hPa Jet Stream
    "10":  ({'shortName': 'u', 'typeOfLevel': 'isobaricInhPa', 'level': 10},
            {'shortName': 'v', 'typeOfLevel': 'isobaricInhPa', 'level': 10}, 31000.0) # 10 hPa
}

while current_time <= end_time:
    # 1. Determine base 6-hour cycle run (00, 06, 12, 18)
    cycle_hour = (current_time.hour // 6) * 6
    date_folder = current_time.strftime("%Y%m%d")
    cycle_str = f"{cycle_hour:02d}"

    # 2. Calculate forecast offset hour (f000, f001, ..., f005)
    f_hr = current_time.hour - cycle_hour
    f_str = f"f{f_hr:03d}"

    # 3. Calculate exact valid timestamp & OpenSpace J2000 trigger time
    trigger_time_j2000 = iso_to_j2000_seconds(current_time.strftime("%Y-%m-%dT%H:%M:%S.000"))
    filename_osfls = current_time.strftime("%Y-%m-%dT%H-%M-%S-000.osfls")

    print(f"Target Time: {current_time.strftime('%Y-%m-%d %H:00 UTC')} | Cycle: {date_folder} t{cycle_str}z | Offset: {f_str}")

    # 4. Construct S3 URL with forecast offset
    grib_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date_folder}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.{f_str}"
    local_grib_file = f"gfs.t{cycle_str}z.pgrb2.0p25.{f_str}"

    try:
        if not os.path.exists(local_grib_file):
            urllib.request.urlretrieve(grib_url, local_grib_file)

        for data_id, (filter_u, filter_v, alt_m) in LEVEL_CONFIGS.items():
                # Save files directly into server subfolder matching data_id
                output_dir = f"./osfls_files/{data_id}"
                os.makedirs(output_dir, exist_ok=True)

                output_filepath = os.path.join(output_dir, filename_osfls)

                # Open datasets with specific filters
                ds_u = xr.open_dataset(local_grib_file, engine='cfgrib', filter_by_keys=filter_u)
                ds_v = xr.open_dataset(local_grib_file, engine='cfgrib', filter_by_keys=filter_v)
                ds = xr.merge([ds_u, ds_v], compat='override')

                # Trace fieldlines using base ALTITUDE + layer offset to prevent terrain clipping
                effective_altitude = ALTITUDE + alt_m
                fieldlines = trace_wind_fieldlines(ds, alt_m=effective_altitude)

                # Export
                export_to_osfls(output_filepath, fieldlines, trigger_time_j2000, scalar_name="grid_value")
                print(f"Exported {data_id} ({alt_m}m) to {output_filepath}")

                # Clean up local GRIB download to conserve disk space
                ds.close()
                ds_u.close()
                ds_v.close()

        if os.path.exists(local_grib_file):
            os.remove(local_grib_file)

        for idx_file in glob.glob(f"{local_grib_file}*.idx"):
            if os.path.exists(idx_file):
                os.remove(idx_file)

    except Exception as e:
        print(f"     Failed to fetch {grib_url}: {e}\n")

    # Advance by 1 hour
    current_time += timedelta(hours=1)