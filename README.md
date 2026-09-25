# OpenSpace NOAA GFS Wind Fieldline Visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asabljic-iit/OpenSpace-GFS-Wind-Visualization/blob/main/OpenSpaceAtmosphereWind.ipynb)

A PyTorch-accelerated data pipeline that fetches global 10-meter wind vector fields ($U/V$ components) from NOAA's Global Forecast System (GFS) AWS S3 bucket, traces 3D streamlines, and exports them directly into native OpenSpace binary fieldline (`.osfls`) time-series sequences or JSON files.

<img width="969" height="892" alt="image" src="https://github.com/user-attachments/assets/408a121c-2c86-4738-85dd-ebc4544af293" />

## Features

- **PyTorch CUDA Acceleration:** Traces thousands of streamlines in parallel via 2D bilinear tensor sampling (`torch.nn.functional.grid_sample`).
- **Hourly Historical & Forecast Loops:** Automatically targets published 6-hour GFS cycle runs (`00z`, `06z`, `12z`, `18z`) and resolves forecast offset steps (`f000`–`f005`) to assemble seamless hourly animation sequences.
- **JSON Exporter & Parser:** Generates intermediate OpenSpace-compatible JSON fieldline structures for inspectability, testing, or custom pipeline transformations.
- **Direct `.osfls` Binary Export:** Converts JSON structures or in-memory streamlines directly into OpenSpace `.osfls` binary files, converting ISO-8601 timestamps into J2000 epoch offsets and packing line vertices and scalar attributes into native C-struct byte streams.
- **OpenSpace Asset Included:** Includes a ready-to-use OpenSpace `.asset` script configured with a custom transfer function for color-mapping fieldline flow speed.

## Data Formats & Conversion Pipeline

### OpenSpace Fieldline JSON Structure
The notebook supports exporting streamlines to standard OpenSpace JSON formatting. Each fieldline index contains a valid ISO timestamp and a list of Cartesian ECEF coordinates $[X, Y, Z]$ paired with scalar quantities (e.g., wind speed in m/s):

```json
{
  "0": {
    "time": "2026-09-25T00:00:00.000",
    "trace": {
      "columns": ["x", "y", "z", "grid_value"],
      "data": [
        [-4213501.0, 312040.5, 4781200.0, 12.45],
        [-4213420.0, 312100.2, 4781280.0, 12.80],
        ...
      ]
    }
  },
  ...
}
```

### JSON to OSFLS Conversion (`convert_json_to_osfls`)
For large datasets or animation sequences, OpenSpace loads binary `.osfls` files significantly faster than raw JSON. The `convert_json_to_osfls()` helper parses JSON fieldline files and converts them into the binary spec expected by OpenSpace:

1. **J2000 Timestamp Conversion:** Translates ISO strings (e.g., `2026-09-25T00:00:00.000`) into total seconds relative to the J2000 epoch (`2000-01-01 12:00:00 UTC`).
2. **Binary Header Packing:** Writes C-compatible binary struct headers (`int32`, `uint32`, `float32`, `double`) containing total line counts, vertex counts, scalar quantity counts, and null-terminated attribute variable names.
3. **Contiguous Vertex Offsets:** Packs line start indices (`int32`), point counts per line (`uint32`), interleaved coordinate positions (`float32`), and scalar attribute arrays sequentially into disk storage.

## Quick Start

1. Open `OpenSpaceAtmosphereWind.ipynb` in Google Colab (with a GPU runtime) or run it locally in Jupyter Notebook.
2. Run the **Setup** section to initialize dependencies (`torch`, `cfgrib`, `xarray`, `eccodes`).
3. Execute the **Historical Loop** or **Forecasting Loop** cell to generate your sequence of `.osfls` binary files.
4. Move the exported `.osfls` files and the provided `wind-speed.txt` color table into your OpenSpace asset directory alongside `wind_fieldlines.asset`.
