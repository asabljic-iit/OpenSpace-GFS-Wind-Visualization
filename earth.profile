{
  "addons": {
    "recommended": [
      "addons/asteroids",
      "addons/dwarf_planets",
      "addons/interstellar_objects",
      "addons/minor_moons",
      "addons/desi"
    ]
  },
  "assets": [
    "base",
    "base_keybindings",
    "scene/solarsystem/planets/earth/earth",
    "${USER_ASSETS}/scene/solarsystem/planets/earth/atmosphere/wind_fieldlines"
  ],
  "camera": {
    "altitude": 17000000.0,
    "anchor": "Earth",
    "latitude": 58.5877,
    "longitude": 16.1924,
    "type": "goToGeo"
  },
  "delta_times": [
    1.0,
    5.0,
    30.0,
    60.0,
    300.0,
    1800.0,
    3600.0,
    43200.0,
    86400.0,
    604800.0,
    1209600.0,
    2592000.0,
    5184000.0,
    7776000.0,
    15552000.0,
    31536000.0,
    63072000.0,
    157680000.0,
    315360000.0,
    630720000.0
  ],
  "mark_nodes": [
    "Earth",
    "Sun"
  ],
  "meta": {
    "author": "Amir Sabljic",
    "description": "Earth profile for OpenSpace. Contains wind fieldlines and other visualizations.",
    "license": "MIT License",
    "name": "Earth",
    "url": "https://github.com/asabljic-iit",
    "version": "1.0"
  },
  "properties": [
    {
      "name": "Scene.Earth.Renderable.Layers.Overlays.noaa-sos-overlays-latlon_grid-white.Enabled",
      "type": "setPropertyValueSingle",
      "value": "false"
    },
    {
      "name": "Scene.*Trail.Renderable.Enabled",
      "type": "setPropertyValue",
      "value": "false"
    },
    {
      "name": "Scene.Earth.Renderable.Layers.NightLayers.Earth_at_Night_2012.Opacity",
      "type": "setPropertyValueSingle",
      "value": "0.5000"
    },
    {
      "name": "Scene.EarthAtmosphere.Renderable.SunIntensity",
      "type": "setPropertyValueSingle",
      "value": "3.5000"
    }
  ],
  "time": {
    "is_paused": false,
    "type": "relative",
    "value": "-1d"
  },
  "version": {
    "major": 1,
    "minor": 1
  }
}