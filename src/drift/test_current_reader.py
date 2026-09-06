from datetime import timedelta
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic


# 1. Open the Copernicus current data
reader = reader_netCDF_CF_generic.Reader(
    "data/currents/test_currents.nc"
)

print("Reader created successfully!")
print(reader)


# 2. Create OpenDrift model
o = OceanDrift(loglevel=20)

# 3. Add the current-data reader
o.add_reader(reader)

print("OpenDrift reader connected successfully!")

# 4. Release one test oil particle
o.seed_elements(
    lon=72.5,
    lat=7.5,
    time=reader.start_time,
    number=1,
)

# 5. Run the drift simulation for 1 day
o.run(
    duration=timedelta(days=1),
    time_step=timedelta(hours=1),
    time_step_output=timedelta(hours=1),
)

# 6. Print final position
print("Final longitude:", o.elements.lon[-1])
print("Final latitude:", o.elements.lat[-1])

import pandas as pd

trajectory = pd.DataFrame({
    "time": o.result.time,
    "longitude": o.result.lon[0],
    "latitude": o.result.lat[0],
})

trajectory.to_csv(
    "data/currents/test_drift_trajectory.csv",
    index=False
)

print("Trajectory saved successfully!")
print(trajectory)
