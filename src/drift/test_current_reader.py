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