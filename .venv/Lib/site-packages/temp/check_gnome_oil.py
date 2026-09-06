
from pathlib import Path

import adios_db.scripting as ads

from adios_db.computation.gnome_oil import make_gnome_oil


ID = "AD00005"

oil_path = Path("../../../noaa-oil-data/data/oil/AD/AD00005.json")

oil = ads.Oil.from_file(oil_path)

go = make_gnome_oil(oil)


