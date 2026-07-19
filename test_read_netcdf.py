from metpy.cbook import get_test_data
from xarray_sql import XarrayContext

path = get_test_data('irma_gfs_example.nc', False)
print(type(path), path)

ctx = XarrayContext.read_netcdf(get_test_data('irma_gfs_example.nc', False))
print(ctx._registered_datasets.keys())

result = ctx.sql("SELECT * FROM irma_gfs_example.time1_isobaric1_latitude_longitude LIMIT 5")
print(result)