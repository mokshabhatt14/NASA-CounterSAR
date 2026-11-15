import xarray as xr

ds = xr.open_dataset("data/processed/s1_cube.nc")
df = ds[["vv_db", "vh_db", "rainfall", "elevation"]].to_dataframe().reset_index()
df = df.dropna(subset=["vv_db", "vh_db"])
df["time"] = df["time"].dt.strftime("%Y-%m-%d")
df = df[["time", "y", "x", "vv_db", "vh_db", "rainfall", "elevation"]]
df.to_parquet("data/processed/s1_table.parquet", index=False)
print(df.shape, df.columns.tolist())