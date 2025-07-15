import pandas
import argparse
import seaborn
import numpy as np

seaborn.set_theme(font_scale=1.2)
import matplotlib.pyplot as plt
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", "-f", type=Path, required=True)

args = argparser.parse_args()

out_dir = Path(f"{args.file}")
dataframe = pandas.read_csv(out_dir)

filtered_df = dataframe.filter(regex=".max|num_procs|num_dofs|num_iterations")

df_index = filtered_df[filtered_df["num_procs"] == 1].index

op_table = filtered_df.loc[df_index]

fig, ax = plt.subplots()


operations = op_table.filter(regex=".max").columns
new_df = {
    "num_procs": [],
    "num_dofs": [],
    "operation": [],
    "num_iterations": [],
    "time": [],
}
for i, row in op_table.iterrows():
    for operation in operations:
        new_df["num_procs"].append(int(row["num_procs"]))
        new_df["num_dofs"].append(int(row["num_dofs"]))
        new_df["num_iterations"].append(int(row["num_iterations"]))
        new_df["operation"].append(operation.split("_")[0])
        new_df["time"].append(row[operation])

new_df = pandas.DataFrame.from_dict(new_df)
num_processes = np.unique(new_df["num_procs"].array)
assert len(num_processes) == 1, "Expected only one number of processes in the data"
plt.title(f"Timing breakdown for {num_processes[0]} process")

seaborn.lineplot(new_df, x="num_dofs", y="time", hue="operation", ax=ax)

ax.set_xscale("log")
ax.set_yscale("log")
out_file = "timing.png"
ax.grid(True)

fig.savefig(out_file, bbox_inches="tight")
