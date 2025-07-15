import pandas
import argparse
import seaborn
import numpy as np

seaborn.set_theme(font_scale=1.2)
import matplotlib.pyplot as plt
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", "-f", type=Path, required=True)
argparser.add_argument("--degree", "-d", type=int, default=1, help="Polynomial degree")

args = argparser.parse_args()

out_dir = Path(f"{args.file}")
dataframe = pandas.read_csv(out_dir)

filtered_df = dataframe.filter(regex=".max|num_procs|num_dofs|num_iterations|degree")

df_index = filtered_df[filtered_df["num_procs"] == 1].index

op_table = filtered_df.loc[df_index]

fig, ax = plt.subplots(figsize=(10, 6), dpi=400)


operations = op_table.filter(regex=".max").columns
new_df = {
    "Num processes": [],
    "#Dofs": [],
    "Operation": [],
    "num_iterations": [],
    "Time (s)": [],
    "degree": [],
}

for i, row in op_table.iterrows():
    for operation in operations:
        new_df["Num processes"].append(str(int(row["num_procs"])))
        new_df["#Dofs"].append(int(row["num_dofs"]))
        new_df["num_iterations"].append(int(row["num_iterations"]))
        new_df["Operation"].append(operation.split("_")[0])
        new_df["Time (s)"].append(row[operation])
        new_df["degree"].append(int(row["degree"]))

new_df = pandas.DataFrame.from_dict(new_df)

num_processes = np.unique(new_df["Num processes"].array)
assert len(num_processes) == 1, "Expected only one number of processes in the data"
new_df = new_df[new_df["degree"] == args.degree]
plt.title(f"Timing breakdown for {num_processes[0]} process for P{args.degree}")

seaborn.lineplot(
    new_df,
    x="#Dofs",
    y="Time (s)",
    hue="Operation",
    ax=ax,
    style="Operation",
    markers=True,
    dashes=False,
    markersize=10,
    zorder=1,
)

for pos, row in new_df.iterrows():
    if row["Operation"] == "solve":
        ax.annotate(
            f"{row['num_iterations']}",
            (row["#Dofs"], row["Time (s)"] * 1.5),
            color="k",
            va="top",
            ha="center",
            zorder=10,
        )

ax.set_xscale("log")
ax.set_yscale("log")
out_file = f"timing_{args.degree}.png"
ax.grid(True)

fig.savefig(out_file, bbox_inches="tight")
