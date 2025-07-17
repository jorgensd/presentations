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
filtered_df = filtered_df[filtered_df["degree"] == args.degree]
max_dofs = np.max(filtered_df["num_dofs"].array)


df_index = filtered_df[filtered_df["num_dofs"] == max_dofs].index

op_table = filtered_df.loc[df_index]

fig, ax = plt.subplots(figsize=(10, 6), dpi=400)


operations = op_table.filter(regex=".max").columns
new_df = {
    "Num processes": [],
    "num_dofs": [],
    "Operation": [],
    "num_iterations": [],
    "Time (s)": [],
    "degree": [],
}
for i, row in op_table.iterrows():
    for operation in operations:
        new_df["Num processes"].append(str(int(row["num_procs"])))
        new_df["num_dofs"].append(int(row["num_dofs"]))
        new_df["num_iterations"].append(int(row["num_iterations"]))
        new_df["Operation"].append(operation.split("_")[0])
        new_df["Time (s)"].append(row[operation])
        new_df["degree"].append(int(row["degree"]))

new_df = pandas.DataFrame.from_dict(new_df)
num_dofs = np.unique(new_df["num_dofs"].array)
assert len(num_dofs) == 1, "Expected only one number of processes in the data"
plt.title(f"Timing breakdown for {num_dofs[0]:.2e} dofs for P{args.degree}")
new_df = new_df[new_df["degree"] == args.degree]

breakpoint()
operations = np.unique(new_df["Operation"].array)

seaborn.lineplot(
    new_df,
    x="Num processes",
    y="Time (s)",
    hue="Operation",
    hue_order=operations,
    ax=ax,
    style="Operation",
    markers=True,
    dashes=False,
    markersize=10,
)


proc_counts = np.unique(new_df["Num processes"].array).astype(np.int64)
operations = np.unique(new_df["Operation"].array)
for i, operation in enumerate(operations):
    filtered_op = new_df[new_df["Operation"] == operation]
    op_procs = filtered_op["Num processes"].astype(np.int64)
    ref_time = filtered_op[filtered_op["Num processes"] == str(np.min(op_procs))][
        "Time (s)"
    ]
    ref_line = np.array(
        [ref_time * 1 / (proc / np.min(op_procs)) for proc in proc_counts]
    )
    arg_sort = np.argsort(proc_counts)
    ax.plot(
        proc_counts[arg_sort].astype(str),
        ref_line[arg_sort],
        label=f"{operation} reference",
        linestyle="--",
        color=seaborn.color_palette()[i],
        zorder=0,
    )

for pos, row in new_df.iterrows():
    if row["Operation"] == "solve":
        ax.annotate(
            f"{row['num_iterations']}",
            (row["Num processes"], row["Time (s)"] * 1.5),
            color="k",
            va="top",
            ha="center",
            zorder=10,
        )

# ax.set_xscale("log")
ax.set_yscale("log")
out_file = f"timing_processes_{args.degree}.png"
ax.grid(True, which="both", ls="--", c="gray")

fig.savefig(out_file)  # , bbox_inches="tight")
