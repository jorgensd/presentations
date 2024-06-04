import pandas
import argparse
import seaborn.objects
seaborn.set_theme(font_scale=1.2)
import matplotlib.pyplot as plt
from pathlib import Path
argparser = argparse.ArgumentParser()
argparser.add_argument("--folder", "-f", type=str, required=True)
argparser.add_argument("-p", "--problem", type=str, choices=["heat", "curl"], required=True)
argparser.add_argument("--degree", "-d", type=int, required=True)
argparser.add_argument("-N", type=int, required=True)
argparser.add_argument("--mpi_size",  "-m", type=int, required=True)
argparser.add_argument("--ymin", type=float, default=None)
argparser.add_argument("--ymax", type=float, default=None)

args = argparser.parse_args()
ymin = args.ymin
ymax = args.ymax

out_dir = Path(f"{args.folder}")
tables = []
for backend in ["dolfin", "dolfinx"]:
    in_file = (out_dir / f"{backend}_{args.N}_{args.degree}_{args.problem}_{args.mpi_size}").with_suffix(".txt")
    tables.append(pandas.read_csv(in_file, sep=" "))
table = pandas.concat(tables, ignore_index=True)
op_table = table.drop((table[table["Operation"] == "Total"]).index).drop((table[table["Operation"] == "Solve"]).index)

fig, ax = plt.subplots()


plot = seaborn.objects.Plot(op_table, x="Backend", y="Avg", color="Operation").add(
    seaborn.objects.Bar(), seaborn.objects.Stack(), legend=True)
# plot_min = plot.add(seaborn.objects.Line(marker="s"), x="Backend", y="Min", color="Operation")
# plot_max = plot_min.add(seaborn.objects.Line(marker="o"), x="Backend", y="Max", color="Operation")
plot_t = plot.label(title=f"Timing breakdown for {args.problem} with N={args.N}, degree={args.degree}, mpi_size={args.mpi_size}",
                    y="Time (s)")
plot_t.on(ax).plot()


out_file = (out_dir / f"timing_{args.N=}_{args.degree=}_{args.problem=}_{args.mpi_size=}").with_suffix(".png")

fig.savefig(out_file, bbox_inches="tight")
#plot_t.save(out_file,  bbox_inches='tight')