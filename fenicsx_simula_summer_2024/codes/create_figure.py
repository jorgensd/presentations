import pandas
import argparse
import seaborn
import matplotlib.pyplot as plt
from pathlib import Path
argparser = argparse.ArgumentParser()
argparser.add_argument("--input", type=str, required=True)
args = argparser.parse_args()

table = pandas.read_csv(args.input, sep=" ")
f = plt.figure()
ax = plt.gca()

ax.set(yscale="log")
seaborn.set(style="ticks")
seaborn.set(font_scale=1.2)
seaborn.set_style("darkgrid")
g = seaborn.catplot(x="Operation", y="Avg", kind="bar", data=table)
g.fig.set_size_inches(16, 4)

g.set(title=f"Timing breakdown for {args.input}")
g.map_dataframe(seaborn.swarmplot, x="Operation", y="Min", color="r", data=table)
g.map_dataframe(seaborn.swarmplot, x="Operation", y="Max", color="g", data=table)



plt.savefig(Path(args.input).with_suffix(".png"))