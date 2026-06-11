# IDA mechanical-presentation

Presentation for IDA (https://ida.dk/en/arrangementer-og-kurser/arrangementer/open-source-simulation-tools-opportunities-and-limitations-363941)

Link to code used in live demo (https://github.com/jorgensd/ida-presentation/tree/main/live_demo)

## Resources regarding FEniCS

Thank you for giving us the opportunity to speak to the IDA Mechanical members yesterday.
Following is a list of sources that could be useful for the members.

Link to FEniCS webpage: https://fenicsproject.org/
Link to user forum: https://fenicsproject.discourse.group/

For people that really want to know how to use DOLFINx, I would go through material in this order:

1. DOLFINx tutorial: https://jsdokken.com/dolfinx-tutorial/
2. Mechanical tours in DOLFINx: https://bleyerj.github.io/comet-fenicsx/
3. Shell models in DOLFINx: https://fenics-shells.github.io/fenicsx-shells/
4. MultiPointConstraints in DOLFINx: https://jsdokken.com/dolfinx_mpc/https://github.com/jorgensd/dolfinx_mpc
5. Workshop notes on how DOLFINx works (https://jsdokken.com/FEniCS-workshop/README.html)
6. Non-conforming 3D-1D coupling: https://scientificcomputing.github.io/fenicsx_ii
7. Toolbox with convenience tools on top of FEniCS: https://scientificcomputing.github.io/scifem
8. Further toolbox on top of FEniCS: https://dolfiny.uni.lu/

You can either use marp for vscode, or generate the presentation from commandline with:

```bash
npx @marp-team/marp-cli@latest presentation.md -o presentation.html --html
```

Server with live updates

```bash
npx @marp-team/marp-cli@latest -s .  --html
```
