from pathlib import Path

import arviz as az

from vangja_simple.components import FourierSeasonality, LinearTrend

model = (
    LinearTrend()
    + FourierSeasonality(period=365.25, series_order=10)
    + FourierSeasonality(period=7, series_order=3)
)

out = ""
ess_out = ""

for method in [
    "metropolis",
    "demetropolisz",
    # "nuts",
    # "fullrank_advi",
    # "advi",
    # "svgd",
    # "asvgd",
]:
    model.load_model(Path("./") / "models" / "methods_2" / f"{method}")
    ess = az.ess(model.trace)
    rhat = az.rhat(model.trace)
    # print(f"Method: {method}, ess: {ess}, rhat: {rhat}")
    out += f"\\textit{{{method}}}"
    ess_out += f"\\textit{{{method}}}"
    for key in [
        "lt_0 - slope",
        "lt_0 - intercept",
        "lt_0 - delta",
        "fs_0 - beta(p=365.25,n=10)",
        "fs_1 - beta(p=7,n=3)",
    ]:
        out += f" & {float(rhat[key].mean()):.4f}"
        ess_out += f" & {float(ess[key].mean()):.2f}"

    out += "\\\\ \n \\hline \n"
    ess_out += "\\\\ \n \\hline \n"

print("R-hat")
print(out)

print("ESS")
print(ess_out)
