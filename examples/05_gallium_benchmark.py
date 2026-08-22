"""Gau–Viskanta / Brent gallium cavity plots from results/validate.json."""

import json
from pathlib import Path

from pcm.report import plot_validate
from pcm.published import GAU_VISKANTA_1986


def main() -> None:
    path = Path("results/validate.json")
    if not path.exists():
        from pcm.validate import check_gallium, check_neumann_series, check_stefan

        report = {
            "stefan": check_stefan(),
            "neumann_series": check_neumann_series(),
            "gallium_gau_viskanta": check_gallium(),
        }
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    else:
        report = json.loads(path.read_text())
    plot_validate(report, Path("paper/figures"))
    gal = report["gallium_gau_viskanta"]
    print("stefan", report.get("stefan"))
    print("gallium convection ratio", gal["convection_volume_ratio_17min"])
    print("Gau–Viskanta factor", GAU_VISKANTA_1986["late_time_volume_over_neumann"])
    print("wrote paper/figures/neumann_series.png and gallium_gau_viskanta.png")


if __name__ == "__main__":
    main()
