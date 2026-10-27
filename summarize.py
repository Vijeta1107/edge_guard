"""Print the headline numbers from results/results_<arch>.json (used for the slides)."""
import json, sys
import pandas as pd
arch = sys.argv[1] if len(sys.argv) > 1 else "edgenet"
R = json.load(open(f"results/results_{arch}.json"))
a = R["accuracy"]; print("clean acc", round(a["clean"], 4)); print({c: round(v, 3) for c, v in a["per_corruption"].items()}); print({k: round(v, 3) for k, v in a["per_severity"].items()})
for k in ["failure_detection_all_ID", "failure_detection_seen_corruptions", "failure_detection_heldout_corruptions"]:
    print(k)
    for m, v in R[k].items(): print(f"  {m:22s} AUROC {v['AUROC']:.3f} AUPR {v['AUPR']:.3f} FPR95 {v['FPR95']:.3f}")
for on, t in R["ood_detection"].items():
    print("OOD", on)
    for m, v in t.items(): print(f"  {m:22s} AUROC {v['AUROC']:.3f} FPR95 {v['FPR95']:.3f}")
for k, v in R["decision_metrics"].items(): print(k, {kk: round(vv, 3) for kk, vv in v.items()})
print(R["plain_model_silent_failures"])
for f, v in R["fixed_vs_adaptive"].items(): print(f, v)
print({m: round(v["AURC"], 4) for m, v in R["risk_coverage"].items()})
print({k: {kk: round(vv["ece"], 4) for kk, vv in v.items()} for k, v in R["calibration_ece"].items()})
print("explanation", R["explanation"]["accuracy"], R["explanation"]["n"], R["explanation"]["per_class"])
for k, v in R["ablation"].items(): print(k, {kk: round(vv, 3) for kk, vv in v.items()})
for s, r in R["blind_spot_memory"]["streams"].items():
    print(s, {k: (round(v["missed_failure_rate"], 4), round(v["false_alarm"], 4), v["memory_entries"], v["memory_kb"]) for k, v in r.items()})
print({k: R["calibration"][k] for k in ["tau_fixed", "tau_by_condition", "fusion_weights", "fail_rate"]})
print(R["efficiency"])
