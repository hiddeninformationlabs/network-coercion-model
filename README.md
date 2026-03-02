\# Structural Unfreedom in Social Contagion

> How network topology determines coerced adoption

**Paper:** Matziorinis, A.M. (2026).
**Author:** Anna Maria Matziorinis — anna.matziorinis@uib.no
**Affiliations:** Hidden Information Labs Institute, Vancouver, Canada | Department of Clinical and Biological Psychology, University of Bergen

---

## The problem

When 80% of a population adopts a new technology, standard diffusion models report one number: 80%. They don't tell you how many of those people adopted because they wanted to, and how many adopted because their network position left them no realistic alternative.

This matters. A coerced adopter and a willing adopter look identical in every existing contagion metric. But they represent fundamentally different outcomes for individual autonomy, and they respond to different structural features of the network.

## What this model does

We split adoption into two measurable categories:

- **Willing adoption**: the agent adopted and their private preference was favorable
- **Coerced adoption**: the agent adopted but their private preference was against it

This operationalizes the philosopher Harry Frankfurt's (1971) concept of *structural unfreedom*: you act unfreely when you could not have done otherwise, given the structural conditions you're in, even if you don't feel constrained.

We then ask: what predicts which outcome an agent gets? The answer turns out to be network topology.

## Core findings

**Clustering protects.** Agents in tightly-knit neighborhoods (high clustering coefficient) are significantly less likely to be coerced. This inverts a canonical result from Centola (2007), who showed clustering *facilitates* complex contagion. The same structural feature that helps things spread also shields people from being overridden. The mechanism is symmetric: clustering keeps neighbors aligned, and whether that alignment accelerates adoption or blocks coercion depends on the composition of the local neighborhood.

**Who your neighbors are matters more than who you are.** Resistant neighbor fraction (the proportion of your contacts who privately oppose adoption) is the strongest structural predictor of protection from coercion (OR = 0.638, p < 10⁻¹¹¹). Stronger than degree, betweenness, or clustering alone.

**Topology effects are regime-dependent.** At low social pressure, everyone resists regardless of network shape. At extreme pressure, everyone capitulates. The window where network topology determines outcomes sits at intermediate pressure, peaking at β₁/β₄ = 4.5 (η² = 0.137). This is where most real social systems likely operate.

**The effect strengthens with scale.** Watts-Strogatz (clustered) networks produce significantly less coercion than Erdős-Rényi (random) networks, and this gap widens from Cohen's d = 0.99 at n = 300 to d = 2.04 at n = 2,000.

**Preference drift makes it worse, not better.** When agent preferences are allowed to shift toward their neighbors' views (DeGroot-style updating), coercion rises from 52.5% to 62.7% of adopters. A small number of agents (155 across all fast-drift runs) have their preferences fully rewritten: they end up adopting *willingly* by the time they adopt, but only because sustained network pressure changed what they wanted. These agents are invisible to any post-hoc measurement.

**The findings don't depend on how you define coercion.** Three different thresholds for classifying coercion (strict, original, lenient) produce the same predictor rankings, same directions, same significance levels.

**Counterfactual experiments confirm structural causation.** Reshuffling utilities across network positions (breaking any preference-topology correlation) preserves RNF's predictive power (OR = 0.581, p < 0.001). Rewiring networks to matched-density random graphs eliminates the topology gap entirely. The protective effect is genuinely structural, not compositional. Spatial autocorrelation analysis (Moran's I) further shows that coercion is spatially clustered in ways a null model cannot reproduce.

## Reproducing the results

### Setup

```bash
git clone https://github.com/hiddeninformationlabs/network-coercion-model.git
cd network-coercion-model
pip install -r requirements.txt
```

Python 3.9+. Dependencies: numpy, pandas, networkx, matplotlib, scipy, statsmodels, scikit-learn.

### Option A: Run analysis on pre-generated data

All datasets are included in `data/`. Every number in the paper traces to one of two scripts:

```bash
python scripts/analyze_production.py   # Tables 1-5, regressions, regime analysis
python scripts/analyze_extended.py     # Drift, scale robustness, threshold robustness
```

### Option B: Regenerate everything from scratch

```bash
python scripts/run_production.py       # ~15 min. Datasets 1-3.
python scripts/run_extended.py         # ~45 min. Datasets 4-6 (drift, scale to n=2000, thresholds).
python scripts/run_counterfactual.py   # ~2 min. Datasets 7-8 (decomposition, Moran's I).
```

Then run the analysis scripts above on the fresh output.

## Verifying specific claims

Every number in the paper maps to a computation you can run. Two examples:

**Table 2: RNF OR = 0.638**
```python
import pandas as pd
from scipy.stats import zscore
import statsmodels.api as sm

df = pd.read_csv('data/prod_agents.csv')
peak = df[df['regime'] == 'peak']
resistant = peak[peak['private_utility'] < 0].copy()

features = ['resistant_neighbor_frac', 'clustering_coeff', 'betweenness', 'degree']
for f in features:
    resistant[f] = zscore(resistant[f])
resistant['abs_private_utility'] = zscore(resistant['private_utility'].abs())

X = sm.add_constant(resistant[['abs_private_utility'] + features])
y = resistant['is_coerced']
model = sm.Logit(y, X).fit()
print(model.summary())
```

**Table 5: ER vs WS at n = 2,000**
```python
scale = pd.read_csv('data/ext_scale_summaries.csv')
large = scale[scale['n_agents'] == 2000]
er = large[large['topology'] == 'ER']['coercion_rate']
ws = large[large['topology'] == 'WS']['coercion_rate']

from scipy.stats import ttest_ind
t, p = ttest_ind(er, ws)
d = (er.mean() - ws.mean()) / ((er.std()**2 + ws.std()**2) / 2)**0.5
print(f"Cohen's d = {d:.2f}, p = {p:.1e}")
```

Full implementations in `scripts/analyze_production.py` and `scripts/analyze_extended.py`.

## How the model works

Each agent has a **private utility** drawn from N(-0.1, 0.3), representing genuine preference. Negative means they oppose adoption. Social pressure accumulates from four sources:

```
U_i(t) = u_i + β_local · ℓ_i(t) · exp(-f_i(t)) + β_global · g(t) + β_memory · m_i(t) - β_resist · r_i
```

| Term | What it captures |
|---|---|
| u_i | Private preference |
| β_local · ℓ_i(t) | Peer pressure from adopted neighbors, with fatigue decay |
| β_global · g(t) | Global adoption signal |
| β_memory · m_i(t) | Cumulative exposure history |
| β_resist · r_i | Individual conviction drag |

When total utility exceeds the agent's threshold and their private utility is negative, they're classified as **coerced**. Same threshold crossing with positive private utility is **willing**. This is the Frankfurt operationalization: the agent adopted, but the structural conditions of their network position determined the outcome, not their own preference.

Three topologies, all matched on mean degree (k ≈ 6):

| Topology | Properties | Coercion behavior |
|---|---|---|
| Erdős-Rényi | Random, low clustering | Highest coercion, low variance |
| Barabási-Albert | Scale-free, hubs | Mid coercion, high variance (hub-dependent) |
| Watts-Strogatz | High clustering, small-world | Lowest coercion, protective |

## Repository structure

```
├── network_coercion/          # Simulation library
│   ├── agents.py              # Agent class, 6-state model
│   ├── networks.py            # ER, BA, WS generation
│   ├── world.py               # Agents + topology, structural features
│   ├── engine.py              # Contagion dynamics, utility equation
│   ├── metrics.py             # Output DataFrames
│   ├── sweeps.py              # Parameter sweeps
│   └── analysis.py            # Regression, effect sizes
├── scripts/
│   ├── run_production.py      # Generate datasets 1-3
│   ├── run_extended.py        # Generate datasets 4-6
│   ├── run_counterfactual.py  # Generate datasets 7-8
│   ├── analyze_production.py  # Paper Tables 1-5, regressions
│   └── analyze_extended.py    # Drift, scale, threshold analysis
├── data/                      # Pre-generated CSVs (~57 MB)
│   └── README.md              # Complete data dictionary
├── paper/
│   ├── network_coercion_paper.tex
│   └── network_coercion_paper.pdf
├── tests/                     # Development verification scripts
├── figures/                   # All 7 paper figures (PDF + PNG)
├── requirements.txt
├── CITATION.cff
└── LICENSE                    # MIT
```

## Datasets

| # | File | Rows | What it tests |
|---|---|---|---|
| 1 | `prod_agents.csv` | 80,925 | Core: 3 topologies × 3 regimes × 30 seeds, n = 300 |
| 2 | `prod_beta_sweep.csv` | 2,700 | Regime mapping across 29 β₁/β₄ ratios |
| 3 | `prod_scale_sweep.csv` | ~900 | Scale robustness: n = 200, 400, 800 |
| 4 | `ext_drift_agents.csv` | 81,001 | Preference drift: δ = 0.0, 0.02, 0.05 |
| 5 | `ext_scale_summaries.csv` | 181 | Large scale: n = 500, 1,000, 2,000 |
| 6 | `ext_threshold_agents.csv` | 27,001 | Three coercion definitions |
| 7 | `counterfactual_agents.csv` | 81,000 | Structural vs compositional decomposition |
| 8 | `counterfactual_morans.csv` | 90 | Spatial autocorrelation: model vs null |

Full column definitions in [`data/README.md`](data/README.md).

## Citation

```bibtex
@article{matziorinis2026structural,
  title={Structural Unfreedom in Social Contagion:
         How Network Topology Determines Coerced Adoption},
  author={Matziorinis, Anna Maria},
  year={2026},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

Anna Maria Matziorinis — anna.matziorinis@uib.no
Hidden Information Labs Institute, Vancouver, Canada
Department of Clinical and Biological Psychology, University of Bergen
