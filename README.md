# A/B Experimentation Toolkit (abtk)

A beginner-friendly Python toolkit for analyzing A/B tests.

This repo is to demonstrate the following:
- how online experiments (A/B tests) are analyzed,
- what p-values and confidence intervals mean 

---

## What is an A/B test?

An **A/B test** compares two versions of something:
- **A** = control (current experience)
- **B** = treatment (new experience)

Example:
- A: existing booking button
- B: new booking button

We send some users to A and some users to B, then compare outcomes like:
- conversion rate (did they book?)
- revenue (how much they spent?)

---

## What will this toolkit include?

- Mean metrics (e.g., average revenue) analysis with confidence intervals
- Binary metrics (e.g., conversion rate) analysis
- Ratio metrics (common in product analytics)
- Experiment health checks (e.g., SRM detection)
- Sample size / power calculators
- CUPED variance reduction
- Multiple testing corrections
- Beginner-friendly reporting outputs

---

## Installation (dev)

This project uses Python 3.10+.

Clone the repo and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install pytest ruff black