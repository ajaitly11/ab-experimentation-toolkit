# A/B Experimentation Toolkit (`abtk`)

A beginner-friendly Python toolkit for analysing A/B tests.

The goal of this repository is to provide a small set of tools for analysing experiments in a clean, reliable way.

---

## Current status

**Implemented**
- Mean (average) metric analysis: compare group means with an effect estimate, confidence interval, and p-value.
- Conversion metric analysis
- Ratio Metrics
- Sample Ratio Mismatch (SRM) Health Checks

**In progress**
- This repo will continue to grow in small, reviewable steps, with tests added alongside features.

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
- time on page (how long they stayed?)

The key idea is that user behaviour is noisy, so we need statistical tools to separate:
- real changes caused by the product change, from
- random variation.

---

## What this toolkit can do today

### 1) Mean metric analysis (average revenue, average time, etc.)

Use this when your metric is an “average per user” type of number, such as:
- revenue per visitor
- time on page (seconds)
- number of items added to cart

The main function is:

- `mean_diff(group_a, group_b)`

It returns:
- the sample size and mean of each group
- the **effect**: mean(B) - mean(A)
- a **95% confidence interval** for the effect
- a **p-value** for the hypothesis “the true effect is 0”

#### Example

```python
from abtk import mean_diff

# Example: revenue per visitor (in £)
# Realistic shape: lots of zeros, occasional purchases.
group_a = [0, 0, 0, 10, 0, 0, 50, 0, 0, 15, 0]
group_b = [0, 0, 5, 12, 0, 0, 60, 0, 0, 20, 0]

result = mean_diff(group_a, group_b)

print("Mean(A):", result.mean_a)
print("Mean(B):", result.mean_b)
print("Effect (B - A):", result.effect)
print("95% confidence interval:", (result.ci_low, result.ci_high))
print("p-value:", result.p_value)
```

### 2) Conversion rate analysis (binary metrics)

Use this when each user either converts or does not convert (for example: “made a booking”).

The main function is:

- `conversion_diff(group_a, group_b)`

It returns:
- conversion rate in each group
- effect = rate(B) minus rate(A)
- a confidence interval for the effect
- a p-value for the hypothesis “the true effect is 0”

#### Example

```python
from abtk import conversion_diff

# Example: booking conversion (1 = booked, 0 = did not book)
group_a = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
group_b = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1]

result = conversion_diff(group_a, group_b)

print("Rate(A):", result.rate_a)
print("Rate(B):", result.rate_b)
print("Effect (B - A):", result.effect)
print("95% confidence interval:", (result.ci_low, result.ci_high))
print("p-value:", result.p_value)
```
### 3) Ratio metric analysis (revenue per visitor, revenue per booking, and similar metrics)

Use this when your metric is a ratio of totals, such as:

- revenue per visitor
- bookings per visitor
- revenue per booking

The main function is:

- `ratio_diff(numerators_a, denominators_a, numerators_b, denominators_b)`

It supports two uncertainty methods:
- `method="delta"` (fast approximation)
- `method="bootstrap"` (resampling-based)

#### Example (revenue per visitor)

```python
from abtk import ratio_diff

# Each user contributes:
# numerator   = revenue
# denominator = 1 visitor
a_num = [0, 0, 50, 0, 0, 10, 0]
a_den = [1, 1, 1, 1, 1, 1, 1]

b_num = [0, 0, 60, 0, 0, 12, 0]
b_den = [1, 1, 1, 1, 1, 1, 1]

result = ratio_diff(a_num, a_den, b_num, b_den, method="delta")

print("Ratio(A):", result.ratio_a)
print("Ratio(B):", result.ratio_b)
print("Effect (B - A):", result.effect)
print("95% confidence interval:", (result.ci_low, result.ci_high))
print("p-value:", result.p_value)
```
### 4) Experiment health check: Sample Ratio Mismatch (SRM)

Before interpreting metric results, it is common to check whether the traffic split
matches the intended randomisation.

Use:

- `srm_check(count_a, count_b, expected_split=(...))`

Example:

```python
from abtk import srm_check

result = srm_check(60000, 40000, expected_split=(0.5, 0.5))

print("Expected A:", result.expected_a)
print("Expected B:", result.expected_b)
print("Chi-square:", result.chi2)
print("p-value:", result.p_value)
# If the p-value is extremely small, that is a signal to pause and investigate the
# experiment setup before trusting metric outcomes.
```

### 4) Power and sample size planning

The toolkit includes utilities to plan experiments and estimate power for:

- conversion rates (two-proportion planning)
- mean metrics (two-mean planning)

These functions support typical A/B planning questions like:
- “How many users per group do we need to detect a +1 percentage point lift?”
- “Given 20,000 users per group, what power do we have to detect this effect?”

### 5) CUPED variance reduction for mean metrics

CUPED uses a pre-experiment covariate (measured before treatment assignment) to reduce
noise in a mean metric.

Typical example:
- metric: revenue per visitor during the experiment
- covariate: revenue per visitor in the week before the experiment

Use:
- `cuped_mean_diff(metric_a, metric_b, covariate_a, covariate_b)`

It returns the CUPED coefficient (theta) and the usual mean test outputs on the adjusted metric.

⸻

What this toolkit aims to include (planned scope)

This section is the intended scope of the toolkit.

	-	Mean metrics (e.g., average revenue) analysis with confidence intervals ✅
	-	Binary metrics (e.g., conversion rate) analysis ✅
	-	Ratio metrics (common in product analytics) ✅
	-	Experiment health checks (Sample Ratio Mismatch (SRM) detection) ✅
	-	Sample size and power calculators ✅
	-	CUPED variance reduction ✅
	-	Multiple testing corrections
	-	Beginner-friendly reporting outputs

⸻

Installation (development)

This project uses Python 3.10+.

Clone the repo and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install development tools:

```bash
pip install pytest ruff black
```

Run tests:
```bash
pytest
```

Style checks:
```bash
ruff check .
black --check .
```

Format code:
```bash
black .
```

⸻

Repository structure

	-	src/abtk/ — library code
	-	tests/ — unit tests
	-	.github/workflows/ — CI checks (lint, formatting, tests)
