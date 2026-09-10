# Check A reproduction notes

Bastani, Bastani, Sungu, Ge, Kabakcı, and Mariman (2025), *Generative AI without guardrails can harm learning: Evidence from high school mathematics*, PNAS. Replication package: `GenAICanHarmLearning-main/` (GitHub `obastani/GenAICanHarmLearning`).

This folder contains **only Check A**. Authors' scripts and raw data were not modified. `additional_results/` and `text_analysis/` were not used.

## 1. Official reproduction instructions (README)

The README is primarily a data dictionary, not a numbered cookbook.

- Main analyses live in `main_regressions/` (R scripts plus analysis datasets).
- `additional_results/` covers balance, perception, heterogeneity, dispersion, absenteeism (Python/Stata). **Not used here.**
- `text_analysis/` covers student–GPT messages and GPT error rates. **Not used here.**
- The student-level analysis file is `main_regressions/final_data.csv` (also referenced as `final_data.csv` at the repo root in the README; in this package it is under `main_regressions/`).
- Headline ITT is student-level Part 2 (assisted practice) and Part 3 (unassisted exam).
- Treatment indicators in the data dictionary: `GPTBase`, `GPTTutor`; `Treatment_arm` / `Treatment arm` is the three-arm label.
- Outcomes: `Part2Tot`, `Part3Tot`.
- Key covariates/IDs: `gpa_prev`, `teacher`, `Session`, `Grader`, `Year`, `Class`, `Honors`.

Scripts must be run with working directory `main_regressions/` because they read `final_data.csv` with a relative path.

## 2. `main_regressions/` inventory (Check A)

### Analysis datasets

| File | Role for Check A |
| --- | --- |
| `final_data.csv` | **Headline student-level ITT.** 3,255 rows (student × session). |
| `problem_part2.csv`, `problem_part3.csv` | Problem-level robustness (`problem_level_analysis.R`). Not required for headline Check A. |
| `problem_mapping.csv`, `gpt_answers_full.csv` | GPT error-rate mechanism. Not used. |

### R scripts

| Script | Role |
| --- | --- |
| `main_analysis.R` | **Headline Table 1 ITT** (first regression block). Also: drop non-compliers, survey covariates, include honors, pre-registered t-tests. |
| `problem_level_analysis.R` | Problem-level ITT and GPT error-rate interactions. Robustness / mechanism, not headline Check A. |

### Treatment coding

- `Treatment arm`: `control`, `vanilla`, `augmented`.
- `GPTBase` = 1 if vanilla (GPT Base); `GPTTutor` = 1 if augmented (GPT Tutor).
- **Control** = omitted category (`GPTBase = 0`, `GPTTutor = 0`); textbooks/notes, no GPT.
- **GPT Base** = unfettered ChatGPT-like tutor (`vanilla`).
- **GPT Tutor** = safeguarded tutor with hints and teacher-provided solutions (`augmented`).

### Outcomes

- **Assisted / practice:** `Part2Tot` ∈ [0, 1] (Part 2, GPT available in treated arms).
- **Later unassisted exam:** `Part3Tot` ∈ [0, 1] (Part 3, no AI).
- Scores are already normalized to the unit interval. Check A does **not** further standardize and does **not** subtract Part 2 from Part 3.

### Covariates, fixed effects, clustering (headline Eq. 1)

```
Part2Tot / Part3Tot ~ GPTBase + GPTTutor + gpa_prev + teacher + Session + Grader + Year
```

- Sample: drop `Honors == 1`.
- Covariate: previous GPA (`gpa_prev`).
- Fixed effects: `teacher`, `Session`, `Grader`, `Year`.
- Cluster-robust SEs: `sandwich::vcovCL(..., cluster = ~Class)`.
- Analysis N = 2,848 (2,899 non-honors rows minus 51 incomplete cases, almost all missing `gpa_prev`). 44 classrooms.

## 3. Which scripts reproduce the headline findings

| Headline | Script | Block | DV |
| --- | --- | --- | --- |
| **A.** Performance while AI is available | `main_analysis.R` | first `reg2` | `Part2Tot` |
| **B.** Later performance after AI is removed | `main_analysis.R` | first `reg3` | `Part3Tot` |

Published comparator: PNAS Table 1 (and the accompanying text). Practice: GPT Base 0.137 (0.031), GPT Tutor 0.361 (0.032). Exam: GPT Base −0.054 (0.022), GPT Tutor −0.004 (0.013). N = 2,848.

## 4. Original scripts run

Command (authors' file unmodified):

```
cd GenAICanHarmLearning-main/main_regressions
Rscript main_analysis.R
```

Full stdout: `checkA/original_main_analysis_output.txt`.

The first stargazer table (headline) is:

| | Part2Tot | Part3Tot |
| --- | ---: | ---: |
| GPTBase | 0.137*** (0.031) | −0.054** (0.022) |
| GPTTutor | 0.361*** (0.032) | −0.004 (0.013) |
| Observations | 2,848 | 2,848 |
| R² | 0.389 | 0.386 |

This matches PNAS Table 1 at the reported three-decimal rounding. (Stargazer stars use p<0.01 / 0.05 / 0.1; the PNAS table uses a slightly different star convention. Point estimates and SEs match.)

Later blocks in the same script (non-compliers, survey covariates, include honors, t-tests) were executed but are not the Check A headline.

Exact (unrounded) coefficients, cluster-robust SEs, 95% CIs, and p-values were extracted in `extract_checkA.R` using the **same** `lm` + `vcovCL` specification. CIs and p-values use a two-sided t-test with residual df = 2,813 (stargazer convention). With this df, t* ≈ 1.961, so CIs are essentially identical to ±1.96×SE.

## 5. Headline reproduction log

See `reproduction_log.csv`. Summary:

| Comparison | DV | Estimate | SE | 95% CI | p | N | Matches Table 1? |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| GPT Base vs Control | Part2Tot | 0.13702 | 0.03098 | [0.076, 0.198] | 1.01e-05 | 2848 | yes |
| GPT Tutor vs Control | Part2Tot | 0.36128 | 0.03173 | [0.299, 0.423] | 2.13e-29 | 2848 | yes |
| GPT Base vs Control | Part3Tot | −0.05435 | 0.02244 | [−0.098, −0.010] | 0.0155 | 2848 | yes |
| GPT Tutor vs Control | Part3Tot | −0.00432 | 0.01338 | [−0.031, 0.022] | 0.747 | 2848 | yes |

## 6. Check A model-constraint contrasts

Same original specification; no cross-assessment differencing; no extra standardization.

| Outcome | Contrast | Estimate | SE | 95% CI | p | N |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Assisted (Part 2) | GPT Base vs Control | 0.13702 | 0.03098 | [0.076, 0.198] | 1.01e-05 | 2848 |
| Assisted (Part 2) | GPT Tutor vs Control | 0.36128 | 0.03173 | [0.299, 0.423] | 2.13e-29 | 2848 |
| Later unassisted (Part 3) | GPT Base vs Control | −0.05435 | 0.02244 | [−0.098, −0.010] | 0.0155 | 2848 |
| Later unassisted (Part 3) | GPT Tutor vs Control | −0.00432 | 0.01338 | [−0.031, 0.022] | 0.747 | 2848 |
| Later unassisted (Part 3) | GPT Tutor vs GPT Base | 0.05003 | 0.02331 | [0.004, 0.096] | 0.0319 | 2848 |

The Tutor−Base exam contrast is a linear combination `GPTTutor − GPTBase` from the Part 3 regression, using the clustered covariance matrix (not a new specification).

Files: `checkA_results.csv`, `checkA_coefficient_plot.png`.

## 7. Interpretation as an M2 qualitative constraint only

Tentative **proxies** (not exact M2 workflows):

- Control ≈ H-like independent work
- GPT Base ≈ L/D-like answer-oriented AI support
- GPT Tutor ≈ A/W-like cognitively engaged / scaffolded AI support

**Do not read numerical attractors from these coefficients.**

The object that maps to a production-competence *transition* is later **unassisted** performance (Part 3), not assisted practice scores. Part 2 is contemporaneous output with the tool available.

Qualitative pattern on Part 3:

- GPT Base < Control (harm relative to independent practice).
- GPT Tutor ≈ Control (no detectable exam gain or loss vs independent practice).
- GPT Tutor > GPT Base (p = 0.032).

That last contrast is the relevant ordering for

\[
m_A^c > m_L^c
\]

as a **qualitative constraint**: later unassisted performance is higher after scaffolded/co-production-like support than after answer-oriented/delegation-like support.

What the data do **not** support, and we do not claim:

- A numerical value for \(m_A^c\) or \(m_L^c\).
- \(m_A^c > m_H^c\): Tutor is not better than Control on the exam.
- That assisted Part 2 gaps are competence transitions.

## 8. Methodological notes / discrepancies

1. **Headline match:** no discrepancy with PNAS Table 1 at published rounding.
2. **Working-paper OCR:** some scraped Table 1 renderings show GPT Tutor exam as +0.004; the PNAS text and the authors' script both give **−0.004**.
3. **Unit of observation:** rows are student × session; clustering is at `Class` (assignment level), not student. This is the authors' spec.
4. **Missingness:** 51/2,899 non-honors rows dropped from OLS, primarily missing `gpa_prev`.
5. **CI/p-value construction:** not printed in the original stargazer table; we computed them from `vcovCL` with residual-df t critical values. Using G−1 = 43 cluster df does not change 5% significance of the five Check A contrasts.
6. **R was not preinstalled** on this machine; R 4.5.3 and CRAN packages (`readr`, `sandwich`, `lmtest`, `stargazer`, `fastDummies`, `readxl`) were installed via conda in order to run the original script. The original `.R` files were not edited.
)
