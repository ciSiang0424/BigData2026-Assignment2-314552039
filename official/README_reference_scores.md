# Official assignment README — reference scores (quoted)

Source: https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/af9f335301d9badb482dfafa10c4534eabac845a/README.md

Commit: `af9f335301d9badb482dfafa10c4534eabac845a`

Quoted from `README.md` lines 113–122 (the official reference scores on the week of 2026/09/27–10/04):

> **Reference scores.** These are the scores of five baseline methods that we tested on the
> week of 2026/09/27 – 10/04.
>
> | Method | macro-F1 | micro-F1 | Score |
> |---|---|---|---|
> | Random | 19.77% | 22.74% | 21.26% |
> | Predict all 0 | 0.00% | 0.00% | 0.00% |
> | Copy last week | 43.01% | 56.20% | 49.61% |
> | GNN without time encoding | 22.80% | 48.61% | 35.70% |
> | GNN with time encoding | 50.90% | 58.74% | 54.82% |

These numbers are the course's published baselines, not results of this repository's runs. Our reproduction of Copy last week on the same week is in `results/task1_copy_last_week_refweek.csv` (50.20% on test users, 48.83% on the full cohort).
