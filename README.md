# Group 6: Synergy Board (MGT 8040 Data Mining, Fall 2026)

Shared repository for Assignments 5, 6 and 7 in MGT 8040 Data Mining, Section 002, Georgia State University.

## Team

| Member | Last name | Role |
|---|---|---|
| Isis | Harrison | Devil's advocate reviewer for A6 and A7 |
| Nana | Amoah | Owner of the selected dataset |
| Tejas | Vaidya | Repository owner; Section 2 distance analysis |

## Selected Asset (advances to A6 and A7)

- **Tracking domain:** Campus learning center usage (Piedmont North and Commons)
- **Target Y:** Weekly sign-ins (visits) per center and time block
- **A3 interaction being scaled:** flyer_week × piedmont
- **Real data:** 40 logged rows (Spring 2026 sign-ins, 2 centers × 10 week-periods × 2 time blocks)
- **Real A3 result:** β3 = −3.625, p = 0.510 (verdict: artifact)
- **Synthetic sample size:** 720 rows (18 simulated centers × 2 simulated semesters × 10 week-periods × 2 time blocks)

## Data Disclosure

The scaled dataset in this repository is **synthetic**. It is generated from an explicit data-generating process (DGP) fitted to Nana's 40 real rows. It is not real student data and must not be presented as real. Every row carries `data_source = "synthetic"`.

**Generating equation** (predictors centered at the real means):

```
visits = 17.08 + 2.67·flyer + 15.71·piedmont − 0.31·flyer×piedmont
       + 7.54·snack + 9.94·snack×piedmont      (snack terms: exploratory, post-hoc)
       + 20.19·event + 0.82·week − 5.56·evening + 0.65·midterm − 3.40·break
       + noise   (SD 7.44 first-year halls, 4.32 upperclass halls; seed 42)
```

- **Why the flyer interaction is −0.31, not −3.625:** once the snack terms are in the model, most of the A3 β3 turns out to be the snack-week effect. Building in both would double count it. Because the real calendar is kept, refitting A3 on the synthetic data still recovers about −3.6.
- **extended_hours:** always equals midterm + spring break in the real calendar, so its effect is folded into those two terms.
- **Negative draws:** 9 of 720 came out below 0 and were set to 0 (all in upperclass evening blocks).

**Verification (seed 42)**

| Check | Real | Synthetic |
|---|---|---|
| Visits mean / SD | 20.64 / 12.64 | 20.45 / 12.60 |
| r(visits, piedmont) | 0.71 | 0.71 |
| r(visits, snack_week) | 0.22 | 0.21 |
| r(visits, event_week) | 0.46 | 0.47 |
| Largest correlation gap | | 0.016 |
| A3 β3 | −3.625 (p = 0.51) | −4.10 (p = 0.0003) |
| Snack effect, first-year / upperclass | +12.5 / +2.6 | +12.3 / +2.1 |

Seed 42 is a typical draw: across 1,000 seeds, β3 averages −3.61, falls within ±2 of −3.625 in 95% of runs and never comes back positive.

**Limitations**

- **The synthetic p = 0.0003 is not evidence.** It is significant only because the 40-row pattern is repeated 18 times. The real p is 0.51 for the flyer interaction and 0.11 for the snack interaction.
- **Halls are copies of one real hall per type.** C01 to C09 copy Piedmont North and C10 to C18 copy Commons, so there is no real hall-to-hall variation.
- **Both semesters use the Spring 2026 calendar,** including spring break, so they are labeled Sim-Semester-1 and Sim-Semester-2, not Fall.

## Repository Structure

```
.
├── README.md
├── section2_distance/      # Jaccard and Gower team-diversity matrices (A5 Section 2)
├── synthesis/              # DGP script that scales the real 40 rows to 720 synthetic rows (A5 Section 4)
├── data/
│   ├── real/               # Real 40-row source data (shared only with team consent)
│   └── synthetic/          # Generated synthetic datasets, clearly labeled
└── a6_a7/                  # Clustering (A6) and classification (A7) work
```

## Assignment Map

| Assignment | Focus | Folder |
|---|---|---|
| A5 | Selection matrix, team diversity audit, scale-up plan | `section2_distance/`, `synthesis/` |
| A6 | Clustering on the scaled dataset | `a6_a7/` |
| A7 | Classification on the scaled dataset | `a6_a7/` |

## Contribution Ledger

| Member | Contribution |
|---|---|
| Isis Harrison | [to be filled in] |
| Nana Amoah | Selected dataset; A3 refit and grounded synthesis (720-row synthetic dataset and generator) |
| Tejas Vaidya | Section 2 Jaccard and Gower distance analysis; repository setup |

## AI Interaction Transcripts

- Synthesis and scale-up chat (Nana): https://claude.ai/share/f136b920-e820-4af3-8b2b-fa3998b45704
- Team distance matrices, Prompt 1 (Tejas): https://chatgpt.com/share/6afb5dff-f988-83e9-852b-651a1bc1ef84

## AI Disclosure

AI co-pilots (Claude, ChatGPT) were used with the course-approved prompts to compute distance matrices, write the synthesis code and draft documentation. All outputs were checked by team members before use.
