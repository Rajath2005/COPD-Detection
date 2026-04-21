# 📅 Project Roadmap

## Table of Contents
1. [Overview](#overview)
2. [Phase I — Foundation (Feb–May 2026)](#phase-i--foundation-febmay-2026)
3. [Phase II — Development (Jun–Sep 2026)](#phase-ii--development-junsep-2026)
4. [Phase III — Finalisation (Oct–Nov 2026)](#phase-iii--finalisation-octnov-2026)
5. [Milestone Tracker](#milestone-tracker)
6. [Deliverables Summary](#deliverables-summary)
7. [Risk Register](#risk-register)

---

## Overview

The COPD-Detection project is a **two-semester VTU Major Project** conducted at VCET Puttur for the academic year 2025–26. It is structured in three phases aligned with VTU guidelines for major project submissions and presentations.

```
Feb 2026    Mar       Apr       May       Jun–Aug   Sep       Oct       Nov 2026
   │         │         │         │          │        │         │          │
   ●─────────●─────────●─────────●──────────●────────●─────────●──────────●
   │Synopsis │Synopsis │Progress │Pres. 1   │Design+ │Pres. 2  │Testing   │Final
   │Submit   │Present  │Rpts 1+2 │          │Impl.   │         │Results   │Report
```

---

## Phase I — Foundation (Feb–May 2026)

### Synopsis & Project Definition

| Task | Deadline | Status |
|------|----------|--------|
| Literature review | Jan 2026 | ✅ Done |
| Project synopsis preparation | Feb 2026 | ✅ Done |
| Synopsis submission | 27 Feb 2026 | ✅ Done |
| Synopsis presentation | 03–10 Mar 2026 | ✅ Done |

### Progress Reports 1 & 2

| Task | Deadline | Status |
|------|----------|--------|
| Progress Report 1 — Introduction | 14 Apr 2026 | 🔄 In Progress |
| Progress Report 2 — Requirements | 14 Apr 2026 | 🔄 In Progress |
| Dataset acquisition (ICBHI 2017) | Apr 2026 | 🔄 In Progress |
| Initial preprocessing pipeline | Apr 2026 | 🔄 In Progress |
| Feature extraction (MFCC, Mel) | Apr 2026 | 🔄 In Progress |
| EDA notebook | Apr 2026 | 🔄 In Progress |

### Presentation 1

| Task | Deadline | Status |
|------|----------|--------|
| Baseline CNN training & evaluation | May 2026 | ⏳ Upcoming |
| Presentation slides preparation | May 2026 | ⏳ Upcoming |
| Presentation 1 | 2nd Week May 2026 | ⏳ Upcoming |

---

## Phase II — Development (Jun–Sep 2026)

### Progress Reports 3 & 4

| Task | Deadline | Status |
|------|----------|--------|
| CNN-LSTM architecture design | Jun 2026 | ⏳ Upcoming |
| Hybrid model training | Jul 2026 | ⏳ Upcoming |
| Hyperparameter tuning | Jul–Aug 2026 | ⏳ Upcoming |
| Progress Report 3 — System Design | Aug 2026 | ⏳ Upcoming |
| Progress Report 4 — Implementation | Aug 2026 | ⏳ Upcoming |
| Comparative analysis (CNN vs CNN-LSTM) | Aug 2026 | ⏳ Upcoming |

### Presentation 2

| Task | Deadline | Status |
|------|----------|--------|
| Implementation demo | Sep 2026 | ⏳ Upcoming |
| Presentation 2 | 2nd Week Sep 2026 | ⏳ Upcoming |

---

## Phase III — Finalisation (Oct–Nov 2026)

### Progress Reports 5 & 6

| Task | Deadline | Status |
|------|----------|--------|
| Testing & validation | Oct 2026 | ⏳ Upcoming |
| Progress Report 5 — Testing | Oct 2026 | ⏳ Upcoming |
| Progress Report 6 — Results | Oct 2026 | ⏳ Upcoming |

### Final Submission & Viva

| Task | Deadline | Status |
|------|----------|--------|
| Final project report writing | Nov 2026 | ⏳ Upcoming |
| Journal paper draft | Nov 2026 | ⏳ Upcoming |
| Final Report + Journal Paper submission | 20 Nov 2026 | ⏳ Upcoming |
| Internal viva / project exhibition | 4th Week Nov 2026 | ⏳ Upcoming |

---

## Milestone Tracker

| # | Milestone | Deadline | Status |
|---|-----------|----------|--------|
| M1 | Synopsis submission | 27 Feb 2026 | ✅ Done |
| M2 | Synopsis presentation | 03–10 Mar 2026 | ✅ Done |
| M3 | Progress Reports 1 & 2 | 14 Apr 2026 | 🔄 In Progress |
| M4 | Presentation 1 | 2nd Week May 2026 | ⏳ Upcoming |
| M5 | Progress Reports 3 & 4 | 4th Week Aug 2026 | ⏳ Upcoming |
| M6 | Presentation 2 | 2nd Week Sep 2026 | ⏳ Upcoming |
| M7 | Progress Reports 5 & 6 | 2nd Week Oct 2026 | ⏳ Upcoming |
| M8 | Final Report + Journal Paper | 20 Nov 2026 | ⏳ Upcoming |
| M9 | Internal Viva / Exhibition | 4th Week Nov 2026 | ⏳ Upcoming |

**Status Legend:**
- ✅ Done — Completed and submitted
- 🔄 In Progress — Currently active
- ⏳ Upcoming — Scheduled but not yet started
- ❌ Blocked — Blocked by dependency or issue

---

## Deliverables Summary

| Deliverable | Type | Target Date |
|-------------|------|-------------|
| Repository with preprocessing + EDA | Code | Apr 2026 |
| Baseline CNN notebook + results | Code + Report | May 2026 |
| Hybrid CNN-LSTM notebook + results | Code + Report | Aug 2026 |
| Comparative analysis report | Report | Sep 2026 |
| Progress Reports 1–6 | Document | Rolling |
| Final Project Report | Document | Nov 2026 |
| Journal Paper Draft | Document | Nov 2026 |
| Project Demo | Presentation | Nov 2026 |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Dataset download issues (access restrictions) | Low | High | Cache dataset locally; document access procedure |
| Insufficient compute for GPU training | Medium | Medium | Use Google Colab or Kaggle Notebooks as fallback |
| Overfitting on small dataset | High | High | Strong regularisation (dropout, augmentation, class weights) |
| Team member unavailability | Low | Medium | Cross-train all members on each module |
| Model below target accuracy | Medium | Medium | Investigate ensembling, transfer learning as contingency |
| VTU deadline changes | Low | High | Track VTU circulars; build 2-week buffer before each deadline |

---

*← [Results](Results.md) | Next: [Team →](Team.md)*
