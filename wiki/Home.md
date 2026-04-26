# 🫁 COPD-Detection Wiki — Home

Welcome to the official wiki for the **COPD-Detection** project — a deep learning-based system for early detection and severity classification of Chronic Obstructive Pulmonary Disease (COPD) using respiratory sound analysis.

> **Project:** VTU Major Project | Department of Computer Science | VCET Puttur | 2025–26  
> **Status:** Phase I — Active

---

## 📚 Wiki Contents

| Page | Description |
|------|-------------|
| [Project Overview](Project-Overview.md) | Background, problem statement, objectives, and scope |
| [Architecture](Architecture.md) | CNN-LSTM model design, pipeline, and technical details |
| [Dataset](Dataset.md) | ICBHI 2017 Respiratory Sound Database — details and structure |
| [Installation](Installation.md) | Environment setup, dependencies, and getting started |
| [Usage](Usage.md) | Step-by-step guide to run the full pipeline |
| [Objective 1 Completion Report](Objective-1-Completion-Report.md) | Detailed implementation log, results, issues faced, fixes, and final completion status for Objective 1 |
| [Model Training](Model-Training.md) | Training procedure, hyperparameters, and evaluation |
| [Results](Results.md) | Expected performance metrics and output interpretation |
| [Roadmap](Roadmap.md) | Project timeline, milestones, and current status |
| [Team](Team.md) | Team members, roles, and contacts |

---

## 🔍 Quick Summary

**COPD** is a chronic, progressive lung disease affecting hundreds of millions worldwide. Most patients are diagnosed too late because conventional diagnostic tools (spirometry, CT scans) require specialised equipment and clinical visits.

This project addresses that gap by using **AI to analyse respiratory sounds** — a low-cost, non-invasive approach that can be deployed even in remote settings.

### Key Highlights

- 🎙️ **Input**: Raw lung auscultation audio (WAV format)
- 🤖 **Model**: Hybrid CNN-LSTM architecture
- 🎯 **Tasks**: Binary COPD detection + 4-class severity classification
- 📊 **Dataset**: ICBHI 2017 Respiratory Sound Database (920 recordings, 6,898 breathing cycles)
- 🏆 **Target Accuracy**: > 90% binary classification

---

## 🚀 Getting Started Fast

```bash
git clone https://github.com/Rajath2005/COPD-Detection.git
cd COPD-Detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py
jupyter notebook notebooks/
```

See the [Installation](Installation.md) and [Usage](Usage.md) pages for full details.

---

## 📄 License

MIT License — see [LICENSE](../LICENSE) for details.
