# Multimodal Sleep Disordered Breathing Event Classification

This project develops an end-to-end machine-learning workflow for detecting and
localizing sleep-disordered breathing (SDB) events in pediatric physiological
recordings. The goal is to make home sleep apnea testing more useful for
vulnerable populations while producing transparent metrics and visualizations
that researchers, developers, and clinical collaborators can evaluate.

## Project Profile

- **Project name:** Multimodal Sleep Disordered Breathing Event Classification
- **Question, problem, or opportunity:** The purpose of this project is to develop and evaluate a machine-learning workflow for detecting and classifying sleep-disordered breathing events in pediatric physiological recordings. The system will analyze multiple synchronized signals, including thoracic effort, airflow, blood oxygen saturation, EEG-derived arousal information, and sleep stages. Combining these signals may help identify apnea and hypopnea events more reliably when recordings contain noise, motion artifacts, or displaced sensors.
This work is motivated by the need for more accessible sleep-apnea evaluation among vulnerable populations, including children, long-term survivors of childhood cancer, and people with sickle cell disease. Home Sleep Apnea Testing could help patients who have limited access to specialized sleep clinics or in-lab studies. However, existing automated scoring systems are largely designed for adults from the general population and may not perform equally well in pediatric or medically complex groups. The project will compare model predictions with expert annotations using event-level F1 scores and session-level AHI error, while documenting limitations and requirements for future clinical validation.

- **Data, inputs, or evidence:** The project uses preprocessed baseline recordings from the
  Pediatric Adenotonsillectomy Trial for Snoring (PATS), shared through the
  National Sleep Research Resource (NSRR). The challenge dataset contains 494
  baseline recordings derived from the original study dataset. Each MAT file contains
  synchronized physiological signals, event boundaries, sleep-stage information, and summary
  metrics.
- **Expected output:** A reproducible prototype that detects and localizes
  obstructive apnea and hypopnea events, calculates session-level
  Apnea-Hypopnea Index (AHI), and visualizes predictions alongside the
  underlying signals.
- **Tools and stack:** Python, Matplotlib, Github Copilot, Github Repository, VS code
- **Team lead:** Miguel Navarrete (Github ID: mnavarretem) and Lu Xie 
(lxie3)
- **Team members and roles:** 
  - Miguel Navarrete (Github ID: mnavarretem): Project genesis/Subject matter expert
  - Lu Xie (Github ID: lxie3): Team lead/Model development and testing
  - Anna Pittman (Github ID: PittmanAEP ): Model development and testing
  - Paul VanGilder (Github ID: pvangild): Documentation
  - Richa Singh (Github ID: richa-singhx): Model development and testing
  - Pragyee Neupane (Github ID: pragyeeneupane): Model development and testing 
  - Prajvala Mysore (Github ID: pmysore-stjude): Documentation 
  - Juliette Kippen (Github ID: juliettekip): Documentation and organization
- **Communication:** Miguel Navarrete and Lu Xie 
 

## Vision and Mission

- **Vision:** Improve access to timely and appropriately escalated sleep-apnea
  evaluation for children, childhood-cancer survivors, people with sickle cell
  disease, and other vulnerable populations whose needs may not be represented
  by models trained on adults from the general population.
- **Mission:** Build and evaluate a multimodal event-classification workflow
  that can handle noisy recordings, combine respiratory and oxygenation signals
  with arousal and sleep-stage information, and provide clinically meaningful
  event- and session-level summaries.

## About

The purpose of this project is to create a reproducible machine-learning workflow for identifying and classifying sleep-disordered breathing events in pediatric physiological recordings. This project addresses important access and reliability problems. Recent evidence from St. Jude APNEA, CSALTS medical trials involving long-term survivors of childhood cancer, and preliminary findings from cHOD17 in pediatric patients suggests that obstructive sleep apnea is highly prevalent in these populations, even though its burden is not explained by traditional risk factors. Obstructive sleep apnea also appears to affect other vulnerable populations, including individuals with sickle cell disease. At St. Jude, the absence of a dedicated sleep clinic can make home sleep apnea testing an important first step for evaluating patients and determining whether an in-lab study or specialized follow-up is needed. Existing commercial scoring algorithms are proprietary and primarily designed for adults from the general population. As such, they may be less reliable when applied to pediatric patients, medically complex populations, or recordings affected by noise, motion, and sensor displacement.

The workflow will process synchronized, time-series signals such as thoracic respiratory effort, airflow, oxygen saturation, EEG-derived arousal features, calculated breathing amplitude, and sleep-stage information. By combining evidence across these signals, the project seeks to detect obstructive, central, and mixed apnea events as well as hypopneas, while accounting for timing differences between respiratory changes, oxygen desaturation, and arousals.

The project will use annotated pediatric sleep data to develop a baseline detector and machine-learning prototype. Performance will be measured by how accurately the system localizes individual events and how closely its estimated Apnea-Hypopnea Index matches expert-derived values. The project will also examine robustness to artifacts, integration of multiple signal types, and the usefulness of visualizations for reviewing model predictions. The intended result is not a clinical diagnostic device, but a transparent research prototype that demonstrates technical feasibility, identifies model limitations, and establishes a foundation for future validation with home sleep apnea recordings and certified sleep-technologist review.

## Analyze Recordings

`analyze_recordings.py` reads every `.mat` recording in a folder and writes a Markdown event-count table. It requires Python with `numpy`, `scipy`, and `matplotlib` installed.

```powershell
python analyze_recordings.py --data-dir "Z:\path\to\recordings"
```

To plot a particular event, choose files and events using 1-based numbering after sorting files alphabetically. For example, this plots apnea event 3 from the second file:

```powershell
python analyze_recordings.py --data-dir "Z:\path\to\recordings" --plot-apnea --file 2 --event 3
```

Use `--plot-hypopnea` to plot a hypopnea instead. By default, the summary is saved as `event_summary.md`; change that path with `--summary-file`.

## Roadmap and Milestones

| When | Focus | Expected outcome |
| --- | --- | --- |
| Day 1 | Agree on the question, inputs, stack, roles, and first tasks | A shared plan and a first small change in the repository |
| Day 1 | Inspect MAT-file structure, channel quality, event labels, sleep stages, and AHI metrics | A documented data-quality baseline and reproducible exploratory summary |
| Day 1 | Analysis of the literature to determine pre-existing model compatability | A shared list of potential models to be adapted |
| Day 2 | Build, test, and compare approaches | A working result or clear evidence about what does not work |
| Day 2 | Establish a baseline detector and compare model outputs with expert event boundaries | Initial event-level F1 and a clear account of timing errors |
| Day 2 | Test noisy and artifact-heavy windows and inspect cross-modal relationships | Documented robustness findings and representative visualizations |
| Day 3 | Stabilize, document, and present | A demo or handoff with methods, limitations, and next steps |
| Day 3 | Calculate session-level AHI and MAE on held-out data | A reproducible clinical-metric report |
| Day 3 | Define certified-technologist review and future HSAT validation requirements | A responsible plan for clinical translation |

The goal is not a perfect production system. The goal is a clear, honest, useful
result that the team can explain and others can build on.



