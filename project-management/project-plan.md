# Project Plan

## Goal
This project aims to develop an end-to-end machine-learning workflow to detect sleep disordered breathing (SDB) events in pediatric polysomnography data and estimate the clinical Apnea-Hypoapnea Index (AHI) in order to enhance the utility of at home testing for vulnerable populations, including pediatric patients.

## Tools

 Python, Matplotlib, Github Copilot, Github Repository, VS code

## First Tasks

- [ ] [Access to data for all members]
- [ ] [Visualization of datasets]
- [ ] [Understanding of normal baseline, and abnormal SBDs]

## Milestones


- **Day 1:** [Assess question, determine tasks, gain access, visualize data, analyze literature]
- **Day 2:** [Build, test, and compare approaches]
- **Day 3:** [Model finalization, Presentation design]

## Definition of Done

The project can be considered done when the end-to-end pipeline reproducibly ingests synchronized PSG data, outputs localized apnea/hypopnea event annotations with clinically aligned timing, and achieves the target performance benchmarks (event-level F1 > 0.75 and AHI MAE < 5 events/hour) while handling noisy or artifact-heavy recordings without failure. If completed early, potential next steps would be to continue refining model above the target benchmarks and to develop visualization tools to increase ease of use for clinicians.

## Risks and Questions
- Project data is a subset of data from a larger study; data validity needs to be assessed to validate suitability for training purposes
- Need to ensure sufficient data is available to sufficiently train the model 
- Training data is from in-hospital tests vs at home testing kits (target use)

