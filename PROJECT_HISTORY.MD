# Face Detection Project History

## Project Overview

This project aims to build a face detection system using YOLO models and expose the model through a FastAPI service for real-time inference.

The project currently contains:

- Dataset preparation and enrichment
- Dataset splitting
- Model training experiments
- Model evaluation
- Inference engine
- FastAPI API service
- Docker containerization

Repository Status:
- Research / Experiment Workspace

Future repositories planned:

- face-detection-training (research and experiments)
- face-detection-api (production inference service)

---

# Dataset Evolution

## dataset_v1

Purpose:
- Initial training dataset.

Notes:
- First complete labeled dataset used for experiments.
- Used to establish baseline model performance.

Outcome:
- Baseline model successfully trained.

---

## dataset_v2

Purpose:
- Improve dataset quality and quantity.

Changes:
- Additional images added.
- Label corrections performed.
- Dataset restructuring.

Reason:
- Baseline model performance was not satisfactory.
- More training samples were required.

Outcome:
- Improved validation results compared to dataset_v1.

---

## dataset_enriched

Purpose:
- Further improve model generalization.

Changes:
- Added enrichment data.
- Increased diversity of training samples.
- Improved representation of real-world scenarios.

Reason:
- Reduce overfitting.
- Improve robustness.

Outcome:
- Used for advanced training experiments.

---

## add_on_data_v1

Purpose:
- Supplementary dataset.

Notes:
- Created to support enrichment process.

---

# Data Processing Scripts

## split_dataset_v1.py

Purpose:
- Generate train/validation/test splits for dataset_v1.

Status:
- Completed.

---

## split_dataset_v2.py

Purpose:
- Generate train/validation/test splits for dataset_v2.

Status:
- Completed.

---

## add_enrichment_data.py

Purpose:
- Merge enrichment data into training dataset.

Status:
- Completed.

---

# Model Training Experiments

## train_baseline.py

Purpose:
- First training experiment.

Goal:
- Establish initial benchmark performance.

Notes:
- Used dataset_v1.

Result:
- Baseline established.

---

## train_enrichment.py

Purpose:
- Train using enriched dataset.

Goal:
- Evaluate impact of additional training samples.

Result:
- Performance improved compared to baseline.

---

## train_v2_img800.py

Purpose:
- Increase image size to 800.

Hypothesis:
- Larger image size may improve small-face detection.

Result:
- To be reviewed.

---

## train_v2_800_augmented.py

Purpose:
- Combine larger image size with data augmentation.

Hypothesis:
- Improve model robustness.

Result:
- To be reviewed.

---

## train_v2_stratified.py

Purpose:
- Experiment with different dataset splitting strategy.

Hypothesis:
- Better data distribution may improve generalization.

Result:
- To be reviewed.

---

## train_v2_yolo11m.py

Purpose:
- Upgrade from smaller YOLO model to YOLO11m.

Hypothesis:
- Larger model may achieve higher detection accuracy.

Result:
- To be reviewed.

---

## train_v2_yolo11m_800_aug.py

Purpose:
- YOLO11m + image size 800 + augmentation.

Hypothesis:
- Combine multiple improvements into one training run.

Result:
- To be reviewed.

---

# Model Weights

## yolo11s.pt

Purpose:
- YOLO11s pretrained weights.

Usage:
- Transfer learning.

Status:
- Reference model.

---

## yolo11m.pt

Purpose:
- YOLO11m pretrained weights.

Usage:
- Transfer learning.

Status:
- Main experiment candidate.

---

## yolo26n.pt

Purpose:
- Alternative YOLO architecture experiments.

Status:
- Experimental.

---

# Inference Layer

## inference_engine.py

Purpose:
- Central inference logic.

Responsibilities:
- Load model.
- Process image input.
- Perform face detection.
- Return prediction results.

Status:
- Active.

---

## infer_v2.py

Purpose:
- Inference testing and experimentation.

Status:
- Development utility.

---

# API Layer

## main.py

Purpose:
- FastAPI application entry point.

Responsibilities:
- Receive requests.
- Execute inference.
- Return detection results.

Status:
- Active.

---

# Containerization

## Dockerfile.v1

Purpose:
- First containerization attempt.

Status:
- Legacy version.

---

## Dockerfile.v2

Purpose:
- Improved containerization approach.

Status:
- Current candidate.

---

# Testing Assets

## test.py

Purpose:
- Local testing script.

Status:
- Development utility.

---

## test.jpg / test1.jpg / test2.jpg / ...

Purpose:
- Quick inference validation.

Status:
- Test assets.

---

# Lessons Learned

## Git

Topics learned:

- git init
- git add
- git commit
- git remote
- git push
- git restore
- .gitignore
- commit amend
- force push

Outcome:
- Successfully pushed project to personal GitHub repository.

---

## Docker

Topics learned:

- Dockerfile creation
- Image builds
- Container execution
- API containerization

Outcome:
- FastAPI service successfully containerized.

---

## Deployment Planning

Future roadmap:

1. Repository cleanup
2. GitHub Actions
3. CI pipeline
4. Automated Docker builds
5. Container registry
6. Continuous deployment
7. Monitoring
8. Production MLOps practices

---

# Current Best Candidate

Date:
- 2026-10-07

Dataset:
- TBD

Training Script:
- TBD

Model:
- TBD

Reason:
- TBD

This section must always reflect the best known model at the current time.

---

# Future Improvements

- Repository separation
- Automated testing
- GitHub Actions CI
- Docker build automation
- Registry integration
- Health checks
- Monitoring
- Logging
- Model versioning
- MLOps workflow