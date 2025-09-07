# Data Setup

## Primary Dataset: OULAD
Download the **Open University Learning Analytics Dataset (OULAD)** ZIP and extract all CSVs into `data/raw/oulad/`.

- OULAD homepage: https://analyse.kmi.open.ac.uk/open-dataset
- UCI entry: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Scientific Data paper: Kuzilek et al., 2017

### Minimal Files Used (suggested)
- `studentVle.csv` – daily click counts per student per VLE activity
- `vle.csv` – mapping for VLE activities (ids, types, urls)
- `courses.csv` / `modules.csv` (depending on version) – metadata
- `studentRegistration.csv` – enrollment/withdrawal timestamps
- `studentAssessment.csv` – outcomes (optional for labels)
- `assessments.csv` – assessment metadata

## Optional Datasets
- **ASSISTments (2009/2017)** – skill builder logs for math; place under `data/raw/assistments/`
- **EdNet** – large‑scale multi‑activity logs; place under `data/raw/ednet/`

## Output
Running `src/data_prep.py` writes a unified interaction log to:
`data/processed/interactions.parquet`
