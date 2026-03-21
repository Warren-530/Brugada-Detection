# Brugada Syndrome Clinical AI Assistant

## 1. Project Overview
This project is a clinical decision-support web application for Brugada syndrome risk triage from 12-lead ECG WFDB records.

It combines:
- Multi-view deep feature extraction
- Handcrafted clinical morphology/statistical features
- A classical meta-learner stack on top of selected features
- A Streamlit reporting layer with evidence visualization and triage-oriented recommendations

The application is designed for high-recall triage behavior and explicit physician-facing explanations. It supports single-record diagnosis and batch queue prioritization.

Important note: this tool supports triage and workflow prioritization. It does not replace physician diagnosis.

## 2. Repository Contents
- `app.py`: Streamlit web interface and report rendering
- `inference.py`: End-to-end inference pipeline, feature extraction, recommendation logic
- `requirements.txt`: Python dependencies
- `extractor_resnet.keras`: 1D ResNet feature extractor
- `extractor_eegnet.keras`: EEGNet-style feature extractor
- `extractor_bilstm.keras`: Attention-BiLSTM feature extractor
- `extractor_cwt_cnn.keras`: CWT-CNN feature extractor
- `brugada_scaler.pkl`: StandardScaler fitted on training data
- `brugada_selector.pkl`: Feature selector fitted on training data
- `brugada_meta_learner.pkl`: Trained meta-learner for final risk probability

## 3. Model Architecture and Inference Pipeline
The deployed pipeline in `inference.py` follows this sequence:

1. WFDB loading and preprocessing
- Input format: paired `.hea` and `.dat`
- Bandpass filter: 0.5 to 40 Hz, order 3 Butterworth
- Sequence length normalization: truncate or zero-pad to 1200 samples

2. Multi-view feature extraction
- Deep latent features from 4 extractors (32 dimensions each):
  - 1D ResNet latent feature
  - EEGNet latent feature
  - Attention-BiLSTM latent feature
  - CWT-CNN latent feature (from V1 to V3 scalograms)
- Clinical handcrafted features:
  - 84 statistical time-domain features (7 per lead over 12 leads)
  - 9 expert morphology features from V1 to V3 (J height, ST slope, curvature)

3. Feature assembly
- Concatenation order is strict:
  - Statistical 84
  - Expert 9
  - ResNet 32
  - EEGNet 32
  - BiLSTM 32
  - CWT-CNN 32
- Total: 221 dimensions

4. Classical ML post-processing
- Standardization with saved scaler
- Feature selection with saved selector
- Final probability from saved meta-learner

5. Decision policy
- Decision threshold: 0.05
- Borderline-positive zone: [0.05, 0.06]
- Output includes:
  - Brugada risk probability
  - Decision confidence (derived from threshold margin)
  - Threshold distance (percentage points from threshold)
  - Predicted-class support
  - Evidence table and recommendation tier

## 4. Training Method Summary
Training was performed in a separate notebook workflow (artifact generation already completed and included in this folder as `.keras` and `.pkl` files).

High-level training strategy:
- ECG preprocessing with the same filtering and length normalization policy
- Multi-view representation learning
- Strict out-of-fold deep feature extraction to reduce leakage in stacked learning
- Feature selection on concatenated feature space
- Meta-learning with a soft-voting/stacking style ensemble of classical models
- Threshold scanning and clinical uncertainty analysis during evaluation

Deployed artifacts in this folder are already trained and ready for inference.

## 5. Explainability and Clinical Reporting
The web report provides:
- 12-lead ECG visualization with V1 to V3 highlighted windows in relevant cases
- Decision margin chart showing threshold and borderline-positive zone
- V1 to V3 morphology evidence table and heatmap
- Per-lead evidence summaries:
  - Evidence Strength: strong, moderate, weak
  - Extraction Reliability: good, fair, poor
- Deep-view contribution share (descriptive, not causal attribution)
- Recommendation tier and actionable checklist
- Batch triage queues:
  - Urgent Review Queue
  - Gray-Zone Priority Queue
  - Discordant Cases Queue

## 6. Web Logic Summary
Single-record output logic is designed to be explicit:
- Risk status from model probability and threshold policy
- Borderline protocol card appears for borderline cases or near-threshold distance
- Discordance warning appears when model-level decision and morphology strength diverge
- Recommendation banner is tier-driven and consistent with backend policy

Batch logic:
- Prioritization order is recommendation tier first, then probability, then decision stability
- Batch schema includes recommendation tier and discordance flag to support review workflow

## 7. Is the Current Version Enough?
For demo and competition-oriented deployment, this implementation is strong and coherent.

For production-like clinical deployment, additional work is recommended:
- Prospective validation and calibration by site/population
- Governance around threshold policy and escalation pathway
- Data quality controls and audit logging
- Regulatory, legal, and safety review

## 8. Environment Setup and First Run (Fresh Pull)
This section assumes a new machine or a clean environment.

### 8.1 Prerequisites
- Python 3.10 or 3.11 recommended
- pip latest available
- Git installed

### 8.2 Clone and enter project
PowerShell (Windows):

```powershell
git clone <your-repository-url>
cd "Web App"
```

### 8.3 Create and activate virtual environment
PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 8.4 Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 8.5 Verify core files exist
Confirm these files are present before running:
- `extractor_resnet.keras`
- `extractor_eegnet.keras`
- `extractor_bilstm.keras`
- `extractor_cwt_cnn.keras`
- `brugada_scaler.pkl`
- `brugada_selector.pkl`
- `brugada_meta_learner.pkl`

### 8.6 Start web app

```powershell
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually:
- `http://localhost:8501`

## 9. Routine Daily Run (Environment Already Set)
If setup is already done and dependencies are installed:

### 9.1 Open project and activate venv

```powershell
cd "Web App"
.\.venv\Scripts\Activate.ps1
```

### 9.2 Optional cache clear
Use this if you suspect stale Streamlit state:

```powershell
streamlit cache clear
```

### 9.3 Run app

```powershell
python -m streamlit run app.py
```

### 9.4 Use the app
Single-record mode:
- Upload matched `.hea` and `.dat`
- Click Run Diagnosis
- Review risk metrics, recommendation, and evidence

Batch mode:
- Upload multiple `.hea` and `.dat`
- Click Run Batch Risk List
- Review tiered queues and discordant cases first

## 10. Input Data Requirements
- Input records must be valid WFDB pairs (`.hea` with matching `.dat`)
- Record base names must match (for example, `100.hea` with `100.dat`)
- Signal should be 12-lead ECG compatible with the preprocessing assumptions

## 11. Output Field Semantics
Key report metrics:
- Brugada Risk Probability: model-estimated probability for Brugada class
- Threshold Distance: absolute distance from decision threshold in percentage points
- Decision Confidence (derived): normalized transform of threshold margin for quick readability
- Predicted-Class Support: posterior support for assigned class label

Evidence semantics:
- Evidence Strength indicates morphology support level
- Extraction Reliability indicates robustness of delineation process, not disease severity

## 12. Troubleshooting
### 12.1 Import warnings in editor
If your editor reports unresolved imports for tensorflow, cv2, pywt, or neurokit2:
- Ensure the selected Python interpreter is the project venv
- Reinstall dependencies in that venv
- Restart VS Code Python language server if needed

### 12.2 Streamlit starts but inference fails
- Verify all `.keras` and `.pkl` artifact files are present in project root
- Ensure uploaded input includes both `.hea` and `.dat`
- Check file naming consistency for WFDB pair

### 12.3 Performance issues on CPU
- First inference may be slow due to model loading
- Keep app session running to reuse loaded models

## 13. Known Limitations
- Evidence strength/reliability logic is heuristic and should be interpreted with clinical context
- Deep-view contribution chart is descriptive, not causal model attribution
- Threshold policy is intentionally recall-oriented and may increase false positives
- The tool is not a substitute for cardiologist interpretation or formal diagnosis

## 14. Suggested Next Improvements
- Report export (PDF/CSV) for clinician handoff
- Additional data-quality checks before inference
- Configurable threshold policy profiles (screening, balanced, rule-out)
- External validation dashboard and calibration metrics

## 15. License and Clinical Disclaimer
Use according to your repository license.

Clinical disclaimer:
This software is for decision support and research workflow assistance. It does not provide a definitive diagnosis and must be used with qualified clinical judgment.
