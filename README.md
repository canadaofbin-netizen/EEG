# Experimental Design: EEG Motor Imagery Classification for Smart Glasses Form Factor

**Project:** Can a machine learning model infer left/right motor imagery from EEG channels accessible by glasses frames, without direct access to the primary motor cortex?

**Author:** Kyubin Yun  
**Date:** August 28, 2026  
**Version:** 1.0

---

## Table of Contents

1. [Research Motivation & Problem Statement](#1-research-motivation--problem-statement)
2. [Hypothesis & Research Questions](#2-hypothesis--research-questions)
3. [Neuroscientific Foundation](#3-neuroscientific-foundation)
4. [Dataset Selection](#4-dataset-selection)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Model Architecture & Selection](#6-model-architecture--selection)
7. [Experimental Protocol](#7-experimental-protocol)
8. [Evaluation & Statistical Analysis Plan](#8-evaluation--statistical-analysis-plan)
9. [Risk Analysis & Mitigation](#9-risk-analysis--mitigation)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [References](#11-references)

---

## 1. Research Motivation & Problem Statement

### 1.1 The Consumer BCI Gap

Current Brain-Computer Interface (BCI) systems for motor imagery require research-grade EEG caps with electrodes placed directly over the primary motor cortex (C3, Cz, C4). These are impractical for everyday consumer use — they require conductive gel, head caps, and 20+ minutes of setup.

**Smart glasses** represent a compelling consumer form factor. However, the physical contact points of glasses frames (temples, nose bridge, behind ears) correspond to **lateral and frontal EEG positions** (T7/T8, F7/F8, Fp1/Fp2, TP9/TP10) — none of which directly overlie the sensorimotor cortex.

### 1.2 Core Question

> If a well-trained model achieves high accuracy using full-channel EEG data including the motor cortex, can it maintain meaningful inferential performance when restricted to only the channels physically accessible by a glasses frame?

### 1.3 Project Scope

This is a **personal side project** to explore the feasibility of this idea using open-source EEG data. The scope is:

- **Phase 1:** Achieve robust baseline accuracy (target: >90% within-subject) using full-channel data with rigorous methodology.
- **Phase 2:** Systematically degrade channels and measure the accuracy drop when excluding motor cortex electrodes.
- **Phase 3:** Test classification using only "glasses-accessible" channels (T7, T8, F7, F8 + optionally Fp1, Fp2, TP9, TP10).

---

## 2. Hypothesis & Research Questions

### 2.1 Primary Hypothesis

**H₀ (Null):** A model trained on glasses-accessible EEG channels (T7, T8, F7, F8) cannot classify left vs. right hand motor imagery above chance level (50%) after removing primary motor cortex channels (C3, C4, Cz).

**H₁ (Alternative):** Volume-conducted sensorimotor rhythms, captured at lateral/temporal electrodes, retain sufficient inter-hemispheric asymmetry to support above-chance motor imagery classification.

### 2.2 Research Questions

| # | Research Question | Phase |
|:---:|:---|:---:|
| **RQ1** | What is the maximum achievable within-subject accuracy for binary (Left vs Right hand) motor imagery using full-channel data and rigorous methodology? | 1 |
| **RQ2** | How does classification accuracy degrade as a function of distance from the motor cortex — i.e., which channels contribute most to decoding? | 2 |
| **RQ3** | Is above-chance classification possible using only glasses-accessible channels (T7, T8, F7, F8)? If so, what is the accuracy ceiling? | 3 |
| **RQ4** | Does cross-hemisphere referencing (T7 vs T8 bipolar) preserve lateralized SMR gradients better than common average reference at peripheral sites? | 3 |
| **RQ5** | Can artifact confounds (EMG from temporalis muscle, EOG) inflate lateral-channel accuracy, and how can we control for this? | 3 |

### 2.3 Realistic Expectations (Based on Literature)

> [!WARNING]
> **Critical Context from Literature Review:**
> - **Full-channel within-subject accuracy** for 2-class MI (Left vs Right) typically ranges **78%–86%** on average across subjects, with top-performing subjects reaching **92%–98%**.
> - **>95% average across all subjects** is physiologically unrealistic due to the "BCI Illiteracy" phenomenon (15–30% of subjects produce weak or absent sensorimotor rhythms).
> - **Lateral-only channels (T7, T8, F7, F8) without C3/C4** are expected to degrade accuracy to **52%–62%** (near chance level) for genuine neural MI decoding.
> - Any study claiming >80% MI accuracy from purely frontal/temporal channels should be scrutinized for **EMG/EOG artifact confounds**.

**Revised Phase 1 Target:** Achieve **>85% within-subject accuracy** on high-performing subjects, and **>75% population average**, using full-channel data with methodologically sound preprocessing.

---

## 3. Neuroscientific Foundation

### 3.1 Sensorimotor Rhythms (SMR) and Motor Imagery

Motor imagery activates the primary motor cortex (M1, Brodmann Area 4) and supplementary motor area (SMA). This produces two key electrophysiological phenomena:

- **Event-Related Desynchronization (ERD):** Power decrease in mu (8–13 Hz) and beta (13–30 Hz) bands over the **contralateral** motor cortex during imagery.
- **Event-Related Synchronization (ERS):** Power increase (or maintenance) over the **ipsilateral** cortex — "surround inhibition."

$$\text{ERD/ERS\%} = \frac{P_{\text{active}} - P_{\text{baseline}}}{P_{\text{baseline}}} \times 100$$

| Imagery Task | Contralateral ERD | Ipsilateral ERS |
|:---|:---|:---|
| Right Hand | C3, FC3, CP3 (mu/beta ↓) | C4, FC4, CP4 (mu/beta ↑ or baseline) |
| Left Hand | C4, FC4, CP4 (mu/beta ↓) | C3, FC3, CP3 (mu/beta ↑ or baseline) |

### 3.2 Volume Conduction and Lateral Detection

The skull acts as a spatial low-pass filter with conductivity ratio $\sigma_{\text{brain}} / \sigma_{\text{skull}} \approx 15:1$ to $80:1$. Signals from C3/C4 are attenuated at lateral sites:

- **Amplitude attenuation at T7/T8:** 70%–90% reduction (3×–10× amplitude drop)
- **Power attenuation:** 90%–99% ($-10$ dB to $-20$ dB)

```
  Cortical Dipole at C3 (M1 Hand Knob)
           |
  [CSF] σ = 1.79 S/m → Current spreading
           |
  [Skull] σ = 0.01–0.03 S/m → Severe spatial low-pass filter
           |
  [Scalp] σ = 0.33 S/m
    /                       \
 C3 (100% SNR)         T7 (~7 cm away: 10–30% amplitude)
                        + Temporalis muscle EMG contamination
```

### 3.3 Electrode Positions and Glasses Mapping

```
                        [ Nose Bridge ]
                         Fpz / Glabella
                           /        \
                  [ Left Brow ]    [ Right Brow ]
                   Fp1 / AF7        Fp2 / AF8
                       |                |
                  [ Frame Rim ]    [ Frame Rim ]
                      F7               F8
                       |                |
                 [ Temple Arms ]  [ Temple Arms ]
                   FT7 / T7         FT8 / T8
                       |                |
                 [ Behind Ears ]  [ Behind Ears ]
                  TP9 / A1          TP10 / A2
```

| Glasses Contact Point | 10-20 Position | Underlying Cortex | Relevance to MI |
|:---|:---|:---|:---|
| Temple arms | T7 / T8 | Middle Temporal Gyrus (BA 21/22) | Auditory cortex; far-field volume-conducted SMR |
| Frame hinges | F7 / F8 | Inferior Frontal Gyrus (BA 45/47) | Broca's area; no direct motor representation |
| Nose bridge | Fpz / Fp1 / Fp2 | Frontal Pole (BA 10) | Executive control; EOG artifacts |
| Behind ears | TP9 / TP10 | Mastoid / Retroauricular | Reference site; far-field alpha/SMR |

---

## 4. Dataset Selection

### 4.1 Selection Criteria

For this experiment, the dataset **must** satisfy:

1. ✅ Binary left/right hand motor imagery task
2. ✅ Contains channels C3, C4, Cz (for Phase 1 baseline)
3. ✅ Contains channels T7/T3, T8/T4, F7, F8 (for Phase 3 glasses simulation)
4. ✅ Sufficient trial count per subject (≥100 trials for reliable within-subject evaluation)
5. ✅ Available via Python (MNE / MOABB / braindecode)

### 4.2 Dataset Comparison

| Dataset | Subjects | Channels | Has T7/T8, F7/F8? | Trials (L/R MI) | Fs (Hz) | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **PhysioNet EEGMMI** | 109 (103 clean) | 64 | ✅ Yes | ~42–45 / subj | 160 | ⚠️ Low trial count |
| **BCI Comp. IV 2a** | 9 | 22 | ❌ No | 288 / subj | 250 | ❌ Missing lateral channels |
| **Lee2019_MI** | 54 | 62 | ✅ Yes | 400 / subj | 1000 | ✅ **Primary choice** |
| **Cho2017** | 52 | 64 | ✅ Yes | 100–120 / subj | 512 | ✅ **Secondary choice** |
| **Schirrmeister2017 (HGD)** | 14 | 128 | ✅ Yes | ~500 L/R / subj | 500 | ✅ **DL validation** |
| **BCI Comp. IV 2b** | 9 | 3 | ❌ No | 720 / subj | 250 | ❌ Only C3/Cz/C4 |

### 4.3 Selected Datasets

#### Primary Dataset: Lee2019_MI

**Rationale:** Largest modern Left/Right MI dataset with 54 subjects, 400 trials per subject, multi-session design, full 62-channel coverage including all target channels. Ideal for both within-subject and cross-subject evaluation.

```python
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery

dataset = Lee2019_MI()
paradigm = LeftRightImagery()
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=list(range(1, 55)))
```

#### Secondary Dataset: Cho2017

**Rationale:** 52 subjects with 64-channel coverage, includes EMG verification channels to confirm no actual muscle activation during imagery — critical for artifact control in Phase 3.

```python
from moabb.datasets import Cho2017
dataset = Cho2017()
```

#### Validation Dataset: Schirrmeister2017 (High-Gamma Dataset)

**Rationale:** 128 channels at 500 Hz with ~1000 trials per subject. Best available dataset for training deep learning models without overfitting. Used to validate findings from Lee2019.

```python
from moabb.datasets import Schirrmeister2017
dataset = Schirrmeister2017()
```

---

## 5. Preprocessing Pipeline

### 5.1 Pipeline Overview

```
Raw EEG (Continuous)
    │
    ├─ [1] Load & Channel Selection
    │       Select target montage subset or use full channels
    │
    ├─ [2] Band-Pass Filter (Continuous)
    │       ★ MUST filter before epoching to avoid edge ringing
    │       IIR Butterworth 4th order, zero-phase (filtfilt)
    │       Passband: 4–40 Hz (broad for DL) or 8–30 Hz (CSP/Riemannian)
    │
    ├─ [3] Notch Filter
    │       50 Hz (Korea/Europe) or 60 Hz (US), Q ≥ 30
    │
    ├─ [4] Bad Channel Detection & Interpolation
    │       Detect flat/noisy channels → spherical spline interpolation
    │       ★ MUST do before CAR to prevent noise injection
    │
    ├─ [5] Re-Referencing
    │       Common Average Reference (CAR) for ≥32 channels
    │       Or Surface Laplacian (CSD) for high spatial specificity
    │
    ├─ [6] ICA Artifact Removal
    │       Algorithm: Picard or Extended Infomax
    │       High-pass ≥1.0 Hz before ICA fitting
    │       Identify EOG components (correlation with Fp1/Fp2 > 0.4)
    │       ★ Fit ICA on training data only (no leakage)
    │
    ├─ [7] Epoching
    │       Window: [0.5s, 4.0s] post-cue (avoid cue-evoked VEP in first 0.5s)
    │       Baseline: [-1.0s, 0.0s] relative to cue onset
    │
    ├─ [8] Artifact Rejection
    │       Peak-to-peak threshold: 80–120 µV (sensorimotor channels)
    │       Or use autoreject for automated Bayesian thresholding
    │
    ├─ [9] Normalization
    │       Per-channel z-score (fit on training set only)
    │       Or Exponential Moving Standardization (for braindecode)
    │
    └─ [10] Feature Extraction / Model Input
            CSP: Covariance matrix → spatial filters → log-variance
            DL: Raw normalized epochs (1, C, T) tensor
```

### 5.2 Critical Anti-Patterns to Avoid

| # | Anti-Pattern | Consequence | Correct Approach |
|:---:|:---|:---|:---|
| 1 | Filtering after epoching | Edge ringing artifacts | Filter continuous raw data, then epoch |
| 2 | Fitting CSP on full dataset | Data leakage → inflated accuracy | Fit CSP strictly inside training fold |
| 3 | Computing global z-score before split | Leaks test statistics | Fit scaler on training set only |
| 4 | Overlapping sliding windows split randomly | Temporal autocorrelation leakage | Split at trial level first, then window |
| 5 | ICA fitted on train + test | Distribution leakage | Fit ICA per-subject on training data |
| 6 | Using Cz as reference for MI | Extinguishes foot MI, distorts C3/C4 | Re-reference to CAR or Laplacian |
| 7 | CAR with bad channels included | Noise injected into all channels | Interpolate bad channels before CAR |

### 5.3 Preprocessing Parameters Summary

| Parameter | CSP/Riemannian Pipeline | Deep Learning Pipeline |
|:---|:---|:---|
| **Band-pass** | 8–30 Hz | 4–40 Hz |
| **Filter type** | Butterworth 4th order, zero-phase | FIR or Butterworth, zero-phase |
| **Re-reference** | CAR | CAR |
| **Epoch window** | [0.5s, 3.5s] post-cue | [0.5s, 4.0s] post-cue |
| **Baseline** | [-1.0s, 0.0s] subtractive | [-1.0s, 0.0s] subtractive |
| **Artifact threshold** | 100 µV peak-to-peak | autoreject |
| **Normalization** | None (CSP handles internally) | Per-channel z-score or EMS |
| **Resample** | 250 Hz | 128 Hz (EEGNet) or 250 Hz |

---

## 6. Model Architecture & Selection

### 6.1 Model Comparison Matrix

| Model | Type | Parameters | BCI IV-2a 2-Class Acc | PhysioNet 2-Class Acc | Best For |
|:---|:---|:---:|:---:|:---:|:---|
| **CSP + LDA** | Traditional | ~12 weights | 79%–83% | 76%–82% | Baseline reference |
| **FBCSP** | Traditional | Variable | 82%–86% | 80%–85% | Robust traditional baseline |
| **Riemannian TS + LR** | Traditional | $C(C+1)/2$ | 84%–89% | 83%–88% | Best classical; small datasets |
| **EEGNet-8,2** | CNN | ~2.5k | 80%–84% | 81%–86% | Low-param DL baseline |
| **ShallowConvNet** | CNN | ~45k | 82%–87% | 82%–87% | SMR-optimized DL |
| **EEG Conformer** | Hybrid CNN+Transformer | ~1M–2M | 86%–91% | 86%–91% | Max performance (large data) |

### 6.2 Selected Models for Each Phase

#### Phase 1 (Full-Channel Baseline)

All five models will be evaluated to establish comprehensive baselines:

1. **CSP + LDA** — Gold standard traditional baseline
2. **Riemannian Tangent Space + Logistic Regression** — State-of-the-art traditional
3. **EEGNet-8,2** — Compact deep learning baseline
4. **ShallowConvNet** — SMR-optimized deep learning
5. **EEG Conformer** — Maximum capacity model (if dataset size permits)

```python
# Classical Pipelines
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

# Pipeline 1: CSP + LDA
pipe_csp = make_pipeline(
    CSP(n_components=6, log=True, norm_trace=False),
    LinearDiscriminantAnalysis()
)

# Pipeline 2: Riemannian TS + Logistic Regression
pipe_riemann = make_pipeline(
    Covariances(estimator='lwf'),
    TangentSpace(metric='riemann'),
    LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
)
```

#### Phase 2 (Channel Degradation)

Use the **best-performing model from Phase 1** and systematically reduce channels.

#### Phase 3 (Glasses-Only)

Re-train all models from Phase 1 using only the glasses-accessible channel subset.

### 6.3 Deep Learning Hyperparameters

| Hyperparameter | EEGNet-8,2 | ShallowConvNet | EEG Conformer |
|:---|:---|:---|:---|
| **Temporal kernel** | $f_s / 2$ (64 @ 128 Hz) | $(1, 25)$ | $(1, 31)$ |
| **Spatial filters** | $F_1=8$, $D=2$, $F_2=16$ | 40 | 40 |
| **Dropout** | 0.25–0.5 | 0.5 | 0.3–0.5 |
| **Optimizer** | Adam | Adam | AdamW |
| **Learning rate** | $1 \times 10^{-3}$ | $6.25 \times 10^{-4}$ | $5 \times 10^{-4}$ |
| **Weight decay** | $1 \times 10^{-4}$ | 0 | $5 \times 10^{-2}$ |
| **Epochs** | 300–500 | 500 | 500 |
| **Batch size** | 64 | 64 | 32–64 |
| **Early stopping** | Patience 50 | Patience 50 | Patience 30 |

### 6.4 Data Augmentation Strategy (Deep Learning)

| Technique | Parameters | Apply When |
|:---|:---|:---|
| **Sliding window** | Window: 2.5s, Stride: 0.25s | After trial-level train/test split |
| **Gaussian noise** | σ = 0.02 × std(X) | Training only |
| **Channel dropout** | $p_{\text{drop}} = 0.1$ | Training only |
| **EEG Mixup** | $\lambda \sim \text{Beta}(0.3, 0.3)$ | Training only, same class |

---

## 7. Experimental Protocol

### 7.1 Phase 1 — Full-Channel Baseline

**Objective:** Establish robust, methodologically sound baseline accuracies using all available EEG channels.

#### Channel Configuration
- **Full montage:** All 62 channels (Lee2019_MI) or all 64 channels (Cho2017)

#### Cross-Validation Strategy
- **Within-subject:** 5-fold stratified cross-validation (preserving class balance)
- **Cross-session:** Train on Session 1, test on Session 2 (Lee2019 only)
- **Cross-subject (LOSO):** Leave-One-Subject-Out for generalization assessment

#### Metrics
- Classification Accuracy (%)
- Cohen's Kappa ($\kappa$)
- Confusion Matrix
- Per-subject accuracy distribution (histogram + box plot)

#### Success Criteria
- Population average accuracy **≥75%** across all subjects
- Top-quartile subjects **≥85%**
- Identification of "BCI-literate" vs "BCI-illiterate" subjects for Phase 2/3 analysis

---

### 7.2 Phase 2 — Systematic Channel Degradation

**Objective:** Map the relationship between electrode proximity to motor cortex and classification accuracy.

#### Channel Degradation Tiers

| Tier | Channels Included | # Channels | Rationale |
|:---:|:---|:---:|:---|
| **T0** | Full montage (all 62) | 62 | Phase 1 baseline |
| **T1** | Sensorimotor core: C3, C1, Cz, C2, C4, FC3, FC1, FCz, FC2, FC4, CP3, CP1, CPz, CP2, CP4 | 15 | Motor cortex + immediate surround |
| **T2** | Central strip only: C3, Cz, C4, FC3, FC4, CP3, CP4 | 7 | Minimal motor cortex coverage |
| **T3** | Minimal central: C3, Cz, C4 | 3 | BCI Competition IV 2b equivalent |
| **T4** | Central + Lateral: C3, Cz, C4, T7, T8, F7, F8 | 7 | Hybrid: motor cortex + glasses |
| **T5** | Lateral only (glasses): T7, T8, F7, F8 | 4 | Glasses frame simulation — **KEY TEST** |
| **T6** | Extended glasses: T7, T8, F7, F8, Fp1, Fp2, TP9, TP10 | 8 | Glasses + nose bridge + behind ears |

#### Protocol
1. For each tier, **re-train** all Phase 1 models using only the specified channel subset.
2. CSP filters and Riemannian covariances are recomputed for each channel configuration.
3. Plot **Degradation Curve:** Accuracy (y-axis) vs. Channel Tier (x-axis).

#### Key Comparisons
- **T0 → T3:** How much accuracy is retained with just 3 central channels?
- **T3 → T5:** The critical gap — what is lost when motor cortex channels are completely removed?
- **T4 vs T5:** Does adding C3/C4 to the glasses channels significantly improve accuracy?
- **T5 vs T6:** Does adding nose bridge (Fp1/Fp2) and behind-ear (TP9/TP10) channels help?

---

### 7.3 Phase 3 — Glasses-Only Classification & Artifact Control

**Objective:** Rigorously evaluate whether genuine neural MI can be decoded from glasses-accessible channels, controlling for artifact confounds.

#### 3A. Classification with Glasses Channels

Channel set: **T7, T8, F7, F8** (Tier T5) and extended **T7, T8, F7, F8, Fp1, Fp2, TP9, TP10** (Tier T6).

#### 3B. Re-referencing Experiment

Test different referencing strategies optimized for lateral montages:

| Strategy | Formulation | Hypothesis |
|:---|:---|:---|
| CAR (4-ch) | $v_i - \frac{1}{4}\sum v_k$ | Likely ineffective — too few channels for CAR |
| Cross-hemisphere bipolar | $T7 - T8$ and $F7 - F8$ | May preserve lateralized SMR asymmetry |
| Linked mastoids | $v_i - \frac{A1 + A2}{2}$ | Standard sparse-montage reference |
| Ipsilateral bipolar | $T7 - F7$ (left) and $T8 - F8$ (right) | Tests local gradient |

#### 3C. Artifact Control Protocol

> [!CAUTION]
> **This is the most critical section of the entire experiment.**
> Temporal channels (T7/T8) sit directly on the temporalis muscle. Any MI decoding accuracy above chance from these channels could be driven by **subconscious jaw clenching or muscle tension** rather than genuine neural activity.

**Artifact Control Measures:**

1. **EMG Frequency Analysis:**
   - Compare classification accuracy using **8–13 Hz only** (pure mu-rhythm, below EMG range) vs. **20–45 Hz** (EMG-contaminated range).
   - If accuracy is driven by the 20–45 Hz band at lateral channels, it is likely EMG-based.

2. **EMG Amplitude Monitoring (Cho2017 only):**
   - Use the simultaneous forearm EMG channels to verify no actual motor execution.
   - Correlate temporal-channel high-gamma power with EMG amplitude.

3. **ICA Component Inspection:**
   - After ICA decomposition, classify components as neural vs. myogenic.
   - Re-run classification with and without muscle-artifact ICA components to measure artifact contribution.

4. **Permutation Testing:**
   - For any lateral-channel accuracy >55%, run 1000 permutation tests (shuffled labels) to establish the null distribution and compute the exact p-value.

5. **Topographic Validation:**
   - Visualize CSP spatial filter topomaps. Genuine MI patterns should show clear contralateral lateralization centered on C3/C4. If topomaps show temporal/frontal maxima, the signal is likely artifactual.

---

## 8. Evaluation & Statistical Analysis Plan

### 8.1 Metrics

| Metric | Formula | Purpose |
|:---|:---|:---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Primary performance measure |
| **Cohen's Kappa (κ)** | $\kappa = \frac{p_o - p_e}{1 - p_e}$ | Chance-adjusted agreement (critical for 2-class MI) |
| **AUC-ROC** | Area under ROC curve | Threshold-independent discrimination |
| **Per-subject accuracy** | Individual subject scores | Identify BCI-literate subgroup |

### 8.2 Statistical Tests

| Comparison | Test | Rationale |
|:---|:---|:---|
| Model A vs Model B (same subjects) | **Paired Wilcoxon signed-rank test** | Non-parametric, paired comparison of accuracy distributions |
| Channel Tier X vs chance (50%) | **One-sample Wilcoxon** + **Permutation test (n=1000)** | Conservative non-parametric test against chance |
| Accuracy across channel tiers | **Friedman test** (repeated measures) | Non-parametric ANOVA for multiple related groups |
| Post-hoc pairwise tier comparisons | **Nemenyi test** or **Wilcoxon with Bonferroni correction** | Multiple comparison correction |

### 8.3 Reporting Standards

- Report **mean ± std** accuracy across subjects
- Report **median** and **IQR** (interquartile range) to account for BCI-illiterate outliers
- Provide **per-subject accuracy table** (no cherry-picking)
- Include **box plots** showing full distribution for each channel tier
- Report **exact p-values** for all statistical comparisons
- Use **α = 0.05** (Bonferroni-corrected for multiple comparisons)

---

## 9. Risk Analysis & Mitigation

### 9.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|:---|:---:|:---:|:---|
| Phase 1 accuracy below 75% average | Medium | High | Try multiple preprocessing pipelines; use subject screening; consider adding motor execution trials |
| Phase 3 accuracy at chance level | **High** | Medium | This is a valid negative result. Document rigorously with artifact controls to confirm it is genuine. |
| EMG artifacts inflating lateral accuracy | **High** | **Critical** | Implement full artifact control protocol (Section 7.3C) |
| Data leakage inflating all results | Medium | Critical | Use sklearn Pipeline; fit all transforms inside CV folds; verify with MOABB benchmark |
| Overfitting DL models on small trial counts | Medium | High | Use EEGNet (2.5k params); early stopping; data augmentation; report training curves |
| Computational resource constraints | Low | Medium | Start with CSP/Riemannian (fast); use EEGNet before Conformer |

### 9.2 Methodological Risks

| Risk | Mitigation |
|:---|:---|
| Confirmation bias (wanting lateral channels to work) | Pre-register hypothesis; implement rigorous artifact controls; report negative results honestly |
| Comparing different preprocessing for different tiers | Use **identical** preprocessing pipeline for all channel tiers |
| Session/order effects in cross-session evaluation | Randomize session assignment; report both directions |

---

## 10. Implementation Roadmap

### 10.1 Phase 1: Foundation (Weeks 1–3)

```
Week 1: Environment Setup & Data Loading
├── Set up Python environment (MNE, MOABB, braindecode, PyTorch, pyriemann)
├── Download and validate Lee2019_MI dataset
├── Implement preprocessing pipeline with unit tests
└── Verify event codes and epoch extraction

Week 2: Classical Models
├── Implement CSP + LDA pipeline
├── Implement Riemannian TS + LogisticRegression pipeline
├── Run 5-fold within-subject CV on all 54 subjects
├── Generate per-subject accuracy table and box plots
└── Identify BCI-literate subgroup (accuracy > 70%)

Week 3: Deep Learning Models
├── Implement EEGNet-8,2 (PyTorch / braindecode)
├── Implement ShallowConvNet
├── Train with proper CV, early stopping, augmentation
├── Compare all 4 models; select best for Phase 2
└── Document Phase 1 results
```

### 10.2 Phase 2: Degradation Analysis (Weeks 4–5)

```
Week 4: Channel Degradation Experiment
├── Define 7 channel tiers (T0–T6)
├── Re-train best model for each tier
├── Plot degradation curve (accuracy vs channel tier)
└── Statistical tests for each tier comparison

Week 5: Analysis & Interpretation
├── Identify critical accuracy drop points
├── Compute channel importance rankings (CSP filter weights, attention maps)
├── ERD/ERS topographic visualization per tier
└── Document Phase 2 results
```

### 10.3 Phase 3: Glasses Simulation (Weeks 6–8)

```
Week 6: Lateral-Only Classification
├── Re-train all models with T5 and T6 channel sets
├── Test different re-referencing strategies
├── Run permutation tests for significance
└── Compare results with Cho2017 (EMG validation)

Week 7: Artifact Control
├── Frequency-band analysis (mu-only vs broadband)
├── ICA component classification (neural vs muscle)
├── CSP topographic validation
├── Cross-reference with Cho2017 EMG channels
└── Document artifact control results

Week 8: Synthesis & Reporting
├── Compile all results into final report
├── Generate publication-quality figures
├── Write conclusions and future directions
└── Assess feasibility of glasses-based MI-BCI
```

### 10.4 Python Environment

```
# requirements.txt
mne>=1.7.0
moabb>=1.1.0
braindecode>=0.8.0
pyriemann>=0.5
scikit-learn>=1.5.0
torch>=2.3.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.9.0
seaborn>=0.13.0
autoreject>=0.4.0
scipy>=1.14.0
```

---

## 11. References

### Datasets
- Lee, M.-H., et al. (2019). EEG dataset and OpenBMI toolbox for three BCI paradigms. *GigaScience*, 8(5).
- Cho, H., et al. (2017). EEG datasets for motor imagery brain–computer interface. *GigaScience*, 6(7).
- Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11).
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
- Brunner, C., et al. (2008). BCI Competition 2008 – Graz data set A. *Institute for Knowledge Discovery*, Graz University of Technology.

### Models & Methods
- Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5).
- Barachant, A., et al. (2013). Classification of covariance matrices using a Riemannian-based kernel for BCI applications. *Neurocomputing*, 112.
- Ang, K. K., et al. (2008). Filter bank common spatial pattern (FBCSP) in brain-computer interface. *IEEE IJCNN*.
- Song, Y., et al. (2022). EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization. *IEEE TNSRE*.

### Neuroscience
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11).
- Blankertz, B., et al. (2010). The Berlin Brain-Computer Interface: Non-medical uses of BCI technology. *Frontiers in Neuroscience*, 4.

### Wearable BCI
- Debener, S., et al. (2015). Unobtrusive ambulatory EEG using a smartphone and flexible printed electrodes around the ear. *Scientific Reports*, 5.
- Kosmyna, N., & Maes, P. (2019). AttentivU: An EEG-based closed-loop biofeedback system for real-time monitoring and improvement of engagement. *CHI EA '19*.

---

> **Next Steps:** After review and approval of this experimental design, proceed to set up the Python environment and begin Phase 1 data loading and preprocessing pipeline implementation.
