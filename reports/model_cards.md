# Model Cards

Generated 2026-10-30 18:07:30 UTC.

## 1. Delivery delay classifier

- **Purpose:** estimate the probability that a trip is not delivered on time.
- **Target:** `is_delayed` (1 = Delayed or Failed).
- **Features:** distance, driver score, load, vehicle age, month, monsoon flag,
  origin, destination, vehicle class and day of week.
- **Split:** chronological 80/20 (4,000 train /
  1,000 test).
- **Best model:** calibrated_hist_gradient_boosting.
- **Metrics:** accuracy 0.923, precision 0.496,
  recall 0.750, F1 0.597, ROC-AUC 0.952.
- **Intended use:** prioritise proactive customer communication and dispatch
  support. Not a basis for penalising drivers.
- **Limitations:** synthetic data; no traffic, weather or customer-priority
  features; class imbalance limits recall.

## 2. Fuel anomaly detector

- **Purpose:** flag unusual fuel readings for investigation.
- **Primary detector:** robust z-score per vehicle class (|z| > 3.5).
- **Primary metrics:** precision 0.909, recall 0.714,
  F1 0.800, flagged rate 0.022.
- **Secondary model:** IsolationForest trained on clean trips; precision
  0.317, recall 0.464,
  F1 0.377, ROC-AUC 0.855.
- **Intended use:** maintenance/leak/theft investigation queue.
- **Limitations:** anomalies are injected; real sensor drift and refuelling
  events need domain validation.

## 3. Driver behaviour score

- **Purpose:** transparent coaching and recognition score.
- **Formula:** 60 + 40 × (0.40·on-time + 0.25·fuel efficiency + 0.20·incident-free
  + 0.15·consistency).
- **Range:** 60–100; grades A–D.
- **Drivers scored:** 50; mean 91.76, std 2.81.
- **Fairness:** depot mean gap 2.3 points;
  flag = False.
- **Intended use:** coaching, recognition and support. **Not** an automated
  employment decision. Appeals and human review are mandatory.
- **Limitations:** outcome-based; does not observe harsh braking, speeding or
  hours-of-service because the dataset has no telematics.

## 4. Route optimiser

- **Method:** exact permutations for <=7 stops; nearest-neighbour + 2-opt otherwise.
- **Average distance saving on benchmark routes:**
  5.94%.
- **Intended use:** multi-stop delivery sequencing.
- **Limitations:** static road-distance matrix; no live traffic or time windows.
