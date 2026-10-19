# Locating Compromised Smart Meters in EV-Integrated Smart Grids

Code for the paper **"Trust-Based Localization of Compromised Meters in EV-Integrated Smart Grids under Energy Theft Attacks"** (link TBD).

## Overview

Energy theft attacks compromise smart meters so that they report less electricity consumption than was actually delivered. In public EV charging locations, dynamic charging loads make these attacks hard to detect and hard to locate. This project proposes a semi-supervised framework that identifies *which* charging locations contain compromised meters. It has three parts:

1. **Rule-Based Clustering (RBC):** groups locations with similar load profiles by seasonal peak and average load, so one model can serve many locations.
2. **LSTM Load Forecasting:** trains one load forecasting model per cluster, which cuts the number of models.
3. **Trust Assessment:** compares reported and predicted load against a location-specific tolerance. Deviations raise an alarm count, and each week a location is labeled *Trusted* or *Compromised*.

The framework is evaluated on synthetic data that combines real utility base load with a custom EV charging simulator, under several attack intensities.

## Paper

- **Title:** Trust-Based Localization of Compromised Meters in EV-Integrated Smart Grids under Energy Theft Attacks
- **Authors:** Shengwen Ding and Mai Abdelhakim
- **Venue:** Workshop on Cyber-Physical Power System Resilience: Challenges and Emerging Solutions, held at IEEE SmartGridComm 2026 (IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids), College Station, TX, USA, October 2026
- **Link:** TBD

## Citation

```bibtex
@inproceedings{ding2026trust,
  author    = {Ding, S. and Abdelhakim, M.},
  title     = {Trust-Based Localization of Compromised Meters in {EV}-Integrated Smart Grids under Energy Theft Attacks},
  booktitle = {Proc. Workshop on Cyber-Physical Power System Resilience: Challenges and Emerging Solutions, IEEE SmartGridComm 2026: IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids},
  address   = {College Station, TX, USA},
  month     = oct,
  year      = {2026}
}
```
