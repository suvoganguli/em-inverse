# em-inverse

A small exploration of an electromagnetic inverse problem using Maxwell-based simulations and neural networks.

The goal is to understand how well a model can recover tumor parameters from multistatic EM measurements — and more importantly, when and why it fails.

---

## Overview

This project simulates a 2D electromagnetic setup:

- Circular antenna array surrounding a square domain  
- A small circular inclusion (tumor) with different permittivity and conductivity  
- Forward simulation using a simplified Maxwell (FDTD) solver  
- Neural networks trained to recover tumor parameters from measurements  

Three inverse problems are studied:

- **XY** → predict tumor location `(x, y)`  
- **XYR** → predict location + radius `(x, y, r)`  
- **XYRC** → predict location + radius + contrast `(x, y, r, c)`  

---

## Key Insight

Even when training appears successful, the inverse problem is not uniformly well-posed.

- Predictions show a **bias toward the center of the domain**
- The effect becomes much stronger as more parameters are added
- Sensitivity analysis reveals that the forward mapping becomes **locally flat near the center**

This creates a **low-information region**, making accurate recovery intrinsically difficult.

> The model does not fail to learn — it learns exactly what the system allows it to learn.

---

## Notebooks

### 01_datagen.ipynb
Generates datasets using the Maxwell-based simulation.

- Simulates multistatic measurements
- Saves datasets for:
  - XY
  - XYR
  - XYRC

---

### 02_model.ipynb
Trains convolutional neural networks on the generated datasets.

- CNN-based regression models
- Separate training for:
  - XY
  - XYR
  - XYRC
- Includes:
  - loss curves
  - prediction plots
  - error vector visualization

---

### 03_analysis.ipynb (optional / exploratory)
Analyzes why the inverse problem becomes difficult.

- Local sensitivity analysis
- Finite-difference approximation of Jacobian
- Cross-section plots of:
  - ‖∂M/∂x‖
  - ‖∂M/∂y‖
  - smallest singular value

---

## Config

`config.py` contains:

- Domain size and grid resolution  
- Antenna configuration  
- Material properties  
- Dataset sizes  
- Utility functions for saving figures and outputs  

---

## Requirements

- Python 3.x  
- NumPy  
- Matplotlib  
- PyTorch  

---

## Notes

- This is a compact research-style exploration, not a production EM solver  
- Boundary conditions and numerical schemes are simplified  
- The focus is on **understanding inverse behavior**, not achieving high-fidelity imaging  

---

## Reference

Accompanying article:

**"Why AI Can’t See: A Physics Perspective on an Inverse Problem"**

---
