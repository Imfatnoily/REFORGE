# REFORGE
**REFORGE: Multi-modal Attacks Reveal Vulnerable Concept Unlearning in Image Generation Models**

> **WARNING**: This repository contains data or model outputs that may be offensive in nature.

## Environment Setup
To set up the environment:
```bash
conda env create -f environment.yml
conda activate reforge
```

## Usage

### 1. Generate Reference Images
Run the following command to generate target reference images:
```bash
python reference_generate.py
```

### 2. Generate Stroke Images
Run the following command to simulate stroke-based images:
```bash
python Stroke_Simulation.py
```

### 3. Run REFORGE Attack
To execute the REFORGE attack (e.g., evaluating the 'nudity' concept with the 'ESD' unlearning method), run:
```bash
python attack.py --concept nudity --unlearn_method ESD
```
