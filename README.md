# A Guide to a Modular Causal Inference & Experimentation Platform

This document provides a comprehensive blueprint for a modular platform designed to measure causality. It outlines the core causal inference methods, the specific problems they solve, and the required input data structures for analysis.  

The goal is to create a clear decision tree, guiding an analyst from their business problem to the most appropriate and robust causal method.

---

## 🧠 Causal Inference Methods: A Method-per-Module Guide

The platform is organized around two primary pathways, determined by the fundamental question of experimental control: **was the treatment actively randomized, or was it observed in the wild?**

### Pathway A: Designed Experiments (RCTs)

Used when you can actively control and randomize treatment assignment.  
This is the gold standard for establishing causality because randomization ensures that, on average, the treatment and control groups are identical across all characteristics.

➡️ [Click here for the detailed guide to RCT Methods](#)

---

### Pathway B: Observational & Quasi-Experimental Studies

Used when you cannot randomize and must infer causality from observational data.  
These methods rely on strong, testable assumptions to approximate the conditions of an RCT.

➡️ [Click here for the detailed guide to Observational Methods](#)

---

## 💾 Input Data Designs for Causal Analysis

The required structure of your input DataFrame is fundamentally tied to the causal method you intend to use.

➡️ [Click here for the detailed guide to Input Data Designs](#)
