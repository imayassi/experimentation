## Pathway B: In-Depth Guide to Observational & Quasi-Experimental Studies

Used when you **cannot randomize** and must infer causality from observational data.  
These methods rely on strong, testable assumptions to approximate the conditions of an RCT.

---

### 1. Staggered Rollout Analysis

**When to Use:**  
When a feature is rolled out to different groups at different times — common for enterprise software or gradual feature releases.  
This leverages **early adopters** as the treatment group and **late adopters** (before they are treated) as the control group.

**Core Method:**  
- **Difference-in-Differences (DiD)** using a **Two-Way Fixed Effects (TWFE)** model.  
- Controls for unobserved, time-invariant confounders (e.g., a user’s innate skill) and unobserved time-varying shocks that affect everyone (e.g., a holiday).

**Key Diagnostic:**  
- **Event Study plot** required.  
- Visualizes treatment effects before and after treatment.  
- Strong evidence for causality is shown when **pre-treatment effects are statistically zero**, supporting the **Parallel Trends assumption**.

---

### 2. General Observational Study (Selection on Observables)

**When to Use:**  
When you have **cross-sectional data** with no natural experiment and must assume you have measured all key confounders that affect both treatment assignment and outcomes.  
Common in retrospective analyses of user behavior.

**Core Method (multi-step pipeline):**  
1. **Matching (CEM):** Coarsen and exactly match users on key confounders, discarding incomparable units.  
2. **Weighting (IPW):** Fine-tune balance by creating a pseudo-population where confounders no longer predict treatment.  
3. **Estimation (AIPW or Multi-Treatment DML):**  
   - **AIPW** is *doubly robust*: unbiased if either the matching or weighting model is correct.  
   - **DML** is ideal for high-dimensional confounders and multiple overlapping treatments.

**Key Diagnostic:**  
- **Standardized Mean Difference (SMD) plots** after balancing.  
- Treatment vs. control covariate distributions should be nearly identical (**|SMD| < 0.1**).

---

### 3. Regression Discontinuity Design (RDD)

**When to Use:**  
When treatment assignment is based on a **sharp cutoff or threshold** in a continuous variable (e.g., users who spend > $1000 become VIPs).  
This creates a highly credible *local* randomized experiment by comparing users just above and just below the cutoff.

**Core Method:**  
- **Local Linear Regression** around the cutoff to estimate the discontinuity (the “jump”) in outcomes.

---

## Universal Follow-up & Optimization Modules

Used after an initial ATE has been estimated to gain deeper insights or optimize in real-time.

---

### 1. Heterogeneity Analysis

**When to Use:**  
As a **secondary analysis** to move beyond the *average treatment effect* (ATE) and understand *for whom* the treatment worked best, worst, or even negatively.  
Often more valuable for business strategy than the ATE alone.

**Core Method:**  
- **Causal Tree** or **Causal Forest** to estimate **Conditional Average Treatment Effects (CATEs)**.  
- Partitions the user base to reveal statistically significant segments with different treatment effects.  
- Informs **personalization** and **targeted rollouts**.

---

### 2. Real-Time Optimization

**When to Use:**  
For **short-term optimizations** (e.g., headlines, promotions, UI layouts) where the goal is to dynamically find the best option and minimize **regret** (lost opportunity from showing a suboptimal variant).  
Contrasts with A/B tests, which focus on *statistical inference*.

**Core Method:**  
- **Multi-Armed Bandit (MAB)** algorithms.  
- Dynamically allocate more traffic to better-performing variants over time.  
- Balance **exploration vs. exploitation** to maximize cumulative reward.
