## Pathway A: In-Depth Guide to Designed Experiments (RCTs)

Used when you can actively control and randomize treatment assignment.  
This is the gold standard for establishing causality because randomization ensures that, on average, the treatment and control groups are identical across all characteristics — both observed and unobserved.

---

### 1. Standard A/B/n Test

**When to Use:**  
The default workhorse for most product and marketing experiments. Ideal for simple product/UI changes (e.g., button colors, new recommendation carousels, email subject lines) where users' experiences are independent of each other and there is a low risk of one user's treatment affecting another's behavior (**low interference**).

**Core Method:**  
- **ANCOVA (CUPED)** for variance reduction.  
- While a simple t-test is valid, ANCOVA is more powerful (regression-based: `outcome ~ treatment + pre_experiment_covariate`).  
- By including a pre-experiment variable highly correlated with the outcome (e.g., past purchases), the model explains away predictable "noise."  
- This reduces variance, tightens confidence intervals, and increases the ability to detect smaller effects with the same sample size.

---

### 2. Clustered RCT

**When to Use:**  
Essential when **network effects or interference** exist, common in two-sided marketplaces.  
For example: a promotion given to one user might take a delivery rider who would have otherwise served another user. Randomizing by individual users would bias estimates.  
Instead, randomization is done at the **group level** (e.g., city, neighborhood, or sales team) to contain spillovers within the cluster.

**Core Method:**  
- **Generalized Linear Mixed Model (GLMM)**.  
- Designed for nested data structures.  
- Accounts for correlation among users within the same cluster by estimating a *random effect* for each cluster.  
- Correctly adjusts standard errors to avoid false positives and provides an unbiased estimate of the treatment effect.

---

### 3. Switchback RCT (Crossover Design)

**When to Use:**  
Used for **system-wide changes** where a concurrent control group is impossible because the treatment affects the entire system.  
Examples: logistics, pricing, or backend algorithm changes (e.g., a new rider dispatching algorithm).  
The entire system (e.g., a city) is alternated between treatment and control over time (e.g., every hour).

**Core Method:**  
- **Generalized Linear Mixed Model (GLMM)** with controls for:  
  - Time periods (e.g., lunch rush vs. dinner).  
  - Carryover effects (treatment lingering into the next period).  
- Ensures seasonality and crossover contamination are correctly modeled.
