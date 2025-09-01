## 💾 Input Data Designs for Causal Analysis

The required structure of your input DataFrame is fundamentally tied to the causal method you intend to use.

---

### 1. Cross-Sectional Data (The "Snapshot")

**Description:**  
Each row is a single unit (e.g., user) observed at a single point in time or over a single, fixed experimental period.  
This is the most common and basic data structure.

**Structure Example:**

| user_id | treatment_group | converted | past_purchases |
|---------|-----------------|-----------|----------------|
| 101     | A (Control)     | 0         | 5              |
| 102     | B (Treatment)   | 1         | 2              |
| 103     | B (Treatment)   | 0         | 11             |

**Used For:**  
- Standard **A/B/n Tests** (analyzed with ANCOVA/CUPED).  
- Observational studies (analyzed with **Matching, IPW, AIPW, DML**).

---

### 2. Panel / Longitudinal Data (The "Movie")

**Description:**  
Tracks multiple units over multiple time periods, creating a history for each unit.  
This structure is powerful because it allows you to control for stable, unobserved differences between units (e.g., innate skill).

**Structure Example:**

| user_id | month   | treatment_flag | outcome_orders |
|---------|---------|----------------|----------------|
| 101     | 2025-01 | 0              | 5              |
| 101     | 2025-02 | 1              | 8              |
| 102     | 2025-01 | 0              | 12             |
| 102     | 2025-02 | 0              | 11             |

**Used For:**  
- **Difference-in-Differences / TWFE** for staggered rollouts.  
- **Switchback RCTs**.

---

### 3. Clustered Data (The "Grouped Snapshot")

**Description:**  
Cross-sectional data that includes a column grouping individual units into non-independent clusters.  
This structure explicitly defines the **nested nature** of the data.

**Structure Example:**

| user_id | cluster_id | treatment_group | converted |
|---------|------------|-----------------|-----------|
| 101     | Dubai      | B               | 1         |
| 102     | Dubai      | B               | 1         |
| 201     | Abu Dhabi  | A               | 0         |

**Used For:**  
- **Clustered RCTs** (analyzed with **GLMM**).

---

### 4. Regression Discontinuity Data (The "Cutoff")

**Description:**  
Cross-sectional data with a **continuous running variable** that determines treatment assignment.  
The running variable should not be easily manipulable by subjects around the cutoff.

**Structure Example:**

| user_id | past_spending | is_vip | future_spending |
|---------|---------------|--------|-----------------|
| 101     | 995           | 0      | 150             |
| 102     | 1001          | 1      | 350             |

**Used For:**  
- **Regression Discontinuity Design (RDD)**.
