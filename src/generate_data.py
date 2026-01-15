import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 1000  # number of employees

data = {
    "employee_id": range(1, N + 1),
    "gender": np.random.choice(["Male", "Female"], size=N, p=[0.55, 0.45]),
    "department": np.random.choice(
        ["Engineering", "Sales", "HR", "Finance", "Operations"],
    xsize=N,
        p=[0.35, 0.25, 0.1, 0.15, 0.15]
    ),
    "role_level": np.random.choice(
        ["Junior", "Mid", "Senior", "Lead"],
        size=N,
        p=[0.3, 0.4, 0.2, 0.1]
    )
    "tenure_years": np.round(np.random.exponential(scale=4, size=N), 1),
    "performance_rating": np.random.randint(1, 6, size=N),
    "absenteeism_days": np.random.poisson(lam=5, size=N),
    "monthly_salary": np.random.normal(4500, 1200, size=N).astype(int),
}

df = pd.DataFrame(data)

# Attrition logic (realistic but synthetic)
risk = (
    (df["tenure_years"] < 2).astype(int) * 0.4 +
    (df["performance_rating"] <= 2).astype(int) * 0.3 +
    (df["absenteeism_days"] > 8).astype(int) * 0.3
)

df["attrition"] = (np.random.rand(N) < risk).astype(int)

# Ensure no negative salaries
df["monthly_salary"] = df["monthly_salary"].clip(lower=2000)

# Save
output_path = Path(__file__).resolve().parents[1] / "data" / "sample_data.csv"
output_path.parent.mkdir(exist_ok=True)

df.to_csv(output_path, index=False)
print(f"Synthetic dataset saved to {output_path}")
