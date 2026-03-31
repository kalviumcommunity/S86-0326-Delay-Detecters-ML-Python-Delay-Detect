"""
Script to generate sample data for testing the ML pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Create sample delivery data
n_samples = 200

data = {
    "distance_km": np.random.uniform(1, 50, n_samples),
    "items_count": np.random.randint(1, 20, n_samples),
    "order_value": np.random.uniform(10, 500, n_samples),
    "day_of_week": np.random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], n_samples),
    "day_of_month": np.random.randint(1, 31, n_samples),
    "zone": np.random.choice(["Zone_A", "Zone_B", "Zone_C", "Zone_D"], n_samples),
    "peak_hour": np.random.choice(["Morning", "Afternoon", "Evening", "Night"], n_samples),
    "is_delayed": np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 40% delayed
}

df = pd.DataFrame(data)

# Create output directory
output_dir = Path(__file__).parent / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

# Save to CSV
output_path = output_dir / "delivery_data.csv"
df.to_csv(output_path, index=False)

print(f"Sample data created: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
