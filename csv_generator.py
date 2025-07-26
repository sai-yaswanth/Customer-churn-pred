import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Define number of samples for churned (1) and non-churned (0)
num_samples = 80000
half_samples = num_samples // 2

# Generate values for churn = 1 (Churned)
churned_data = {
    "customer_name": ["Customer_" + str(i) for i in range(half_samples)],  # Unique customer names
    "age": np.random.randint(18, 70, size=half_samples),  # Random ages between 18 and 70
    "gender": np.random.choice(["Male", "Female"], size=half_samples),  # Random gender
    "subscription_length": np.random.randint(1, 7, size=half_samples),  # 1 to 6 months
    "subscription_type": np.random.choice(["Basic", "Standard", "Premium"], size=half_samples),  # Random subscription types
    "number_of_logins": np.random.randint(0, 11, size=half_samples),  # 0 to 10 logins
    "login_activity": np.random.choice(["Low", "Medium", "High"], size=half_samples),  # Random login activity
    "customer_ratings": np.random.uniform(1, 3, size=half_samples),  # Rating between 1 and 3
    "churn": np.ones(half_samples)  # Churned customers
}

# Generate values for churn = 0 (Not Churned)
not_churned_data = {
    "customer_name": ["Customer_" + str(i + half_samples) for i in range(half_samples)],  # Unique customer names
    "age": np.random.randint(18, 70, size=half_samples),  # Random ages between 18 and 70
    "gender": np.random.choice(["Male", "Female"], size=half_samples),  # Random gender
    "subscription_length": np.random.randint(12, 25, size=half_samples), # 12 to 24 months
    "subscription_type": np.random.choice(["Basic", "Standard", "Premium"], size=half_samples),  # Random subscription types
    "number_of_logins": np.random.randint(20, 51, size=half_samples),  # 20 to 50 logins
    "login_activity": np.random.choice(["Low", "Medium", "High"], size=half_samples),  # Random login activity
    "customer_ratings": np.random.uniform(4, 5, size=half_samples),  # Rating between 4 and 5
    "churn": np.zeros(half_samples)  # Non-churned customers
}

# Create DataFrames
churned_df = pd.DataFrame(churned_data)
not_churned_df = pd.DataFrame(not_churned_data)

# Combine the datasets
final_df = pd.concat([churned_df, not_churned_df])

# Shuffle the dataset to mix churned and non-churned customers
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save to CSV
final_df.to_csv("balanced_customer_churn.csv", index=False)
