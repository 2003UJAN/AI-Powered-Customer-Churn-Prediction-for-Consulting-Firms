import pandas as pd
import numpy as np

def generate_customer_churn_data(rows=1000):
    np.random.seed(42)
    data = {
        "CustomerID": np.arange(1, rows + 1),
        "Tenure": np.random.randint(1, 60, size=rows),
        "MonthlySpend": np.random.uniform(100, 10000, size=rows).round(2),
        "ConsultingHours": np.random.randint(1, 100, size=rows),
        "SatisfactionScore": np.random.randint(1, 6, size=rows),
        "ServiceUsage": np.random.uniform(0.2, 1.0, size=rows),
        "ContractType": np.random.choice(["Fixed", "Flexible"], size=rows),
        "Churn": np.random.choice([0, 1], size=rows, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    df.to_csv("customer_churn.csv", index=False)
    print("✅ Synthetic churn data saved to 'customer_churn.csv'")

if __name__ == "__main__":
    generate_customer_churn_data()
