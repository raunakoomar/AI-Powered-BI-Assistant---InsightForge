import pandas as pd

def load_data():
    df = pd.read_csv("datasets/sales_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def generate_summary(df):
    summary = {}
    df['Month'] = df['Date'].dt.to_period('M')
    summary['total_sales'] = df['Sales'].sum()
    summary['avg_sales'] = df['Sales'].mean()
    summary['monthly_sales'] = df.groupby('Month')['Sales'].sum().to_dict()
    summary['region_sales'] = df.groupby('Region')['Sales'].sum().to_dict()
    summary['product_sales'] = df.groupby('Product')['Sales'].sum().to_dict()
    summary['customer_age_avg'] = df['Customer_Age'].mean() if 'Customer_Age' in df.columns else None
    return summary