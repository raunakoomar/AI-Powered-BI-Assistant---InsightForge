import matplotlib.pyplot as plt
import seaborn as sns

def plot_monthly_sales(df):
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('Month')['Sales'].sum()
    fig, ax = plt.subplots()
    monthly_sales.plot(kind='bar', ax=ax)
    ax.set_title("Monthly Sales")
    ax.set_ylabel("Sales")
    return fig

def plot_region_comparison(df):
    region_sales = df.groupby('Region')['Sales'].sum()
    fig, ax = plt.subplots()
    region_sales.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title("Sales by Region")
    ax.set_ylabel("")
    return fig

def plot_product_performance(df):
    product_sales = df.groupby('Product')['Sales'].sum()
    fig, ax = plt.subplots()
    product_sales.plot(kind='bar', ax=ax)
    ax.set_title("Product Performance")
    ax.set_ylabel("Sales")
    return fig

def plot_customer_demographics(df):
    fig, ax = plt.subplots()
    sns.histplot(df['Customer_Age'], bins=10, kde=True, ax=ax)
    ax.set_title("Customer Age Distribution")
    ax.set_xlabel("Age")
    return fig