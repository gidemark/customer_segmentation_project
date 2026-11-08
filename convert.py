import pandas as pd

src = r"C:\Users\opolot\Desktop\customer_segmentation_project\online_retail_II.xlsx"
dst = r"C:\Users\opolot\Desktop\customer_segmentation_project\data\raw\online_retail_II.csv"

df = pd.read_excel(src)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())
df.to_csv(dst, index=False)
print("CSV saved to", dst)