from langchain_openai import ChatOpenAI
import csv
import pandas as pd

# with open("C:\\Users\\User\Documents\\terrorism.csv", mode='r', newline='', encoding='utf-8') as file:

df = pd.read_csv("C:\\Users\\User\Documents\\terrorism.csv")

print(df.head())
print(df.columns)

for index, row in df.iterrows():
    para = ''
    para += f"On the date of {row['iyear']}-{row['imonth']}-{row['iday']} in the region of {row['region_txt']} in the country of {row['country_txt']}"
    para += f"in the region of {row['region_txt']} and city of {row['city']} at {row['location']}"
    para += {row['summary']}
    para += f"it was a {row['attack_type']} aimed at {row['targtype1_txt']}, {row['targsubtype1_txt']}, {row['corp1']}, {row['target1']} who is from {row['natlty1_txt'
]}"
    if row["targtype2_txt"]:
        para += f"and {row['targtype2_txt']}, {row['targsubtype2_txt']}, {row['corp2']}, {row['target2']} who is from {row['natlty1_txt']}"
    break
# print(f"{df['iyear']}-{df['imonth']}-{df['iday']}")