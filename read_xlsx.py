#%%
import pandas as pd

# 读取xlsx文件
df_radii = pd.read_excel('E:\\ResearchGroup\\AdsorbEPred\\pre_set.xlsx',sheet_name='Radii_X')
df_ip = pd.read_excel('E:\\ResearchGroup\\AdsorbEPred\\pre_set.xlsx',sheet_name='IP')

element_dict = df_radii.iloc[:,:5].set_index('symbol').T.to_dict('dict')
for idx,col in enumerate(df_ip.columns):
    if col in element_dict:
        for i in range(9):
            if df_ip.iloc[i,idx]==200000:
                element_dict[col][df_ip.iloc[i,0]] = -1
            else:
                element_dict[col][df_ip.iloc[i,0]] = df_ip.iloc[i,idx]
for j in element_dict:
    print(j, element_dict[j])
#%%
print(df_ip.iloc[19,0])