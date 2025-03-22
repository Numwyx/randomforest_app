import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

import matplotlib.font_manager as fm  

# 添加字体文件到 Matplotlib 的字体管理器  
font_files = fm.findSystemFonts(fontpaths=['.'], fontext='ttf')  
for font_file in font_files:  
    fm.fontManager.addfont(font_file)  

# 设置 Matplotlib 使用中文字体  
plt.rcParams['font.family'] = ['SimHei']  # 替换为你实际使用的字体名称  
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

st.set_page_config(    
    page_title="重症创伤性脊髓损伤患者 28 天生存状况评估",
    page_icon="💠",
    layout="wide"
)

st.markdown('''
    <h1 style="font-size: 20px; text-align: center; color: black; background: #1E88E5; border-radius: .5rem; margin-bottom: 1rem;">
    重症创伤性脊髓损伤患者 28 天生存状况评估
    </h1>''', unsafe_allow_html=True)

# 导入模型文件
with open("best_rf(1).pkl", 'rb') as f:
    model = pickle.load(f)

with st.expander("**预测输入**", True):
    col = st.columns(4)
    
colnames = {
    "injury_site":"脊髓损伤部位",
    "injury_type":"脊髓损伤性质",
    "vasoactive_drug":"脊髓损伤后24小时内患者是否使用血管活性药物",
    "charlson_comorbidity_index":"查尔森合并症指数",
    "sbp_min":"脊髓损伤后24小时内患者收缩压最低值",
    "ph_min":"脊髓损伤后24小时内患者PH最低值",
    "temperature_min":"脊髓损伤后24小时内患者体温最低值",
    "wbc_max":"脊髓损伤后24小时内患者白细胞计数最高值",
    "sodium_max":"脊髓损伤后24小时内患者血钠最高值",
    "lactate_min":"脊髓损伤后24小时内患者血乳酸最低值"}
    
d1 = {"颈椎":1, "胸部":2, "腰椎":3, "不详":4}
d2 = {"完全损伤":1, "不完全损伤":2, "不详":3}
d3 = {"是":0, "否":1}
dd = [d1, d2, d3]

x = list(colnames.keys())
x1 = list(colnames.values())
y = [4, 1, 1, 2, 86, 7.36, 36.56, 9.9, 137, 1]

inputdata = {}

for i, j, k in zip(x, y, x1):
    if x.index(i) in [0, 1, 2]:
        inputdata[i] = dd[x.index(i)][col[x.index(i)%4].selectbox(k, dd[x.index(i)])]
    elif x.index(i)==3:
        inputdata[i] = col[x.index(i)%4].number_input(k, value=int(j), min_value=0, max_value=37, step=1)
    else:
        inputdata[i] = col[x.index(i)%4].number_input(k, value=float(j), min_value=0.00,)

predata = pd.DataFrame([inputdata])
data = predata.copy()

with st.expander("**预测结果**", True):
    d = model.predict_proba(predata).flatten()
    
    # SHAP 值的计算  
    explainer = shap.TreeExplainer(model)  # 使用TreeExplainer来解释随机森林模型  
    shap_values = explainer.shap_values(predata.iloc[0, :])
    
    pre = predata
    pre.columns = [colnames[i] for i in pre.columns.tolist()]

    shap_plot = shap.plots.force(
        explainer.expected_value[1], 
        shap_values[:,1].flatten(), 
        pre.iloc[0, :], 
        show=False, 
        matplotlib=True)
    
    for text in plt.gca().texts:  # 遍历当前坐标轴的所有文本对象 
        if "=" in text.get_text():
            text.set_rotation(-15)  # 设置旋转角度，修改为你希望的角度 
            text.set_va('top')
        text.set_bbox(dict(facecolor='none', alpha=0.5, edgecolor='none'))
    
    plt.tight_layout()  
    st.pyplot(plt.gcf())
    
    st.markdown(f'''
    <div style="font-size: 20px; text-align: center; color: red; background: transparent; border-radius: .5rem; margin-bottom: 1rem;">
    这名重症创伤性脊髓损伤患者的28天存活率为：{round(d[1]*100, 2)}%
    </div>''', unsafe_allow_html=True)









