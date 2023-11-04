import streamlit as st
import joblib
import numpy as np
import warnings
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

model = joblib.load("classifier_RF final 8-feature model.m")

with st.form("my_form"):
   slider_Bodyweight = st.number_input('Body weight')
   slider_PRI = st.number_input('Pediatric Risk of Mortality III (PRISM Ⅲ)')
   slider_MAP = st.number_input('Minimum mean arterial pressure (MAP_min)')
   slider_APTT = st.number_input('Activated partial thromboplastin time (APTT)')
   num_TBil = st.number_input('Total bilirubin (TBil)')
   num_eGFR = st.number_input('Estimated glomerular filtration rate (eGFR)')
   slider_BUN = st.number_input('Blood urea nitrogen (BUN)')
   num_UO = st.number_input('Urine output (UO)')

   submitted = st.form_submit_button("Predict")
   if submitted:
      x_test = np.array([[slider_Bodyweight, slider_PRI, slider_MAP, slider_APTT,
         num_TBil, num_eGFR, slider_BUN, num_UO]])
      explainer = shap.TreeExplainer(model) #创建解释器，本例针对树模型创建解释器
      shap_values = explainer.shap_values(x_test) #生成SHAP值
      temp = np.round(x_test, 2)
      shap.force_plot(explainer.expected_value[1], shap_values[1],temp,
         feature_names = ['Body weight','PRISM Ⅲ','MAP_min','APTT','TBil','eGFR','BUN','UO'], matplotlib=True, show=False)
      plt.xticks(fontproperties='Times New Roman', size=15)
      plt.yticks(fontproperties='Times New Roman', size=20)
      plt.tight_layout()
      plt.savefig("19 AKI force plot.png",dpi=600) #可以保存图
      pred = model.predict_proba(x_test)
      st.markdown("#### _Based on feature values, predicted possibility of AKI is {}%_".format(round(pred[0][1], 4)*100))
      st.image('19 AKI force plot.png')
