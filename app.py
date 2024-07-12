import streamlit as st
import joblib
import pandas as pd
import sklearn
import numpy as np

from catboost import CatBoostRegressor
from my_module import MyTransformer, GroupMedianImputer, SomeCustomShit

sklearn.set_config(transform_output="pandas")

cb_pipeline = joblib.load('models/cb_pipeline.pkl')
st_pipeline = joblib.load('models/stacking_pipeline.pkl')

st.title('Real estate kaggle')
st.caption('Серёжа²')
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('Pipeline с использованием Stacking моделей **Catboost**, **SVR**, **GBoost**')
with col3:
    st.metric(label='RMSLE', value=0.12262)

st.image('reports/Screenshot 2024-07-12 at 12.16.36.png')

with st.sidebar:
    st.write('Панель управления')
    file = st.file_uploader(label='Грузи свой датасет, бро', type='csv')
    if file:
        df = pd.read_csv(file)
        answer0 = np.exp(cb_pipeline.predict(df))
        answer1 = np.exp(st_pipeline.predict(df))
        answer = 0.5 * answer0 + 0.5 * answer1
        result_df = pd.DataFrame({
            'Id': df['Id'],
            'SalePrice': answer
        })

        st.download_button(label='Забирай ответы', data=result_df.to_csv(index=False), file_name='submission.csv')
