import streamlit as st
import joblib
import pandas as pd
import sklearn
import numpy as np

from my_module import MyTransformer, GroupMedianImputer, SomeCustomShit

sklearn.set_config(transform_output="pandas")

cb_pipeline = joblib.load('models/cb_pipeline.pkl')

st.title('Real estate kaggle')
st.caption('Серёжа²')
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('Pipeline с использованием модели **Catboost**')
with col3:
    st.metric(label='RMSLE', value=0.12447)

st.image('reports/Screenshot 2024-07-12 at 10.16.21.png')

with st.sidebar:
    st.write('Панель управления')
    file = st.file_uploader(label='Грузи свой датасет, бро', type='csv')
    if file:
        df = pd.read_csv(file)
        answer = np.exp(cb_pipeline.predict(df))
        result_df = pd.DataFrame({
            'Id': df['Id'],
            'SalePrice': answer
        })

        st.download_button(label='Забирай ответы', data=result_df.to_csv(index=False), file_name='submission.csv')

