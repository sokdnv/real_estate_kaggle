import streamlit as st


st.header('Выводы')
st.divider()

st.page_link('app.py', label='На главную', icon='🔙')

st.markdown('- Начинай с самой простой модели и обработки!\n- Чуть меньше окр\n'
            '- Тестировать небольшие гипотезы и смотреть на изменение метрики\n'
            '- Catboost умнее тебя (пока что уж точно)'
)

st.image('reports/model.gif')


