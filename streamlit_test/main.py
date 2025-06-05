import streamlit as st

st.write("Hi there")
st.subheader("Hello")
st.selectbox("languages ", {'python','java'})
st.checkbox("python")

st.slider("java",0,100)
st.select_slider("Select", ["Best","Average","Worst"])
st.progress(10)
st.button("Red")

st.sidebar.title("About")
st.sidebar.selectbox("features",{"a","b","c"})
st.sidebar.markdown("info")
st.sidebar.button("press")