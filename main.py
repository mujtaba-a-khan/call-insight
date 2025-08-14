# main.py
import importlib, streamlit as st

app = importlib.import_module("call_app.app")

if hasattr(app, "main"):
    app.main()
elif hasattr(app, "run"):
    app.run()
else:
    st.error("call_app/app.py needs a main() function.")
