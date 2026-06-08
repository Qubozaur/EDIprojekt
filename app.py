"""
EDI - Eksploracja Danych: analiza muzyki Spotify + tagi Last.fm.
Punkt wejscia aplikacji: nawigacja w naglowku (position='top') laczaca strony.

Uruchomienie:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="EDI - Muzyka Spotify", layout="wide")

# Nawigacja w naglowku (zakladki u gory zamiast paska bocznego)
strony = [
    st.Page("pages/0_Przeglad.py", title="Przeglad", default=True),
    st.Page("pages/1_Klasyfikacja_gatunku.py", title="Klasyfikacja"),
    st.Page("pages/2_Grupowanie_artystow.py", title="Grupowanie"),
    st.Page("pages/3_Regresja_popularnosci.py", title="Regresja"),
]

nawigacja = st.navigation(strony, position="top")
nawigacja.run()
