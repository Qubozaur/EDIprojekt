"""Wspolne funkcje pomocnicze dla stron Streamlit (cache'owane ladowanie)."""
import joblib
import pandas as pd
import streamlit as st

import pipeline as P


@st.cache_resource(show_spinner="Wczytywanie modeli...")
def load_model(nazwa):
    """Laduje artefakt modelu z katalogu models/ (raz, z cache)."""
    sciezka = P.MODELS_DIR / f"{nazwa}.pkl"
    if not sciezka.exists():
        st.error(f"Brak pliku modelu: {sciezka.name}. "
                 f"Uruchom najpierw:  python train_models.py")
        st.stop()
    return joblib.load(sciezka)


@st.cache_data(show_spinner="Wczytywanie danych...")
def load_tracks():
    """Surowa tabela utworow (z cache)."""
    df_tracks, _ = P.load_raw()
    return df_tracks


@st.cache_data(show_spinner="Przygotowanie tabeli artystow...")
def load_classification_df():
    df_artist, meta = P.build_classification()
    return df_artist, meta


def naglowek_strony(tytul, podtytul):
    """Spojny naglowek strony."""
    st.title(tytul)
    st.caption(podtytul)
