"""
Strona: PRZEGLAD / EDA.
Opis projektu, realizacja zalozen i wstepna analiza danych.
"""
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P
from ui_common import load_tracks, load_classification_df

st.title("Eksploracja danych muzycznych - Spotify + Last.fm")
st.caption("Projekt EDI: grupowanie, klasyfikacja i regresja na danych o utworach. "
           "Dane semantyczne (tagi Last.fm) zasilaja klasyfikacje gatunku.")

st.info("Wybierz komponent z nawigacji u gory: **Klasyfikacja**, **Grupowanie** "
        "lub **Regresja**. Kazda strona pobiera dane od uzytkownika i zwraca wynik modelu.")

# --- Realizacja zalozen -----------------------------------------------------
st.subheader("Realizacja zalozen projektu")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
**3 kategorie algorytmow** (wymagane >=3):
- **Klasyfikacja** gatunku - kNN + drzewo decyzyjne
- **Grupowanie** artystow - K-Means
- **Regresja** popularnosci - OLS / Ridge / wielomianowa

**Dane semantyczne (tekstowe):** tagi Last.fm w klasyfikacji
(MultiLabelBinarizer -> 574 cechy tagowe).
""")
with c2:
    st.markdown("""
**Pozostale zagadnienia:**
- Przygotowanie danych: czyszczenie, braki, normalizacja, podzial
- Wizualizacja danych uczacych (ponizej)
- Selekcja atrybutow (usuniecie cech redundantnych)
- Identyfikacja wartosci odstajacych (IQR / DBSCAN)
- Dobor miary podobienstwa (kNN: manhattan vs euclidean)
- Ocena dzialania algorytmow (metryki, walidacja krzyzowa)
- Klasyfikacja nowego komentarza / profilu klienta (strony interaktywne)
""")

st.divider()

# --- Przeglad danych --------------------------------------------------------
st.subheader("Przeglad zbioru danych (EDA)")

df_tracks = load_tracks()
df_artist, meta = load_classification_df()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Utwory (po czyszczeniu)", f"{len(df_tracks):,}".replace(",", " "))
m2.metric("Unikalni artysci", f"{df_tracks['artist_main'].nunique():,}".replace(",", " "))
m3.metric("Gatunki (track_genre)", df_tracks['track_genre'].nunique())
m4.metric("Klasy po filtrach", meta['n_artystow'])

with st.expander("Przygotowanie danych (pipeline)"):
    st.markdown("""
1. **Czyszczenie:** usuniecie duplikatow `track_id`, usuniecie wierszy z brakami.
2. **Agregacja do poziomu artysty:** srednie cech audio, dominanta cech kategorycznych,
   gatunek glowny = najczestszy `track_genre`.
3. **Filtry jakosci:** min. 5 utworow na artyste (grupowanie/regresja),
   min. 50 artystow na klase i min. 6 wystapien tagu (klasyfikacja).
4. **Normalizacja:** `StandardScaler` na cechach numerycznych.
5. **Podzial:** train/test (stratyfikowany dla klasyfikacji) + walidacja krzyzowa.
""")

# --- Wykres: rozklad klas ---------------------------------------------------
st.markdown("##### Rozklad klas - liczba artystow na gatunek")
rozklad = df_artist['gatunek_glowny'].value_counts()
st.bar_chart(rozklad, color="#1DB954", height=320)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("##### Korelacja cech audio")
    cechy = P.CLF_AUDIO_NUM + ['popularity']
    corr = df_artist[cechy].corr()
    st.dataframe(corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
                 .format("{:.2f}"), width='stretch')
    st.caption("Wysoka korelacja energy-loudness -> w klasyfikacji usunieto energy "
               "(selekcja atrybutow).")

with col_b:
    st.markdown("##### Profil audio wybranych gatunkow (z-score)")
    profil = df_artist.groupby('gatunek_glowny')[P.CLF_AUDIO_NUM].mean()
    profil_z = (profil - profil.mean()) / profil.std()
    wybrane = ['black-metal', 'club', 'salsa', 'acoustic', 'anime', 'bluegrass']
    wybrane = [g for g in wybrane if g in profil_z.index]
    st.dataframe(profil_z.loc[wybrane].style.background_gradient(cmap="RdBu_r", axis=None)
                 .format("{:.2f}"), width='stretch')
    st.caption("Kazdy gatunek ma odmienny 'odcisk' audio - to czyni klasyfikacje mozliwa.")

st.divider()
st.caption("Dane: dataset_clean2.csv (Spotify) + tags_cache2.csv (Last.fm). "
           "Modele wytrenowane skryptem train_models.py.")
