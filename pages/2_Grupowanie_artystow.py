"""
Strona: GRUPOWANIE artystow (K-Means).
Kategoria: grupowanie (uczenie nienadzorowane).

Interakcja: uzytkownik wprowadza profil nowego artysty -> przyporzadkowanie do grupy
(realizuje "przyporzadkowanie do grupy nowego profilu klienta").
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P
from ui_common import load_model, naglowek_strony

naglowek_strony("Grupowanie artystow",
                "K-Means dzieli artystow na grupy o podobnym profilu audio.")

M = load_model("clustering")
kmeans, scaler, pca = M['kmeans'], M['scaler'], M['pca']
feat = M['feature_cols']
prof = M['profile']

# --- Charakterystyka klastrow ----------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Liczba klastrow (K)", M['K'])
m2.metric("Silhouette", f"{M['silhouette']:.3f}")
m3.metric("Artysci w analizie", len(prof))

OPISY = {
    0: "spokojne / akustyczne - wyzsza popularnosc",
    1: "taneczne / pozytywne - mainstream",
    2: "ekstremalne / energetyczne - nisza (metal, grindcore)",
}

st.divider()
st.subheader("Przyporzadkuj nowy profil artysty")
st.markdown("Ustaw usrednione cechy artysty - K-Means wskaze najblizsza grupe.")

ranges = M['ranges']
sr = M['scaler_mean']
zakresy01 = {'danceability', 'speechiness', 'instrumentalness', 'liveness', 'valence'}

wart = {}
cols = st.columns(4)
for i, f in enumerate(feat):
    with cols[i % 4]:
        lo, hi = float(ranges.loc[f, 'min']), float(ranges.loc[f, 'max'])
        dom = float(sr.get(f, (lo + hi) / 2))
        if f in zakresy01:
            wart[f] = st.slider(f, 0.0, 1.0, round(dom, 2), 0.01)
        elif f == 'tempo':
            wart[f] = st.slider("tempo (BPM)", 40.0, 220.0, round(dom, 1), 1.0)
        elif f == 'popularity_mean':
            wart[f] = st.slider("popularnosc (sr.)", 0.0, 100.0, round(dom, 1), 1.0)
        elif f == 'energy':
            wart[f] = st.slider("energy", 0.0, 1.0, round(dom, 2), 0.01)
        else:
            wart[f] = st.slider(f, lo, hi, round(dom, 2))

X_new = scaler.transform(pd.DataFrame([wart])[feat])
klaster = int(kmeans.predict(X_new)[0])
odl = kmeans.transform(X_new)[0]
pc = pca.transform(X_new)[0]

st.success(f"### -> Klaster {klaster}  ({OPISY.get(klaster, '')})")
st.caption("Odleglosci do centrow: " +
           "  ".join(f"k{j}={d:.2f}" for j, d in enumerate(odl)))

# --- Wizualizacja PCA -------------------------------------------------------
st.divider()
c1, c2 = st.columns([3, 2])

with c1:
    st.markdown("##### Klastry w przestrzeni PCA (2 skladowe)")
    df_plot = prof.copy()
    df_plot['cluster'] = df_plot['cluster'].astype(str)
    base = alt.Chart(df_plot).mark_circle(size=45, opacity=0.45).encode(
        x=alt.X('PC1:Q'), y=alt.Y('PC2:Q'),
        color=alt.Color('cluster:N', legend=alt.Legend(title="Klaster"),
                        scale=alt.Scale(scheme='tableau10')),
        tooltip=['artist_main', 'main_genre', 'cluster', 'popularity_mean'])
    punkt = alt.Chart(pd.DataFrame({'PC1': [pc[0]], 'PC2': [pc[1]]})).mark_point(
        shape='diamond', size=420, color='black', filled=True).encode(
        x='PC1:Q', y='PC2:Q')
    st.altair_chart(base + punkt, width='stretch')
    st.caption("Czarny romb = wprowadzony profil.")

with c2:
    st.markdown("##### Profil klastrow (srednie cech)")
    st.dataframe(M['cluster_means'].T.style.background_gradient(cmap="RdYlBu_r", axis=1)
                 .format("{:.2f}"), width='stretch')
    st.markdown("##### Licznosc")
    st.bar_chart(pd.Series(M['sizes']), height=180, color="#1DB954")

# --- Dominujace gatunki / przykladowi artysci ------------------------------
st.divider()
st.markdown("##### Co jest w klastrze?")
kolumny = st.columns(M['K'])
for k in range(M['K']):
    sub = prof[prof['cluster'] == k]
    with kolumny[k]:
        st.markdown(f"**Klaster {k}** ({len(sub)})  \n{OPISY.get(k, '')}")
        top_g = sub['main_genre'].value_counts().head(4)
        st.write("Gatunki:", ", ".join(f"{g} ({c})" for g, c in top_g.items()))
        przyklady = sub.nlargest(5, 'popularity_mean')['artist_main'].tolist()
        st.write("Przyklady:", ", ".join(przyklady))
