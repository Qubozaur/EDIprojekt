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

# --- Rekomendacje: 10 najbardziej podobnych artystow z klastra --------------
st.divider()
st.subheader("Rekomendacje - 10 artystow z tej grupy najbardziej podobnych do profilu")
st.markdown("System liczy odleglosc wpisanego profilu do kazdego artysty w grupie "
            "(w przestrzeni standaryzowanych cech) i zwraca najblizszych, wraz z ich "
            "najpopularniejszym utworem.")

prof_klaster = prof[prof['cluster'] == klaster].copy()
X_klaster = scaler.transform(prof_klaster[feat])
prof_klaster['podobienstwo'] = np.linalg.norm(X_klaster - X_new, axis=1)
rekom = prof_klaster.nsmallest(10, 'podobienstwo')

tabela_rek = rekom[['artist_main', 'main_genre', 'top_track',
                    'top_track_pop', 'popularity_mean', 'podobienstwo']].copy()
tabela_rek.columns = ['Artysta', 'Glowny gatunek', 'Najpopularniejszy utwor',
                      'Popularnosc utworu', 'Sr. popularnosc', 'Odleglosc (mniej=lepiej)']
st.dataframe(tabela_rek.style.format({
    'Popularnosc utworu': '{:.0f}', 'Sr. popularnosc': '{:.1f}',
    'Odleglosc (mniej=lepiej)': '{:.3f}'})
    .background_gradient(cmap='Greens_r', subset=['Odleglosc (mniej=lepiej)']),
    width='stretch', hide_index=True)

# --- Dodanie nowego artysty do puli bazy ------------------------------------
st.divider()
st.subheader("Dodaj tego artyste do bazy")
st.markdown("Jesli profil pasuje, mozesz dopisac nowego artyste do puli danych "
            "(z przypisanym klastrem). W razie potrzeby uzupelnij brakujace pola.")

with st.form("dodaj_artyste"):
    cc1, cc2 = st.columns(2)
    nazwa_art = cc1.text_input("Nazwa artysty", placeholder="np. Nowy Wykonawca")
    top_utwor = cc2.text_input("Najpopularniejszy utwor (opcjonalnie)", placeholder="np. Tytul singla")
    wyslij = st.form_submit_button("Dodaj do bazy", type="primary")

if wyslij:
    if not nazwa_art.strip():
        st.warning("Podaj nazwe artysty, aby dodac go do bazy.")
    else:
        ile = P.dodaj_artyste_do_bazy(nazwa_art.strip(), wart, klaster, top_utwor.strip())
        st.success(f"Dodano artyste **{nazwa_art.strip()}** do klastra {klaster}. "
                   f"Pula dodanych artystow liczy teraz {ile}.")

dodani = P.wczytaj_dodanych()
if not dodani.empty:
    with st.expander(f"Artysci dodani z aplikacji ({len(dodani)})"):
        kol_pokaz = [c for c in ['artist_main', 'cluster', 'top_track', 'dodano'] if c in dodani.columns]
        st.dataframe(dodani[kol_pokaz].iloc[::-1], width='stretch', hide_index=True)

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

    wykres = base + punkt
    # Artysci dodani z aplikacji - rzutowani tym samym scaler+PCA
    if not dodani.empty and all(f in dodani.columns for f in feat):
        coords_d = pca.transform(scaler.transform(dodani[feat]))
        df_dod = pd.DataFrame({'PC1': coords_d[:, 0], 'PC2': coords_d[:, 1],
                               'artist_main': dodani['artist_main'].values,
                               'cluster': dodani['cluster'].astype(str).values})
        warstwa_dod = alt.Chart(df_dod).mark_point(
            shape='triangle-up', size=220, color='red', filled=True,
            stroke='black', strokeWidth=0.5).encode(
            x='PC1:Q', y='PC2:Q', tooltip=['artist_main', 'cluster'])
        wykres = base + warstwa_dod + punkt

    st.altair_chart(wykres, width='stretch')
    st.caption("Czarny romb = wprowadzony profil; czerwone trojkaty = artysci dodani do bazy.")

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
