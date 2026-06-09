import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P
from ui_common import load_model, load_classification_df, naglowek_strony

naglowek_strony("Klasyfikacja gatunku",
                "kNN + drzewo decyzyjne na cechach audio i tagach Last.fm (dane semantyczne).")

M = load_model("classification")
df_artist, meta = load_classification_df()
knn, tree, scaler, mlb = M['knn'], M['tree'], M['scaler'], M['mlb']
feat_cols, kol_skal = M['feature_columns'], M['kol_skalowane']
audio_num, audio_dom = M['kol_audio_num'], M['kol_audio_dom']


def zbuduj_wiersz(audio_dict, dom_dict, popularity, tagi_lista):
    """Sklada wektor cech w kolejnosci feature_columns i skaluje cechy numeryczne."""
    wiersz = {}
    wiersz.update(audio_dict)
    wiersz.update(dom_dict)
    wiersz['popularity'] = popularity
    tagi_set = set(tagi_lista)
    for kol in feat_cols:
        if kol.startswith('tag_'):
            wiersz[kol] = 1 if kol[4:] in tagi_set else 0
    X = pd.DataFrame([wiersz])[feat_cols]
    X[kol_skal] = scaler.transform(X[kol_skal])
    return X


def pokaz_predykcje(X, prawda=None):
    """Wyswietla predykcje obu modeli + top-2 + rozklad prawdopodobienstw."""
    for nazwa, model in [("kNN", knn), ("Drzewo decyzyjne", tree)]:
        proba = model.predict_proba(X)[0]
        klasy = model.classes_
        top = np.argsort(proba)[::-1][:3]
        pred = klasy[top[0]]
        if prawda is None:
            status = ""
        else:
            status = "(trafione)" if pred == prawda else "(pudlo)"
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(f"{nazwa} - predykcja", f"{pred} {status}",
                      f"{proba[top[0]]*100:.0f}% pewnosci")
            st.caption(f"Top-2: **{klasy[top[1]]}** ({proba[top[1]]*100:.0f}%)")
        with c2:
            df_top = pd.DataFrame({'gatunek': klasy[top], 'pstwo': proba[top]}).set_index('gatunek')
            st.bar_chart(df_top, color="#1DB954", height=160)


tab1, tab2, tab3 = st.tabs(["Artysta ze zbioru", "Nowy utwor / komentarz",
                            "Ocena modeli"])

with tab1:
    st.markdown("Wybierz artyste - model oszacuje gatunek na podstawie jego cech audio i tagow, "
                "a wynik porownamy z prawdziwa etykieta.")
    artysta = st.selectbox("Artysta", sorted(df_artist.index.tolist()), index=0)
    rzad = df_artist.loc[artysta]
    prawda = rzad['gatunek_glowny']

    audio_dict = {f: float(rzad[f]) for f in audio_num}
    dom_dict = {f: int(rzad[f]) for f in audio_dom}
    tagi = rzad['tagi']

    c1, c2 = st.columns(2)
    c1.info(f"**Prawdziwy gatunek:** {prawda}")
    c2.write(f"**Tagi Last.fm:** {', '.join(tagi[:12]) if tagi else '(brak)'}")

    X = zbuduj_wiersz(audio_dict, dom_dict, float(rzad['popularity']), tagi)
    st.divider()
    pokaz_predykcje(X, prawda)

with tab2:
    st.markdown("Ustaw cechy utworu i opisz go tagami - system **nie zna** tego utworu i "
                "oszacuje gatunek.")

    sr = M['srednie_audio']
    zakresy01 = {'danceability', 'valence', 'acousticness', 'instrumentalness',
                 'liveness', 'speechiness'}
    audio_dict = {}
    cols = st.columns(3)
    for i, f in enumerate(audio_num):
        with cols[i % 3]:
            if f in zakresy01:
                audio_dict[f] = st.slider(f, 0.0, 1.0, float(sr.get(f, 0.3)), 0.01)
            elif f == 'loudness':
                audio_dict[f] = st.slider("loudness (dB)", -60.0, 2.0, float(sr.get(f, -8.0)), 0.5)
            elif f == 'tempo':
                audio_dict[f] = st.slider("tempo (BPM)", 40.0, 220.0, float(sr.get(f, 120.0)), 1.0)
            elif f == 'duration_ms':
                minuty = st.slider("dlugosc (min)", 0.5, 10.0,
                                   float(sr.get(f, 210000) / 60000), 0.1)
                audio_dict[f] = minuty * 60000

    c1, c2, c3 = st.columns(3)
    dom_dict = {
        'mode': c1.selectbox("tonacja (mode)", [0, 1], index=1),
        'time_signature': c2.selectbox("metrum", [1, 3, 4, 5], index=2),
        'explicit': c3.selectbox("explicit", [0, 1], index=0),
    }
    popularity = st.slider("popularnosc (0-100)", 0, 100, 40)

    st.markdown("##### Tagi semantyczne (Last.fm)")
    cc1, cc2 = st.columns(2)
    tagi_wybrane = cc1.multiselect("Wybierz tagi ze slownika",
                                   options=M['tagi_slownik'], default=[])
    komentarz = cc2.text_input("...lub wpisz komentarz/opis (wykryjemy tagi)",
                               placeholder="np. raw black metal, norwegian, brutal")
    tagi_z_tekstu = P.parse_tagi_uzytkownika(komentarz, M['tagi_slownik'])
    tagi_all = list(dict.fromkeys(tagi_wybrane + tagi_z_tekstu))
    if komentarz:
        st.caption(f"Wykryte tagi z komentarza: {tagi_z_tekstu if tagi_z_tekstu else '(zadne nie pasuja do slownika)'}")

    if st.button("Klasyfikuj utwor", type="primary"):
        X = zbuduj_wiersz(audio_dict, dom_dict, popularity, tagi_all)
        st.divider()
        pokaz_predykcje(X)

with tab3:
    st.markdown("##### Metryki na zbiorze testowym")
    tab_metryk = pd.DataFrame(M['metryki']).T[
        ['accuracy', 'top2_accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']]
    st.dataframe(tab_metryk.style.format("{:.3f}").background_gradient(cmap="Greens"),
                 width='stretch')
    st.caption(f"Zbior: {M['n_train']} treningowych / {M['n_test']} testowych artystow, "
               f"{M['n_klas']} klas, {M['wymiar']} cech (w tym {M['n_tagow']} tagow).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Wklad cech: audio vs tagi (ablacja, F1 macro)")
        abl = pd.DataFrame(M['ablation']).T
        st.bar_chart(abl, height=280)
        st.caption("Same tagi bija samo audio - dane semantyczne rozdzielaja nakladajace sie "
                   "gatunki (np. black-metal vs grindcore).")
    with c2:
        st.markdown("##### Dobor miary podobienstwa (kNN)")
        st.markdown("""
GridSearchCV porownal metryki odleglosci; wygrala **manhattan** (k=5, wagi=odleglosc):

| Miara | charakterystyka |
|---|---|
| **manhattan** (wybrana) | suma roznic - odporna w wielowymiarowej, rzadkiej przestrzeni tagow |
| euclidean | wrazliwa na wymiarowosc (574 binarne tagi) |
| minkowski (p=3) | posrednia, bez przewagi |

Drzewo: kryterium **gini**, glebokosc 30.
""")

    st.markdown("##### F1 dla poszczegolnych klas")
    st.bar_chart(M['f1_per_klasa'], height=320)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Top 15 najwazniejszych cech (drzewo)")
        top = M['waznosci'].head(15).sort_values()
        st.bar_chart(top, horizontal=True, height=400, color="#e74c3c")
    with c4:
        st.markdown("##### Macierz pomylek kNN (recall per klasa)")
        cm = pd.DataFrame(M['confusion'], index=M['confusion_klasy'],
                          columns=M['confusion_klasy'])
        st.dataframe(cm.style.background_gradient(cmap="Blues", vmin=0, vmax=1)
                     .format("{:.2f}"), width='stretch', height=400)
