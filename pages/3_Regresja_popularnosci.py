import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P
from ui_common import load_model, naglowek_strony

naglowek_strony("Regresja popularnosci",
                "Przewidywanie popularnosci artysty na podstawie cech audio.")

M = load_model("regression")
model, scaler, feat = M['model'], M['scaler'], M['features']

m1, m2 = st.columns(2)
m1.metric("Najlepszy model", M['model_nazwa'].split('(')[0].strip())
m2.metric("Artysci (po IQR)", M['n'])

st.divider()
st.subheader("Oszacuj popularnosc")
st.markdown("Ustaw usrednione cechy artysty - model oszacuje jego popularnosc.")

sr = M['srednie']
ranges = M['ranges']
zakresy01 = {'danceability', 'speechiness', 'acousticness', 'instrumentalness', 'valence'}

wart = {}
cols = st.columns(3)
for i, f in enumerate(feat):
    with cols[i % 3]:
        if f in zakresy01:
            wart[f] = st.slider(f, 0.0, 1.0, round(float(sr[f]), 2), 0.01)
        elif f == 'tempo':
            wart[f] = st.slider("tempo (BPM)", 40.0, 220.0, round(float(sr[f]), 1), 1.0)
        elif f == 'loudness':
            wart[f] = st.slider("loudness (dB)", -40.0, 2.0, round(float(sr[f]), 1), 0.5)
        elif f == 'mode':
            wart[f] = float(st.selectbox("tonacja (mode)", [0, 1], index=1))
        elif f == 'log_duration':
            minuty = st.slider("dlugosc (min)", 0.5, 10.0,
                               round(float(np.expm1(sr[f]) / 60000), 1), 0.1)
            wart[f] = np.log1p(minuty * 60000)
        else:
            wart[f] = st.slider(f, float(ranges.loc[f, 'min']),
                                float(ranges.loc[f, 'max']), round(float(sr[f]), 2))

X_new = scaler.transform(pd.DataFrame([wart])[feat])
pred = float(model.predict(X_new)[0])
pred = max(0.0, min(100.0, pred))

st.success(f"### Szacowana popularnosc: {pred:.1f} / 100")
st.progress(pred / 100)

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.markdown("##### Porownanie modeli (zbior testowy)")
    tab = pd.DataFrame(M['wyniki']).T[['MAE', 'RMSE', 'R2']]
    st.dataframe(tab.style.format("{:.3f}").background_gradient(cmap="Greens", subset=['R2']),
                 width='stretch')
    st.caption("Niskie R2 to cecha problemu: popularnosc zalezy glownie od czynnikow "
               "pozaaudio (marketing, rok). Model i tak bije baseline (srednia).")

with c2:
    st.markdown("##### Wplyw cech (standaryzowane wspolczynniki OLS)")
    coefs = M['coefs']
    df_c = pd.DataFrame({'cecha': coefs.index, 'wspolczynnik': coefs.values})
    wykres = alt.Chart(df_c).mark_bar().encode(
        x=alt.X('wspolczynnik:Q'),
        y=alt.Y('cecha:N', sort='-x'),
        color=alt.condition(alt.datum.wspolczynnik > 0,
                            alt.value("#1DB954"), alt.value("#e74c3c")),
        tooltip=['cecha', 'wspolczynnik'])
    st.altair_chart(wykres, width='stretch')
    st.caption("Zielony = podnosi popularnosc, czerwony = obniza.")

st.divider()
st.markdown("##### Rzeczywista vs przewidywana popularnosc (test)")
df_vp = pd.DataFrame({'rzeczywista': M['y_test'], 'przewidywana': M['pred_test']})
linia = pd.DataFrame({'x': [0, 100], 'y': [0, 100]})
chart = alt.Chart(df_vp).mark_circle(size=35, opacity=0.4, color="#1DB954").encode(
    x=alt.X('rzeczywista:Q', scale=alt.Scale(domain=[0, 80])),
    y=alt.Y('przewidywana:Q', scale=alt.Scale(domain=[0, 80])),
    tooltip=['rzeczywista', 'przewidywana'])
ideal = alt.Chart(linia).mark_line(color='gray', strokeDash=[5, 5]).encode(x='x', y='y')
st.altair_chart(chart + ideal, width='stretch')
st.caption("Linia przerywana = predykcja idealna (y = x).")
