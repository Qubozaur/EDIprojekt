"""
Wspolny modul danych i preprocessingu dla aplikacji EDI.
Odtwarza logike przygotowania danych z trzech notebookow:
  - klasyfikacja_gatunku.ipynb  (klasyfikacja + dane semantyczne / tagi Last.fm)
  - grupowanie_artystow.ipynb   (grupowanie K-Means)
  - regresja.ipynb              (regresja popularnosci)

Dzieki jednemu zrodlu prawdy aplikacja webowa i skrypt trenujacy uzywaja
dokladnie tych samych transformacji.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# --- Sciezki ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SPOTIFY_CSV = DATA_DIR / "dataset_clean2.csv"
TAGS_CSV = DATA_DIR / "tags_cache2.csv"

RANDOM_STATE = 42

# --- Wspolne zestawy cech --------------------------------------------------
AUDIO_NUM = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
             'liveness', 'speechiness', 'loudness', 'tempo', 'duration_ms']
AUDIO_DOM = ['mode', 'time_signature', 'explicit']

# Klasyfikacja: po usunieciu energy (redundancja |r|>0.7 z loudness)
CLF_AUDIO_NUM = [f for f in AUDIO_NUM if f != 'energy']          # 9 cech
CLF_USUNIETE_GATUNKI = {'malay', 'disney', 'kids', 'study'}      # klasy "mood", nie gatunki
CLF_MIN_ARTYSTOW = 50
CLF_MIN_DF_TAGOW = 6

# Grupowanie: bez loudness i acousticness (korelacja), + popularity_mean
CLUSTER_AUDIO = ['danceability', 'energy', 'loudness', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
CLUSTER_FEATURES = [f for f in CLUSTER_AUDIO
                    if f not in ('loudness', 'acousticness')] + ['popularity_mean']  # 8 cech
CLUSTER_SPOKEN = {'comedy', 'spoken-word'}
CLUSTER_MIN_TRACKS = 5
CLUSTER_K = 3

# Regresja: bazowe cechy audio + inzynieria + usuniecie wielokoliniowych
REG_AUDIO_BASE = ['danceability', 'energy', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo', 'mode', 'loudness']
REG_ENGINEERED = ['log_duration']
REG_REMOVE = ['energy', 'liveness']
REG_FEATURES = [f for f in (REG_AUDIO_BASE + REG_ENGINEERED) if f not in REG_REMOVE]
REG_MIN_TRACKS = 5


# ===========================================================================
#  Wczytanie surowych danych
# ===========================================================================
def load_raw():
    """Zwraca (df_tracks, df_tags) - surowe tabele utworow i tagow."""
    df_tracks = pd.read_csv(SPOTIFY_CSV)
    df_tracks = df_tracks.drop_duplicates(subset='track_id', keep='first')
    df_tracks = df_tracks.dropna()
    df_tracks['artist_main'] = df_tracks['artists'].astype(str).str.split(';').str[0].str.strip()
    df_tracks['explicit'] = df_tracks['explicit'].astype(int)
    df_tags = pd.read_csv(TAGS_CSV)
    return df_tracks, df_tags


def normalizuj_tag(t):
    """Ujednolicenie tagu: male litery, bez myslnikow."""
    return str(t).strip().lower().replace('-', ' ')


# ===========================================================================
#  KLASYFIKACJA - budowa tabeli artysta -> gatunek + tagi semantyczne
# ===========================================================================
def build_classification(df_tracks=None, df_tags=None):
    """
    Buduje df_artist dla klasyfikacji oraz zwraca metadane (lista klas, slownik tagow).
    Replikuje komorki preprocessingu z klasyfikacja_gatunku.ipynb.
    """
    if df_tracks is None:
        df_tracks, df_tags = load_raw()

    aggr_num = df_tracks.groupby('artist_main')[AUDIO_NUM + ['popularity']].mean()
    aggr_dom = df_tracks.groupby('artist_main')[AUDIO_DOM].agg(lambda x: x.mode().iloc[0])
    df_artist = aggr_num.join(aggr_dom)

    gatunek_glowny = df_tracks.groupby('artist_main')['track_genre'].agg(
        lambda s: Counter(s).most_common(1)[0][0])
    df_artist['gatunek_glowny'] = gatunek_glowny

    # Tagi Last.fm (dane semantyczne)
    df_tags = df_tags.copy()
    df_tags['artist_main'] = df_tags['artist'].astype(str).str.strip()
    df_tags['tagi'] = df_tags['tags'].fillna('').apply(
        lambda s: list(dict.fromkeys(
            [normalizuj_tag(t) for t in s.split('|') if t.strip()])))
    mapa_tagow = df_tags.set_index('artist_main')['tagi'].to_dict()
    df_artist['tagi'] = df_artist.index.map(lambda a: mapa_tagow.get(a, []))

    # Usun tag identyczny z etykieta gatunku (zapobiega "wycieku" celu)
    df_artist['tagi'] = df_artist.apply(
        lambda r: [t for t in r['tagi'] if t != normalizuj_tag(r['gatunek_glowny'])], axis=1)

    # Filtry klas i tagow
    df_artist = df_artist[~df_artist['gatunek_glowny'].isin(CLF_USUNIETE_GATUNKI)].copy()
    liczba = df_artist['gatunek_glowny'].value_counts()
    klasy_ok = liczba[liczba >= CLF_MIN_ARTYSTOW].index.tolist()
    df_artist = df_artist[df_artist['gatunek_glowny'].isin(klasy_ok)].copy()

    licznik = Counter(t for lista in df_artist['tagi'] for t in lista)
    tagi_ok = {t for t, c in licznik.items() if c >= CLF_MIN_DF_TAGOW}
    df_artist['tagi'] = df_artist['tagi'].apply(lambda l: [t for t in l if t in tagi_ok])

    meta = {
        'klasy': sorted(df_artist['gatunek_glowny'].unique().tolist()),
        'tagi_slownik': sorted(tagi_ok),
        'n_artystow': len(df_artist),
    }
    return df_artist, meta


def parse_tagi_uzytkownika(tekst, tagi_slownik):
    """
    Zamienia wolny tekst / komentarz uzytkownika na liste znanych tagow.
    Dzieli po przecinkach, '|' i spacjach-frazach; dopasowuje do slownika.
    """
    if not tekst:
        return []
    surowe = []
    for kawalek in tekst.replace('|', ',').split(','):
        surowe.append(normalizuj_tag(kawalek))
    znalezione = []
    slownik_set = set(tagi_slownik)
    for frag in surowe:
        if frag in slownik_set and frag not in znalezione:
            znalezione.append(frag)
        else:
            # sprobuj dopasowac pojedyncze slowa/bigramy z komentarza
            slowa = frag.split()
            for n in (2, 1):
                for i in range(len(slowa) - n + 1):
                    kandydat = ' '.join(slowa[i:i + n])
                    if kandydat in slownik_set and kandydat not in znalezione:
                        znalezione.append(kandydat)
    return znalezione


# ===========================================================================
#  GRUPOWANIE - profile artystow
# ===========================================================================
def build_clustering(df_tracks=None):
    """Buduje profile artystow dla grupowania (grupowanie_artystow.ipynb)."""
    if df_tracks is None:
        df_tracks, _ = load_raw()

    prof = df_tracks.groupby('artist_main').agg(
        danceability=('danceability', 'mean'),
        energy=('energy', 'mean'),
        loudness=('loudness', 'mean'),
        speechiness=('speechiness', 'mean'),
        acousticness=('acousticness', 'mean'),
        instrumentalness=('instrumentalness', 'mean'),
        liveness=('liveness', 'mean'),
        valence=('valence', 'mean'),
        tempo=('tempo', 'mean'),
        popularity_mean=('popularity', 'mean'),
        n_tracks=('track_id', 'count'),
        n_genres=('track_genre', 'nunique'),
    ).reset_index()

    prof = prof[prof['n_tracks'] >= CLUSTER_MIN_TRACKS].reset_index(drop=True)
    main_genre = df_tracks.groupby('artist_main')['track_genre'].agg(lambda x: x.mode()[0])
    prof['main_genre'] = prof['artist_main'].map(main_genre)
    prof = prof[~prof['main_genre'].isin(CLUSTER_SPOKEN)].reset_index(drop=True)
    return prof


# ===========================================================================
#  REGRESJA - profile artystow (mediana popularnosci jako cel)
# ===========================================================================
def build_regression(df_tracks=None):
    """Buduje tabele artystow dla regresji (regresja.ipynb)."""
    if df_tracks is None:
        df_tracks, _ = load_raw()

    df = df_tracks.copy()
    df['log_duration'] = np.log1p(df['duration_ms'])
    all_feat = REG_AUDIO_BASE + REG_ENGINEERED

    artist_df = df.groupby('artist_main')[all_feat].mean().reset_index()
    artist_df['popularity'] = df.groupby('artist_main')['popularity'].median().values
    n_tracks = df.groupby('artist_main')['track_id'].count()
    artist_df = artist_df[artist_df['artist_main'].map(n_tracks) >= REG_MIN_TRACKS]
    artist_df = artist_df.reset_index(drop=True)

    # Filtracja outlierow 3xIQR po finalnych cechach (wariant z notebooka)
    mask = pd.Series(True, index=artist_df.index)
    for feat in REG_FEATURES:
        q1, q3 = artist_df[feat].quantile(0.25), artist_df[feat].quantile(0.75)
        iqr = q3 - q1
        mask &= (artist_df[feat] >= q1 - 3 * iqr) & (artist_df[feat] <= q3 + 3 * iqr)
    artist_clean = artist_df[mask].reset_index(drop=True)
    return artist_df, artist_clean
