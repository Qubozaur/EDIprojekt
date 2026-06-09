import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, top_k_accuracy_score,
                             confusion_matrix, silhouette_score, r2_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor

import pipeline as P

P.MODELS_DIR.mkdir(exist_ok=True, parents=True)


def train_classification():
    print("[1/3] Klasyfikacja gatunku ...")
    df_artist, meta = P.build_classification()

    mlb = MultiLabelBinarizer(sparse_output=False)
    X_tagi = mlb.fit_transform(df_artist['tagi'])
    nazwy_tagow = [f"tag_{t}" for t in mlb.classes_]
    X_tagi_df = pd.DataFrame(X_tagi, index=df_artist.index, columns=nazwy_tagow)

    kol_audio = P.CLF_AUDIO_NUM + P.AUDIO_DOM + ['popularity']
    X = pd.concat([df_artist[kol_audio], X_tagi_df], axis=1)
    y = df_artist['gatunek_glowny']
    kol_skalowane = P.CLF_AUDIO_NUM + ['popularity']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=P.RANDOM_STATE)

    scaler = StandardScaler()
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    X_tr_s[kol_skalowane] = scaler.fit_transform(X_tr[kol_skalowane])
    X_te_s[kol_skalowane] = scaler.transform(X_te[kol_skalowane])

    knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
    tree = DecisionTreeClassifier(criterion='gini', max_depth=30,
                                  min_samples_leaf=10, min_samples_split=2,
                                  random_state=P.RANDOM_STATE)
    knn.fit(X_tr_s, y_tr)
    tree.fit(X_tr_s, y_tr)

    def ocena(model):
        pred = model.predict(X_te_s)
        proba = model.predict_proba(X_te_s)
        return {
            'accuracy': accuracy_score(y_te, pred),
            'precision_macro': precision_score(y_te, pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_te, pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_te, pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_te, pred, average='weighted', zero_division=0),
            'top2_accuracy': top_k_accuracy_score(y_te, proba, k=2, labels=model.classes_),
            'roc_auc': roc_auc_score(y_te, proba, average='macro', multi_class='ovr',
                                     labels=model.classes_),
        }

    metryki = {'kNN': ocena(knn), 'Drzewo decyzyjne': ocena(tree)}

    n_audio = len(kol_audio)
    ablation = {}
    for nazwa, cols in [('Tylko audio', slice(0, n_audio)),
                        ('Tylko tagi', slice(n_audio, None)),
                        ('Razem', slice(0, None))]:
        Xa_tr, Xa_te = X_tr_s.iloc[:, cols], X_te_s.iloc[:, cols]
        k = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance').fit(Xa_tr, y_tr)
        d = DecisionTreeClassifier(criterion='gini', max_depth=30, min_samples_leaf=10,
                                   random_state=P.RANDOM_STATE).fit(Xa_tr, y_tr)
        ablation[nazwa] = {
            'kNN': f1_score(y_te, k.predict(Xa_te), average='macro', zero_division=0),
            'Drzewo': f1_score(y_te, d.predict(Xa_te), average='macro', zero_division=0),
        }

    klasy = sorted(y_te.unique())
    f1_klasy = pd.DataFrame({
        'kNN': f1_score(y_te, knn.predict(X_te_s), average=None, labels=klasy, zero_division=0),
        'Drzewo': f1_score(y_te, tree.predict(X_te_s), average=None, labels=klasy, zero_division=0),
    }, index=klasy).sort_values('kNN', ascending=False)

    waznosci = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)

    cm = confusion_matrix(y_te, knn.predict(X_te_s), labels=knn.classes_, normalize='true')

    joblib.dump({
        'knn': knn, 'tree': tree, 'scaler': scaler, 'mlb': mlb,
        'feature_columns': X.columns.tolist(),
        'kol_skalowane': kol_skalowane,
        'kol_audio_num': P.CLF_AUDIO_NUM, 'kol_audio_dom': P.AUDIO_DOM,
        'klasy': knn.classes_.tolist(),
        'tagi_slownik': meta['tagi_slownik'],
        'metryki': metryki, 'ablation': ablation,
        'f1_per_klasa': f1_klasy, 'waznosci': waznosci,
        'confusion': cm, 'confusion_klasy': knn.classes_.tolist(),
        'srednie_audio': df_artist[P.CLF_AUDIO_NUM + ['popularity']].mean().to_dict(),
        'n_artystow': meta['n_artystow'], 'n_klas': len(meta['klasy']),
        'n_tagow': len(mlb.classes_), 'wymiar': X.shape[1],
        'n_train': len(X_tr), 'n_test': len(X_te),
    }, P.MODELS_DIR / 'classification.pkl')
    print(f"      kNN F1={metryki['kNN']['f1_macro']:.3f}  Drzewo F1={metryki['Drzewo decyzyjne']['f1_macro']:.3f}")


def train_clustering():
    print("[2/3] Grupowanie artystow ...")
    prof = P.build_clustering()
    feat = P.CLUSTER_FEATURES

    scaler = StandardScaler()
    X = scaler.fit_transform(prof[feat])

    db = DBSCAN(eps=2.5, min_samples=5).fit(X)
    keep = db.labels_ != -1
    X = X[keep]
    prof = prof[keep].reset_index(drop=True)

    kmeans = KMeans(n_clusters=P.CLUSTER_K, random_state=P.RANDOM_STATE, n_init=10)
    prof['cluster'] = kmeans.fit_predict(X)
    sil = silhouette_score(X, prof['cluster'])

    pca = PCA(n_components=2, random_state=P.RANDOM_STATE)
    coords = pca.fit_transform(X)
    prof['PC1'], prof['PC2'] = coords[:, 0], coords[:, 1]
    centers_pca = pca.transform(kmeans.cluster_centers_)

    cluster_means = prof.groupby('cluster')[feat].mean().round(3)
    ranges = prof[feat].agg(['min', 'max']).T

    df_tracks, _ = P.load_raw()
    idx_top = df_tracks.groupby('artist_main')['popularity'].idxmax()
    top_tracks = df_tracks.loc[idx_top, ['artist_main', 'track_name', 'popularity']]
    top_tracks = top_tracks.rename(columns={'track_name': 'top_track',
                                            'popularity': 'top_track_pop'})
    prof = prof.merge(top_tracks, on='artist_main', how='left')

    kol_profil = (['artist_main', 'main_genre', 'popularity_mean', 'cluster',
                   'PC1', 'PC2', 'top_track', 'top_track_pop'] + feat)
    kol_profil = list(dict.fromkeys(kol_profil))

    joblib.dump({
        'kmeans': kmeans, 'scaler': scaler, 'pca': pca,
        'feature_cols': feat,
        'profile': prof[kol_profil],
        'cluster_means': cluster_means,
        'centers_pca': centers_pca,
        'ranges': ranges,
        'silhouette': sil, 'K': P.CLUSTER_K,
        'sizes': prof['cluster'].value_counts().sort_index().to_dict(),
        'scaler_mean': dict(zip(feat, scaler.mean_)),
    }, P.MODELS_DIR / 'clustering.pkl')
    print(f"      K={P.CLUSTER_K}  Silhouette={sil:.3f}  n={len(prof)}")


def train_regression():
    print("[3/3] Regresja popularnosci ...")
    _, artist_clean = P.build_regression()
    feat = P.REG_FEATURES

    X = artist_clean[feat].values
    y = artist_clean['popularity'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=P.RANDOM_STATE)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    from sklearn.model_selection import cross_val_score
    X_all_s = StandardScaler().fit_transform(X)
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_a_ridge = max(alphas, key=lambda a: cross_val_score(
        Ridge(alpha=a), X_all_s, y, cv=5, scoring='r2').mean())
    best_a_poly = max(alphas, key=lambda a: cross_val_score(
        make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=a)),
        X_all_s, y, cv=5, scoring='r2').mean())

    modele = {
        'Baseline (srednia)': DummyRegressor(strategy='mean'),
        'Regresja liniowa (OLS)': LinearRegression(),
        f'Ridge (a={best_a_ridge})': Ridge(alpha=best_a_ridge),
        f'Wielomianowa st.2 + Ridge (a={best_a_poly})':
            make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=best_a_poly)),
    }
    wyniki = {}
    najlepszy_nazwa, najlepszy_model, najlepszy_r2 = None, None, -1e9
    for nazwa, m in modele.items():
        m.fit(X_tr_s, y_tr)
        pred = m.predict(X_te_s)
        r2 = r2_score(y_te, pred)
        wyniki[nazwa] = {
            'MAE': mean_absolute_error(y_te, pred),
            'RMSE': np.sqrt(mean_squared_error(y_te, pred)),
            'R2': r2,
        }
        if nazwa != 'Baseline (srednia)' and r2 > najlepszy_r2:
            najlepszy_r2, najlepszy_nazwa, najlepszy_model = r2, nazwa, m

    ols = LinearRegression().fit(X_tr_s, y_tr)
    coefs = pd.Series(ols.coef_, index=feat).sort_values()

    pred_best = najlepszy_model.predict(X_te_s)

    joblib.dump({
        'model': najlepszy_model, 'model_nazwa': najlepszy_nazwa,
        'scaler': scaler, 'features': feat,
        'wyniki': wyniki, 'coefs': coefs,
        'y_test': y_te, 'pred_test': pred_best,
        'srednie': artist_clean[feat].mean().to_dict(),
        'ranges': artist_clean[feat].agg(['min', 'max']).T,
        'n': len(artist_clean),
    }, P.MODELS_DIR / 'regression.pkl')
    print(f"      Najlepszy: {najlepszy_nazwa}  R2={najlepszy_r2:.3f}")


if __name__ == '__main__':
    train_classification()
    train_clustering()
    train_regression()
    print("\nGotowe. Artefakty w:", P.MODELS_DIR)
