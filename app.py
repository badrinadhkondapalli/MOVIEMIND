"""
MovieMind - A Linear-Algebra Based Movie Recommendation System
Maths for AI & DS (UE25MA241B) mini project. NO model training: only matrices,
dot products, norms, SVD, eigen-decomposition and least squares.

Run:  streamlit run app.py
Data: tmdb_5000_movies.csv + tmdb_5000_credits.csv (Kaggle) in the same folder.
"""
import html, json, os, re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import sparse
from scipy.linalg import lu
from scipy.sparse.linalg import svds

st.set_page_config(page_title="MovieMind", page_icon="🎬", layout="wide")

BLOCKS = ["overview", "genres", "keywords", "cast", "director"]
DEFAULT_W = {"overview": 1.0, "genres": 1.5, "keywords": 1.5, "cast": 0.7, "director": 0.7}
K_SVD = 60
STOP = set("""the and for with that this from into his her their they them who which was were are been have has had
not but out over after before when while about than then also its one two him she you your will would can could
more most only other some such what where there these those being him get got just like make made new""".split())

CSS = """<style>
.stApp{background:radial-gradient(circle at 15% 0%,#1b1038 0%,#0b0b14 55%)}
section[data-testid=stSidebar]{background:#0f0f1a;border-right:1px solid #2a2a40}
.hero{padding:2rem 2.2rem;border-radius:20px;background:linear-gradient(135deg,#7c3aed 0%,#ec4899 60%,#f59e0b 100%);color:#fff;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.6rem;color:#fff}.hero p{margin:.4rem 0 0;font-size:1.05rem;opacity:.95}
.card{background:#141422cc;border:1px solid #2e2e48;border-radius:16px;padding:1rem 1.2rem;margin-bottom:1rem;position:relative;transition:.2s}
.card:hover{border-color:#a855f7;transform:translateY(-2px)}
.rank{position:absolute;top:10px;right:14px;font-weight:800;color:#a855f7}
.ctitle{font-size:1.12rem;font-weight:700;padding-right:2rem}.yr{color:#9ca3af;font-weight:400}
.chip{display:inline-block;background:#2a1f4d;color:#c4b5fd;border-radius:999px;padding:2px 10px;font-size:.72rem;margin:4px 4px 4px 0}
.ov{color:#cbd5e1;font-size:.86rem;margin:.4rem 0}.meta{color:#9ca3af;font-size:.8rem}
.bar{height:6px;background:#2a2a40;border-radius:9px;margin-top:.6rem}
.bar div{height:6px;border-radius:9px;background:linear-gradient(90deg,#7c3aed,#ec4899)}
.sc{font-size:.76rem;color:#c4b5fd;margin-top:.3rem;font-family:monospace}
.pipe{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.5rem 0 1rem}
.step{background:#1c1c30;border:1px solid #3b3b5c;border-radius:12px;padding:.5rem .9rem;font-size:.85rem}
.arrow{color:#a855f7;font-weight:700}
</style>"""


# ───────────────────────── 1. DATA  →  FEATURE MATRIX ─────────────────────────
def jl(s, key="name", n=None):
    try:
        out = [d[key] for d in json.loads(s)]
    except Exception:
        out = []
    return out[:n] if n else out


def director_of(s):
    try:
        return next((d["name"] for d in json.loads(s) if d.get("job") == "Director"), "")
    except Exception:
        return ""


def tag(x):
    return re.sub(r"[^a-z0-9]", "", x.lower())


def words(text):
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP]


@st.cache_data(show_spinner="Loading dataset…")
def load_df():
    m = pd.read_csv("tmdb_5000_movies.csv")
    c = pd.read_csv("tmdb_5000_credits.csv").rename(columns={"movie_id": "id"})[["id", "cast", "crew"]]
    df = m.merge(c, on="id").dropna(subset=["overview"]).drop_duplicates("title").reset_index(drop=True)
    df["year"] = pd.to_datetime(df.release_date, errors="coerce").dt.year
    df["genre_list"] = df.genres.apply(jl)
    df["kw_list"] = df.keywords.apply(jl)
    df["cast_list"] = df.cast.apply(lambda s: jl(s, n=4))
    df["director"] = df.crew.apply(director_of)
    df["main_genre"] = df.genre_list.apply(lambda g: g[0] if g else "Unknown")
    return df.drop(columns=["cast", "crew", "genres", "keywords"])


def tfidf_block(docs, min_df=2, max_f=4000):
    """Hand-made TF-IDF: rows = movies, cols = terms, every row scaled to unit length."""
    N = len(docs)
    dfc = Counter()
    for d in docs:
        dfc.update(set(d))
    vocab = [t for t, c in dfc.most_common(max_f) if c >= min_df]
    idx = {t: i for i, t in enumerate(vocab)}
    r, c, v = [], [], []
    for i, d in enumerate(docs):
        for t, cnt in Counter(d).items():
            if t in idx:
                r.append(i); c.append(idx[t]); v.append(1 + np.log(cnt))
    X = sparse.csr_matrix((v, (r, c)), shape=(N, len(vocab)), dtype=np.float64)
    idf = np.log((1 + N) / (1 + np.array([dfc[t] for t in vocab]))) + 1
    X = sparse.csr_matrix(X.multiply(idf))
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sparse.diags(1 / np.maximum(nrm, 1e-12)) @ X
    return sparse.csr_matrix(X), vocab


@st.cache_resource(show_spinner="Building feature matrices, SVD and PCA…")
def build():
    df = load_df()
    docs = {
        "overview": [words(o) for o in df.overview],
        "genres": [[tag(g) for g in L] for L in df.genre_list],
        "keywords": [[w for k in L for w in words(k)] for L in df.kw_list],
        "cast": [[tag(a) for a in L] for L in df.cast_list],
        "director": [[tag(d)] if d else [] for d in df.director],
    }
    cfg = {"overview": (2, 5000), "genres": (1, 50), "keywords": (2, 4000), "cast": (2, 3000), "director": (2, 1500)}
    B, V = {}, {}
    for k in BLOCKS:
        B[k], V[k] = tfidf_block(docs[k], *cfg[k])
    # Full feature matrix A  (N x D): blocks side by side, scaled by sqrt(weight)
    A = sparse.hstack([B[k] * float(np.sqrt(DEFAULT_W[k])) for k in BLOCKS]).tocsr()
    # Truncated SVD  A ≈ U Σ Vᵀ  (rank K_SVD)
    U, s, Vt = svds(A, k=K_SVD, v0=np.random.RandomState(0).rand(min(A.shape)))
    o = np.argsort(-s)
    U, s, Vt = U[:, o], s[o], Vt[o]
    Z = U * s                                            # latent coordinates of each movie
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    # PCA on latent space: covariance matrix + eigen-decomposition
    Zc = Z - Z.mean(0)
    C = Zc.T @ Zc / (len(Z) - 1)
    ev, EV = np.linalg.eigh(C)                           # symmetric  ->  eigh
    ev, EV = ev[::-1], EV[:, ::-1]
    return dict(df=df, B=B, V=V, A=A, U=U, s=s, Vt=Vt, Z=Z, Zn=Zn, C=C, ev=ev, EV=EV, P2=Zc @ EV[:, :2],
                fro2=float(A.multiply(A).sum()), pos={t: i for i, t in enumerate(df.title)},
                vocab_all=[f"{k}: {t}" for k in BLOCKS for t in V[k]])


# ───────────────────────── 2. SIMILARITY  =  COSINE (dot product / norms) ─────────────────────────
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def scores(S, idx, mode, w):
    """Cosine similarity of every movie to the 'taste vector' (mean of the chosen movies)."""
    N = len(S["df"])
    if mode == "latent":
        return S["Zn"] @ unit(S["Zn"][idx].mean(0)), {}
    parts, total = {}, np.zeros(N)
    for k in BLOCKS:
        q = np.asarray(S["B"][k][idx].mean(axis=0)).ravel()
        nq = np.linalg.norm(q)
        parts[k] = S["B"][k] @ (q / nq) if nq > 1e-12 else np.zeros(N)   # matrix–vector product
        total += w[k] * parts[k]
    wsum = sum(w.values()) or 1
    return total / wsum, {k: w[k] * v / wsum for k, v in parts.items()}


def recommend(S, idx, mode, w, n, flt):
    sc, parts = scores(S, idx, mode, w)
    df = S["df"]
    mask = ((df.vote_count >= flt["votes"]) & df.year.between(*flt["years"])).to_numpy().copy()
    if flt["genres"]:
        mask &= df.genre_list.apply(lambda g: any(x in g for x in flt["genres"])).to_numpy()
    mask[idx] = False
    cand = np.where(mask)[0]
    return cand[np.argsort(-sc[cand])[:n]], sc, parts


def angle(c):
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def card(r, score, rank):
    chips = "".join(f"<span class='chip'>{html.escape(g)}</span>" for g in r.genre_list[:3])
    yr = int(r.year) if pd.notna(r.year) else "—"
    ov = html.escape(r.overview[:170]) + "…"
    return (f"<div class='card'><div class='rank'>#{rank}</div><div class='ctitle'>{html.escape(r.title)} "
            f"<span class='yr'>({yr})</span></div>{chips}<div class='ov'>{ov}</div>"
            f"<div class='meta'>⭐ {r.vote_average:.1f} &nbsp;·&nbsp; 🎬 {html.escape(r.director or 'N/A')}</div>"
            f"<div class='bar'><div style='width:{max(score, 0) * 100:.0f}%'></div></div>"
            f"<div class='sc'>cos θ = {score:.3f} &nbsp;|&nbsp; θ = {angle(score):.1f}°</div></div>")


def style(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def gram_schmidt(V):
    Q = []
    for v in V:
        w = v - sum((v @ q) * q for q in Q)
        if np.linalg.norm(w) > 1e-9:
            Q.append(w / np.linalg.norm(w))
    return np.array(Q)


# ───────────────────────── 3. PAGES ─────────────────────────
def page_home(S, c):
    A, df = S["A"], S["df"]
    st.markdown("<div class='hero'><h1>🎬 MovieMind</h1><p>A movie recommender built purely on linear algebra — "
                "vectors, dot products, SVD and eigen-decomposition. <b>Zero model training.</b></p></div>",
                unsafe_allow_html=True)
    a, b, d, e = st.columns(4)
    a.metric("Movies (rows)", f"{A.shape[0]:,}")
    b.metric("Features (columns)", f"{A.shape[1]:,}")
    d.metric("Matrix sparsity", f"{100 * (1 - A.nnz / (A.shape[0] * A.shape[1])):.2f}% zeros")
    e.metric(f"Energy kept by rank-{K_SVD} SVD", f"{100 * (S['s'] ** 2).sum() / S['fro2']:.1f}%")
    st.subheader("Pipeline")
    steps = ["Movie dataset", "Text → tokens", "TF-IDF feature matrix A", "Row vectors", "Dot products & norms",
             "Cosine similarity", "Rank & filter", "Top-N picks"]
    st.markdown("<div class='pipe'>" + "<span class='arrow'>→</span>".join(f"<span class='step'>{x}</span>" for x in steps)
                + "</div>", unsafe_allow_html=True)
    st.subheader("Syllabus concepts used")
    st.dataframe(pd.DataFrame({
        "Concept (Unit)": ["Matrices & vectors (U1)", "Rank, subspaces, LU (U1)", "Dot product, norm, orthogonality (U2)",
                           "Projection (U2)", "Gram–Schmidt (U2)", "Least squares (U2)", "Covariance, eigenvalues (U2)", "SVD & PCA (U2)"],
        "Where in MovieMind": ["Feature matrix A (movies × terms)", "Lab → Rank & LU tabs", "Cosine similarity engine",
                               "Math page demo; taste vector", "Lab → Gram–Schmidt tab", "Lab → predict ratings",
                               "Movie Map: covariance matrix + eigh", "Latent space & 2-D Movie Map"]}),
        hide_index=True, use_container_width=True)
    st.info("👈 Pick movies in the sidebar, then open **Recommender**.")


def page_reco(S, c):
    df = S["df"]
    st.title("🎯 Recommendations")
    if not c["idx"]:
        st.info("Select at least one movie in the sidebar.")
        return
    top, sc, parts = recommend(S, c["idx"], c["mode"], c["w"], c["n"], c["flt"])
    st.caption("Because you picked: " + ", ".join(f"**{t}**" for t in c["sel"]))
    cols = st.columns(2)
    for r, i in enumerate(top):
        with cols[r % 2]:
            st.markdown(card(df.iloc[i], sc[i], r + 1), unsafe_allow_html=True)
    if len(top) == 0:
        st.warning("No movies match the filters.")
        return
    g1, g2 = st.columns(2)
    names = [df.title[i] for i in top]
    with g1:
        f = px.bar(x=sc[top][::-1], y=names[::-1], orientation="h", title="Cosine similarity scores",
                   color=sc[top][::-1], color_continuous_scale="Purples", labels={"x": "cos θ", "y": ""})
        st.plotly_chart(style(f, 60 + 38 * len(top)).update_coloraxes(showscale=False), use_container_width=True)
    with g2:
        if parts:
            rows = [{"movie": df.title[i], "block": k.title(), "contribution": parts[k][i]} for i in top for k in BLOCKS]
            f = px.bar(pd.DataFrame(rows), x="contribution", y="movie", color="block", orientation="h",
                       title="Why? Dot-product contribution per feature block")
            st.plotly_chart(style(f, 60 + 38 * len(top)).update_yaxes(autorange="reversed"), use_container_width=True)
        else:
            st.info("Latent (SVD) mode mixes all blocks into 60 hidden dimensions, so per-block contributions "
                    "aren't shown. Switch to *Full TF-IDF space* to see them.")


def page_math(S, c):
    df, A = S["df"], S["A"]
    st.title("🧮 The Mathematics Behind It")
    t1, t2 = st.tabs(["Theory", "Live demo: two movies"])
    with t1:
        st.markdown("**1. Feature matrix** — each movie is a row vector in ℝᴰ (TF-IDF weights, Unit 1 matrices).")
        st.latex(r"A\in\mathbb{R}^{N\times D},\qquad A_{ij}=(1+\ln tf_{ij})\cdot\Big(\ln\tfrac{1+N}{1+df_j}+1\Big)")
        st.markdown("**2. Dot product & norm** (Unit 2)")
        st.latex(r"a\cdot b=\sum_j a_jb_j,\qquad \|a\|=\sqrt{a\cdot a}")
        st.markdown("**3. Cosine similarity** — the angle between two movie vectors; `0` ⇒ orthogonal (nothing in common).")
        st.latex(r"\cos\theta=\frac{a\cdot b}{\|a\|\,\|b\|}\in[0,1]")
        st.markdown("**4. Whole library in ONE matrix product** — with unit rows, similarity of all movies to a taste vector *q*:")
        st.latex(r"s=\hat A\,\hat q,\qquad q=\tfrac1m\sum_{i\in\text{liked}}\hat a_i")
        st.markdown("**5. Weighted blocks** — concatenating blocks scaled by √wₖ gives a weighted sum of block dot products:")
        st.latex(r"a\cdot b=\sum_k w_k\,(a_k\cdot b_k)")
        st.markdown("**6. Projection** — how much of *a* lies along *b* (Unit 2):")
        st.latex(r"\operatorname{proj}_b a=\frac{a\cdot b}{b\cdot b}\,b,\qquad (a-\operatorname{proj}_b a)\perp b")
        st.markdown("**7. SVD** — compresses A into K hidden 'taste' dimensions:")
        st.latex(r"A\approx U_K\Sigma_K V_K^{T},\qquad Z=U_K\Sigma_K")
        st.markdown("**8. PCA** — eigen-decomposition of the covariance matrix (symmetric, positive semi-definite):")
        st.latex(r"C=\tfrac{1}{N-1}Z_c^{T}Z_c=Q\Lambda Q^{T},\qquad \text{2-D map}=Z_cQ_{[:,1:2]}")
    with t2:
        titles = list(df.title)
        d0 = c["sel"][0] if c["sel"] else titles[0]
        x, y = st.columns(2)
        ta = x.selectbox("Movie a", titles, index=titles.index(d0))
        tb = y.selectbox("Movie b", titles, index=min(1, len(titles) - 1) if ta != titles[1] else 2)
        ia, ib = S["pos"][ta], S["pos"][tb]
        a, b = A[ia].toarray().ravel(), A[ib].toarray().ravel()
        dot, na, nb = a @ b, np.linalg.norm(a), np.linalg.norm(b)
        cs = dot / (na * nb)
        k = (a @ b) / (b @ b)
        p = k * b
        m = st.columns(5)
        m[0].metric("a · b", f"{dot:.4f}"); m[1].metric("‖a‖", f"{na:.3f}"); m[2].metric("‖b‖", f"{nb:.3f}")
        m[3].metric("cos θ", f"{cs:.4f}"); m[4].metric("θ", f"{angle(cs):.1f}°")
        st.markdown(f"**Projection:** proj_b(a) = **{k:.4f}**·b, length {np.linalg.norm(p):.4f}. "
                    f"Check (a − proj)·b = `{(a - p) @ b:.2e}` ≈ 0 ✔ (residual is orthogonal to b).")
        if cs < 1e-9:
            st.warning("a·b = 0 → these two vectors are **orthogonal**: no shared features.")
        prod = A[ia].multiply(A[ib]).tocoo()
        if prod.nnz:
            tab = pd.DataFrame({"shared feature": [S["vocab_all"][j] for j in prod.col], "aⱼ·bⱼ": prod.data})
            st.markdown("**Terms that make up the dot product** (largest first):")
            st.dataframe(tab.sort_values("aⱼ·bⱼ", ascending=False).head(15), hide_index=True, use_container_width=True)


def page_map(S, c):
    df, P = S["df"], S["P2"]
    st.title("🗺️ Movie Map (PCA)")
    st.caption("Latent vectors → covariance matrix → eigenvectors. Movies close together on the map are similar.")
    top = [g for g, _ in Counter(df.main_genre).most_common(8)]
    d = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "title": df.title,
                      "genre": df.main_genre.where(df.main_genre.isin(top), "Other")})
    f = px.scatter(d, x="PC1", y="PC2", color="genre", hover_name="title", opacity=.65)
    if c["idx"]:
        f.add_trace(go.Scatter(x=P[c["idx"], 0], y=P[c["idx"], 1], mode="markers+text", text=c["sel"],
                               textposition="top center", name="your picks",
                               marker=dict(symbol="star", size=18, color="#fbbf24", line=dict(width=1, color="white"))))
    st.plotly_chart(style(f, 560), use_container_width=True)
    ev = S["ev"]
    r = ev / ev.sum()
    g = go.Figure([go.Bar(x=list(range(1, 16)), y=r[:15], name="variance ratio"),
                   go.Scatter(x=list(range(1, 16)), y=np.cumsum(r)[:15], name="cumulative")])
    st.plotly_chart(style(g.update_layout(title="Scree plot — eigenvalues of the covariance matrix"), 350), use_container_width=True)
    st.success(f"PC1 + PC2 explain {100 * r[:2].sum():.1f}% of latent variance. "
               f"Smallest eigenvalue = {ev.min():.2e} ≥ 0 → covariance matrix is positive semi-definite ✔")


def page_lab(S, c):
    df, Z = S["df"], S["Z"]
    st.title("🔬 Linear Algebra Lab")
    sel = c["idx"] or [0, 1, 2]
    top, _, _ = recommend(S, sel, c["mode"], c["w"], 8 - len(sel) if len(sel) < 8 else 1, c["flt"])
    ids = (sel + list(top))[:8]
    nd = 6
    M = Z[ids, :nd]
    st.caption(f"M = first {nd} latent coordinates of {len(ids)} movies (your picks + their top matches)")
    lab = [df.title[i][:22] for i in ids]
    tabs = st.tabs(["Similarity matrix", "Rank & subspaces", "LU", "Gram–Schmidt", "Least squares"])
    with tabs[0]:
        Zn = S["Zn"][ids]
        G = Zn @ Zn.T
        st.plotly_chart(style(px.imshow(G, x=lab, y=lab, color_continuous_scale="Purples", zmin=0, zmax=1, text_auto=".2f",
                                        title="Gram matrix G = Ẑ Ẑᵀ (cosine of every pair)"), 520), use_container_width=True)
        st.write(f"G is symmetric: **{np.allclose(G, G.T)}** · smallest eigenvalue {np.linalg.eigvalsh(G).min():.2e} (≥ 0 ⇒ PSD)")
    with tabs[1]:
        r = np.linalg.matrix_rank(M)
        m, n = M.shape
        st.dataframe(pd.DataFrame(M, index=lab).round(3), use_container_width=True)
        st.table(pd.DataFrame({"Subspace": ["Column space C(M)", "Row space C(Mᵀ)", "Null space N(M)", "Left null space N(Mᵀ)"],
                               "Lives in": [f"ℝ^{m}", f"ℝ^{n}", f"ℝ^{n}", f"ℝ^{m}"],
                               "Dimension": [r, r, n - r, m - r]}))
        st.write(f"rank(M) = **{r}**; singular values: {np.round(np.linalg.svd(M, compute_uv=False), 3)}")
    with tabs[2]:
        P, L, U = lu(M)
        st.write("Gaussian elimination as a factorisation:  **M = P · L · U**")
        a, b = st.columns(2)
        a.write("L (lower triangular, multipliers)"); a.dataframe(pd.DataFrame(L).round(3))
        b.write("U (upper triangular, echelon form)"); b.dataframe(pd.DataFrame(U).round(3))
        st.write(f"P·L·U = M ? **{np.allclose(P @ L @ U, M)}**")
    with tabs[3]:
        Q = gram_schmidt(M)
        st.write(f"Orthonormal basis of the row space of M from {len(ids)} movie vectors → **{len(Q)}** orthonormal vectors.")
        st.dataframe(pd.DataFrame(Q).round(3), use_container_width=True)
        st.write(f"Q Qᵀ = I ? **{np.allclose(Q @ Q.T, np.eye(len(Q)))}**")
    with tabs[4]:
        st.write("Predict each movie's rating from its latent coordinates by solving the **normal equations** "
                 "(XᵀX)x̂ = Xᵀy — one closed-form solve, no iterative training.")
        k = st.slider("Latent dimensions used", 2, K_SVD, 20)
        X = np.hstack([np.ones((len(Z), 1)), Z[:, :k]])
        y = df.vote_average.to_numpy()
        xh = np.linalg.solve(X.T @ X, X.T @ y)
        yh = X @ xh
        r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        st.write(f"R² = **{r2:.3f}** · residual ⟂ column space: ‖Xᵀ(y − Xx̂)‖ = `{np.linalg.norm(X.T @ (y - yh)):.2e}` ≈ 0 ✔ · "
                 f"matches np.linalg.lstsq: **{np.allclose(xh, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-5)}**")
        f = px.scatter(x=y, y=yh, opacity=.4, labels={"x": "actual rating", "y": "projected rating"},
                       title="y vs projection of y onto the column space")
        f.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(color="#ec4899", dash="dash"))
        st.plotly_chart(style(f, 420), use_container_width=True)


def page_stats(S, c):
    df = S["df"]
    st.title("📈 Dataset Statistics")
    a, b, d = st.columns(3)
    a.metric("Movies", f"{len(df):,}"); b.metric("Avg rating", f"{df.vote_average.mean():.2f}")
    d.metric("Years", f"{int(df.year.min())} – {int(df.year.max())}")
    g = pd.Series(Counter(x for L in df.genre_list for x in L)).sort_values()
    x, y = st.columns(2)
    x.plotly_chart(style(px.bar(g, orientation="h", title="Movies per genre").update_layout(showlegend=False), 480), use_container_width=True)
    y.plotly_chart(style(px.histogram(df, x="year", nbins=50, title="Releases by year"), 480), use_container_width=True)
    x, y = st.columns(2)
    x.plotly_chart(style(px.histogram(df[df.vote_count > 50], x="vote_average", nbins=30, title="Rating distribution (>50 votes)")), use_container_width=True)
    y.plotly_chart(style(px.bar(df.nlargest(10, "popularity").iloc[::-1], x="popularity", y="title", orientation="h", title="Top 10 by popularity")), use_container_width=True)


def page_about(S, c):
    st.title("ℹ️ About & Methodology")
    st.markdown("""
**MovieMind** recommends movies *content-based*: it never learns from ratings or trains a network. Every number comes from a matrix operation.

1. **Preprocess** – overview, genres, keywords, top-4 cast and director are turned into tokens.
2. **Vectorise** – each field becomes a TF-IDF block; blocks sit side by side to form the feature matrix **A** (N × D).
3. **Compare** – cosine similarity = dot product ÷ product of norms. For several liked movies, we average their unit vectors into a *taste vector*.
4. **Rank** – sort scores, apply filters, show top-N.
5. **Compress (optional)** – truncated SVD gives a 60-D latent space that captures hidden themes; PCA of that space draws the Movie Map.

**Limitations:** purely content-based (no user ratings), English text only, popularity is not considered.
**Data:** TMDB 5000 Movie Dataset (Kaggle).
""")


# ───────────────────────── 4. MAIN ─────────────────────────
PAGES = {"🏠 Home": page_home, "🎯 Recommender": page_reco, "🧮 Math Behind It": page_math,
         "🗺️ Movie Map": page_map, "🔬 Linear Algebra Lab": page_lab, "📈 Dataset Stats": page_stats, "ℹ️ About": page_about}


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    if not all(os.path.exists(f) for f in ["tmdb_5000_movies.csv", "tmdb_5000_credits.csv"]):
        st.error("Dataset not found. Download **tmdb_5000_movies.csv** and **tmdb_5000_credits.csv** from Kaggle "
                 "(“TMDB 5000 Movie Dataset”) and place them next to app.py.")
        st.stop()
    S = build()
    df = S["df"]
    with st.sidebar:
        st.markdown("## 🎬 MovieMind")
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        st.divider()
        default = ["Interstellar"] if "Interstellar" in S["pos"] else [df.title[0]]
        sel = st.multiselect("🎞️ Movies you like (1–5)", list(df.title), default=default, max_selections=5)
        mode_label = st.radio("Similarity space", ["Full TF-IDF space", f"SVD latent space (rank {K_SVD})"])
        n = st.slider("Number of recommendations", 3, 20, 10)
        with st.expander("⚖️ Feature weights (TF-IDF mode)"):
            w = {k: st.slider(k.title(), 0.0, 3.0, DEFAULT_W[k], 0.1) for k in BLOCKS}
        with st.expander("🔧 Filters"):
            genres = st.multiselect("Genres", sorted({g for L in df.genre_list for g in L}))
            votes = st.slider("Min. vote count", 0, 3000, 50, 10)
            years = st.slider("Release year", int(df.year.min()), int(df.year.max()), (int(df.year.min()), int(df.year.max())))
    ctx = dict(sel=sel, idx=[S["pos"][t] for t in sel], mode="tfidf" if mode_label.startswith("Full") else "latent",
               w=w, n=n, flt=dict(genres=genres, votes=votes, years=years))
    PAGES[page](S, ctx)


if __name__ == "__main__":
    main()
