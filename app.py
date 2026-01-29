import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, hypergeom, norm
import math

st.set_page_config(page_title="Distribusi Probabilitas", layout="centered")
st.title("📘 Distribusi Probabilitas")
st.write("Binomial, Poisson, Hipergeometrik, dan Normal")

menu = st.selectbox(
    "Pilih Distribusi",
    ["Binomial", "Poisson", "Hipergeometrik", "Normal"]
)

# =========================================================
# BINOMIAL
# =========================================================
if menu == "Binomial":
    st.header("Distribusi Binomial")

    n = st.number_input("Jumlah percobaan (n)", min_value=1, value=10)
    x = st.number_input("Jumlah sukses (x)", min_value=0, max_value=n, value=3)
    p = st.number_input("Probabilitas sukses (p)", min_value=0.0, max_value=1.0, value=0.5)

    prob = binom.pmf(x, n, p)

    st.subheader("📐 Rumus")
    st.latex(r"P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}")

    st.subheader("✏️ Penjabaran")
    st.latex(fr"P(X={x})=\binom{{{n}}}{{{x}}}({p})^{x}(1-{p})^{{{n-x}}}")
    st.latex(fr"\binom{{{n}}}{{{x}}}=\frac{{{n}!}}{{{x}!({n-x})!}}")
    st.latex(fr"P(X={x})={math.comb(n,x)}\times {p**x:.5f}\times {(1-p)**(n-x):.5f}")
    st.latex(fr"P(X={x})={prob:.6f}")

    st.success(f"Hasil akhir: P(X={x}) = {prob:.5f}")

    X = np.arange(0, n + 1)
    Y = binom.pmf(X, n, p)
    st.subheader("📊 Grafik Distribusi Binomial")
    fig, ax = plt.subplots()
    ax.bar(X, Y)
    ax.set_title("Distribusi Binomial")
    ax.grid(True)
    st.pyplot(fig)

# =========================================================
# POISSON
# =========================================================
elif menu == "Poisson":
    st.header("Distribusi Poisson")

    lam = st.number_input("Rata-rata kejadian (λ)", min_value=0.1, value=3.0)
    x = st.number_input("Jumlah kejadian (x)", min_value=0, value=2)

    prob = poisson.pmf(x, lam)

    st.subheader("📐 Rumus")
    st.latex(r"P(X=x)=\frac{e^{-\lambda}\lambda^x}{x!}")

    st.subheader("✏️ Penjabaran")
    st.latex(fr"P(X={x})=\frac{{e^{{-{lam}}}{lam}^{x}}}{{{x}!}}")
    st.latex(fr"e^{{-{lam}}}={math.exp(-lam):.6f}")
    st.latex(fr"{lam}^{x}={lam**x}")
    st.latex(fr"{x}!={math.factorial(x)}")
    st.latex(fr"P(X={x})=\frac{{{math.exp(-lam):.6f}\times{lam**x}}}{{{math.factorial(x)}}}")
    st.latex(fr"P(X={x})={prob:.6f}")

    st.success(f"Hasil akhir: P(X={x}) = {prob:.5f}")

    X = np.arange(0, int(lam*5)+1)
    Y = poisson.pmf(X, lam)
    st.subheader("📊 Grafik Distribusi Poisson")
    fig, ax = plt.subplots()
    ax.bar(X, Y)
    ax.set_title("Distribusi Poisson")
    ax.grid(True)
    st.pyplot(fig)

# =========================================================
# HIPERGEOMETRIK
# =========================================================
elif menu == "Hipergeometrik":
    st.header("Distribusi Hipergeometrik")

    N = st.number_input("Total populasi (N)", min_value=1, value=50)
    K = st.number_input("Jumlah sukses populasi (K)", min_value=0, max_value=N, value=20)
    n = st.number_input("Ukuran sampel (n)", min_value=1, max_value=N, value=10)
    x = st.number_input("Sukses dalam sampel (x)", min_value=0, max_value=n, value=3)

    prob = hypergeom.pmf(x, N, K, n)

    st.subheader("📐 Rumus")
    st.latex(r"P(X=x)=\frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}")

    st.subheader("✏️ Penjabaran")
    st.latex(fr"P(X={x})=\frac{{\binom{{{K}}}{{{x}}}\binom{{{N-K}}}{{{n-x}}}}}{{\binom{{{N}}}{{{n}}}}}")
    st.latex(fr"=\frac{{{math.comb(K,x)}\times{math.comb(N-K,n-x)}}}{{{math.comb(N,n)}}}")
    st.latex(fr"P(X={x})={prob:.6f}")

    st.success(f"Hasil akhir: P(X={x}) = {prob:.5f}")

    X = np.arange(0, n + 1)
    Y = hypergeom.pmf(X, N, K, n)
    st.subheader("📊 Grafik Distribusi Hipergeometrik")
    fig, ax = plt.subplots()
    ax.bar(X, Y)
    ax.set_title("Distribusi Hipergeometrik")
    ax.grid(True)
    st.pyplot(fig)

# =========================================================
# NORMAL
# =========================================================
elif menu == "Normal":
    st.header("Distribusi Normal")

    mu = st.number_input("Rata-rata (μ)", value=50.0)
    sigma = st.number_input("Standar deviasi (σ)", min_value=0.1, value=10.0)
    x = st.number_input("Nilai X", value=60.0)

    z = (x - mu) / sigma
    prob = norm.cdf(z)

    st.subheader("📐 Rumus")
    st.latex(r"Z=\frac{X-\mu}{\sigma}")

    st.subheader("✏️ Penjabaran")
    st.latex(fr"Z=\frac{{{x}-{mu}}}{{{sigma}}}")
    st.latex(fr"Z={z:.2f}")
    st.latex(fr"P(X\le {x})=P(Z\le {z:.2f})")
    st.latex(fr"P(X\le {x})={prob:.6f}")

    st.success(f"Hasil akhir: P(X ≤ {x}) = {prob:.5f}")

    X = np.linspace(mu-4*sigma, mu+4*sigma, 1000)
    Y = norm.pdf(X, mu, sigma)

    st.subheader("📊 Grafik Distribusi Normal")
    fig, ax = plt.subplots()
    ax.plot(X, Y)
    ax.axvline(x, linestyle="--")
    ax.fill_between(X, Y, where=(X <= x), alpha=0.4)
    ax.set_title("Distribusi Normal")
    ax.grid(True)
    st.pyplot(fig)
