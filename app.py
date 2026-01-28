import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, hypergeom, norm

st.set_page_config(page_title="Kalkulator Distribusi Probabilitas", layout="centered")

st.title("📊 Distribusi Probabilitas")
st.write("Binomial, Poisson, Hipergeometrik, dan Normal")

menu = st.selectbox(
    "Pilih Distribusi",
    ["Binomial", "Poisson", "Hipergeometrik", "Normal"]
)

# ================= BINOMIAL =================
if menu == "Binomial":
    st.subheader("Distribusi Binomial")

    n = st.number_input("Jumlah percobaan (n)", min_value=1, value=10)
    x = st.number_input("Jumlah sukses (x)", min_value=0, max_value=n, value=3)
    p = st.number_input("Probabilitas sukses (p)", min_value=0.0, max_value=1.0, value=0.5)

    prob = binom.pmf(x, n, p)
    st.success(f"P(X = {x}) = {prob:.5f}")

    fig, ax = plt.subplots()
    X = np.arange(0, n + 1)
    Y = binom.pmf(X, n, p)
    ax.bar(X, Y)
    ax.set_title("Distribusi Binomial")
    st.pyplot(fig)

# ================= POISSON =================
elif menu == "Poisson":
    st.subheader("Distribusi Poisson")

    lam = st.number_input("Rata-rata kejadian (λ)", min_value=0.1, value=4.0)
    x = st.number_input("Jumlah kejadian (x)", min_value=0, value=3)

    prob = poisson.pmf(x, lam)
    st.success(f"P(X = {x}) = {prob:.5f}")

    fig, ax = plt.subplots()
    X = np.arange(0, int(lam * 4))
    Y = poisson.pmf(X, lam)
    ax.bar(X, Y)
    ax.set_title("Distribusi Poisson")
    st.pyplot(fig)

# ================= HIPERGEOMETRIK =================
elif menu == "Hipergeometrik":
    st.subheader("Distribusi Hipergeometrik")

    N = st.number_input("Total populasi (N)", min_value=1, value=50)
    K = st.number_input("Jumlah sukses populasi (K)", min_value=0, max_value=N, value=20)
    n = st.number_input("Ukuran sampel (n)", min_value=1, max_value=N, value=10)
    x = st.number_input("Sukses dalam sampel (x)", min_value=0, max_value=n, value=3)

    prob = hypergeom.pmf(x, N, K, n)
    st.success(f"P(X = {x}) = {prob:.5f}")

    fig, ax = plt.subplots()
    X = np.arange(0, n + 1)
    Y = hypergeom.pmf(X, N, K, n)
    ax.bar(X, Y)
    ax.set_title("Distribusi Hipergeometrik")
    st.pyplot(fig)

# ================= NORMAL =================
elif menu == "Normal":
    st.subheader("Distribusi Normal")

    mu = st.number_input("Rata-rata (μ)", value=0.0)
    sigma = st.number_input("Standar deviasi (σ)", min_value=0.1, value=1.0)

    jenis = st.selectbox(
        "Pilih jenis probabilitas",
        ["P(X ≤ a)", "P(X ≥ b)", "P(a ≤ X ≤ b)"]
    )

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = norm.pdf(x, mu, sigma)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    if jenis == "P(X ≤ a)":
        a = st.number_input("Nilai a", value=mu)
        prob = norm.cdf(a, mu, sigma)
        ax.fill_between(x, y, where=(x <= a), alpha=0.4)
        st.success(f"P(X ≤ {a}) = {prob:.5f}")

    elif jenis == "P(X ≥ b)":
        b = st.number_input("Nilai b", value=mu)
        prob = 1 - norm.cdf(b, mu, sigma)
        ax.fill_between(x, y, where=(x >= b), alpha=0.4)
        st.success(f"P(X ≥ {b}) = {prob:.5f}")

    else:
        a = st.number_input("Nilai a", value=mu - sigma)
        b = st.number_input("Nilai b", value=mu + sigma)
        prob = norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma)
        ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.4)
        st.success(f"P({a} ≤ X ≤ {b}) = {prob:.5f}")

    ax.set_title("Distribusi Normal")
    st.pyplot(fig)
