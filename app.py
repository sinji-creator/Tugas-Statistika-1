import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.stats import binom, poisson, hypergeom, norm

st.set_page_config(page_title="Kalkulator Distribusi Probabilitas", layout="centered")

st.title("📊 Menghitung Distribusi Probabilitas")
st.write("Disertai langkah perhitungan matematis (substitusi angka)")

menu = st.selectbox(
    "Pilih Distribusi",
    ["Binomial", "Poisson", "Hipergeometrik", "Normal"]
)

tampilkan_langkah = st.checkbox("🔥 Tampilkan langkah perhitungan")

# ================= BINOMIAL =================
if menu == "Binomial":
    st.subheader("Distribusi Binomial")

    n = st.number_input("Jumlah percobaan (n)", min_value=1, value=10)
    x = st.number_input("Jumlah sukses (x)", min_value=0, max_value=n, value=3)
    p = st.number_input("Probabilitas sukses (p)", min_value=0.0, max_value=1.0, value=0.5)

    prob = binom.pmf(x, n, p)
    st.success(f"P(X = {x}) = {prob:.5f}")

    if tampilkan_langkah:
        kombinasi = factorial(n) / (factorial(x) * factorial(n - x))
        st.latex(r"P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}")
        st.latex(
            rf"P(X={x})=\binom{{{n}}}{{{x}}}({p})^{x}(1-{p})^{{{n-x}}}"
        )
        st.latex(
            rf"=\frac{{{n}!}}{{{x}!({n-x})!}} \times {p**x:.5f} \times {(1-p)**(n-x):.5f}"
        )
        st.latex(
            rf"={kombinasi:.0f} \times {p**x:.5f} \times {(1-p)**(n-x):.5f}"
        )
        st.latex(rf"={prob:.5f}")

    X = np.arange(0, n + 1)
    Y = binom.pmf(X, n, p)
    fig, ax = plt.subplots()
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

    if tampilkan_langkah:
        st.latex(r"P(X=x)=\frac{e^{-\lambda}\lambda^x}{x!}")
        st.latex(
            rf"P(X={x})=\frac{{e^{{-{lam}}}\cdot {lam}^{x}}}{{{x}!}}"
        )
        st.latex(
            rf"=\frac{{{np.exp(-lam):.5f}\cdot {lam**x:.0f}}}{{{factorial(x)}}}"
        )
        st.latex(rf"={prob:.5f}")

    X = np.arange(0, int(lam * 4))
    Y = poisson.pmf(X, lam)
    fig, ax = plt.subplots()
    ax.bar(X, Y)
    ax.set_title("Distribusi Poisson")
    st.pyplot(fig)

# ================= HIPERGEOMETRIK =================
elif menu == "Hipergeometrik":
    st.subheader("Distribusi Hipergeometrik")

    N = st.number_input("Total populasi (N)", min_value=1, value=50)
    K = st.number_input("Sukses dalam populasi (K)", min_value=0, max_value=N, value=20)
    n = st.number_input("Ukuran sampel (n)", min_value=1, max_value=N, value=10)
    x = st.number_input("Sukses dalam sampel (x)", min_value=0, max_value=n, value=3)

    prob = hypergeom.pmf(x, N, K, n)
    st.success(f"P(X = {x}) = {prob:.5f}")

    if tampilkan_langkah:
        st.latex(
            r"P(X=x)=\frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}"
        )
        st.latex(
            rf"P(X={x})=\frac{{\binom{{{K}}}{{{x}}}\binom{{{N-K}}}{{{n-x}}}}}{{\binom{{{N}}}{{{n}}}}}"
        )
        st.latex(rf"={prob:.5f}")

    X = np.arange(0, n + 1)
    Y = hypergeom.pmf(X, N, K, n)
    fig, ax = plt.subplots()
    ax.bar(X, Y)
    ax.set_title("Distribusi Hipergeometrik")
    st.pyplot(fig)

# ================= NORMAL =================
elif menu == "Normal":
    st.subheader("Distribusi Normal")

    mu = st.number_input("Rata-rata (μ)", value=70.0)
    sigma = st.number_input("Standar deviasi (σ)", min_value=0.1, value=5.0)

    jenis = st.selectbox(
        "Jenis probabilitas",
        ["P(X ≤ a)", "P(X ≥ b)", "P(a ≤ X ≤ b)"]
    )

    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y_vals = norm.pdf(x_vals, mu, sigma)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals)

    if jenis == "P(X ≤ a)":
        a = st.number_input("Nilai a", value=mu)
        z = (a - mu) / sigma
        prob = norm.cdf(a, mu, sigma)

        if tampilkan_langkah:
            st.latex(r"Z=\frac{a-\mu}{\sigma}")
            st.latex(rf"Z=\frac{{{a}-{mu}}}{{{sigma}}}={z:.3f}")
            st.latex(rf"P(X\le {a})=P(Z\le {z:.3f})={prob:.5f}")

        ax.fill_between(x_vals, y_vals, where=(x_vals <= a), alpha=0.4)

    elif jenis == "P(X ≥ b)":
        b = st.number_input("Nilai b", value=mu)
        z = (b - mu) / sigma
        prob = 1 - norm.cdf(b, mu, sigma)

        if tampilkan_langkah:
            st.latex(rf"P(X\ge {b})=1-P(X\le {b})")
            st.latex(rf"Z=\frac{{{b}-{mu}}}{{{sigma}}}={z:.3f}")
            st.latex(rf"P(X\ge {b})={prob:.5f}")

        ax.fill_between(x_vals, y_vals, where=(x_vals >= b), alpha=0.4)

    else:
        a = st.number_input("Nilai a", value=mu - sigma)
        b = st.number_input("Nilai b", value=mu + sigma)
        z1 = (a - mu) / sigma
        z2 = (b - mu) / sigma
        prob = norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma)

        if tampilkan_langkah:
            st.latex(r"P(a\le X\le b)=P(X\le b)-P(X\le a)")
            st.latex(rf"Z_1=\frac{{{a}-{mu}}}{{{sigma}}}={z1:.3f}")
            st.latex(rf"Z_2=\frac{{{b}-{mu}}}{{{sigma}}}={z2:.3f}")
            st.latex(rf"P({a}\le X\le {b})={prob:.5f}")

        ax.fill_between(x_vals, y_vals, where=(x_vals >= a) & (x_vals <= b), alpha=0.4)

    ax.set_title("Distribusi Normal")
    st.pyplot(fig)
