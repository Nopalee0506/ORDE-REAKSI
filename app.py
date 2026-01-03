import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# ----- Page config -----
st.set_page_config(page_title="Penentuan Orde Reaksi", layout="centered")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = None

# ----- Sidebar Navigation (format yang kamu minta) -----
page = st.sidebar.selectbox(
    "📄 Go to Page",
    ("Dashboard", "Upload Data", "Kinetics Chatbot", "Settings")
)

# Sidebar: global small settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Global Settings")
default_orders = st.sidebar.multiselect(
    "Default orders to analyze",
    ["Orde 0", "Orde 1", "Orde 2"],
    default=["Orde 1", "Orde 2"]
)
show_graph_by_default = st.sidebar.checkbox("Show graph by default", value=True)


# ----- Helper functions -----
def safe_read_csv(uploaded_file):
    # Try common delimiters
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(uploaded_file, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            uploaded_file.seek(0)
            continue
    # fallback
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, engine="python")


def regresi(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    sx, sy = np.sum(x), np.sum(y)
    sxy = np.sum(x * y)
    sx2 = np.sum(x * x)
    denom = (n * sx2 - sx * sx)
    if denom == 0:
        m = np.nan
    else:
        m = (n * sxy - sx * sy) / denom
    # intercept
    b = (sy - m * sx) / n if not np.isnan(m) else np.nan
    y_pred = m * x + b
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return float(m), float(r2)


def analyze_dataframe(df, orders_to_compute=None):
    if orders_to_compute is None:
        orders_to_compute = ["Orde 1", "Orde 2"]
    # assume first two columns are t and A, allow user to rename earlier
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    A = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    mask = ~np.isnan(t) & ~np.isnan(A)
    t = t[mask]
    A = A[mask]
    if len(t) < 2:
        raise ValueError("Data terlalu sedikit setelah pembersihan (butuh minimal 2 titik).")
    if (A <= 0).any():
        raise ValueError("Nilai absorbansi harus semuanya > 0.")
    results = []
    if "Orde 0" in orders_to_compute:
        k0, r20 = regresi(t, A)
        results.append(("Orde 0", k0, r20))
    if "Orde 1" in orders_to_compute:
        y1 = np.log(A[0] / A)  # ln(A0/A)
        k1, r21 = regresi(t, y1)
        results.append(("Orde 1", k1, r21))
    if "Orde 2" in orders_to_compute:
        y2 = (1 / A - 1 / A[0])
        k2, r22 = regresi(t, y2)
        results.append(("Orde 2", k2, r22))
    return results, t, A


def kinetics_bot(question, df):
    if df is None:
        return "Silakan upload data reaksi pada halaman 'Upload Data' terlebih dahulu."
    try:
        results, t, A = analyze_dataframe(df, orders_to_compute=["Orde 0", "Orde 1", "Orde 2"])
    except Exception as e:
        return f"Terjadi error saat menganalisis data: {e}"

    q = question.lower()
    if "orde terbaik" in q or "best order" in q:
        df_res = pd.DataFrame(results, columns=["Orde", "k", "R2"])
        df_res = df_res.dropna(subset=["R2"])
        if df_res.empty:
            return "Tidak dapat menentukan orde terbaik (R² tidak tersedia)."
        best = df_res.sort_values("R2", ascending=False).iloc[0]
        return f"Orde terbaik menurut R² adalah {best['Orde']} (k = {best['k']:.4f}, R² = {best['R2']:.4f})."
    if "k orde 1" in q or "k1" in q:
        for o, k, r in results:
            if o == "Orde 1":
                return f"k (Orde 1) = {k:.6f}, R² = {r:.4f}"
        return "Data untuk Orde 1 tidak tersedia."
    if "plot" in q or "grafik" in q:
        return "Gunakan halaman Dashboard untuk melihat grafik kinetika."
    return "Maaf, saya belum memahami pertanyaan itu. Coba tanyakan 'orde terbaik' atau 'k orde 1'."


# ----- Page: Dashboard -----
if page == "Dashboard":
    st.title("📊 Kinetika — Dashboard Analisis")
    if st.session_state.data is None:
        st.info("Silakan upload data reaksi pada halaman 'Upload Data' terlebih dahulu.")
    else:
        df = st.session_state.data
        st.subheader("Data Ringkasan")
        st.write(f"Jumlah titik data: {len(df)}")
        st.dataframe(df.head(10))

        # Orders selection (allow override of sidebar default)
        orders = st.multiselect("Pilih orde yang ingin dianalisis (filter)", ["Orde 0", "Orde 1", "Orde 2"], default=default_orders)
        show_graph = st.checkbox("Tampilkan grafik kinetika", value=show_graph_by_default)

        try:
            results, t, A = analyze_dataframe(df, orders_to_compute=orders)
        except Exception as e:
            st.error(f"Analisis gagal: {e}")
            st.stop()

        # Show results table
        df_res = pd.DataFrame(results, columns=["Orde", "k", "R²"])
        st.subheader("Hasil Regresi")
        st.dataframe(df_res)

        # Best order
        df_res_valid = df_res.dropna(subset=["R²"])
        if not df_res_valid.empty:
            best = df_res_valid.sort_values("R²", ascending=False).iloc[0]
            st.success(f"📌 Orde reaksi terbaik: **{best['Orde']}** (k = {best['k']:.6f}, R² = {best['R²']:.4f})")
        else:
            st.warning("Tidak ada nilai R² valid untuk menentukan orde terbaik.")

        # Plot interactive with Plotly
        if show_graph:
            st.subheader("Grafik Transformasi Kinetika")
            fig = go.Figure()
            # raw A vs t (Orde 0)
            if "Orde 0" in orders:
                fig.add_trace(go.Scatter(x=t, y=A, mode="markers+lines", name="Orde 0: A(t)"))
            if "Orde 1" in orders:
                fig.add_trace(go.Scatter(x=t, y=np.log(A[0] / A), mode="markers+lines", name="Orde 1: ln(A0/A)"))
            if "Orde 2" in orders:
                fig.add_trace(go.Scatter(x=t, y=(1 / A - 1 / A[0]), mode="markers+lines", name="Orde 2: 1/A - 1/A0"))

            fig.update_layout(
                xaxis_title="Waktu (t)",
                yaxis_title="Transformasi Kinetika",
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

# ----- Page: Upload Data -----
elif page == "Upload Data":
    st.title("📁 Upload Data Kinetika")
    st.markdown("Format file: CSV dengan minimal 2 kolom (kolom pertama = waktu, kolom kedua = absorbansi).")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = safe_read_csv(uploaded_file)
            # If there are more than 2 columns, ask user to pick
            if df.shape[1] > 2:
                st.info("Terdeteksi lebih dari 2 kolom. Pilih kolom untuk waktu dan absorbansi:")
                col_time = st.selectbox("Kolom waktu (t)", df.columns, index=0)
                col_A = st.selectbox("Kolom absorbansi (A)", df.columns, index=1)
                df = df[[col_time, col_A]].copy()
                df.columns = ["t", "A"]
            else:
                df.columns = ["t", "A"]
            # convert types
            df["t"] = pd.to_numeric(df["t"], errors="coerce")
            df["A"] = pd.to_numeric(df["A"], errors="coerce")
            if df["A"].isnull().any() or df["t"].isnull().any():
                st.warning("Terdapat nilai non-numerik yang akan diabaikan saat analisis.")
            st.dataframe(df.head(20))
            # store in session
            st.session_state.data = df.dropna().reset_index(drop=True)
            st.success("Data berhasil diupload dan disimpan pada session.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

# ----- Page: Kinetics Chatbot -----
elif page == "Kinetics Chatbot":
    st.title("💬 Kinetics Chatbot")
    st.chat_message("assistant").write("Hi! Saya KineticsBot. Tanyakan hal seperti 'orde terbaik' atau 'k orde 1'.")
    if prompt := st.chat_input("Tulis pertanyaan Anda..."):
        st.chat_message("user").write(prompt)
        reply = kinetics_bot(prompt, st.session_state.data)
        st.chat_message("assistant").write(reply)

# ----- Page: Settings -----
elif page == "Settings":
    st.title("⚙️ Settings")
    st.markdown("Pengaturan aplikasi dan preferensi.")
    # Let user tune defaults (persist to session)
    default_orders_new = st.multiselect(
        "Default orders to analyze (sidebar default)",
        ["Orde 0", "Orde 1", "Orde 2"],
        default=default_orders
    )
    show_graph_def_new = st.checkbox("Show graph by default", value=show_graph_by_default)
    if st.button("Simpan preferensi"):
        # we cannot directly modify outer-scope variables, but we can save to session_state
        st.session_state.ui_defaults = {
            "default_orders": default_orders_new,
            "show_graph": show_graph_def_new
        }
        st.success("Preferensi disimpan (session).")

    st.markdown("---")
    if st.button("Reset uploaded data (session)"):
        st.session_state.data = None
        st.success("Data dihapus dari session.")
