# Mini-app de datos · Vuelos: ¿qué hace que un vuelo llegue tarde?
# Ejecutar en local:  streamlit run app.py
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Vuelos · Correlaciones", page_icon="✈️", layout="wide")


@st.cache_data
def cargar_datos():
    return pd.read_csv("vuelos_limpio.csv.gz")


df = cargar_datos()

VARIABLES = ["retraso_llegada", "retraso_salida", "hora_programada", "mes", "distancia_km", "edad_avion",
             "asientos", "vuelos_misma_hora", "temp_c", "humedad", "viento_kmh", "lluvia_mm", "visibilidad_km"]

# ---------- Barra lateral: filtros ----------
st.sidebar.header("Filtros")
aerolineas = st.sidebar.multiselect("Aerolínea", sorted(df["aerolinea"].unique()), placeholder="Todas")
origenes = st.sidebar.multiselect("Aeropuerto de salida", sorted(df["origen"].unique()), placeholder="Todos")
meses = st.sidebar.slider("Meses", 1, 12, (1, 12))
metodo = st.sidebar.radio("Método de correlación", ["pearson", "spearman"])

datos = df[df["mes"].between(meses[0], meses[1])]
if aerolineas:
    datos = datos[datos["aerolinea"].isin(aerolineas)]
if origenes:
    datos = datos[datos["origen"].isin(origenes)]

st.title("✈️ ¿Qué hace que un vuelo llegue tarde?")
st.caption("Vuelos que salieron de Nueva York (JFK, LaGuardia y Newark) en 2013. Deje los filtros vacíos para ver todo.")

if len(datos) < 30:
    st.warning("Hay muy pocos vuelos con estos filtros. Amplíe la selección.")
    st.stop()

# ---------- Indicadores ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vuelos", f"{len(datos):,}".replace(",", "."))
c2.metric("Llegan tarde (15 min o más)", f"{datos['llego_tarde'].mean() * 100:.1f} %")
c3.metric("Retraso promedio", f"{datos['retraso_llegada'].mean():.1f} min")
r = datos["hora_programada"].corr(datos["retraso_llegada"], method=metodo)
c4.metric("Correlación hora–retraso", f"{r:.2f}")

# ---------- Pestañas ----------
tab1, tab2, tab3 = st.tabs(["Matriz de correlación", "Hora y clima", "Explorar una variable"])

with tab1:
    corr = datos[VARIABLES].corr(method=metodo).round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Matriz de correlación ({metodo})")
    fig.update_layout(height=700)
    st.plotly_chart(fig)

with tab2:
    izq, der = st.columns(2)
    franjas = datos.groupby("franja_horaria").agg(vuelos=("llego_tarde", "count"),
                                                  pct_tarde=("llego_tarde", "mean")).reset_index()
    franjas["pct_tarde"] = franjas["pct_tarde"] * 100
    fig = px.bar(franjas, x="franja_horaria", y="pct_tarde", text_auto=".1f", hover_data=["vuelos"],
                 color_discrete_sequence=["#2F5597"], labels={"franja_horaria": "", "pct_tarde": "% que llega tarde"},
                 title="% de vuelos tarde según la hora de salida")
    izq.plotly_chart(fig)

    lluvia = datos.groupby("llueve").agg(vuelos=("llego_tarde", "count"),
                                         pct_tarde=("llego_tarde", "mean")).reset_index()
    lluvia["condicion"] = lluvia["llueve"].map({0: "Sin lluvia", 1: "Con lluvia"})
    lluvia["pct_tarde"] = lluvia["pct_tarde"] * 100
    fig = px.bar(lluvia, x="condicion", y="pct_tarde", text_auto=".1f", hover_data=["vuelos"],
                 color_discrete_sequence=["#12366B"], labels={"condicion": "", "pct_tarde": "% que llega tarde"},
                 title="% de vuelos tarde con y sin lluvia")
    der.plotly_chart(fig)
    st.info("Una correlación baja no significa ausencia de relación: el % de vuelos tarde cambia mucho con la hora y la lluvia.")

with tab3:
    x = st.selectbox("Variable a comparar", [v for v in VARIABLES if v not in ("retraso_llegada", "retraso_salida")])
    tramos = pd.qcut(datos[x], q=5, duplicates="drop")
    if len(tramos.cat.categories) < 3:  # variables con muchos valores repetidos (lluvia, visibilidad)
        tramos = pd.cut(datos[x], bins=5)
    resumen = datos.groupby(tramos, observed=True).agg(vuelos=("llego_tarde", "count"),
                                                       pct_tarde=("llego_tarde", "mean")).reset_index()
    resumen[x] = resumen[x].astype(str)
    resumen["pct_tarde"] = resumen["pct_tarde"] * 100
    fig = px.bar(resumen, x=x, y="pct_tarde", text_auto=".1f", hover_data=["vuelos"],
                 color_discrete_sequence=["#4472C4"], labels={x: f"{x} (tramos)", "pct_tarde": "% que llega tarde"},
                 title=f"% de vuelos tarde según {x}")
    st.plotly_chart(fig)

with st.expander("Ver los datos filtrados"):
    st.dataframe(datos.head(500))
    st.download_button("Descargar CSV filtrado", datos.to_csv(index=False).encode("utf-8"),
                       file_name="vuelos_filtrado.csv", mime="text/csv")
