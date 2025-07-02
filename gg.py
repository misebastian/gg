import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import io
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering

# -------------------------
# DATOS SIMULADOS
# -------------------------
@st.cache_data
def cargar_datos():
    np.random.seed(42)
    actividades = ["Transporte", "Tecnología", "Papelería", "Servicios Generales", "Limpieza",
                   "Alimentos", "Mobiliario", "Consultoría", "Publicidad", "Energía",
                   "Seguridad", "Logística", "Marketing", "Contabilidad", "Legal"]

    portcos = [f"Portco {i}" for i in range(1, 101)]
    suppliers = pd.DataFrame({
        "supplier": [f"Supplier {i}" for i in range(1, 1001)],
        "actividad_economica": np.random.choice(actividades, 1000)
    })

    clientes_simulados = []
    for _ in range(3000):
        actividad = np.random.choice(actividades)
        portco = np.random.choice(portcos)
        clientes_simulados.append({
            "actividad_necesaria": actividad,
            "portco": portco
        })

    clientes = pd.DataFrame(clientes_simulados)
    relaciones = pd.merge(clientes, suppliers, left_on="actividad_necesaria", right_on="actividad_economica")
    relaciones["monto_usd"] = np.random.randint(1000, 50000, size=len(relaciones))

    G = nx.Graph()
    for _, row in relaciones.iterrows():
        p, s, monto = row["portco"], row["supplier"], row["monto_usd"]
        G.add_node(p, tipo="portco")
        G.add_node(s, tipo="supplier")
        if G.has_edge(p, s):
            G[p][s]["weight"] += monto
        else:
            G.add_edge(p, s, weight=monto)

    return G, relaciones

G, relaciones = cargar_datos()

# -------------------------
# FILTRAR SUPPLIERS DEL 90% GASTO
# -------------------------
supplier_gasto = relaciones.groupby("supplier")["monto_usd"].sum().sort_values(ascending=False).reset_index()
supplier_gasto["cumsum"] = supplier_gasto["monto_usd"].cumsum()
total_gasto = supplier_gasto["monto_usd"].sum()
supplier_gasto["cumsum_pct"] = supplier_gasto["cumsum"] / total_gasto
top_90_suppliers = supplier_gasto[supplier_gasto["cumsum_pct"] <= 0.9]["supplier"].tolist()

# Filtrar relaciones
relaciones_top = relaciones[relaciones["supplier"].isin(top_90_suppliers)]

# -------------------------
# AGRUPAR PORTCOS POR SUPPLIERS COMPARTIDOS
# -------------------------
def encontrar_clusters_por_supplier(relaciones):
    df = relaciones.groupby(["supplier", "portco"])["monto_usd"].sum().reset_index()
    matriz = pd.crosstab(df["supplier"], df["portco"])
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage='ward').fit(matriz.T)
    grupos = {}
    for i, cluster in enumerate(clustering.labels_):
        grupos.setdefault(cluster, []).append(matriz.columns[i])
    return [v for v in grupos.values() if len(v) > 1]

clusters_portcos = encontrar_clusters_por_supplier(relaciones_top)

# -------------------------
# GENERAR TOP 10 ACCIONES ESTRATÉGICAS
# -------------------------
def generar_acciones(relaciones, clusters):
    acciones = []
    for idx, grupo in enumerate(clusters):
        subset = relaciones[relaciones["portco"].isin(grupo)]
        supplier_comun = subset["supplier"].value_counts().idxmax()
        monto_total = subset[subset["supplier"] == supplier_comun]["monto_usd"].sum()
        impacto = "Alto" if monto_total > 700000 else "Medio" if monto_total > 300000 else "Bajo"
        acciones.append({
            "Acción Recomendada": f"Unificar compras entre {', '.join(grupo[:4])} con {supplier_comun}",
            "Beneficio Estimado": f"${monto_total:,.0f} USD estimado",
            "Nivel de Impacto": impacto
        })
    acciones_df = pd.DataFrame(acciones).sort_values(by="Nivel de Impacto", ascending=True).head(10)
    return acciones_df

top_acciones = generar_acciones(relaciones_top, clusters_portcos)

# -------------------------
# STREAMLIT DASHBOARD
# -------------------------
st.set_page_config(layout="wide")
st.title("📊 Dashboard Estratégico: Portcos & Suppliers")

col1, col2 = st.columns(2)
with col1:
    st.metric("🔢 Portcos activos", len({n for n, d in G.nodes(data=True) if d['tipo'] == 'portco'}))
    st.metric("🏢 Suppliers en top 90%", len(top_90_suppliers))
with col2:
    st.metric("💰 Gasto total analizado", f"${relaciones_top['monto_usd'].sum():,.0f}")
    st.metric("🔗 Acciones estratégicas generadas", len(top_acciones))

# Tabla de Acciones Estratégicas
st.markdown("## 🧩 Top 10 Acciones Estratégicas")
st.dataframe(top_acciones, use_container_width=True)

# Exportar
def exportar_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Acciones")
        writer.save()
    return output.getvalue()

st.download_button(
    label="📥 Descargar Acciones en Excel",
    data=exportar_excel(top_acciones),
    file_name="acciones_estrategicas.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
