import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px 
# -------------------------
# CARGA Y SIMULACIÓN DE DATOS
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
        G.add_node(s, tipo="supplier", actividad=row["actividad_economica"])
        if G.has_edge(p, s):
            G[p][s]["weight"] += monto
        else:
            G.add_edge(p, s, weight=monto)

    return G, relaciones


# -------------------------
# CARGA DE DATOS
# -------------------------
G, relaciones = cargar_datos()

# -------------------------
# SIDEBAR: FILTROS
# -------------------------
st.sidebar.header("🔎 Filtros")
actividades_opc = st.sidebar.multiselect("Actividad económica", sorted(relaciones["actividad_economica"].unique()))
portcos_opc = st.sidebar.multiselect("Portcos", sorted(relaciones["portco"].unique()))

relaciones_filtradas = relaciones.copy()
if actividades_opc:
    relaciones_filtradas = relaciones_filtradas[relaciones_filtradas["actividad_economica"].isin(actividades_opc)]
if portcos_opc:
    relaciones_filtradas = relaciones_filtradas[relaciones_filtradas["portco"].isin(portcos_opc)]

# -------------------------
# TOP 90% SUPPLIERS POR GASTO
# -------------------------
supplier_gasto = relaciones_filtradas.groupby("supplier")["monto_usd"].sum().sort_values(ascending=False).reset_index()
supplier_gasto["cumsum"] = supplier_gasto["monto_usd"].cumsum()
total_gasto = supplier_gasto["monto_usd"].sum()
supplier_gasto["cumsum_pct"] = supplier_gasto["cumsum"] / total_gasto
top_90_suppliers = supplier_gasto[supplier_gasto["cumsum_pct"] <= 0.9]["supplier"].tolist()
relaciones_top = relaciones_filtradas[relaciones_filtradas["supplier"].isin(top_90_suppliers)]

# -------------------------
# MAPA DE CALOR DE GASTOS POR ACTIVIDAD
# -------------------------
st.subheader("📊 Monto total por actividad económica")
actividad_gasto = relaciones_top.groupby("actividad_economica")["monto_usd"].sum().reset_index()

fig_heatmap = px.bar(
    actividad_gasto.sort_values("monto_usd"),
    x="monto_usd",
    y="actividad_economica",
    orientation="h",
    labels={"monto_usd": "Gasto Total (USD)", "actividad_economica": "Actividad"},
    title="Gasto Total por Actividad Económica"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# -------------------------
# GRAFO DE CONEXIONES
# -------------------------
st.subheader("🌐 Red de Conexiones Portcos-Suppliers")

# Crear subgrafo con los nodos filtrados
nodos_utiles = set(relaciones_top["portco"]) | set(relaciones_top["supplier"])
subG = G.subgraph(nodos_utiles)

# Posiciones para layout de grafo
pos = nx.spring_layout(subG, seed=42)

edge_x = []
edge_y = []
for edge in subG.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text = []
node_color = []

for node in subG.nodes(data=True):
    x, y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    tipo = node[1].get("tipo", "")
    color = "skyblue" if tipo == "portco" else "lightgreen"
    node_color.append(color)
    node_text.append(node[0])

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    marker=dict(color=node_color, size=10),
    text=node_text,
    textposition="top center"
)

fig_grafo = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          layout=go.Layout(
    title=dict(text='Red de Conexiones Portcos-Suppliers'),
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False)
)
))

st.plotly_chart(fig_grafo, use_container_width=True)
