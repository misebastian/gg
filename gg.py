# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. CARGA DE DATOS ===
@st.cache_data
def cargar_datos():
    df = pd.read_csv("gastos_portcos.csv")  # columnas: portco, supplier, gasto
    return df

df = cargar_datos()

# === 2. FILTROS ===
st.set_page_config(page_title="Dashboard de Compras Estratégicas", layout="wide")
st.title("💼 Dashboard de Conexiones Estratégicas entre Portcos y Suppliers")

col1, col2, col3 = st.columns(3)
with col1:
    min_gasto = st.slider("Filtrar mínimo gasto", 0, int(df['gasto'].max()), 10000)
with col2:
    portcos_seleccionados = st.multiselect("Portcos", df["portco"].unique())
with col3:
    suppliers_seleccionados = st.multiselect("Suppliers", df["supplier"].unique())

df_filtrado = df[df['gasto'] >= min_gasto]
if portcos_seleccionados:
    df_filtrado = df_filtrado[df_filtrado["portco"].isin(portcos_seleccionados)]
if suppliers_seleccionados:
    df_filtrado = df_filtrado[df_filtrado["supplier"].isin(suppliers_seleccionados)]

# === 3. KPIs ===
st.markdown("### 📊 Indicadores Clave")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Gasto Total", f"${df_filtrado['gasto'].sum():,.0f}")
kpi2.metric("Suppliers Compartidos", df_filtrado["supplier"].nunique())
kpi3.metric("Promedio Gasto por Portco", f"${df_filtrado.groupby('portco')['gasto'].sum().mean():,.0f}")

# === 4. MAPA DE CALOR ===
st.subheader("🔥 Mapa de Calor de Gasto entre Portcos y Suppliers")
pivot = df_filtrado.pivot_table(index="portco", columns="supplier", values="gasto", aggfunc='sum', fill_value=0)

fig_heatmap, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot, cmap="YlGnBu", linewidths=.5, ax=ax)
st.pyplot(fig_heatmap)

# === 5. GRAFO DE CONEXIONES ===
st.subheader("🧠 Mapa Interactivo de Conexiones Portcos-Suppliers")

G = nx.Graph()
for _, row in df_filtrado.iterrows():
    G.add_node(row['portco'], type='portco')
    G.add_node(row['supplier'], type='supplier')
    G.add_edge(row['portco'], row['supplier'], weight=row['gasto'])

pos = nx.spring_layout(G, seed=42, k=0.5)
edge_trace = go.Scatter(
    x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += [x0, x1, None]
    edge_trace['y'] += [y0, y1, None]

node_trace = go.Scatter(
    x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
    marker=dict(color=[], size=[], line_width=2))

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'].append(x)
    node_trace['y'].append(y)
    tipo = G.nodes[node]['type']
    color = 'lightblue' if tipo == 'portco' else 'lightgreen'
    size = 10 + np.log1p(df_filtrado[df_filtrado['portco'] == node]['gasto'].sum() + 1)
    node_trace['marker']['color'].append(color)
    node_trace['marker']['size'].append(size)
    node_trace['text'].append(str(node))

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title=dict(text='Red de Conexiones Portcos-Suppliers', font=dict(size=20)),
    showlegend=False, hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    plot_bgcolor='white',
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False))
st.plotly_chart(fig, use_container_width=True)

# === 6. TOP 10 ACCIONES ESTRATÉGICAS ===
st.subheader("🎯 Top 10 Acciones Estratégicas (Generadas Automáticamente)")

acciones = []
proveedores_top = df_filtrado.groupby('supplier')['gasto'].sum().sort_values(ascending=False).head(20)
for supplier in proveedores_top.index:
    portcos_usando = df_filtrado[df_filtrado["supplier"] == supplier]["portco"].unique()
    ahorro_estimado = df_filtrado[df_filtrado["supplier"] == supplier]["gasto"].sum() * 0.1
    acciones.append({
        "Acción Recomendable": f"Negociar compra conjunta de {supplier}",
        "Beneficio Estimado": f"${ahorro_estimado:,.0f} USD (10%)",
        "Portcos Involucrados": ", ".join(portcos_usando),
        "Nivel de Impacto": "Alto" if ahorro_estimado > 300000 else "Medio"
    })

df_acciones = pd.DataFrame(acciones).head(10)
st.dataframe(df_acciones)

# === 7. ESCENARIOS (SIMULACIÓN) ===
st.subheader("📈 Escenario de Ahorro Potencial")

selected_supplier = st.selectbox("Selecciona un Supplier para Simular", proveedores_top.index)
involucrados = df_filtrado[df_filtrado["supplier"] == selected_supplier]["portco"].unique()
total_gasto = df_filtrado[df_filtrado["supplier"] == selected_supplier]["gasto"].sum()
ahorro = total_gasto * 0.1

st.markdown(f"""
**Supplier:** `{selected_supplier}`  
**Portcos Involucrados:** {', '.join(involucrados)}  
**Gasto Total:** ${total_gasto:,.0f} USD  
**Ahorro Potencial (10%):** 🟢 ${ahorro:,.0f} USD
""")
