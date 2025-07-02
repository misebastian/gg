import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go

# Cargar los datos del grafo G y relaciones
@st.cache_data
def cargar_datos():
    # Simulación (reemplaza con tu dataset real)
    actividades = ["Transporte", "Tecnología", "Papelería", "Servicios Generales", "Limpieza",
                   "Alimentos", "Mobiliario", "Consultoría", "Publicidad", "Energía",
                   "Seguridad", "Logística", "Marketing", "Contabilidad", "Legal"]

    portcos = [f"Portco {i}" for i in range(1, 101)]
    suppliers = pd.DataFrame({
        "supplier": [f"Supplier {i}" for i in range(1, 1001)],
        "actividad_economica": np.random.choice(actividades, 1000)
    })

    clientes_simulados = []
    for i in range(3000):
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

# Función para graficar un portco
def graficar_interactivo_plotly(nombre_nodo, top_n=20):
    vecinos = list(G[nombre_nodo].items())
    vecinos_ordenados = sorted(vecinos, key=lambda x: x[1]['weight'], reverse=True)[:top_n]
    subnodos = [nombre_nodo] + [v[0] for v in vecinos_ordenados]
    subgrafo = G.subgraph(subnodos)
    pos = nx.kamada_kawai_layout(subgrafo)

    node_x, node_y, node_text, node_color, node_size, hover_texts = [], [], [], [], [], []

    pesos = [G[nombre_nodo][n[0]]["weight"] for n in vecinos_ordenados]
    max_peso = max(pesos)
    min_peso = min(pesos)

    for node in subgrafo.nodes():
        x, y = pos[node]
        tipo = subgrafo.nodes[node]["tipo"]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if node == nombre_nodo:
            node_color.append("deepskyblue")
            node_size.append(40)
            hover_texts.append(f"<b>{node}</b> (Central)")
        else:
            peso = G[nombre_nodo][node]["weight"]
            size = 20 + 40 * ((peso - min_peso) / (max_peso - min_peso + 1e-6))
            node_color.append("tomato" if tipo == "supplier" else "lightsteelblue")
            node_size.append(size)
            hover_texts.append(f"<b>{node}</b><br>Tipo: {tipo}<br>Monto: ${peso:,.0f}")

    edge_x, edge_y = [], []
    for edge in subgrafo.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'),
                            hoverinfo='none', mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hovertext=hover_texts,
        hoverinfo="text",
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='black')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"<b>Top {top_n} conexiones de {nombre_nodo}</b>",
                        titlefont_size=20,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        plot_bgcolor='lightsteelblue',
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🔗 Análisis de Relaciones entre Portcos y Suppliers")
portcos = [n for n, d in G.nodes(data=True) if d["tipo"] == "portco"]
seleccionado = st.selectbox("Selecciona un Portco:", sorted(portcos))

fig = graficar_interactivo_plotly(seleccionado)
st.plotly_chart(fig, use_container_width=True)

# --- Panel de insights ---
vecinos = list(G[seleccionado].items())
suppliers_conectados = [(n, v["weight"]) for n, v in vecinos if G.nodes[n]["tipo"] == "supplier"]
df_suppliers = pd.DataFrame(suppliers_conectados, columns=["supplier", "monto_usd"]).sort_values(by="monto_usd", ascending=False)

actividades = relaciones[relaciones["portco"] == seleccionado]["actividad_necesaria"].value_counts().head(5)

st.markdown("### 📌 Insights para el equipo de Deals")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total de Suppliers Conectados", len(df_suppliers))
    st.metric("Monto Total", f"${df_suppliers['monto_usd'].sum():,.0f}")
with col2:
    st.markdown("**Top 5 actividades económicas del Portco:**")
    st.dataframe(actividades.rename_axis("Actividad").reset_index(name="Frecuencia"))

st.markdown("**🔝 Ranking de Suppliers por monto compartido:**")
st.dataframe(df_suppliers.head(10))

