import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import gudhi as gd
import gudhi.clustering.tomato as tomato
import gudhi.representations  # <--- ADD THIS LINE HERE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
import networkx as nx
from scipy.stats import zscore
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TDA River Analysis", layout="wide", page_icon="🌊")

# --- CSS FOR BETTER TABS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Topological Data Analysis: Water Level Analytics")

# --- SIDEBAR: GLOBAL CONFIGURATION ---
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Station Data (CSV)", type=['csv'])

# Initialize Session State
if 'tau' not in st.session_state: st.session_state['tau'] = 1
if 'm' not in st.session_state: st.session_state['m'] = 3
if 'df_subset' not in st.session_state: st.session_state['df_subset'] = None
if 'df_embedded' not in st.session_state: st.session_state['df_embedded'] = None
if 'point_cloud' not in st.session_state: st.session_state['point_cloud'] = None
if 'l2_results' not in st.session_state: st.session_state['l2_results'] = None

# --- HELPER FUNCTIONS ---
def save_plot_to_bytes(fig, format='png'):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "1. Load Data", "2. Param Est (τ/m)", "3. Point Cloud", "4. Persistence",
    "5. Clustering", "6. State ID", "7. Transitions", "8. Infographic",
    "9. L2 Norm", "10. Export Features"
])

# ==========================================
# TAB 1: LOAD DATA
# ==========================================
with tab1:
    st.header("Step 1: Data Loading & Time Series Inspection")
    
    if uploaded_file is not None:
        # Load Data
        df = pd.read_csv(uploaded_file)
        
        # Simple column handling
        if len(df.columns) >= 3:
            df.columns = ['Date_Original', 'Raw', 'Level']
        else:
            st.warning("CSV format unclear. Taking first column as Date, last as Level.")
            df = df.iloc[:, [0, -1]]
            df.columns = ['Date_Original', 'Level']
            
        df['Date'] = pd.to_datetime(df['Date_Original'], dayfirst=True)
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Flood Event Range")
            flood_start = st.date_input("Flood Start", pd.to_datetime("2023-01-04"))
            flood_end = st.date_input("Flood End", pd.to_datetime("2023-01-10"))
        with col2:
            st.subheader("Analysis Window Params")
            days_before = st.number_input("Days Before", value=750)
            days_after = st.number_input("Days After", value=720)
            
        # Filter
        flood_start_dt = pd.to_datetime(flood_start)
        flood_end_dt = pd.to_datetime(flood_end)
        start_win = flood_start_dt - pd.Timedelta(days=days_before)
        end_win = flood_end_dt + pd.Timedelta(days=days_after)
        
        df_subset = df[(df['Date'] >= start_win) & (df['Date'] <= end_win)].copy()
        df_subset = df_subset.sort_values('Date').reset_index(drop=True)
        
        # Save to session
        st.session_state['df_subset'] = df_subset
        st.session_state['flood_dates'] = (flood_start_dt, flood_end_dt)
        
        st.success(f"Data Loaded: {len(df_subset)} rows ({start_win.date()} to {end_win.date()})")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_subset['Date'], df_subset['Level'], label='Water Level', color='blue')
        ax.axvspan(flood_start_dt, flood_end_dt, color='red', alpha=0.3, label='Flood Event')
        ax.set_title("Water Level Time Series (Analysis Window)")
        ax.legend()
        st.pyplot(fig)
        
    else:
        st.info("Please upload a CSV file in the Sidebar.")

# ==========================================
# TAB 2: PARAMETER ESTIMATION (PERSISTENT)
# ==========================================
with tab2:
    st.header("Step 2: Estimate Optimal τ (Delay) and m (Dimension)")
    
    # 1. Logic to Run Calculation
    if st.session_state.get('df_subset') is not None:
        if st.button("Run Parameter Estimation", key="btn_run_param"):
            with st.spinner("Calculating AMI and FNN..."):
                series = st.session_state['df_subset']['Level'].values
                
                # --- AMI Calculation ---
                amis = []
                n_bins = int(np.sqrt(len(series)))
                for delay in range(1, 101):
                    s1, s2 = series[:-delay], series[delay:]
                    c_xy = np.histogram2d(s1, s2, bins=n_bins)[0]
                    amis.append(mutual_info_score(None, None, contingency=c_xy))
                amis = np.array(amis)

                # Find Tau
                tau = np.argmin(amis) + 1
                for i in range(1, len(amis)-1):
                    if amis[i] < amis[i-1] and amis[i] < amis[i+1]:
                        tau = i + 1; break
                
                # --- FNN Calculation ---
                fnn_fractions = []
                R_tol, sigma = 15, np.std(series)
                for m_dim in range(1, 11):
                    n_vectors = len(series) - m_dim * tau
                    if n_vectors < 1: break
                    vectors_m = np.array([series[i:i + m_dim*tau:tau] for i in range(n_vectors)])
                    nbrs = NearestNeighbors(n_neighbors=2).fit(vectors_m)
                    distances, indices = nbrs.kneighbors(vectors_m)
                    
                    false_neighbors = 0
                    for i in range(len(vectors_m)):
                        nn_idx = indices[i, 1]
                        if i + m_dim*tau >= len(series) or nn_idx + m_dim*tau >= len(series): continue
                        val_i_next, val_nn_next = series[i + m_dim*tau], series[nn_idx + m_dim*tau]
                        dist_m = distances[i, 1]
                        if dist_m > 0:
                            if abs(val_i_next - val_nn_next) / dist_m > R_tol: false_neighbors += 1
                        elif abs(val_i_next - val_nn_next) > sigma * 0.01: false_neighbors += 1
                    
                    fnn_fractions.append(false_neighbors / len(vectors_m) if len(vectors_m) > 0 else 0)

                # Find m
                m = 1
                for i, frac in enumerate(fnn_fractions):
                    if frac < 0.10: m = i + 1; break
                
                # SAVE RESULTS TO SESSION STATE
                st.session_state['tau'] = tau
                st.session_state['m'] = m
                st.session_state['amis_data'] = amis
                st.session_state['fnn_data'] = fnn_fractions
                st.success(f"Calculated: τ = {tau}, m = {m}")

        # 2. Logic to Display Results (Runs if data exists)
        if 'amis_data' in st.session_state:
            tau = st.session_state['tau']
            m = st.session_state['m']
            
            st.info(f"Current Parameters: τ = {tau}, m = {m}")
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # AMI Plot
            ax[0].plot(range(1, len(st.session_state['amis_data'])+1), st.session_state['amis_data'])
            ax[0].axvline(tau, color='r', linestyle='--', label=f'Tau={tau}')
            ax[0].set_title('Average Mutual Information')
            ax[0].legend()
            
            # FNN Plot
            ax[1].plot(range(1, len(st.session_state['fnn_data'])+1), st.session_state['fnn_data'], marker='o')
            ax[1].axvline(m, color='r', linestyle='--', label=f'm={m}')
            ax[1].set_title('False Nearest Neighbors')
            ax[1].legend()
            
            st.pyplot(fig)
            
    else:
        st.warning("Please Load Data in Tab 1 first.")

# ==========================================
# TAB 3: POINT CLOUD GENERATION (PERSISTENT)
# ==========================================
with tab3:
    st.header("Step 3: Time Delay Embedding")
    
    col1, col2 = st.columns(2)
    # Allows user to override the calculated values
    tau_input = col1.number_input("Tau (Time Delay)", value=st.session_state.get('tau', 1), key="tau_input_tab3")
    m_input = col2.number_input("m (Dimension)", value=st.session_state.get('m', 3), key="m_input_tab3")
    
    # 1. Run Calculation
    if st.button("Generate Embedding", key="btn_gen_embed"):
        if st.session_state.get('df_subset') is not None:
            series = st.session_state['df_subset']['Level'].values
            N = len(series)
            
            embedding, indices = [], []
            for i in range(N - (m_input - 1) * tau_input):
                vec = [series[i + k * tau_input] for k in range(m_input)]
                embedding.append(vec)
                indices.append(i + (m_input - 1) * tau_input)
                
            point_cloud = np.array(embedding)
            df_embedded = st.session_state['df_subset'].iloc[indices].copy()
            
            # Save to Session
            st.session_state['point_cloud'] = point_cloud
            st.session_state['df_embedded'] = df_embedded
            st.session_state['tau'] = tau_input
            st.session_state['m'] = m_input
            st.rerun() # Force reload to show results immediately

    # 2. Show Results (Persistent)
    if st.session_state.get('point_cloud') is not None:
        pc = st.session_state['point_cloud']
        df_emb = st.session_state['df_embedded']
        current_tau = st.session_state['tau']
        
        st.write(f"✅ Point Cloud Generated. Shape: {pc.shape}")
        
        # 3D Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use only first 3 dimensions for visualization
        sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2] if pc.shape[1] > 2 else np.zeros(len(pc)), 
                    c=df_emb['Level'], cmap='viridis', s=20, alpha=0.7)
        
        ax.set_title(f"3D Embedding (Tau={current_tau}, m={st.session_state['m']})")
        
        # --- ADDED AXIS LABELS HERE ---
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t + {current_tau})")
        ax.set_zlabel(f"x(t + {2*current_tau})")
        # ------------------------------

        plt.colorbar(sc, label="Water Level")
        st.pyplot(fig)
        
        st.download_button("Download 3D Plot", save_plot_to_bytes(fig), "3d_embedding.png", "image/png", key="dl_3d")

# ==========================================
# TAB 4: PERSISTENCE HOMOLOGY (PERSISTENT & DOWNLOADABLE)
# ==========================================
with tab4:
    st.header("Step 4: Persistence Homology")
    
    if st.session_state.get('point_cloud') is None:
        st.warning("⚠️ Please Generate Embedding in Tab 3 first.")
    else:
        # 1. Run Calculation
        if st.button("Run Persistence Analysis", key="btn_run_pers"):
            with st.spinner("Computing Alpha Complex (this can take a minute)..."):
                pc = st.session_state['point_cloud']
                
                # Limit points for speed if dataset is huge
                if len(pc) > 2000:
                    st.warning(f"Subsampling point cloud (from {len(pc)} to 2000 points) for performance.")
                    idx = np.random.choice(len(pc), 2000, replace=False)
                    pc_calc = pc[idx]
                else:
                    pc_calc = pc
                
                ac = gd.AlphaComplex(points=pc_calc)
                stree = ac.create_simplex_tree()
                persistence = stree.persistence()
                
                # Save to session
                st.session_state['persistence_result'] = persistence
                st.success("Persistence computed!")

        # 2. Show Results (Persistent)
        if st.session_state.get('persistence_result') is not None:
            persistence = st.session_state['persistence_result']
            
            # Filtering Logic
            finite = [p for p in persistence if p[1][1] < float('inf')]
            max_death = max(p[1][1] for p in finite) if finite else 1.0
            min_life = max_death * 0.02
            filtered = [p for p in persistence if (p[1][1] - p[1][0] > min_life) or p[1][1] == float('inf')]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Barcode")
                fig, ax = plt.subplots(figsize=(6, 5))
                gd.plot_persistence_barcode(filtered, axes=ax, legend=True)
                ax.set_xlim(0, max_death * 1.1)
                ax.set_title("Persistence Barcode", fontsize=14)
                ax.tick_params(labelsize=10)
                st.pyplot(fig)
                
                # Download Button for Barcode
                st.download_button(
                    "Download Barcode (PNG)",
                    save_plot_to_bytes(fig),
                    "persistence_barcode.png",
                    "image/png",
                    key="dl_barcode"
                )
            
            with col2:
                st.subheader("Diagram")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                gd.plot_persistence_diagram(persistence, axes=ax2, legend=True)
                ax2.set_xlim(0, max_death * 1.1)
                ax2.set_ylim(0, max_death * 1.1)
                ax2.set_title("Persistence Diagram", fontsize=14)
                ax2.tick_params(labelsize=10)
                st.pyplot(fig2)
                
                # Download Button for Diagram
                st.download_button(
                    "Download Diagram (PNG)",
                    save_plot_to_bytes(fig2),
                    "persistence_diagram.png",
                    "image/png",
                    key="dl_diagram"
                )

# ==========================================
# TAB 5: TOMATO CLUSTERING (PERSISTENT)
# ==========================================
with tab5:
    st.header("Step 5: ToMATo Clustering")
    
    # 1. Inputs
    # We use a unique key 'k_val_input' to prevent conflicts
    k_val = st.number_input("k (Nearest Neighbors)", value=420, min_value=1, key="k_val_input")
    
    # 2. Logic to Run Calculation
    if st.button("Run Clustering", key="btn_run_cluster"):
        if st.session_state.get('point_cloud') is None:
            st.error("⚠️ Please generate Point Cloud in Tab 3 first.")
        else:
            with st.spinner("Running ToMATo Clustering..."):
                pc = st.session_state['point_cloud']
                
                # Run Algorithm
                t = tomato.Tomato(k=k_val)
                t.fit(pc)
                
                # Update Dataframe in Session
                # We refer to the 'df_embedded' saved in session
                st.session_state['df_embedded']['Cluster'] = t.labels_
                
                # Save specific results to session so they persist
                st.session_state['n_clusters'] = t.n_clusters_
                st.session_state['cluster_labels'] = t.labels_
                
                # Force reload to show results immediately
                st.rerun()

    # 3. Logic to Display Results (Always runs if data exists)
    if st.session_state.get('cluster_labels') is not None:
        # Load from memory
        labels = st.session_state['cluster_labels']
        n_clusters = st.session_state.get('n_clusters', 0)
        pc = st.session_state['point_cloud']
        
        st.success(f"✅ Clustering Complete. Found {n_clusters} clusters.")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        # Use first 2 dimensions for 2D projection
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=labels, cmap='tab10', s=15)
        plt.colorbar(sc, label='Cluster ID')
        ax.set_title(f"ToMATo Clustering (k={k_val})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        
        st.pyplot(fig)
        
        # Download button
        st.download_button("Download Cluster Plot", save_plot_to_bytes(fig), "cluster_plot.png", "image/png", key="dl_cluster_png")
    
    elif st.session_state.get('point_cloud') is not None:
        st.info("Ready to run. Adjust 'k' and click 'Run Clustering'.")

# # ==========================================
# TAB 6: STATE IDENTIFICATION (BASED ON MEAN GRADIENTS)
# ==========================================
with tab6:
    st.header("Step 6: River State Identification")
    
    # Check Prerequisite
    if st.session_state.get('df_embedded') is None or 'Cluster' not in st.session_state['df_embedded'].columns:
        st.warning("⚠️ Please run Clustering in Tab 5 first.")
    else:
        # --- 0. PRE-CALCULATE DATA RANGE (BASED ON CLUSTER MEANS) ---
        temp_df = st.session_state['df_embedded'].copy()
        # 1. Calculate raw gradients first
        temp_df['Gradient'] = temp_df['Level'].diff()
        
        # 2. Group by Cluster to get the MEAN Gradient for each cluster
        cluster_means = temp_df.groupby('Cluster')['Gradient'].mean()
        
        # 3. Get the Min/Max of the MEANS (not raw data)
        # We use min(0, ...) and max(0, ...) to ensure the range always includes 0 for the sliders
        max_grad = max(0.0, float(cluster_means.max())) 
        min_grad = min(0.0, float(cluster_means.min()))
        
        st.info(f"📊 Range of Cluster Mean Gradients: {min_grad:.5f} to {max_grad:.5f} m/h")

        # --- 1. SLIDER INPUTS ---
        st.subheader("Define Thresholds (Negative to Positive)")
        st.markdown("Adjust boundaries based on the **Average Gradient** of your clusters.")
        
        col_neg, col_pos = st.columns(2)
        
        # SLIDER 1: NEGATIVE (DROP) ZONES
        with col_neg:
            st.markdown("### 📉 Falling Limb (Drop)")
            # Add a small buffer to min_grad to prevent slider errors if min_grad is exactly 0
            safe_min = min_grad if min_grad < 0 else -0.001
            
            drop_thresholds = st.slider(
                "Boundaries: [Rapid | Slow | Steady]",
                min_value=safe_min,
                max_value=0.0,
                value=(safe_min * 0.7, safe_min * 0.2), # Default positions
                format="%.5f",
                step=0.0001,
                key="slider_drop"
            )
            rapid_drop_th = drop_thresholds[0]
            slow_drop_th = drop_thresholds[1]
            st.caption(f"**Rapid Drop:** < {rapid_drop_th:.5f}\n\n**Slow Drop:** {rapid_drop_th:.5f} to {slow_drop_th:.5f}")

        # SLIDER 2: POSITIVE (RISE) ZONES
        with col_pos:
            st.markdown("### 📈 Rising Limb (Rise)")
            # Add a small buffer to max_grad
            safe_max = max_grad if max_grad > 0 else 0.001
            
            rise_thresholds = st.slider(
                "Boundaries: [Steady | Slow | Rapid]",
                min_value=0.0,
                max_value=safe_max,
                value=(safe_max * 0.1, safe_max * 0.5), # Default positions
                format="%.5f",
                step=0.0001,
                key="slider_rise"
            )
            slow_rise_th = rise_thresholds[0]
            flood_rise_th = rise_thresholds[1]
            st.caption(f"**Slow Rise:** {slow_rise_th:.5f} to {flood_rise_th:.5f}\n\n**Rapid Rise:** > {flood_rise_th:.5f}")

        # VISUALIZE STEADY ZONE
        st.success(f"✅ **Steady Flow Zone:** Between {slow_drop_th:.5f} and {slow_rise_th:.5f}")

        # --- 2. CALCULATION LOGIC ---
        if st.button("Analyze Cluster Physics", key="btn_analyze_physics"):
            df_emb = st.session_state['df_embedded'].copy()
            df_emb['Gradient'] = df_emb['Level'].diff()
            
            # Logic Function
            def assign_state_by_physics(row):
                # This uses the Mean_Gradient calculated in the aggregation step below
                grad = row['Mean_Gradient']
                
                # 1. Negative Side (Drop)
                if grad < 0:
                    if grad < rapid_drop_th: return "Rapid Decrease"
                    elif grad < slow_drop_th: return "Slow Decrease"
                    else: return "Base Flow (Steady)" # Between slow_drop_th and 0
                
                # 2. Positive Side (Rise)
                else:
                    if grad > flood_rise_th: return "Flood State (Rapid Rise)"
                    elif grad > slow_rise_th: return "Slow Rise"
                    else: return "Base Flow (Steady)" # Between 0 and slow_rise_th
            
            # Aggregation: Calculate Mean Gradient per cluster
            cluster_stats = df_emb.groupby('Cluster').agg({
                'Level': ['mean', 'min', 'max'],
                'Gradient': ['mean', 'std'],
                'Date': 'count'
            }).reset_index()
            cluster_stats.columns = ['Cluster', 'Mean_Level', 'Min_Level', 'Max_Level', 'Mean_Gradient', 'Std_Gradient', 'Count']
            
            # Apply logic
            cluster_stats['State'] = cluster_stats.apply(assign_state_by_physics, axis=1)
            
            # Map back
            state_map = dict(zip(cluster_stats['Cluster'], cluster_stats['State']))
            df_emb['State'] = df_emb['Cluster'].map(state_map)
            
            # Context Gradients
            cluster_avg_grad = df_emb.groupby('Cluster')['Gradient'].mean()
            df_emb['Cluster_Avg_Gradient'] = df_emb['Cluster'].map(cluster_avg_grad)
            df_emb['Gradient_Instant'] = df_emb['Gradient'].fillna(0) 
            
            # SAVE TO SESSION
            st.session_state['df_embedded'] = df_emb
            st.session_state['cluster_stats'] = cluster_stats
            st.rerun()

        # --- 3. DISPLAY LOGIC (PERSISTENT) ---
        if st.session_state.get('cluster_stats') is not None:
            df_emb = st.session_state['df_embedded']
            cluster_stats = st.session_state['cluster_stats']
            
            st.divider()
            st.write("### Cluster Statistics & Assigned States")
            st.dataframe(cluster_stats)
            
            # Plot
            fig, ax = plt.subplots(figsize=(15, 6))
            palette = {
                "Base Flow (Steady)": "green", 
                "Slow Rise": "orange",
                "Flood State (Rapid Rise)": "red", 
                "Rapid Decrease": "purple",
                "Slow Decrease": "blue", 
                "Unknown": "gray"
            }
            
            # Scatter Plot
            sns.scatterplot(data=df_emb, x='Date', y='Level', hue='State', palette=palette, s=15, ax=ax, linewidth=0)
            ax.set_title("Identified Water Level States based on Gradient Physics", fontsize=16)
            ax.set_xlabel("Date", fontsize=14)
            ax.set_ylabel("Water Level (m)", fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            st.pyplot(fig)
            
            st.download_button("Download Plot", save_plot_to_bytes(fig), "state_ts.png", "image/png", key="dl_state_plot")
            
# ==========================================
# TAB 7: TRANSITION NETWORK (SHOW ALL PROBABILITIES)
# ==========================================
with tab7:
    st.header("Step 7: Transition Network Analysis")
    
    # Check Prerequisite
    if st.session_state.get('df_embedded') is None or 'State' not in st.session_state['df_embedded'].columns:
        st.warning("⚠️ Please Identify States in Tab 6 first.")
    else:
        # --- 1. CALCULATION LOGIC ---
        if st.button("Generate Network", key="btn_gen_network"):
            df_emb = st.session_state['df_embedded']
            
            # Create Edges based on sequential clusters
            cluster_seq = df_emb['Cluster'].values
            transitions = list(zip(cluster_seq[:-1], cluster_seq[1:]))
            
            G = nx.DiGraph()
            for s, t in transitions:
                if G.has_edge(s, t):
                    G[s][t]['weight'] += 1
                else:
                    G.add_edge(s, t, weight=1)
            
            # Calculate Probabilities
            for u, v, d in G.edges(data=True):
                # Sum of all weights leaving node 'u'
                total_out = sum([data['weight'] for _, _, data in G.out_edges(u, data=True)])
                d['probability'] = d['weight'] / total_out
            
            # Create Table Data
            state_map = dict(zip(df_emb['Cluster'], df_emb['State']))
            table_data = []
            for u, v, d in G.edges(data=True):
                if d['weight'] > 0:
                    table_data.append({
                        "From_Cluster": u, 
                        "From_State": state_map.get(u),
                        "To_Cluster": v, 
                        "To_State": state_map.get(v),
                        "Weight": d['weight'], 
                        "Prob": d['probability']
                    })
            
            # Create DataFrame and Sort
            df_trans = pd.DataFrame(table_data).sort_values(['From_Cluster', 'Prob'], ascending=[True, False])
            
            # SAVE TO SESSION
            st.session_state['G'] = G
            st.session_state['df_trans'] = df_trans
            st.rerun()

        # --- 2. DISPLAY LOGIC (PERSISTENT) ---
        if st.session_state.get('G') is not None:
            G = st.session_state['G']
            df_trans = st.session_state['df_trans']
            df_emb = st.session_state['df_embedded']
            state_map = dict(zip(df_emb['Cluster'], df_emb['State']))
            
            st.success("Network Generated Successfully.")
            
            # Visualization
            # We filter visual edges to keep the plot clean, but the table will show everything
            fig, ax = plt.subplots(figsize=(12, 10))
            pos = nx.spring_layout(G, k=3.5, seed=42)
            
            palette = {
                "Base Flow (Steady)": "green", "Slow Rise": "orange",
                "Flood State (Rapid Rise)": "red", "Rapid Decrease": "purple",
                "Slow Decrease": "blue", "Unknown": "gray"
            }
            
            # Draw Nodes
            colors = [palette.get(state_map.get(n, "Unknown"), 'grey') for n in G.nodes()]
            counts = df_emb['Cluster'].value_counts()
            sizes = [counts.get(n, 10) * 1 for n in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, ax=ax, alpha=0.9, edgecolors='k')
            nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_weight='bold')
            
            # Draw Edges (Visual filter only: weight > 1)
            edges = [(u,v) for u,v,d in G.edges(data=True) if u!=v and d['weight'] > 1]
            nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', ax=ax, alpha=0.6, connectionstyle="arc3,rad=0.2")
            
            # Draw Loops
            loops = [(u,v) for u,v,d in G.edges(data=True) if u==v and d['weight'] > 1]
            nx.draw_networkx_edges(G, pos, edgelist=loops, arrowstyle='-|>', ax=ax, alpha=0.6, connectionstyle="arc3,rad=2.5")
            
            ax.set_title("Cluster Transition Network (Visualizing Major Paths)")
            ax.axis('off')
            plt.legend(handles=[mpatches.Patch(color=c, label=l) for l,c in palette.items()], loc='upper left')
            st.pyplot(fig)
            st.download_button("Download Network Graph", save_plot_to_bytes(fig), "network.png", "image/png", key="dl_net_png")
            
            # --- FULL TABLE DISPLAY ---
            st.divider()
            st.write("### Full Transition Probability Table")
            st.write("This table shows every transition calculated between clusters.")
            
            # Show ALL rows (removed .head(10))
            st.dataframe(df_trans, use_container_width=True)
            
            # Add download button for the table
            csv = df_trans.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Transition Table (CSV)",
                csv,
                "transition_probabilities.csv",
                "text/csv",
                key="dl_trans_csv"
            )

# ==========================================
# TAB 8: INFOGRAPHIC (RANGES IN TABLE)
# ==========================================
with tab8:
    st.header("Step 8: Results Infographic Dashboard")
    
    # Check Prerequisites
    if st.session_state.get('cluster_stats') is None or st.session_state.get('df_trans') is None:
        st.warning("⚠️ Please complete Step 6 (States) and Step 7 (Network) first.")
    else:
        # 1. Logic Button
        if st.button("Generate Dashboard", key="btn_gen_dash"):
            st.session_state['show_dashboard'] = True
            st.rerun()

        # 2. Display Logic
        if st.session_state.get('show_dashboard'):
            # Load data
            stats_df = st.session_state['cluster_stats']
            df_emb = st.session_state['df_embedded']
            df_trans = st.session_state['df_trans']
            
            # --- CALCULATIONS ---
            # 1. State Summary (Updated for RANGES)
            state_summary = df_emb.groupby('State').agg({
                'Level': ['min', 'max'],      # Get range for Level
                'Gradient': ['min', 'max'],   # Get range for Gradient
                'Date': 'count'
            }).reset_index()
            
            # Flatten MultiIndex columns
            state_summary.columns = ['State', 'Level_Min', 'Level_Max', 'Grad_Min', 'Grad_Max', 'Count']
            
            # Calculate Frequency
            total_count = state_summary['Count'].sum()
            state_summary['Freq'] = (state_summary['Count'] / total_count * 100).round(2)
            
            # Create Formatted "Range" Columns for the Table
            state_summary['Level_Range (m)'] = state_summary.apply(
                lambda x: f"{x['Level_Min']:.2f} - {x['Level_Max']:.2f}", axis=1
            )
            state_summary['Gradient_Range (m/h)'] = state_summary.apply(
                lambda x: f"{x['Grad_Min']:.4f} - {x['Grad_Max']:.4f}", axis=1
            )
            
            # 2. Stability
            self_loops = df_trans[df_trans['From_Cluster'] == df_trans['To_Cluster']]
            stability = self_loops.groupby('From_State').apply(
                lambda x: (x['Prob']*x['Weight']).sum()/x['Weight'].sum() if x['Weight'].sum() > 0 else 0
            ).reset_index(name='Stability')

            # 3. Macro Transitions
            path_counts = df_trans.groupby(['From_State', 'To_State'])['Weight'].sum().reset_index()
            total_out = df_trans.groupby('From_State')['Weight'].sum().reset_index(name='Total')
            macro_full = path_counts.merge(total_out, on='From_State')
            macro_full['Probability'] = macro_full['Weight'] / macro_full['Total']

            # --- PLOTTING ---
            fig = plt.figure(figsize=(22, 18), constrained_layout=True)
            gs = GridSpec(3, 2, figure=fig, height_ratios=[0.25, 1, 1.2])
            fig.suptitle("Physical Characteristics & Transition Dynamics", fontsize=28, fontweight='bold')
            
            # ------------------------------------------
            # PANEL 1: TABLE (Top Row - UPDATED)
            # ------------------------------------------
            ax_tbl = fig.add_subplot(gs[0, :])
            ax_tbl.axis('off')
            
            # Select the formatted range columns
            tbl_data = state_summary[['State', 'Level_Range (m)', 'Gradient_Range (m/h)', 'Freq']]
            
            # Rename columns for display
            tbl_data.columns = ['State', 'Level Range (m)', 'Gradient Range (m/h)', 'Freq (%)']
            
            # Create Table
            tbl = ax_tbl.table(cellText=tbl_data.values, colLabels=tbl_data.columns, loc='center', cellLoc='center')
            tbl.scale(1, 2.5)
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(16)
            
            # ------------------------------------------
            # PANEL 2: FREQUENCY (Middle Left)
            # ------------------------------------------
            ax_freq = fig.add_subplot(gs[1, 0])
            sns.barplot(data=state_summary, x='Freq', y='State', ax=ax_freq, palette='viridis')
            ax_freq.set_title("State Frequency (%)", fontsize=18, fontweight='bold')
            ax_freq.tick_params(axis='both', which='major', labelsize=14)
            ax_freq.set_xlabel("Frequency (%)", fontsize=16)
            ax_freq.set_ylabel("", fontsize=16)
            
            # ------------------------------------------
            # PANEL 3: STABILITY (Middle Right)
            # ------------------------------------------
            ax_stab = fig.add_subplot(gs[1, 1])
            if not stability.empty:
                sns.barplot(data=stability, x='Stability', y='From_State', ax=ax_stab, palette='coolwarm')
            ax_stab.set_title("Stability Index (0-1)", fontsize=18, fontweight='bold')
            ax_stab.set_xlim(0, 1)
            ax_stab.tick_params(axis='both', which='major', labelsize=14)
            ax_stab.set_xlabel("Probability of Remaining in State", fontsize=16)
            ax_stab.set_ylabel("", fontsize=16)
            
            # ------------------------------------------
            # PANEL 4: HEATMAP (Bottom Left)
            # ------------------------------------------
            ax_heat = fig.add_subplot(gs[2, 0])
            piv = macro_full.pivot(index='From_State', columns='To_State', values='Probability').fillna(0)
            sns.heatmap(piv, annot=True, fmt=".1%", cmap="Blues", ax=ax_heat, annot_kws={"size": 14})
            ax_heat.set_title("Macro Transition Heatmap", fontsize=18, fontweight='bold')
            ax_heat.tick_params(axis='both', which='major', labelsize=14)
            ax_heat.set_xlabel("To State", fontsize=16)
            ax_heat.set_ylabel("From State", fontsize=16)
            
            # ------------------------------------------
            # PANEL 5: PATHWAYS ANALYSIS (Bottom Right)
            # ------------------------------------------
            ax_path = fig.add_subplot(gs[2, 1])
            cross_state_paths = macro_full[macro_full['From_State'] != macro_full['To_State']].copy()
            cross_state_paths['Path_Label'] = cross_state_paths['From_State'] + " → " + cross_state_paths['To_State']
            
            top_paths = cross_state_paths.sort_values('Probability', ascending=False).head(5)
            warning_targets = ['Slow Rise', 'Flood State (Rapid Rise)']
            warning_paths = cross_state_paths[
                (cross_state_paths['To_State'].isin(warning_targets)) &
                (cross_state_paths['From_State'] != 'Flood State (Rapid Rise)')
            ]
            
            if not cross_state_paths.empty:
                final_paths = pd.concat([top_paths, warning_paths]).drop_duplicates().sort_values('Probability', ascending=False)
                def get_path_color(row):
                    if row['To_State'] in warning_targets: return 'orangered'
                    return 'silver'
                path_colors = final_paths.apply(get_path_color, axis=1).tolist()
                sns.barplot(data=final_paths, x='Probability', y='Path_Label', ax=ax_path, palette=path_colors, edgecolor='black')
                for i, v in enumerate(final_paths['Probability']):
                    ax_path.text(v + 0.002, i, f"{v:.1%}", va='center', fontweight='bold', fontsize=12)
            else:
                ax_path.text(0.5, 0.5, "No significant cross-state transitions found", ha='center', fontsize=14)
            
            ax_path.set_title("Transition Pathways: Highways (Grey) vs Warnings (Red)", fontsize=18, fontweight='bold')
            ax_path.set_xlabel("Transition Probability", fontsize=16, fontweight='bold')
            ax_path.set_ylabel("", fontsize=16)
            ax_path.tick_params(axis='both', labelsize=14)
            
            st.pyplot(fig)
            st.download_button("Download Dashboard", save_plot_to_bytes(fig), "dashboard.png", "image/png", key="dl_dash_png")


# ==========================================
# TAB 9: L2 NORM ANALYSIS (FIXED NUMPY ERROR)
# ==========================================
with tab9:
    st.header("Step 9: Sliding Window L2 Norm Analysis")
    st.markdown("Analyze local instability (L2 Norm) over a specific time range.")
    
    if st.session_state.get('df_subset') is None:
        st.error("⚠️ Load Data First (Tab 1)")
    else:
        # --- 1. CONFIGURATION ---
        st.subheader("Analysis Parameters")
        
        col_d1, col_d2 = st.columns(2)
        # Default dates based on loaded data
        min_date = st.session_state['df_subset']['Date'].min()
        max_date = st.session_state['df_subset']['Date'].max()
        
        # User Date Selection
        start_date_l2 = col_d1.date_input("Analysis Start Date", value=min_date, min_value=min_date, max_value=max_date, key="l2_start")
        end_date_l2 = col_d2.date_input("Analysis End Date", value=max_date, min_value=min_date, max_value=max_date, key="l2_end")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        win_size = col1.number_input("Window Size (Hours)", value=24, min_value=10, key="win_l2")
        step_size = col2.number_input("Step Size (Hours)", value=1, min_value=1, key="step_l2")
        
        col3, col4 = st.columns(2)
        tau_l2_input = col3.number_input("Tau (Delay)", value=st.session_state.get('tau', 6), key="tau_l2_widget")
        m_l2_input = col4.number_input("m (Dimension)", value=st.session_state.get('m', 3), key="m_l2_widget")
        
        # --- 2. CALCULATION BLOCK ---
        if st.button("Run L2 Analysis", key="run_l2_btn"):
            # Filter Data based on new date range
            mask = (st.session_state['df_subset']['Date'] >= pd.to_datetime(start_date_l2)) & \
                   (st.session_state['df_subset']['Date'] <= pd.to_datetime(end_date_l2))
            df_slice = st.session_state['df_subset'].loc[mask].copy().reset_index(drop=True)
            
            if len(df_slice) < win_size * 2:
                st.error(f"Selected date range is too short ({len(df_slice)} hours). Needs at least {win_size*2} hours.")
            else:
                with st.spinner("Calculating L2 Norms..."):
                    series = df_slice['Level'].values
                    dates = df_slice['Date'].values
                    tau_l2, m_l2 = tau_l2_input, m_l2_input
                    
                    l2_norms = []
                    aligned_dates = []
                    progress_bar = st.progress(0)
                    total_steps = len(series) - win_size
                    
                    # Main Loop
                    for i in range(0, total_steps, step_size):
                        window = series[i : i + win_size]
                        if np.std(window) == 0: score = 0
                        else:
                            norm_w = zscore(window)
                            pc_l2 = []
                            N_w = len(norm_w)
                            for j in range(N_w - (m_l2 - 1) * tau_l2):
                                vec = norm_w[j : j + m_l2*tau_l2 : tau_l2]
                                if len(vec) == m_l2: pc_l2.append(vec)
                            
                            if len(pc_l2) < 5: score = 0
                            else:
                                try:
                                    rips = gd.RipsComplex(points=pc_l2)
                                    stree = rips.create_simplex_tree(max_dimension=2)
                                    diag = stree.persistence(homology_coeff_field=2, min_persistence=0.01)
                                    # Extract pairs for dimension 1
                                    diag_1 = [p[1] for p in diag if p[0] == 1 and p[1][1] != float('inf')]
                                    
                                    if len(diag_1) == 0: 
                                        score = 0
                                    else:
                                        # --- CRITICAL FIX: Convert list to NumPy Array ---
                                        diag_1_arr = np.array(diag_1)
                                        land = gd.representations.Landscape(num_landscapes=1, resolution=100).fit_transform([diag_1_arr])
                                        score = np.linalg.norm(land)
                                except Exception:
                                    score = 0
                        
                        l2_norms.append(score)
                        aligned_dates.append(dates[i + win_size])
                        if i % 20 == 0: progress_bar.progress(min(i / total_steps, 1.0))
                    
                    progress_bar.empty()
                    
                    # Create Result DataFrame
                    df_l2_res = pd.DataFrame({'Date': aligned_dates, 'TDA_L2_Norm': l2_norms})
                    # Merge back with the specific slice
                    df_merged = pd.merge(df_slice, df_l2_res, on='Date', how='inner')
                    
                    # Normalize & Smooth
                    df_merged['TDA_Smoothed'] = df_merged['TDA_L2_Norm'].rolling(window=6, center=True).mean()
                    df_merged['TDA_Signal_Normalized'] = df_merged['TDA_L2_Norm'] # Rename for plot consistency
                    
                    st.session_state['l2_results'] = df_merged
                    st.session_state['l2_params'] = f"W={win_size}, τ={tau_l2}, m={m_l2}" 
                    st.rerun() 

        # --- 3. VISUALIZATION BLOCK (PERSISTENT) ---
        if st.session_state.get('l2_results') is not None:
            df_merged = st.session_state['l2_results']
            params_str = st.session_state.get('l2_params', "")
            
            st.divider()
            st.subheader("Threshold Tuning & Visualization")
            
            # Statistics
            median_val = df_merged['TDA_L2_Norm'].median()
            std_val = df_merged['TDA_L2_Norm'].std()
            
            col_t1, col_t2 = st.columns([1, 3])
            with col_t1: st.info(f"Median: {median_val:.4f}\nStd: {std_val:.4f}")
            with col_t2:
                multiplier = st.slider("Noise Threshold (Standard Deviations above Median)", 
                                     0.0, 5.0, 1.5, 0.1, key="threshold_slider")
            
            threshold = median_val + (std_val * multiplier)
            st.write(f"**Current Threshold Level:** {threshold:.4f}")

            # --- PLOT 1: INTERACTIVE PLOTLY ---
            st.write("### Interactive Analysis")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['Level'], name="Water Level", line=dict(color='blue')), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['TDA_Smoothed'], name="TDA Signal (Smoothed)", line=dict(color='red')), secondary_y=True)
            fig.add_hline(y=threshold, line_dash="dot", line_color="black", annotation_text=f"Threshold ({multiplier}σ)")
            
            if 'flood_dates' in st.session_state:
                flood_s, flood_e = st.session_state['flood_dates']
                # Only draw rectangle if flood range is inside our analysis range
                if flood_s >= df_merged['Date'].min() and flood_e <= df_merged['Date'].max():
                    fig.add_vrect(x0=flood_s, x1=flood_e, fillcolor="gray", opacity=0.1, layer="below")
            
            fig.update_layout(title=f"Water Level vs Complexity ({params_str})", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # --- PLOT 2: MATPLOTLIB (STATIC REPORT QUALITY) ---
            st.write("### Static Report Plot (High Quality)")
            
            fig_static, ax1 = plt.subplots(figsize=(15, 6))

            # Left Axis: Water Level
            ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Water Level (m)', color='tab:blue', fontweight='bold', fontsize=12)
            ax1.plot(df_merged['Date'], df_merged['Level'], color='tab:blue', label='Water Level', linewidth=1.5, alpha=0.8)
            ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=10)
            ax1.tick_params(axis='x', labelsize=10)
            ax1.grid(True, alpha=0.3)

            # Highlight Flood Range (if it exists)
            if 'flood_dates' in st.session_state:
                flood_s, flood_e = st.session_state['flood_dates']
                ax1.axvspan(flood_s, flood_e, color='gray', alpha=0.15, label='Flood Event Range')

            # Right Axis: TDA Signal
            ax2 = ax1.twinx()
            ax2.set_ylabel('TDA Complexity (L2 Norm)', color='tab:red', fontweight='bold', fontsize=12)

            # Plot Raw Signal (Faint)
            ax2.plot(df_merged['Date'], df_merged['TDA_Signal_Normalized'],
                     color='red', alpha=0.15, linewidth=1, label='Raw Noise')

            # Plot Smoothed Signal (Bold)
            ax2.plot(df_merged['Date'], df_merged['TDA_Smoothed'],
                     color='darkred', linewidth=2, label='Smoothed Signal')

            # Plot Threshold Line
            ax2.axhline(y=threshold, color='black', linestyle=':', alpha=0.6, label=f'Threshold ({multiplier}σ)')

            ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=10)

            # Custom Legend (Combine both axes)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

            plt.title(f"TDA Analysis (Smoothed): Detecting Major Instability\nParameters: {params_str}", fontsize=14)
            
            st.pyplot(fig_static)
            st.download_button("Download Report Plot (PNG)", save_plot_to_bytes(fig_static), "tda_l2_report.png", "image/png", key="dl_l2_static")
            
            # --- DOWNLOAD DATA ---
            csv = df_merged.to_csv(index=False).encode('utf-8')
            st.download_button("Download L2 Analysis Results (CSV)", csv, "l2_analysis_results.csv", "text/csv", key="dl_l2_csv")

# ==========================================
# TAB 10: EXPORT
# ==========================================
with tab10:
    st.header("Step 10: Feature Extraction & Download")
    
    if st.button("Generate Final Dataset"):
        # Check prerequisites
        if st.session_state['df_embedded'] is None: st.error("Missing Clustering Data (Tab 5/6)")
        elif st.session_state['l2_results'] is None: st.error("Missing L2 Norm Data (Tab 9)")
        else:
            df_emb = st.session_state['df_embedded']
            df_l2 = st.session_state['l2_results']
            
            # We need Transition Probability and Stability Score mapped back to the main DF
            # Create mapping dicts from Cluster Stats
            if 'df_trans' in st.session_state and 'cluster_stats' in st.session_state:
                # 1. Stability Map (From self-loops in df_trans)
                # Recalculate node stability
                G = st.session_state['G']
                stability_map = {}
                for node in G.nodes():
                    if G.has_edge(node, node):
                         # Prob of self-loop
                         stability_map[node] = G[node][node]['probability']
                    else:
                        stability_map[node] = 0.0
                
                df_emb['Stability_Score'] = df_emb['Cluster'].map(stability_map)
                
                # 2. Transition Probability (Next Step)
                # This requires look-ahead, which we already captured implicitly in the network construction
                # For a simple feature, we can use the max outgoing probability of the current cluster
                max_prob_map = {}
                for u in G.nodes():
                    probs = [d['probability'] for _,_,d in G.out_edges(u, data=True)]
                    max_prob_map[u] = max(probs) if probs else 0.0
                df_emb['Transition_Probability'] = df_emb['Cluster'].map(max_prob_map)
            
            else:
                df_emb['Stability_Score'] = 0
                df_emb['Transition_Probability'] = 0

            # Merge Embedded (States) with L2 (Signals)
            # Both have 'Date'
            df_final = pd.merge(df_emb, df_l2[['Date', 'TDA_L2_Norm']], on='Date', how='inner')
            
            # Select Columns
            cols = ['Date', 'Level', 'Gradient_Instant', 'Cluster', 'State', 
                    'Cluster_Avg_Gradient', 'Stability_Score', 'Transition_Probability', 'TDA_L2_Norm']
            
            # Safety check if columns exist
            final_cols = [c for c in cols if c in df_final.columns]
            df_final = df_final[final_cols]
            
            st.success(f"Final Dataset Generated: {len(df_final)} rows.")
            st.dataframe(df_final.head())
            
            # Convert to CSV
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Deep Learning Features (CSV)",
                csv,
                "deep_learning_features.csv",
                "text/csv"
            )
