import traceback
import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import os
import math
import tempfile
from openpyxl import Workbook, load_workbook
from ase.cluster import FaceCenteredCubic, BodyCenteredCubic, HexagonalClosedPacked
from ase.io import write, read
from ase.neighborlist import build_neighbor_list
from sklearn.preprocessing import LabelEncoder  # if needed for uploaded logs
import matplotlib.pyplot as plt
import base64
from io import BytesIO

st.set_page_config(page_title="Monte Carlo Nanoparticle Simulator", layout="wide")

# --- Utility functions extracted/adapted from your script ---
BOLTZMANN_K = 8.617333262e-5
BULK_COORD = 12

lattice_map = {
    'fcc': FaceCenteredCubic,
    'bcc': BodyCenteredCubic,
    'hcp': HexagonalClosedPacked
}

def symbol_type(sym, A):
    return 'A' if sym == A else 'B'

def calculate_energy(p, A, coeffs):
    nl = build_neighbor_list(p, bothways=True, self_interaction=False)
    count = {k: 0 for k in coeffs}
    for i in range(len(p)):
        t_i = symbol_type(p[i].symbol, A)
        neighbors = nl.get_neighbors(i)[0]
        if len(neighbors) < BULK_COORD:
            count[f'x{t_i}-S'] += 1
        for j in neighbors:
            if i < j:
                t_j = symbol_type(p[j].symbol, A)
                if t_i == t_j:
                    count[f'x{t_i}-{t_j}'] += 1
                else:
                    count['xA-B'] += 1
    return sum(coeffs[k] * count.get(k, 0) for k in count)

def count_surface(p, A):
    nl = build_neighbor_list(p, bothways=True, self_interaction=False)
    total_A, surf, surf_A = 0, 0, 0
    for i in range(len(p)):
        neighbors = nl.get_neighbors(i)[0]
        t = symbol_type(p[i].symbol, A)
        if t == 'A':
            total_A += 1
        if len(neighbors) < BULK_COORD:
            surf += 1
            if t == 'A':
                surf_A += 1
    ratio = surf_A / surf if surf else 0
    return total_A, surf, surf_A, ratio

def run_simulation(params, progress_callback=None):
    """
    Run the Monte Carlo simulation. Returns a dict of results and file paths.
    progress_callback(step, energy, ratio) can be used to stream progress.
    """
    A = params['element_A']
    B = params['element_B']
    composition_A = params['composition_A']
    T = params['temperature']
    N_STEPS = params['n_steps']
    SAVE_INTERVAL = params['save_interval']
    LAYERS = params['layers']
    SURFACES = params['surfaces']
    coeffs = params['coefficients']
    lattice_type = params.get('lattice_type', 'fcc').lower()

    ClusterBuilder = lattice_map.get(lattice_type)
    if ClusterBuilder is None:
        raise ValueError(f"Unsupported lattice type '{lattice_type}'. Choose from 'fcc', 'bcc', or 'hcp'.")

    # Build initial particle
    particle = ClusterBuilder(A, surfaces=SURFACES, layers=LAYERS)
    n_atoms = len(particle)
    n_A = int(n_atoms * composition_A)
    indices_A = np.random.choice(range(n_atoms), size=n_A, replace=False)
    for i in range(n_atoms):
        particle[i].symbol = A if i in indices_A else B

    initial_surface_data = count_surface(particle, A)
    initial_xyz = f"initial_{A}{B}_{n_atoms}.xyz"
    write(initial_xyz, particle)

    # Prepare storage
    os.makedirs("trajectory", exist_ok=True)
    log = []

    energy = calculate_energy(particle, A, coeffs)
    start_time = time.time()

    for step in range(1, N_STEPS + 1):
        i = np.random.randint(0, n_atoms)
        neighbors = build_neighbor_list(particle).get_neighbors(i)[0]
        if len(neighbors) == 0:
            continue
        j = np.random.choice(neighbors)
        if particle[i].symbol == particle[j].symbol:
            continue

        trial = particle.copy()
        trial[i].symbol, trial[j].symbol = trial[j].symbol, trial[i].symbol
        dE = calculate_energy(trial, A, coeffs) - energy

        if dE < 0 or np.random.random() < math.exp(-dE / (BOLTZMANN_K * T)):
            particle = trial
            energy += dE

        if step % SAVE_INTERVAL == 0:
            total_A, surf, surf_A, ratio = count_surface(particle, A)
            log.append({
                'Step': step,
                'Energy (eV)': energy,
                f'Total {A}': total_A,
                'Surface Atoms': surf,
                f'{A} on Surface': surf_A,
                f'Surface {A} Ratio': ratio
            })
            if progress_callback:
                progress_callback(step, energy, ratio)
        if step % params.get('snapshot_interval', 500) == 0:
            snapshot_file = f"trajectory/step_{step:05d}.xyz"
            write(snapshot_file, particle)

    final_xyz = f"final_{A}{B}_{n_atoms}.xyz"
    write(final_xyz, particle)

    # Save Excel log
    xlsx_file = f"MMC_{A}{B}_log.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Simulation Log"

    # Header params
    meta = [
        ("Element A", A), ("Element B", B), ("Composition A", composition_A),
        ("Temperature (K)", T), ("MC Steps", N_STEPS),
        ("Save Interval", SAVE_INTERVAL), ("Total Atoms", n_atoms)
    ]
    for i, (k, v) in enumerate(meta, 1):
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=v)

    header_row = len(meta) + 2
    headers = list(log[0].keys()) if log else []
    for j, h in enumerate(headers, 1):
        ws.cell(row=header_row, column=j, value=h)
    for i, entry in enumerate(log, header_row + 1):
        for j, h in enumerate(headers, 1):
            ws.cell(row=i, column=j, value=entry[h])

    wb.save(xlsx_file)

    duration = time.time() - start_time
    return {
        "initial_xyz": initial_xyz,
        "final_xyz": final_xyz,
        "log": pd.DataFrame(log),
        "xlsx_file": xlsx_file,
        "duration": duration,
        "initial_surface_data": initial_surface_data,
        "final_surface_data": count_surface(particle, A)
    }

# --- Streamlit UI ---
st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("Elements & Composition", expanded=True):
    element_A = st.text_input("Element A (e.g., Pt)", value="Pt")
    element_B = st.text_input("Element B (e.g., Ru)", value="Ru")
    composition_A = st.slider("Composition of A (fraction)", 0.0, 1.0, 0.5, step=0.05)

with st.sidebar.expander("Thermodynamics", expanded=False):
    temperature = st.number_input("Temperature (K)", min_value=1.0, value=250.0, step=1.0)
    n_steps = st.number_input("Monte Carlo Steps", min_value=100, value=2000, step=100)
    save_interval = st.number_input("Log Save Interval", min_value=1, value=200, step=1)
    snapshot_interval = st.number_input("Snapshot Interval", min_value=100, value=500, step=100)

with st.sidebar.expander("Lattice / Geometry", expanded=False):
    lattice_type = st.selectbox("Lattice Type", ["fcc", "bcc", "hcp"])
    layers = st.multiselect("Layers (x,y,z)", options=[1,2,3,4,5,6,7,8], default=[7,7,7])
    # Ensure tuple of three
    if len(layers) != 3:
        st.warning("Please pick exactly 3 values for layers; defaulting to (7,7,7).")
        layers = [7,7,7]
    surfaces = st.text_input("Surfaces (list of tuples)", value="[(1,1,1),(1,1,1),(1,1,0)]")
    try:
        surfaces_parsed = eval(surfaces)
    except:
        st.error("Cannot parse surfaces. Use Python list of tuples like [(1,1,1),(1,1,1),(1,1,0)]")
        surfaces_parsed = [(1,1,1),(1,1,1),(1,1,0)]

with st.sidebar.expander("Energy Coefficients", expanded=False):
    # default set from your script
    coeffs = {
        'xA-A': st.number_input("xA-A", value=-0.022078),
        'xB-B': st.number_input("xB-B", value=-0.150000),
        'xA-B': st.number_input("xA-B", value=-0.109575),
        'xA-S': st.number_input("xA-S", value=-0.250717),
        'xB-S': st.number_input("xB-S", value=-0.300000),
        'xA-A-out': st.number_input("xA-A-out", value=0.184150),
        'xB-B-out': st.number_input("xB-B-out", value=0.332228),
        'xA-B-out': st.number_input("xA-B-out", value=0.051042),
    }

run_button = st.sidebar.button("▶️ Run Simulation")

# Placeholder for status / progress
status_placeholder = st.empty()
progress_bar = st.sidebar.progress(0)

# Container for results
results_container = st.container()
plot_container = st.columns(2)

# Threading to avoid blocking
if run_button:
    params = {
        'element_A': element_A,
        'element_B': element_B,
        'composition_A': composition_A,
        'temperature': temperature,
        'n_steps': int(n_steps),
        'save_interval': int(save_interval),
        'snapshot_interval': int(snapshot_interval),
        'layers': tuple(layers),
        'surfaces': surfaces_parsed,
        'coefficients': coeffs,
        'lattice_type': lattice_type
    }

    progress_state = {"last_step": 0, "energy": None, "ratio": None}
    def progress_cb(step, energy, ratio):
        progress_state["last_step"] = step
        progress_state["energy"] = energy
        progress_state["ratio"] = ratio
        pct = min(step / params['n_steps'], 1.0)
        progress_bar.progress(pct)
        status_placeholder.markdown(f"**Step:** {step} | **Energy:** {energy:.4f} eV | **Surface {element_A} Ratio:** {ratio:.4f}")

    run_future = st.spinner("Simulation running... this can take a while.")
    # Run in thread so UI stays responsive
    result_holder = {}
    def target():
        try:
            result_holder.update(run_simulation(params, progress_callback=progress_cb))
        except Exception as e:
            result_holder["error"] = str(e)
    thread = threading.Thread(target=target)
    thread.start()

    # Wait with simple polling
    while thread.is_alive():
        time.sleep(0.5)
    st.success("✅ Simulation completed.")

    if "error" in result_holder:
        st.error(f"Simulation failed: {result_holder['error']}")
    else:
        res = result_holder
        df_log = res["log"]

        # Show summary metrics
        st.subheader("Simulation Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration (s)", f"{res['duration']:.1f}")
        init_total, init_surf, init_surf_A, init_ratio = res["initial_surface_data"]
        final_total, final_surf, final_surf_A, final_ratio = res["final_surface_data"]
        col2.metric(f"Initial Surface {element_A} Ratio", f"{init_ratio:.4f}")
        col3.metric(f"Final Surface {element_A} Ratio", f"{final_ratio:.4f}")

        # Plots
        st.subheader("Evolution Plots")
        fig1, ax1 = plt.subplots()
        if not df_log.empty:
            ax1.plot(df_log["Step"], df_log["Energy (eV)"], label="Energy (eV)")
            ax1.set_xlabel("MC Step")
            ax1.set_ylabel("Energy (eV)")
            ax1.grid(True)
            ax1.set_title("Energy vs Step")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(df_log["Step"], df_log[f"Surface {element_A} Ratio"], label=f"Surface {element_A} Ratio", color="orange")
            ax2.set_xlabel("MC Step")
            ax2.set_ylabel(f"Surface {element_A} Ratio")
            ax2.grid(True)
            ax2.set_title(f"Surface {element_A} Ratio vs Step")
            st.pyplot(fig2)
        else:
            st.info("No log entries to plot (check save interval relative to total steps).")

        # Downloads
        st.subheader("Download Artifacts")
        with st.expander("Download Files"):
            def make_download_link(path, label=None):
                label = label or os.path.basename(path)
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(path)}">📥 {label}</a>'
                st.markdown(href, unsafe_allow_html=True)

            make_download_link(res["initial_xyz"], "Initial structure (.xyz)")
            make_download_link(res["final_xyz"], "Final structure (.xyz)")
            make_download_link(res["xlsx_file"], "Simulation log (.xlsx)")

        # Show log table
        st.subheader("Raw Log Data")
        st.dataframe(df_log)

# Optional: Upload existing log or structures for comparison
st.markdown("---")
st.subheader("Upload & Inspect Existing Results")
uploaded_xlsx = st.file_uploader("Upload previous MMC Excel log (.xlsx)", type=["xlsx"])
if uploaded_xlsx:
    try:
        wb = load_workbook(uploaded_xlsx, data_only=True)
        ws = wb.active
        # infer header row (assumes after metadata)
        # naive approach: find row with 'Step' cell
        header_row_idx = None
        for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if row and 'Step' in row:
                header_row_idx = i
                headers = list(row)
                break
        if header_row_idx is None:
            st.error("Could not find header row with 'Step'")
        else:
            data = []
            for row in ws.iter_rows(min_row=header_row_idx+1, values_only=True):
                if all(v is None for v in row):
                    continue
                entry = {headers[j]: row[j] for j in range(len(headers))}
                data.append(entry)
            df_existing = pd.DataFrame(data)
            st.success("Loaded existing log.")
            st.dataframe(df_existing)

            st.subheader("Plots from uploaded log")
            fig_e1, ax_e1 = plt.subplots()
            if 'Energy (eV)' in df_existing.columns:
                ax_e1.plot(df_existing['Step'], df_existing['Energy (eV)'])
                ax_e1.set_title("Energy vs Step")
                ax_e1.set_xlabel("Step")
                ax_e1.set_ylabel("Energy (eV)")
                ax_e1.grid(True)
                st.pyplot(fig_e1)
            if f"Surface {element_A} Ratio" in df_existing.columns:
                fig_e2, ax_e2 = plt.subplots()
                ax_e2.plot(df_existing['Step'], df_existing[f"Surface {element_A} Ratio"])
                ax_e2.set_title(f"Surface {element_A} Ratio vs Step")
                ax_e2.set_xlabel("Step")
                ax_e2.set_ylabel(f"Surface {element_A} Ratio")
                ax_e2.grid(True)
                st.pyplot(fig_e2)

import traceback  # Make sure this is near the top of your file

def target():
    try:
        result_holder.update(run_simulation(params, progress_callback=progress_cb))
    except Exception:
        result_holder["error"] = traceback.format_exc()  # This must be indented under `except`



