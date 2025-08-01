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
from ase.visualize.plot import plot_atoms
from ase.data.colors import jmol_colors
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import imageio.v2 as imageio
import stpy3dmol as stmol # NEW: For 3D visualization

st.set_page_config(page_title="Monte Carlo Nanoparticle Simulator", layout="wide")

# --- CORE SIMULATION LOGIC (Functions from Notebook) ---

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
        raise ValueError(f"Unsupported lattice type '{lattice_type}'.")

    with tempfile.TemporaryDirectory() as temp_dir:
        particle = ClusterBuilder(A, surfaces=SURFACES, layers=LAYERS)
        n_atoms = len(particle)
        n_A = int(n_atoms * composition_A)
        indices_A = np.random.choice(range(n_atoms), size=n_A, replace=False)
        for i in range(n_atoms):
            particle[i].symbol = A if i in indices_A else B

        initial_surface_data = count_surface(particle, A)
        initial_xyz_path = os.path.join(temp_dir, f"initial_{A}{B}_{n_atoms}.xyz")
        write(initial_xyz_path, particle)

        traj_folder = os.path.join(temp_dir, "trajectory")
        os.makedirs(traj_folder, exist_ok=True)
        log = []

        energy = calculate_energy(particle, A, coeffs)
        start_time = time.time()

        for step in range(1, N_STEPS + 1):
            i = np.random.randint(0, n_atoms)
            neighbors = build_neighbor_list(particle).get_neighbors(i)[0]
            if not neighbors.any():
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
                    'Step': step, 'Energy (eV)': energy, f'Total {A}': total_A,
                    'Surface Atoms': surf, f'{A} on Surface': surf_A, f'Surface {A} Ratio': ratio
                })
                if progress_callback:
                    progress_callback(step, energy, ratio)
            
            if step % params.get('snapshot_interval', 500) == 0:
                snapshot_file = os.path.join(traj_folder, f"step_{step:05d}.xyz")
                write(snapshot_file, particle)

        final_xyz_path = os.path.join(temp_dir, f"final_{A}{B}_{n_atoms}.xyz")
        write(final_xyz_path, particle)
        
        xlsx_path = os.path.join(temp_dir, f"MMC_{A}{B}_log.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "Simulation Log"

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
        wb.save(xlsx_path)

        # NEW: Read file contents into memory to return
        with open(initial_xyz_path, 'r') as f: initial_xyz_content = f.read()
        with open(final_xyz_path, 'r') as f: final_xyz_content = f.read()
        with open(xlsx_path, 'rb') as f: xlsx_content = f.read()

        # NEW: Return all results, including the final particle object
        return {
            "initial_xyz_content": initial_xyz_content,
            "final_xyz_content": final_xyz_content,
            "log": pd.DataFrame(log),
            "xlsx_content": xlsx_content,
            "duration": time.time() - start_time,
            "initial_surface_data": initial_surface_data,
            "final_surface_data": count_surface(particle, A),
            "final_particle": particle,
            "n_atoms": n_atoms
        }

# --- NEW: 3D Visualization Function ---
def view_xyz_colored(xyz_data, atom_A, atom_B, radius=1.0):
    view = stmol.view(width=500, height=400)
    view.addModel(xyz_data, 'xyz')
    view.setStyle({"elem": atom_A}, {"sphere": {"color": "gold", "radius": radius}})
    view.setStyle({"elem": atom_B}, {"sphere": {"color": "green", "radius": radius}})
    view.setBackgroundColor("white")
    view.zoomTo()
    return view

# --- NEW: Validation Function ---
def run_validation_checks(results, params):
    st.subheader("🔬 Validation & Sanity Checks")
    errors = 0
    
    # 1. Atom count check
    if len(results['final_particle']) != results['n_atoms']:
        st.error(f"❌ Atom count mismatch: expected {results['n_atoms']}, found {len(results['final_particle'])}")
        errors += 1
    else:
        st.success(f"✅ Atom count is correct: {results['n_atoms']}")

    # 2. Final energy NaN check
    final_energy = results['log'].iloc[-1]['Energy (eV)'] if not results['log'].empty else 0
    if math.isnan(final_energy):
        st.error("❌ Final energy is NaN.")
        errors += 1
    else:
        st.success(f"✅ Final energy is valid: {final_energy:.4f} eV")

    # 3. Surface atom sanity
    _, surf_atoms, surf_A, final_ratio = results["final_surface_data"]
    if surf_atoms == 0:
        st.error("❌ No surface atoms detected.")
        errors += 1
    else:
        st.success(f"✅ Surface atoms present: {surf_atoms} total, {surf_A} are {params['element_A']} atoms.")
        st.info(f"➡️ Final Surface {params['element_A']}%: {final_ratio:.4f}")

    # 4. Surface ratio non-trivial
    if final_ratio == 0 or final_ratio == 1:
        st.warning("⚠️ Surface composition is fully segregated. This may or may not be physical.")
    else:
        st.success("✅ Surface composition is mixed.")
        
    if errors == 0:
        st.success("**Validation PASSED: System appears physically consistent.**")
    else:
        st.error(f"**Validation FAILED with {errors} issue(s). Check configuration.**")

# --- UI Definition ---
st.title("🔬 Monte Carlo Nanoparticle Simulator")
st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("Elements & Composition", expanded=True):
    element_A = st.text_input("Element A (e.g., Pt)", value="Pt")
    element_B = st.text_input("Element B (e.g., Ru)", value="Ru")
    composition_A = st.slider("Composition of A (fraction)", 0.0, 1.0, 0.5, step=0.05)

with st.sidebar.expander("Thermodynamics & Steps", expanded=True):
    temperature = st.number_input("Temperature (K)", min_value=1.0, value=250.0, step=1.0)
    n_steps = st.number_input("Monte Carlo Steps", min_value=100, value=10000, step=100)
    save_interval = st.number_input("Log Save Interval", min_value=1, value=200, step=1)
    snapshot_interval = st.number_input("Snapshot Interval (for movie)", min_value=100, value=500, step=100)

with st.sidebar.expander("Lattice / Geometry", expanded=False):
    lattice_type = st.selectbox("Lattice Type", ["fcc", "bcc", "hcp"])
    layers = st.text_input("Layers (x,y,z)", value="[7,7,7]")
    surfaces = st.text_input("Surfaces (list of tuples)", value="[(1,1,1),(1,1,1),(1,1,0)]")

with st.sidebar.expander("Energy Coefficients", expanded=False):
    coeffs = {
        'xA-A': st.number_input("xA-A", value=-0.022078, format="%.6f"),
        'xB-B': st.number_input("xB-B", value=-0.150000, format="%.6f"),
        'xA-B': st.number_input("xA-B", value=-0.109575, format="%.6f"),
        'xA-S': st.number_input("xA-S", value=-0.250717, format="%.6f"),
        'xB-S': st.number_input("xB-S", value=-0.300000, format="%.6f"),
        # The notebook included these but the energy function doesn't use them.
        # Added here for completeness if you extend the function later.
        'xA-A-out': st.number_input("xA-A-out", value=0.184150, format="%.6f"),
        'xB-B-out': st.number_input("xB-B-out", value=0.332228, format="%.6f"),
        'xA-B-out': st.number_input("xA-B-out", value=0.051042, format="%.6f"),
    }

run_button = st.sidebar.button("▶️ Run Simulation")

# --- Simulation Execution and Display ---

if 'results' not in st.session_state:
    st.session_state.results = None

if run_button:
    try:
        layers_parsed = eval(layers)
        surfaces_parsed = eval(surfaces)
    except Exception as e:
        st.sidebar.error(f"Error parsing layers/surfaces: {e}")
        st.stop()
        
    params = {
        'element_A': element_A, 'element_B': element_B, 'composition_A': composition_A,
        'temperature': temperature, 'n_steps': int(n_steps), 'save_interval': int(save_interval),
        'snapshot_interval': int(snapshot_interval), 'layers': tuple(layers_parsed),
        'surfaces': surfaces_parsed, 'coefficients': coeffs, 'lattice_type': lattice_type
    }
    
    status_placeholder = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)

    def progress_cb(step, energy, ratio):
        pct = min(step / params['n_steps'], 1.0)
        progress_bar.progress(pct)
        status_placeholder.markdown(f"**Step:** {step} | **Energy:** {energy:.4f} | **Surface {element_A} Ratio:** {ratio:.4f}")

    try:
        with st.spinner("Simulation running... this can take a while."):
            st.session_state.results = run_simulation(params, progress_callback=progress_cb)
        st.success("✅ Simulation completed.")
    except Exception as e:
        st.error(f"An error occurred during the simulation: {e}")
        st.exception(e)
        st.session_state.results = None

if st.session_state.results:
    res = st.session_state.results
    params = {
        'element_A': element_A, 'element_B': element_B,
    }
    
    st.header("📊 Results")
    st.subheader("Simulation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration (s)", f"{res['duration']:.1f}")
    init_ratio = res["initial_surface_data"][3]
    final_ratio = res["final_surface_data"][3]
    col2.metric(f"Initial Surface {element_A} Ratio", f"{init_ratio:.4f}")
    col3.metric(f"Final Surface {element_A} Ratio", f"{final_ratio:.4f}", delta=f"{final_ratio - init_ratio:.4f}")

    # --- TABS FOR ORGANIZED OUTPUT ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Plots", "🧬 Structures (3D)", "🔬 Validation", "📄 Log Data", "⬇️ Downloads"])

    with tab1:
        st.subheader("Evolution Plots")
        if not res["log"].empty:
            fig1, ax1 = plt.subplots()
            ax1.plot(res["log"]["Step"], res["log"]["Energy (eV)"], label="Energy (eV)")
            ax1.set_xlabel("MC Step"); ax1.set_ylabel("Energy (eV)"); ax1.grid(True); ax1.set_title("Energy vs Step")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(res["log"]["Step"], res["log"][f"Surface {element_A} Ratio"], color="orange")
            ax2.set_xlabel("MC Step"); ax2.set_ylabel(f"Surface {element_A} Ratio"); ax2.grid(True); ax2.set_title(f"Surface {element_A} Ratio vs Step")
            st.pyplot(fig2)
        else:
            st.warning("Log data is empty, cannot generate plots.")

    with tab2: # NEW: 3D Visualization Tab
        st.subheader("Initial vs. Final Structures")
        col_3d1, col_3d2 = st.columns(2)
        with col_3d1:
            st.markdown("**Initial Structure**")
            view = view_xyz_colored(res['initial_xyz_content'], params['element_A'], params['element_B'])
            stmol.showmol(view, height=400)
        with col_3d2:
            st.markdown("**Final Structure**")
            view = view_xyz_colored(res['final_xyz_content'], params['element_A'], params['element_B'])
            stmol.showmol(view, height=400)
            
    with tab3: # NEW: Validation Tab
        run_validation_checks(res, params)
        
    with tab4:
        st.subheader("Raw Log Data")
        st.dataframe(res["log"])

    with tab5:
        st.subheader("Download Artifacts")
        st.download_button("📥 Initial Structure (.xyz)", res['initial_xyz_content'], f"initial_{element_A}{element_B}.xyz")
        st.download_button("📥 Final Structure (.xyz)", res['final_xyz_content'], f"final_{element_A}{element_B}.xyz")
        st.download_button("📥 Simulation Log (.xlsx)", res['xlsx_content'], f"MMC_{element_A}{element_B}_log.xlsx")
