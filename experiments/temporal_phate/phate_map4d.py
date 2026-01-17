import numpy as np
import pandas as pd
import phate
import umap
from scipy.stats import spearmanr
import os
import json
import plotly.graph_objects as go
from tqdm import tqdm

# ===========================================================
# CONFIGURATION
# ===========================================================

# DATA_PATH = "/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilistm_encoder_consistent_temporal/embeddings_kinetics_data_60frames/expanded_embeddings_with_time.parquet"
# DATA_PATH = "/home/earkfeld/Projects/MitoSpace4D/experiments/temporal_phate/kinetics-20frame_4DMS-2024v2_phate_mean-pooled-mean-frame.parquet"
# DATA_PATH = "/home/earkfeld/Projects/MitoSpace4D/experiments/temporal_phate/kinetics-20frames_4DMS-2024v2_phate_mean-pooled_movie-0-only.parquet"
# DATA_PATH = "/home/earkfeld/Projects/MitoSpace4D/experiments/temporal_phate/kinetics-20frames_4DMS-2024v2_phate_mean-pooled_region-0-only.parquet"
# DATA_PATH = "/home/earkfeld/Projects/MitoSpace4D/experiments/temporal_phate/2024v2-val_4DMS-2024v2_phate_mean-pooled.parquet"

DATA_PATH = ""

# OUTPUT_DIR = "/home/dhruvagarwal/projects/MitoSpace4D/temporal_experiments/3d_trajectories_kinetics_one_region"
# OUTPUT_DIR = "/home/earkfeld/Projects/MitoSpace4D/experiments/temporal_phate/2e_kinetics-20frame_4DMS-2024v2_phate_mean-pooled_movie-0-only"
# OUTPUT_DIR = "/home/earkfeld/Projects/MitoSpace4D/experiments/temporal_phate/2f_kinetics-20frame_4DMS-2024v2_phate-4D_mean-pooled-region-0-only"
# OUTPUT_DIR = "/home/earkfeld/Projects/MitoSpace4D/experiments/temporal_phate/3a_2024v2-val_4DMS-2024v2_phate-4d_mean-pooled"

PHATE_CONFIG = {"n_components": 3, "knn": 30, "decay": 40, "t": "auto", "random_state": 0}
# PHATE_CONFIG = {"n_components": 3, "knn": 30, "decay": 40, "t": 10, "random_state": 0}
UMAP_CONFIG = {"n_components": 3, "n_neighbors": 30, "min_dist": 0.1, "random_state": 0}

# Drug colors (R, G, B) - normalized to 0-1
COLOR_DATA = """
20240729 control 0 240 235 168
20240730 p110 1 0 0 128
20240731 myls22 2 139 187 214
20240801 mfi8 3 255 127 124
20240802 tbhp 4 254 206 238
20240805 h2o2 5 112 128 144
20240806 mitoq 6 167 215 190
20240807 resveratrol 7 152 255 152
20240808 lonidamine 8 50 205 50
20240809 oligomycin 9 0 128 0
20240813 dnp 10 32 120 39
20240814 valinomycin 11 128 0 0
20240815 cccp 12 255 192 203
20240816 mitomycinc 13 0 255 255
20240820 cytochalasind 14 0 255 0
20240821 lantrunculinb 15 255 165 0
20240823 mdivi1 16 128 128 0
20240826 nocodazole 17 255 0 0
20240830 colchicine 18 255 215 0
20240903 antimycina 19 128 0 128
20240904 tiron 20 88 57 130
20240905 cisplatin 21 255 255 0
20240910 rotenone 22 255 0 255
20240911 nigericin 23 221 160 221
20240912 azide 24 255 105 180
20240913 paraquat 25 64 224 208
20240917 metformin 26 0 0 255
"""

# Modern dark theme
THEME = {
    "bg_color": "#0d1117",
    "paper_color": "#161b22",
    "grid_color": "#30363d",
    "text_color": "#c9d1d9",
    "accent": "#58a6ff",
    "border_color": "#30363d",
}


# ===========================================================
# PARSE COLORS
# ===========================================================

def parse_colors(color_data):
    color_dict = {}
    for line in color_data.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        drug_name = parts[1]
        r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
        color_dict[drug_name] = (r / 255, g / 255, b / 255)
    # Add alias for latrinculinb -> lantrunculinb
    if "lantrunculinb" in color_dict:
        color_dict["latrinculinb"] = color_dict["lantrunculinb"]
    return color_dict


COLOR_DICT = parse_colors(COLOR_DATA)

# ===========================================================
# LOAD DATA
# ===========================================================

print("📂 Loading DataFrame...")
df = pd.read_parquet(DATA_PATH, engine="pyarrow")

X = np.stack(df["embedding"].values)
print(f"✓ Embeddings shape: {X.shape}")

if "drug" not in df.columns or "time" not in df.columns:
    raise ValueError("DataFrame must have 'drug' and 'time' columns")

meta = df[["drug", "time"]].copy()

# Match colors to drugs in data
drug_colors = {}
for drug in meta["drug"].unique():
    if drug in COLOR_DICT:
        drug_colors[drug] = COLOR_DICT[drug]
    else:
        # Generate a random color for unknown drugs
        np.random.seed(hash(drug) % 2 ** 32)
        drug_colors[drug] = tuple(np.random.rand(3))
        print(f"⚠ No color for '{drug}', using random color")

print(f"✓ Loaded colors for {len(drug_colors)} drugs")

# ===========================================================
# COMPUTE PHATE & UMAP
# ===========================================================

print("\n🔮 Running PHATE (3D)...")
ph = phate.PHATE(**PHATE_CONFIG)
X_phate = ph.fit_transform(X)
meta["PHATE1"] = X_phate[:, 0]
meta["PHATE2"] = X_phate[:, 1]
meta["PHATE3"] = X_phate[:, 2]
print("✓ PHATE complete")

print("\n🗺️ Running UMAP (3D)...")
um = umap.UMAP(**UMAP_CONFIG)
X_umap = um.fit_transform(X)
meta["UMAP1"] = X_umap[:, 0]
meta["UMAP2"] = X_umap[:, 1]
meta["UMAP3"] = X_umap[:, 2]
print("✓ UMAP complete")


# ===========================================================
# HELPER FUNCTIONS
# ===========================================================


def create_shaded_colors(base_color, num_t):
    """Create shades from light to dark for time progression"""
    if num_t == 1:
        return [base_color]
    base = np.array(base_color)
    return [tuple(base * (0.35 + 0.65 * (i / (num_t - 1)))) for i in range(num_t)]


def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1) to hex string"""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def get_scene_layout(embx, emby, embz):
    """Get consistent 3D scene layout with dark theme"""
    return dict(
        xaxis=dict(
            title=dict(text=embx, font=dict(size=14, color=THEME["text_color"])),
            backgroundcolor=THEME["bg_color"],
            gridcolor=THEME["grid_color"],
            showbackground=True,
            zerolinecolor=THEME["grid_color"],
            tickfont=dict(color=THEME["text_color"]),
        ),
        yaxis=dict(
            title=dict(text=emby, font=dict(size=14, color=THEME["text_color"])),
            backgroundcolor=THEME["bg_color"],
            gridcolor=THEME["grid_color"],
            showbackground=True,
            zerolinecolor=THEME["grid_color"],
            tickfont=dict(color=THEME["text_color"]),
        ),
        zaxis=dict(
            title=dict(text=embz, font=dict(size=14, color=THEME["text_color"])),
            backgroundcolor=THEME["bg_color"],
            gridcolor=THEME["grid_color"],
            showbackground=True,
            zerolinecolor=THEME["grid_color"],
            tickfont=dict(color=THEME["text_color"]),
        ),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        aspectmode="cube",
    )


# ===========================================================
# SINGLE DRUG 3D TRAJECTORY
# ===========================================================


def plot_3d_trajectory(meta, embx, emby, embz, drug_name, base_color, prefix, output_dir):
    """Create 3D trajectory plot for a single drug with modern styling and animation"""
    os.makedirs(output_dir, exist_ok=True)

    subset = meta[meta["drug"] == drug_name]
    centroids = subset.groupby("time")[[embx, emby, embz]].mean().sort_index()
    xs, ys, zs = centroids[embx].to_numpy(), centroids[emby].to_numpy(), centroids[embz].to_numpy()
    time_vals = centroids.index.to_numpy()

    num_t = len(time_vals)
    shades = create_shaded_colors(base_color, num_t)
    shade_hex = [rgb_to_hex(shade) for shade in shades]

    avg_range = np.mean([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()])
    cone_size_ref = avg_range * 0.035  # Reduced from 0.08

    fig = go.Figure()

    # --- STATIC LAYERS (Background Context) ---

    # Draw the full trajectory as a single smooth line first
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(
                color=shade_hex[num_t // 2],  # Use middle shade
                width=4,
            ),
            opacity=0.3,  # Dimmed for background
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add gradient-colored segments on top (thinner, for color effect)
    for i in range(num_t - 1):
        fig.add_trace(
            go.Scatter3d(
                x=xs[i: i + 2],
                y=ys[i: i + 2],
                z=zs[i: i + 2],
                mode="lines",
                line=dict(color=shade_hex[i], width=3),
                showlegend=False,
                name=f"t={time_vals[i]:.2f}",
                opacity=0.4,  # Dimmed for background
            )
        )

    # Only add arrows every 3rd segment to reduce clutter
    arrow_interval = max(1, (num_t - 1) // 4)  # ~4 arrows total
    for i in range(0, num_t - 1, arrow_interval):
        dx, dy, dz = xs[i + 1] - xs[i], ys[i + 1] - ys[i], zs[i + 1] - zs[i]
        vec_len = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if vec_len > 0:
            u, v, w = dx / vec_len, dy / vec_len, dz / vec_len
            cone_pos = 0.7
            fig.add_trace(
                go.Cone(
                    x=[xs[i] + cone_pos * dx],
                    y=[ys[i] + cone_pos * dy],
                    z=[zs[i] + cone_pos * dz],
                    u=[u],
                    v=[v],
                    w=[w],
                    colorscale=[[0, shade_hex[i]], [1, shade_hex[i]]],
                    sizemode="absolute",
                    sizeref=cone_size_ref,
                    showscale=False,
                    hoverinfo="skip",
                    opacity=0.4,  # Dimmed for background
                )
            )

    # Time points with gradient (smaller markers)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                size=8,
                color=time_vals,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Time", font=dict(color=THEME["text_color"])),
                    tickfont=dict(color=THEME["text_color"]),
                    bgcolor=THEME["paper_color"],
                    bordercolor=THEME["border_color"],
                    borderwidth=1,
                    len=0.6,
                    x=1.02,
                ),
                line=dict(width=1, color="white"),
                opacity=0.4  # Dimmed for background
            ),
            name="Time points",
            hovertemplate="<b>Time:</b> %{text:.2f}<br><b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><b>Z:</b> %{z:.3f}<extra></extra>",
            text=time_vals,
        )
    )

    # Start and end markers
    fig.add_trace(
        go.Scatter3d(
            x=[xs[0]],
            y=[ys[0]],
            z=[zs[0]],
            mode="markers+text",
            marker=dict(size=16, color="#2ea043", symbol="diamond", line=dict(width=2, color="white")),
            text=["START"],
            textposition="top center",
            textfont=dict(size=12, color="#2ea043"),
            showlegend=False,
            hovertemplate="<b>START</b><br>Time: %{customdata:.2f}<extra></extra>",
            customdata=[time_vals[0]],
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[xs[-1]],
            y=[ys[-1]],
            z=[zs[-1]],
            mode="markers+text",
            marker=dict(size=16, color="#f85149", symbol="diamond", line=dict(width=2, color="white")),
            text=["END"],
            textposition="top center",
            textfont=dict(size=12, color="#f85149"),
            showlegend=False,
            hovertemplate="<b>END</b><br>Time: %{customdata:.2f}<extra></extra>",
            customdata=[time_vals[-1]],
        )
    )

    # --- ANIMATION LAYERS ---

    # Calculate index where animation traces start
    anim_trace_start_idx = len(fig.data)

    # 1. Growing Line (White/Bright)
    fig.add_trace(
        go.Scatter3d(
            x=[xs[0]],
            y=[ys[0]],
            z=[zs[0]],
            mode="lines",
            line=dict(color="white", width=5),
            name="Evolution",
            showlegend=False,
            opacity=0.9
        )
    )

    # 2. Leading Marker (Head)
    fig.add_trace(
        go.Scatter3d(
            x=[xs[0]],
            y=[ys[0]],
            z=[zs[0]],
            mode="markers",
            marker=dict(size=12, color="white", line=dict(width=2, color=rgb_to_hex(base_color))),
            name="Current Time",
            showlegend=False
        )
    )

    # Create Frames
    frames = []
    for k in range(num_t):
        frames.append(go.Frame(
            data=[
                # Update Growing Line
                go.Scatter3d(x=xs[:k + 1], y=ys[:k + 1], z=zs[:k + 1]),
                # Update Head Marker
                go.Scatter3d(x=[xs[k]], y=[ys[k]], z=[zs[k]])
            ],
            traces=[anim_trace_start_idx, anim_trace_start_idx + 1],
            name=f"frame_{k}"
        ))

    fig.update(frames=frames)

    # Layout with Play Buttons and Slider
    fig.update_layout(
        title=dict(
            text=f"<b>{prefix.upper()}</b> 3D Trajectory: <span style='color:{rgb_to_hex(base_color)}'>{drug_name}</span>",
            font=dict(size=22, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        scene=get_scene_layout(embx, emby, embz),
        paper_bgcolor=THEME["paper_color"],
        plot_bgcolor=THEME["bg_color"],
        width=1100,
        height=850,
        margin=dict(l=20, r=20, t=80, b=100),  # Increased bottom margin for controls
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.1,  # Move buttons below
            x=0.1,
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            bgcolor=THEME["paper_color"],
            bordercolor=THEME["border_color"],
            font=dict(color=THEME["text_color"]),
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="❚❚ Pause",
                     method="animate",
                     args=[[None],
                           dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=14, color=THEME["text_color"]),
                prefix="Time: ",
                visible=True,
                xanchor="right"
            ),
            transition=dict(duration=100, easing="cubic-in-out"),
            pad=dict(b=10, t=0),
            len=0.8,
            x=0.15,
            y=-0.1,  # Move slider below
            font=dict(color=THEME["text_color"]),
            steps=[dict(
                args=[[f"frame_{k}"],
                      dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                label=f"{time_vals[k]:.1f}",
                method="animate"
            ) for k in range(num_t)]
        )]
    )

    fname = os.path.join(output_dir, f"{prefix}_trajectory_3d_{drug_name}.html")
    fig.write_html(fname, include_plotlyjs="cdn")
    return fig


# ===========================================================
# COMBINED 3D TRAJECTORIES WITH ENHANCED UI
# ===========================================================


def plot_3d_trajectories_combined(meta, embx, emby, embz, drug_colors, prefix, output_dir):
    """Create combined 3D trajectory plot with animation and modern UI controls"""
    os.makedirs(output_dir, exist_ok=True)

    fig = go.Figure()
    drug_trace_indices = {}
    current_trace_idx = 0

    # Store pre-calculated data for animation
    drug_data_cache = {}
    all_time_vals = set()

    # ---------------------------------------------------------
    # 1. SETUP TRACES (Ghost, Growth, Head)
    # ---------------------------------------------------------
    for drug_name, base_color in drug_colors.items():
        drug_trace_indices[drug_name] = []

        subset = meta[meta["drug"] == drug_name]
        centroids = subset.groupby("time")[[embx, emby, embz]].mean().sort_index()
        xs, ys, zs = centroids[embx].to_numpy(), centroids[emby].to_numpy(), centroids[embz].to_numpy()
        time_vals = centroids.index.to_numpy()

        for t in time_vals:
            all_time_vals.add(t)

        drug_data_cache[drug_name] = {
            "x": xs, "y": ys, "z": zs, "t": time_vals, "color": rgb_to_hex(base_color)
        }

        # TRACE 1: Ghost Line (Full Static Background)
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=rgb_to_hex(base_color), width=2),
                opacity=0.2,  # Very faint
                showlegend=True,  # Legend entry for visibility toggling
                name=drug_name,
                legendgroup=drug_name,
                hoverinfo="skip"
            )
        )
        drug_trace_indices[drug_name].append(current_trace_idx)  # Index for JS
        current_trace_idx += 1

        # TRACE 2: Growing Line (Animation)
        # Initial state: just first point
        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]],
                mode="lines",
                line=dict(color=rgb_to_hex(base_color), width=5),
                opacity=1.0,
                showlegend=False,
                legendgroup=drug_name,
                hoverinfo="skip"
            )
        )
        drug_trace_indices[drug_name].append(current_trace_idx)
        current_trace_idx += 1

        # TRACE 3: Head Marker (Animation)
        # Initial state: first point
        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]],
                mode="markers",
                marker=dict(
                    size=6,
                    color=rgb_to_hex(base_color),
                    line=dict(width=1, color="white"),
                    symbol="diamond"
                ),
                showlegend=False,
                legendgroup=drug_name,
                hovertemplate=f"<b>{drug_name}</b><br>Time: %{{customdata:.2f}}<extra></extra>",
                customdata=[time_vals[0]]
            )
        )
        drug_trace_indices[drug_name].append(current_trace_idx)
        current_trace_idx += 1

    # ---------------------------------------------------------
    # 2. CREATE ANIMATION FRAMES
    # ---------------------------------------------------------
    sorted_times = sorted(list(all_time_vals))
    frames = []

    # Identify indices for dynamic traces (Growth and Head)

    for k, t_curr in enumerate(sorted_times):
        frame_data = []
        frame_traces = []

        trace_counter = 0
        for drug_name in drug_colors.keys():
            # Skip Ghost (index 0 for this drug)
            trace_counter += 1  # pointing to Growth

            dd = drug_data_cache[drug_name]

            # Simple approach: Mask for times <= t_curr
            mask = dd["t"] <= t_curr
            if not np.any(mask):
                # Before start: just show first point
                curr_x, curr_y, curr_z = [dd["x"][0]], [dd["y"][0]], [dd["z"][0]]
                head_x, head_y, head_z = [dd["x"][0]], [dd["y"][0]], [dd["z"][0]]
                head_t = dd["t"][0]
            else:
                curr_x = dd["x"][mask]
                curr_y = dd["y"][mask]
                curr_z = dd["z"][mask]
                head_x = [curr_x[-1]]
                head_y = [curr_y[-1]]
                head_z = [curr_z[-1]]
                head_t = dd["t"][mask][-1]

            # Update Growing Line
            frame_data.append(go.Scatter3d(x=curr_x, y=curr_y, z=curr_z))
            frame_traces.append(trace_counter)
            trace_counter += 1  # pointing to Head

            # Update Head Marker
            frame_data.append(go.Scatter3d(x=head_x, y=head_y, z=head_z, customdata=[head_t]))
            frame_traces.append(trace_counter)
            trace_counter += 1  # pointing to next drug Ghost

        frames.append(go.Frame(
            data=frame_data,
            traces=frame_traces,
            name=f"frame_{k}"
        ))

    fig.update(frames=frames)

    # ---------------------------------------------------------
    # 3. LAYOUT WITH CONTROLS
    # ---------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f"<b>{prefix.upper()}</b> 3D Trajectories — All Drugs",
            font=dict(size=24, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        scene=get_scene_layout(embx, emby, embz),
        paper_bgcolor=THEME["paper_color"],
        plot_bgcolor=THEME["bg_color"],
        width=1300,
        height=900,
        # KEY FIX: Increase left margin to 260px (panel is 220px)
        # KEY FIX: Increase bottom margin to 120px for controls
        margin=dict(l=260, r=50, t=80, b=120),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.05,  # Push legend further right
            bgcolor="rgba(22,27,34,0.95)",
            bordercolor=THEME["border_color"],
            borderwidth=1,
            font=dict(color=THEME["text_color"], size=12),
            itemsizing="constant",
        ),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            # KEY FIX: Move buttons to bottom margin
            y=-0.05,
            x=0.0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=0, r=10),
            bgcolor=THEME["paper_color"],
            bordercolor=THEME["border_color"],
            font=dict(color=THEME["text_color"]),
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="❚❚ Pause",
                     method="animate",
                     args=[[None],
                           dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=14, color=THEME["text_color"]),
                prefix="Time: ",
                visible=True,
                xanchor="right"
            ),
            transition=dict(duration=150, easing="cubic-in-out"),
            pad=dict(b=10, t=0),
            len=0.8,
            # KEY FIX: Move slider to bottom margin next to buttons
            x=0.15,
            y=-0.05,
            font=dict(color=THEME["text_color"]),
            steps=[dict(
                args=[[f"frame_{k}"],
                      dict(mode="immediate", frame=dict(duration=150, redraw=True), transition=dict(duration=0))],
                label=f"{t:.1f}",
                method="animate"
            ) for k, t in enumerate(sorted_times)]
        )]
    )

    fname = os.path.join(output_dir, f"{prefix}_trajectories_3d_all_drugs.html")
    fig.write_html(fname, include_plotlyjs="cdn")

    # Inject enhanced UI
    with open(fname, "r", encoding="utf-8") as f:
        html_content = f.read()

    trace_indices_js = json.dumps(drug_trace_indices)
    drug_names_js = json.dumps(list(drug_colors.keys()))
    drug_colors_js = json.dumps({k: rgb_to_hex(v) for k, v in drug_colors.items()})

    enhanced_ui = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        * {{ box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: {THEME["bg_color"]};
            margin: 0;
            padding: 0;
        }}

        #control-panel {{
            position: fixed;
            top: 80px;
            left: 16px;
            width: 220px;
            background: linear-gradient(180deg, {THEME["paper_color"]} 0%, rgba(22,27,34,0.98) 100%);
            border: 1px solid {THEME["border_color"]};
            border-radius: 12px;
            padding: 0;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            backdrop-filter: blur(10px);
            overflow: hidden;
            max-height: calc(100vh - 100px);
            display: flex;
            flex-direction: column;
        }}

        .panel-header {{
            background: linear-gradient(90deg, {THEME["accent"]}22, transparent);
            padding: 14px 16px;
            border-bottom: 1px solid {THEME["border_color"]};
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .panel-header svg {{
            width: 18px;
            height: 18px;
            color: {THEME["accent"]};
        }}

        .panel-title {{
            font-size: 13px;
            font-weight: 600;
            color: {THEME["text_color"]};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .panel-actions {{
            padding: 12px 16px;
            border-bottom: 1px solid {THEME["border_color"]};
            display: flex;
            gap: 8px;
        }}

        .action-btn {{
            flex: 1;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 500;
            border: 1px solid {THEME["border_color"]};
            border-radius: 6px;
            background: transparent;
            color: {THEME["text_color"]};
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .action-btn:hover {{
            background: {THEME["accent"]}22;
            border-color: {THEME["accent"]};
            color: {THEME["accent"]};
        }}

        .action-btn.active {{
            background: {THEME["accent"]};
            border-color: {THEME["accent"]};
            color: white;
        }}

        .drug-list {{
            flex: 1;
            overflow-y: auto;
            padding: 8px 0;
        }}

        .drug-list::-webkit-scrollbar {{
            width: 6px;
        }}

        .drug-list::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .drug-list::-webkit-scrollbar-thumb {{
            background: {THEME["border_color"]};
            border-radius: 3px;
        }}

        .drug-item {{
            display: flex;
            align-items: center;
            padding: 10px 16px;
            cursor: pointer;
            transition: all 0.15s ease;
            gap: 12px;
        }}

        .drug-item:hover {{
            background: rgba(255,255,255,0.05);
        }}

        .drug-item.hidden {{
            opacity: 0.4;
        }}

        .color-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
            border: 2px solid rgba(255,255,255,0.2);
            transition: transform 0.15s ease;
        }}

        .drug-item:hover .color-dot {{
            transform: scale(1.2);
        }}

        .drug-name {{
            font-size: 13px;
            color: {THEME["text_color"]};
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .visibility-icon {{
            width: 18px;
            height: 18px;
            color: {THEME["text_color"]};
            opacity: 0.6;
            transition: opacity 0.15s ease;
        }}

        .drug-item:hover .visibility-icon {{
            opacity: 1;
        }}

        .drug-item.hidden .visibility-icon {{
            opacity: 0.3;
        }}

        .search-box {{
            padding: 12px 16px;
            border-bottom: 1px solid {THEME["border_color"]};
        }}

        .search-input {{
            width: 100%;
            padding: 8px 12px;
            font-size: 12px;
            border: 1px solid {THEME["border_color"]};
            border-radius: 6px;
            background: rgba(0,0,0,0.3);
            color: {THEME["text_color"]};
            outline: none;
            transition: border-color 0.2s ease;
        }}

        .search-input:focus {{
            border-color: {THEME["accent"]};
        }}

        .search-input::placeholder {{
            color: rgba(201,209,217,0.5);
        }}

        .stats-bar {{
            padding: 10px 16px;
            background: rgba(0,0,0,0.2);
            font-size: 11px;
            color: rgba(201,209,217,0.7);
            text-align: center;
            border-top: 1px solid {THEME["border_color"]};
        }}
    </style>

    <div id="control-panel">
        <div class="panel-header">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span class="panel-title">Drug Trajectories</span>
        </div>

        <div class="search-box">
            <input type="text" class="search-input" id="drug-search" placeholder="🔍 Search drugs...">
        </div>

        <div class="panel-actions">
            <button class="action-btn active" id="btn-all">All</button>
            <button class="action-btn" id="btn-none">None</button>
            <button class="action-btn" id="btn-invert">Invert</button>
        </div>

        <div class="drug-list" id="drug-list"></div>

        <div class="stats-bar" id="stats-bar">Loading...</div>
    </div>

    <script>
    (function() {{
        const drugTraceIndices = {trace_indices_js};
        const drugNames = {drug_names_js};
        const drugColors = {drug_colors_js};
        let plotDiv = null;
        let originalData = null;
        let originalLayout = null;
        const drugVisibility = {{}};

        drugNames.forEach(d => drugVisibility[d] = true);

        function getPlotDiv() {{
            const divs = document.querySelectorAll('div');
            for (const div of divs) {{
                if (div.data && Array.isArray(div.data) && div.data.length > 0) return div;
                if (div.layout && div._fullLayout) return div;
            }}
            return document.querySelector('.js-plotly-plot');
        }}

        function updatePlot() {{
            if (!plotDiv || !originalData) return;

            const updatedData = JSON.parse(JSON.stringify(originalData));

            drugNames.forEach(drugName => {{
                const indices = drugTraceIndices[drugName];
                const isVisible = drugVisibility[drugName];
                if (indices) {{
                    indices.forEach(idx => {{
                        if (idx < updatedData.length) {{
                            // Trace 0 (Ghost) is 'legendonly' if hidden to keep legend clean?
                            // Actually, let's just use boolean visibility. 
                            // Note: Plotly animation might reset this if not careful, 
                            // but react() usually handles it.
                            updatedData[idx].visible = isVisible ? true : false;
                        }}
                    }});
                }}
            }});

            Plotly.react(plotDiv, updatedData, originalLayout, {{responsive: true}});
            updateStats();
        }}

        function updateStats() {{
            const visible = drugNames.filter(d => drugVisibility[d]).length;
            document.getElementById('stats-bar').textContent = `${{visible}} of ${{drugNames.length}} drugs visible`;
        }}

        function toggleDrug(drugName) {{
            drugVisibility[drugName] = !drugVisibility[drugName];
            updateDrugItem(drugName);
            updatePlot();
        }}

        function updateDrugItem(drugName) {{
            const item = document.querySelector(`[data-drug="${{drugName}}"]`);
            if (item) {{
                item.classList.toggle('hidden', !drugVisibility[drugName]);
                const icon = item.querySelector('.visibility-icon');
                if (icon) {{
                    icon.innerHTML = drugVisibility[drugName]
                        ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />'
                        : '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />';
                }}
            }}
        }}

        function setAll(visible) {{
            drugNames.forEach(d => {{
                drugVisibility[d] = visible;
                updateDrugItem(d);
            }});
            updatePlot();
        }}

        function invertAll() {{
            drugNames.forEach(d => {{
                drugVisibility[d] = !drugVisibility[d];
                updateDrugItem(d);
            }});
            updatePlot();
        }}

        function filterDrugs(query) {{
            const q = query.toLowerCase();
            document.querySelectorAll('.drug-item').forEach(item => {{
                const name = item.getAttribute('data-drug').toLowerCase();
                item.style.display = name.includes(q) ? 'flex' : 'none';
            }});
        }}

        function createUI() {{
            const list = document.getElementById('drug-list');

            drugNames.forEach(drugName => {{
                const item = document.createElement('div');
                item.className = 'drug-item';
                item.setAttribute('data-drug', drugName);
                item.innerHTML = `
                    <div class="color-dot" style="background: ${{drugColors[drugName]}}"></div>
                    <span class="drug-name">${{drugName}}</span>
                    <svg class="visibility-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                `;
                item.addEventListener('click', () => toggleDrug(drugName));
                list.appendChild(item);
            }});

            document.getElementById('btn-all').addEventListener('click', () => setAll(true));
            document.getElementById('btn-none').addEventListener('click', () => setAll(false));
            document.getElementById('btn-invert').addEventListener('click', invertAll);
            document.getElementById('drug-search').addEventListener('input', e => filterDrugs(e.target.value));

            updateStats();
        }}

        function init() {{
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', init);
                return;
            }}

            plotDiv = getPlotDiv();
            if (!plotDiv || !plotDiv.data) {{
                setTimeout(init, 100);
                return;
            }}

            originalData = JSON.parse(JSON.stringify(plotDiv.data));
            originalLayout = JSON.parse(JSON.stringify(plotDiv.layout || plotDiv._fullLayout || {{}}));

            createUI();
        }}

        init();
    }})();
    </script>
    """

    if "</body>" in html_content:
        html_content = html_content.replace("</body>", enhanced_ui + "</body>")

    with open(fname, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  ✓ Saved: {fname}")


# ===========================================================
# GENERATE ALL PLOTS
# ===========================================================

print("\n🎨 Generating 3D trajectory plots...")

for embx, emby, embz, prefix in [("PHATE1", "PHATE2", "PHATE3", "phate"), ("UMAP1", "UMAP2", "UMAP3", "umap")]:
    print(f"\n  [{prefix.upper()}]")

    # Individual trajectories
    for drug_name, base_color in tqdm(drug_colors.items(), desc=f"  Individual ({prefix})", leave=False):
        plot_3d_trajectory(meta, embx, emby, embz, drug_name, base_color, prefix, OUTPUT_DIR)

    # Combined plot
    plot_3d_trajectories_combined(meta, embx, emby, embz, drug_colors, prefix, OUTPUT_DIR)

# ===========================================================
# TEMPORAL METRICS
# ===========================================================

print("\n📊 Computing temporal preservation metrics...")

results = []
for drug_name in tqdm(meta["drug"].unique(), desc="  Computing Spearman ρ"):
    g = meta[meta["drug"] == drug_name]
    rho_ph, _ = spearmanr(g["time"], g["PHATE1"])
    rho_um, _ = spearmanr(g["time"], g["UMAP1"])
    results.append({"drug": drug_name, "rho_PHATE_time": rho_ph, "rho_UMAP_time": rho_um})

df_results = pd.DataFrame(results)
output_csv = os.path.join(OUTPUT_DIR, "temporal_kinetics_correlation_3d.csv")
df_results.to_csv(output_csv, index=False)

print("\n✅ COMPLETE!")
print(f"\n📁 Output directory: {OUTPUT_DIR}/")
print("   • Individual drug trajectories: {prefix}_trajectory_3d_{drug_name}.html")
print("   • Combined trajectories: {prefix}_trajectories_3d_all_drugs.html")
print(f"   • Correlation metrics: {output_csv}")
print(f"\n{df_results.to_string(index=False)}")