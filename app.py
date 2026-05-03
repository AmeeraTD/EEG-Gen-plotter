import streamlit as st
import h5py
import torch
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, stft, welch
from artifact_analyzer import ArtifactAnalyzer

# --- Page Configuration ---
st.set_page_config(page_title="EEG Signal Analyzer", layout="wide")
st.title("🧠 EEG Reconstruction & Frequency Analysis")

FS = 200
FRAME_SIZE = 800
analyzer = ArtifactAnalyzer(fs=FS)

# --- Signal Processing Functions ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if high >= 1: high = 0.99
    if low <= 0: low = 0.001
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def compute_snr_db_numpy(original, reconstruction):
    noise = original - reconstruction
    signal_power = np.mean(np.sum(original**2, axis=-1))
    noise_power = np.mean(np.sum(noise**2, axis=-1))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    return snr

def load_signals(file_obj):
    """Dynamically loads signals from either a .h5 or a .pt file."""
    fname = file_obj.name
    if fname.endswith('.h5'):
        with h5py.File(file_obj, 'r') as f:
            return {k: np.array(f[k][:]) for k in f.keys() if not k.startswith('_')}
    elif fname.endswith('.pt'):
        data = torch.load(file_obj, map_location='cpu')
        if 'eeg_signals' not in data:
            raise ValueError(f"File {fname} is missing the 'eeg_signals' key.")
        signals = data['eeg_signals']
        signals = signals.cpu().numpy() if hasattr(signals, 'cpu') else np.array(signals)
        return {f"trial_{i}": signals[i] for i in range(signals.shape[0])}
    else:
        raise ValueError(f"Unsupported file format: {fname}")

# --- Sidebar: Setup ---
st.sidebar.header("📂 Data Upload")
file1 = st.sidebar.file_uploader("Upload Original signal (.h5 or .pt)", type=["h5", "pt"])
file2 = st.sidebar.file_uploader("Upload Reconstructed signal (.h5 or .pt)", type=["h5", "pt"])

st.sidebar.divider()

with st.sidebar.expander("🔍 Guide: Identifying Artifacts from Graphs", expanded=True):
    st.markdown("""
    Use these guidelines to spot artifacts from plotted data:
    
    1. **Time Domain (High-amplitude > 100 µV):**
        * **Ocular (Eye Blinks):** Sharp, large delta waves (< 4 Hz), especially in frontal channels.
        * **Electrode Pops:** Instantaneous voltage jumps across the whole trial.
        
    2. **STFT Spectrograms:**
        * **Transient Noise:** Sudden vertical bright bursts across all frequencies (indicates muscle/jaw movement).
        * **Continuous Noise:** Horizontal bright lines indicating line noise or high impedance.

    3. **PSD (Power Spectral Density):**
        * **Line Noise:** A sharp, narrow spike at exactly **50 Hz** (or 60 Hz).
        * **EMG Artifacts:** Elevated power at the high-frequency end (> 30 Hz).
    """)

st.sidebar.header("⚡ Processing Settings")
bands = {
    "Full Spectrum (0.5-50Hz)": (0.5, 50), "Delta": (0.5, 4), 
    "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)
}
selected_band_name = st.sidebar.selectbox("Select EEG Band", list(bands.keys()))

if file1 or file2:
    try:
        if file1 and file2:
            signals1 = load_signals(file1)
            signals2 = load_signals(file2)

            keys1 = sorted(list(signals1.keys()))
            keys2 = sorted(list(signals2.keys()))

            st.subheader("📝 File Information and Overview")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"**File 1 (Original):** {file1.name}")
                shape1 = signals1[keys1[0]].shape
                st.caption(f"Shape/Channels: {shape1[0]} Channels, {shape1[1]} Samples")
            with m_col2:
                st.markdown(f"**File 2 (Reconstructed):** {file2.name}")
                shape2 = signals2[keys2[0]].shape
                st.caption(f"Shape/Channels: {shape2[0]} Channels, {shape2[1]} Samples")
            st.divider()

            trial1 = st.sidebar.selectbox("Select Trial from File 1", keys1)
            default_idx2 = keys2.index(trial1) if trial1 in keys2 else 0
            trial2 = st.sidebar.selectbox("Select Trial from File 2", keys2, index=default_idx2)

            ds1 = signals1[trial1]
            ds2 = signals2[trial2]

            num_channels = min(ds1.shape[0], ds2.shape[0])
            selected_ch = st.sidebar.selectbox("Channel", list(range(num_channels)))

            total_samples = min(ds1.shape[1], ds2.shape[1])
            total_frames = int(total_samples // FRAME_SIZE)

            if total_frames > 1:
                current_frame = st.sidebar.number_input(f"Frame (1-{total_frames})", 1, total_frames, 1)
            else:
                current_frame = 1

            start_idx = (current_frame - 1) * FRAME_SIZE
            end_idx = start_idx + FRAME_SIZE

            y1_raw = ds1[selected_ch, start_idx:end_idx]
            y2_raw = ds2[selected_ch, start_idx:end_idx]

            y1_unscaled = (y1_raw / 100.0) if np.max(np.abs(y1_raw)) > 1000 else y1_raw
            y2_unscaled = (y2_raw / 100.0) if np.max(np.abs(y2_raw)) > 1000 else y2_raw

            y1_unscaled = y1_unscaled - np.mean(y1_unscaled)
            y2_unscaled = y2_unscaled - np.mean(y2_unscaled)

            low, high = bands[selected_band_name]
            y1_final_clean = butter_bandpass_filter(y1_unscaled, low, high, FS)
            y2_final_clean = butter_bandpass_filter(y2_unscaled, low, high, FS)

            st.subheader("⚖️ Signal Scaling Configuration")
            scaling_option = st.radio(
                "Select scaling method to compare:",
                ("Auto-Scale (Match Maximum Amplitudes)", "Manual Scale Factors"),
                horizontal=True
            )

            if scaling_option == "Auto-Scale (Match Maximum Amplitudes)":
                max_y1 = np.max(np.abs(y1_final_clean))
                max_y2 = np.max(np.abs(y2_final_clean))
                max_global = max(max_y1, max_y2)
                
                if max_global > 0:
                    y1_final = (y1_final_clean / max_global) * max_global
                    y2_final = (y2_final_clean / max_global) * max_global
                else:
                    y1_final, y2_final = y1_final_clean, y2_final_clean
            else:
                c1, c2 = st.columns(2)
                with c1:
                    scale_1 = st.slider("Signal 1 Multiplier", 0.1, 5.0, 1.0, step=0.1)
                with c2:
                    scale_2 = st.slider("Signal 2 Multiplier", 0.1, 5.0, 1.0, step=0.1)
                    
                y1_final = y1_final_clean * scale_1
                y2_final = y2_final_clean * scale_2

            st.subheader("Time Domain Signal Comparison")
            x = np.arange(start_idx, end_idx)
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=x, y=y1_final, name=f'Signal 1 ({file1.name.split("/")[-1]})', line=dict(color='#2ecc71')))
            fig_time.add_trace(go.Scatter(x=x, y=y2_final, name=f'Signal 2 ({file2.name.split("/")[-1]})', line=dict(color='#e74c3c')))
            fig_time.update_layout(title="Time Domain Signal Comparison", template="plotly_dark", height=400)
            st.plotly_chart(fig_time, use_container_width=True)

            # --- Artifact Analyzer Section ---
            st.divider()
            st.subheader("🤖 Artifact Analysis")
            colA, colB = st.columns(2)
            with colA:
                art1 = analyzer.analyze(y1_final)
                st.markdown(f"#### Signal 1 ({file1.name.split('/')[-1]})")
                st.write(f"**Artifact Type:** {art1['artifact_type']}")
                st.write(f"**Severity:** {art1['severity']}")
                st.write(f"**PTP (Peak-to-Peak):** {art1['ptp']:.2f} µV")
                st.write(f"**Ocular Power (0.5-4 Hz):** {art1['low_freq_power']:.2f} dB/Hz")
                st.write(f"**EMG Power (> 30 Hz):** {art1['emg_power']:.2f} dB/Hz")
                st.write(f"**Line Noise Power (50 Hz):** {art1['line_noise_power']:.2f} dB/Hz")

            with colB:
                art2 = analyzer.analyze(y2_final)
                st.markdown(f"#### Signal 2 ({file2.name.split('/')[-1]})")
                st.write(f"**Artifact Type:** {art2['artifact_type']}")
                st.write(f"**Severity:** {art2['severity']}")
                st.write(f"**PTP (Peak-to-Peak):** {art2['ptp']:.2f} µV")
                st.write(f"**Ocular Power (0.5-4 Hz):** {art2['low_freq_power']:.2f} dB/Hz")
                st.write(f"**EMG Power (> 30 Hz):** {art2['emg_power']:.2f} dB/Hz")
                st.write(f"**Line Noise Power (50 Hz):** {art2['line_noise_power']:.2f} dB/Hz")
            
            # --- End of Artifact Analysis ---

            st.divider()
            st.subheader("📊 Statistical Comparison")
            snr_val = compute_snr_db_numpy(y1_final, y2_final)
            corr = np.corrcoef(y1_final, y2_final)[0, 1]
            mse = np.mean((y1_final - y2_final)**2)

            m1, m2, m3 = st.columns(3)
            m1.metric("SNR (dB)", f"{snr_val:.2f} dB")
            m2.metric("Correlation (R)", f"{corr:.4f}")
            m3.metric("MSE", f"{mse:.6e}")

            st.divider()
            st.subheader("📉 Frequency Domain: STFT Spectrograms")

            def plot_stft(data, title):
                f, t, Zxx = stft(data, fs=FS, nperseg=128)
                fig_stft = go.Figure(data=go.Heatmap(
                    x=t, y=f, z=20 * np.log10(np.abs(Zxx) + 1e-10),
                    colorscale='Viridis', colorbar=dict(title="dB")
                ))
                fig_stft.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Freq (Hz)", height=350, template="plotly_dark")
                return fig_stft

            sc1, sc2 = st.columns(2)
            with sc1: st.plotly_chart(plot_stft(y1_final, f"Signal 1 STFT ({file1.name.split('/')[-1]})"), use_container_width=True)
            with sc2: st.plotly_chart(plot_stft(y2_final, f"Signal 2 STFT ({file2.name.split('/')[-1]})"), use_container_width=True)

            st.divider()
            st.subheader("📈 Power Spectral Density (PSD) Comparison")
            
            f_sig1, psd_sig1 = welch(y1_final, fs=FS, nperseg=256)
            f_sig2, psd_sig2 = welch(y2_final, fs=FS, nperseg=256)

            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(
                x=f_sig1, y=10 * np.log10(psd_sig1 + 1e-12), 
                name=f'Signal 1 PSD ({file1.name.split("/")[-1]})', 
                line=dict(color='#2ecc71', width=2.5)
            ))
            fig_psd.add_trace(go.Scatter(
                x=f_sig2, y=10 * np.log10(psd_sig2 + 1e-12), 
                name=f'Signal 2 PSD ({file2.name.split("/")[-1]})', 
                line=dict(color='#e74c3c', width=2.5)
            ))
            
            fig_psd.update_layout(
                title="PSD Comparison (Welch's Method)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power/Frequency (dB/Hz)",
                template="plotly_dark",
                height=450,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_psd, use_container_width=True)

        else:
            target_file = file1 if file1 else file2
            signals = load_signals(target_file)
            keys = sorted(list(signals.keys()))

            st.subheader("📝 File Information and Overview")
            st.markdown(f"**Signal File:** {target_file.name}")
            shape = signals[keys[0]].shape
            st.caption(f"Shape/Channels: {shape[0]} Channels, {shape[1]} Samples")
            st.divider()

            trial = st.sidebar.selectbox("Select Trial", keys)
            ds = signals[trial]

            num_channels = ds.shape[0]
            selected_ch = st.sidebar.selectbox("Channel", list(range(num_channels)))

            total_samples = ds.shape[1]
            total_frames = int(total_samples // FRAME_SIZE)

            if total_frames > 1:
                current_frame = st.sidebar.number_input(f"Frame (1-{total_frames})", 1, total_frames, 1)
            else:
                current_frame = 1

            start_idx = (current_frame - 1) * FRAME_SIZE
            end_idx = start_idx + FRAME_SIZE

            y_raw = ds[selected_ch, start_idx:end_idx]
            y_unscaled = (y_raw / 100.0) if np.max(np.abs(y_raw)) > 1000 else y_raw
            y_unscaled = y_unscaled - np.mean(y_unscaled)

            low, high = bands[selected_band_name]
            y_final = butter_bandpass_filter(y_unscaled, low, high, FS)

            st.subheader("Time Domain Signal")
            x = np.arange(start_idx, end_idx)
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=x, y=y_final, 
                name=f'Signal ({target_file.name.split("/")[-1]})', 
                line=dict(color='#2ecc71')
            ))
            fig_time.update_layout(
                title=f"Time Domain Signal ({target_file.name.split('/')[-1]})", 
                template="plotly_dark", 
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # --- Single Artifact Analyzer Section ---
            st.divider()
            st.subheader("🤖 Artifact Analysis")
            art_single = analyzer.analyze(y_final)
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.write(f"**Artifact Type:** {art_single['artifact_type']}")
                st.write(f"**Severity:** {art_single['severity']}")
            with s_col2:
                st.write(f"**PTP (Peak-to-Peak):** {art_single['ptp']:.2f} µV")
                st.write(f"**Ocular Power (0.5-4 Hz):** {art_single['low_freq_power']:.2f} dB/Hz")
                st.write(f"**EMG Power (> 30 Hz):** {art_single['emg_power']:.2f} dB/Hz")
                st.write(f"**Line Noise Power (50 Hz):** {art_single['line_noise_power']:.2f} dB/Hz")
            # --- End of Single Artifact Analysis ---

            st.divider()
            st.subheader("📉 Frequency Domain: STFT Spectrogram")

            def plot_stft_single(data, title):
                f, t, Zxx = stft(data, fs=FS, nperseg=128)
                fig_stft = go.Figure(data=go.Heatmap(
                    x=t, y=f, z=20 * np.log10(np.abs(Zxx) + 1e-10),
                    colorscale='Viridis', colorbar=dict(title="dB")
                ))
                fig_stft.update_layout(
                    title=title, 
                    xaxis_title="Time (s)", 
                    yaxis_title="Freq (Hz)", 
                    height=350, 
                    template="plotly_dark"
                )
                return fig_stft

            st.plotly_chart(plot_stft_single(y_final, f"Signal STFT ({target_file.name.split('/')[-1]})"), use_container_width=True)

            st.divider()
            st.subheader("📈 Power Spectral Density (PSD)")

            f_sig, psd_sig = welch(y_final, fs=FS, nperseg=256)

            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(
                x=f_sig, y=10 * np.log10(psd_sig + 1e-12), 
                name=f'Signal PSD ({target_file.name.split("/")[-1]})', 
                line=dict(color='#2ecc71', width=2.5)
            ))
            fig_psd.update_layout(
                title="PSD (Welch's Method)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power/Frequency (dB/Hz)",
                template="plotly_dark",
                height=450
            )
            st.plotly_chart(fig_psd, use_container_width=True)

    except Exception as e:
        st.error(f"Analysis Error: {e}")