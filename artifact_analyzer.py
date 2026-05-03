import numpy as np
from scipy.signal import welch

class ArtifactAnalyzer:
    def __init__(self, fs=200):
        self.fs = fs

    def analyze(self, signal):
        ptp = float(np.ptp(signal))
        mean_val = float(np.mean(signal))
        
        # Estimate Welch's PSD to retrieve the artifact power
        nperseg = min(len(signal) // 2, 256)
        f, psd = welch(signal, fs=self.fs, nperseg=nperseg)
        
        low_freq_power = float(np.mean(psd[(f >= 0.5) & (f <= 4)]))
        emg_power = float(np.mean(psd[f > 30]))
        line_noise_power = float(np.mean(psd[(f >= 49) & (f <= 51)])) if len(psd) > 0 else 0.0

        # Heuristic rules derived from the SEED study
        if ptp > 1000:
            artifact_type = "Definitive Artifact / High Voltage Spike (e.g. Electrode Pop)"
            severity = "Critical"
        elif ptp > 100:
            artifact_type = "Potential Biological / Ocular Artifacts (e.g. Eye Blinks)"
            severity = "Moderate"
        else:
            artifact_type = "Clean Signal"
            severity = "Low"

        return {
            "ptp": ptp,
            "mean": mean_val,
            "artifact_type": artifact_type,
            "severity": severity,
            "low_freq_power": low_freq_power,
            "emg_power": emg_power,
            "line_noise_power": line_noise_power
        }