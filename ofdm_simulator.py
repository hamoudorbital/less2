import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import pandas as pd

st.set_page_config(page_title="OFDM Simulator", layout="wide", page_icon="📶")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📶 OFDM Complete Simulator</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("OFDM Explorer")
app_mode = st.sidebar.selectbox(
    "Choose Module",
    [
        "🎯 Complete OFDM Chain",
        "📊 Orthogonality Demo", 
        "🔄 Cyclic Prefix Impact",
        "📡 Channel Effects",
        "⚡ PAPR Analysis",
        "🎚️ Subcarrier Mapping"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About OFDM")
st.sidebar.info(
    "**OFDM (Orthogonal Frequency Division Multiplexing)** is the foundation of 5G NR, LTE, WiFi 6, and more. "
    "This simulator lets you explore how OFDM works from the ground up!"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_qam_symbols(num_symbols, modulation_order):
    """Generate random QAM symbols"""
    bits_per_symbol = int(np.log2(modulation_order))
    total_bits = num_symbols * bits_per_symbol
    bits = np.random.randint(0, 2, total_bits)
    
    # Simple QAM mapping
    if modulation_order == 4:  # QPSK
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif modulation_order == 16:  # 16-QAM
        I_values = [-3, -1, 1, 3]
        Q_values = [-3, -1, 1, 3]
        constellation = np.array([i + 1j*q for q in Q_values for i in I_values]) / np.sqrt(10)
    elif modulation_order == 64:  # 64-QAM
        I_values = [-7, -5, -3, -1, 1, 3, 5, 7]
        Q_values = [-7, -5, -3, -1, 1, 3, 5, 7]
        constellation = np.array([i + 1j*q for q in Q_values for i in I_values]) / np.sqrt(42)
    else:  # BPSK
        constellation = np.array([1+0j, -1+0j])
    
    symbols = []
    for i in range(0, total_bits, bits_per_symbol):
        bits_chunk = bits[i:i+bits_per_symbol]
        index = int(''.join(map(str, bits_chunk)), 2)
        symbols.append(constellation[index])
    
    return np.array(symbols)

def add_cyclic_prefix(ofdm_symbols, cp_length):
    """Add cyclic prefix to OFDM symbols"""
    if cp_length == 0:
        return ofdm_symbols
    return np.concatenate([ofdm_symbols[-cp_length:], ofdm_symbols])

def remove_cyclic_prefix(received_symbols, cp_length):
    """Remove cyclic prefix from received OFDM symbols"""
    if cp_length == 0:
        return received_symbols
    return received_symbols[cp_length:]

def multipath_channel(signal, delays, gains):
    """Simulate multipath channel"""
    output = np.zeros(len(signal) + max(delays), dtype=complex)
    for delay, gain in zip(delays, gains):
        output[delay:delay+len(signal)] += gain * signal
    return output[:len(signal)]

def awgn_channel(signal, snr_db):
    """Add AWGN to signal"""
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def calculate_ber(transmitted_bits, received_bits):
    """Calculate Bit Error Rate"""
    errors = np.sum(transmitted_bits != received_bits)
    return errors / len(transmitted_bits)

def calculate_papr(signal):
    """Calculate Peak-to-Average Power Ratio"""
    peak_power = np.max(np.abs(signal)**2)
    avg_power = np.mean(np.abs(signal)**2)
    return 10 * np.log10(peak_power / avg_power)

# ============================================================================
# 1. COMPLETE OFDM CHAIN
# ============================================================================
if app_mode == "🎯 Complete OFDM Chain":
    st.markdown('<p class="section-header">🎯 Complete OFDM Transceiver Chain</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### OFDM Parameters")
        
        N_fft = st.selectbox("FFT Size (N)", [64, 128, 256, 512, 1024], index=2)
        num_used_subcarriers = st.slider("Used Subcarriers", N_fft//4, N_fft, int(N_fft*0.75), 1)
        cp_length = st.slider("Cyclic Prefix Length", 0, N_fft//4, N_fft//16)
        num_ofdm_symbols = st.slider("Number of OFDM Symbols", 1, 10, 4)
        
        st.markdown("### Modulation")
        modulation = st.selectbox("Modulation Scheme", ["BPSK", "QPSK", "16-QAM", "64-QAM"])
        mod_order = {"BPSK": 2, "QPSK": 4, "16-QAM": 16, "64-QAM": 64}[modulation]
        
        st.markdown("### Channel Parameters")
        snr_db = st.slider("SNR (dB)", -10, 40, 20, 1)
        
        enable_multipath = st.checkbox("Enable Multipath Channel", value=True)
        if enable_multipath:
            num_paths = st.slider("Number of Paths", 1, 5, 2)
            max_delay = st.slider("Max Delay (samples)", 1, cp_length if cp_length > 0 else 10, 
                                 min(5, cp_length if cp_length > 0 else 5))
    
    # Generate transmit data
    total_data_symbols = num_used_subcarriers * num_ofdm_symbols
    tx_data = generate_qam_symbols(total_data_symbols, mod_order)
    
    # Reshape into OFDM symbols (frequency domain)
    tx_data_matrix = tx_data.reshape(num_ofdm_symbols, num_used_subcarriers)
    
    # Map to subcarriers (zero-pad unused subcarriers)
    tx_freq_domain = np.zeros((num_ofdm_symbols, N_fft), dtype=complex)
    start_idx = (N_fft - num_used_subcarriers) // 2
    tx_freq_domain[:, start_idx:start_idx+num_used_subcarriers] = tx_data_matrix
    
    # IFFT to get time domain OFDM symbols (centered mapping)
    tx_time_domain = np.fft.ifft(np.fft.ifftshift(tx_freq_domain, axes=1), axis=1)
    
    # Add cyclic prefix
    tx_with_cp = []
    for symbol in tx_time_domain:
        tx_with_cp.append(add_cyclic_prefix(symbol, cp_length))
    tx_signal = np.concatenate(tx_with_cp)
    
    # Calculate PAPR
    papr = calculate_papr(tx_signal)
    
    # Channel
    if enable_multipath:
        delays = sorted(np.random.randint(0, max_delay+1, num_paths))
        gains = np.exp(-0.5 * np.array(delays)) * (np.random.randn(num_paths) + 1j*np.random.randn(num_paths))
        gains = gains / np.sqrt(np.sum(np.abs(gains)**2))  # Normalize
        rx_signal = multipath_channel(tx_signal, delays, gains)
        channel_ir = np.zeros(N_fft, dtype=complex)
        channel_ir[delays] = gains
        H = np.fft.fft(channel_ir)
        H_shifted = np.fft.fftshift(H)
    else:
        rx_signal = tx_signal.copy()
        H_shifted = np.ones(N_fft, dtype=complex)
    
    # Add noise
    rx_signal = awgn_channel(rx_signal, snr_db)
    
    # Receiver: Remove CP
    rx_symbols = []
    symbol_length_with_cp = N_fft + cp_length
    for i in range(num_ofdm_symbols):
        start = i * symbol_length_with_cp
        end = start + symbol_length_with_cp
        rx_symbols.append(remove_cyclic_prefix(rx_signal[start:end], cp_length))
    
    # FFT (centered mapping)
    rx_freq_domain = np.fft.fftshift(np.fft.fft(rx_symbols, axis=1), axes=1)
    
    # Extract data subcarriers
    rx_data = rx_freq_domain[:, start_idx:start_idx+num_used_subcarriers].flatten()
    H_used = H_shifted[start_idx:start_idx+num_used_subcarriers]
    rx_data_eq = rx_data / (np.tile(H_used, num_ofdm_symbols) + 1e-8)
    
    # Simple detection (zero-forcing equalization using known channel)
    if modulation == "BPSK":
        rx_detected = (np.real(rx_data_eq) > 0).astype(int)
        tx_bits = (np.real(tx_data) > 0).astype(int)
        ber = calculate_ber(tx_bits, rx_detected)
    else:
        # For QAM, just measure constellation error
        constellation_error = np.mean(np.abs(rx_data_eq - tx_data))
    
    with col2:
        # Create comprehensive visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'TX: Constellation (Freq Domain)', 
                'RX: Constellation (Freq Domain)',
                'TX: Time Domain (Real Part)',
                'RX: Time Domain (Real Part)',
                'TX: Power Spectral Density',
                'RX: Power Spectral Density'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # TX Constellation
        fig.add_trace(
            go.Scatter(x=np.real(tx_data), y=np.imag(tx_data), mode='markers',
                      marker=dict(size=6, color='blue', opacity=0.6),
                      name='TX Symbols'),
            row=1, col=1
        )
        fig.update_xaxes(title_text="I", row=1, col=1)
        fig.update_yaxes(title_text="Q", row=1, col=1)
        
        # RX Constellation
        fig.add_trace(
            go.Scatter(x=np.real(rx_data), y=np.imag(rx_data), mode='markers',
                      marker=dict(size=6, color='red', opacity=0.6),
                      name='RX Symbols'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=np.real(tx_data), y=np.imag(tx_data), mode='markers',
                      marker=dict(size=4, color='blue', opacity=0.3, symbol='x'),
                      name='TX Reference'),
            row=1, col=2
        )
        fig.update_xaxes(title_text="I", row=1, col=2)
        fig.update_yaxes(title_text="Q", row=1, col=2)
        
        # TX Time Domain
        time_samples = np.arange(len(tx_signal))
        fig.add_trace(
            go.Scatter(x=time_samples, y=np.real(tx_signal), mode='lines',
                      line=dict(color='blue', width=1),
                      name='TX Time'),
            row=2, col=1
        )
        # Mark CP regions
        for i in range(num_ofdm_symbols):
            cp_start = i * symbol_length_with_cp
            fig.add_vrect(x0=cp_start, x1=cp_start+cp_length, 
                         fillcolor="green", opacity=0.2, layer="below",
                         annotation_text="CP" if i == 0 else "",
                         row=2, col=1)
        fig.update_xaxes(title_text="Sample", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        
        # RX Time Domain
        fig.add_trace(
            go.Scatter(x=time_samples, y=np.real(rx_signal), mode='lines',
                      line=dict(color='red', width=1),
                      name='RX Time'),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Sample", row=2, col=2)
        fig.update_yaxes(title_text="Amplitude", row=2, col=2)
        
        # TX PSD
        f_tx, psd_tx = signal.welch(tx_signal, nperseg=256)
        fig.add_trace(
            go.Scatter(x=f_tx, y=10*np.log10(psd_tx + 1e-10), mode='lines',
                      line=dict(color='blue', width=2),
                      name='TX PSD'),
            row=3, col=1
        )
        fig.update_xaxes(title_text="Normalized Frequency", row=3, col=1)
        fig.update_yaxes(title_text="PSD (dB)", row=3, col=1)
        
        # RX PSD
        f_rx, psd_rx = signal.welch(rx_signal, nperseg=256)
        fig.add_trace(
            go.Scatter(x=f_rx, y=10*np.log10(psd_rx + 1e-10), mode='lines',
                      line=dict(color='red', width=2),
                      name='RX PSD'),
            row=3, col=2
        )
        fig.update_xaxes(title_text="Normalized Frequency", row=3, col=2)
        fig.update_yaxes(title_text="PSD (dB)", row=3, col=2)
        
        fig.update_layout(height=900, showlegend=False, hovermode='closest')
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    st.markdown("### Performance Metrics")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    with col_a:
        st.metric("FFT Size", N_fft)
    with col_b:
        st.metric("Used Subcarriers", num_used_subcarriers)
    with col_c:
        st.metric("CP Length", cp_length)
    with col_d:
        st.metric("PAPR (dB)", f"{papr:.2f}")
    with col_e:
        if modulation == "BPSK":
            st.metric("BER", f"{ber:.2e}")
        else:
            st.metric("Const. Error", f"{constellation_error:.2e}")
    
    # Channel info
    if enable_multipath:
        st.markdown("### Channel Profile")
        channel_df = pd.DataFrame({
            'Path': range(1, num_paths+1),
            'Delay (samples)': delays,
            'Gain (mag)': np.abs(gains),
            'Gain (dB)': 20*np.log10(np.abs(gains) + 1e-10)
        })
        st.dataframe(channel_df, use_container_width=True)

# ============================================================================
# 2. ORTHOGONALITY DEMO
# ============================================================================
elif app_mode == "📊 Orthogonality Demo":
    st.markdown('<p class="section-header">📊 Subcarrier Orthogonality Demonstration</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **OFDM's key innovation:** Subcarriers are orthogonal, meaning they don't interfere with each other 
    even though they overlap in frequency!
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Parameters")
        N_carriers = st.slider("Number of Subcarriers", 4, 16, 8)
        T_symbol = st.slider("Symbol Duration", 1.0, 10.0, 4.0, 0.5)
        delta_f = 1.0 / T_symbol  # Subcarrier spacing
        
        st.markdown("### Visualization")
        show_subcarrier = st.selectbox("Highlight Subcarrier", ["None"] + list(range(N_carriers)))
        
        show_product = st.checkbox("Show Orthogonality (Inner Product)", value=False)
        
        if show_product and show_subcarrier != "None":
            compare_with = st.selectbox("Compare with Subcarrier", 
                                       [i for i in range(N_carriers) if i != show_subcarrier])
    
    # Generate time vector
    t = np.linspace(0, T_symbol, 1000)
    
    # Generate subcarrier signals
    subcarriers = []
    for k in range(N_carriers):
        freq = k * delta_f
        subcarrier = np.exp(1j * 2 * np.pi * freq * t)
        subcarriers.append(subcarrier)
    
    with col2:
        if not show_product:
            # Show subcarriers in time and frequency
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Subcarriers in Time Domain (Real Part)', 
                              'Subcarriers in Frequency Domain'),
                vertical_spacing=0.15
            )
            
            # Time domain
            for k, sc in enumerate(subcarriers):
                color = 'red' if (show_subcarrier != "None" and k == show_subcarrier) else 'lightblue'
                width = 3 if (show_subcarrier != "None" and k == show_subcarrier) else 1
                opacity = 1.0 if (show_subcarrier != "None" and k == show_subcarrier) else 0.4
                
                fig.add_trace(
                    go.Scatter(x=t, y=np.real(sc), mode='lines',
                              name=f'Subcarrier {k}',
                              line=dict(color=color, width=width),
                              opacity=opacity),
                    row=1, col=1
                )
            
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            
            # Frequency domain (just show delta functions)
            freqs = np.arange(N_carriers) * delta_f
            amplitudes = np.ones(N_carriers)
            
            fig.add_trace(
                go.Bar(x=freqs, y=amplitudes, name='Subcarriers',
                      marker=dict(color=['red' if (show_subcarrier != "None" and k == show_subcarrier) 
                                        else 'blue' for k in range(N_carriers)])),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=False, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Subcarrier Spacing:** Δf = 1/T = {delta_f:.3f} Hz")
        
        else:
            # Show orthogonality proof
            if show_subcarrier != "None":
                sc1 = subcarriers[show_subcarrier]
                sc2 = subcarriers[compare_with]
                
                # Compute product
                product = sc1 * np.conj(sc2)
                
                # Compute integral (inner product)
                dt = t[1] - t[0]
                integral = np.sum(product) * dt
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(
                        f'Subcarrier {show_subcarrier} (Real)',
                        f'Subcarrier {compare_with} (Real)',
                        'Product: s₁(t) × s₂*(t) (Real Part)'
                    ),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(x=t, y=np.real(sc1), mode='lines',
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=t, y=np.real(sc2), mode='lines',
                              line=dict(color='green', width=2)),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=t, y=np.real(product), mode='lines',
                              line=dict(color='red', width=2),
                              fill='tozeroy'),
                    row=3, col=1
                )
                
                # Add integral value annotation
                avg_product = np.mean(np.real(product))
                fig.add_hline(y=avg_product, line_dash="dash", line_color="orange",
                            annotation_text=f"Average: {avg_product:.4f}", row=3, col=1)
                
                for i in range(1, 4):
                    fig.update_xaxes(title_text="Time (s)", row=i, col=1)
                    fig.update_yaxes(title_text="Amplitude", row=i, col=1)
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show result
                if show_subcarrier == compare_with:
                    st.success(f"✅ Inner product = {np.real(integral):.4f} (Same subcarrier → Energy = T)")
                else:
                    if abs(integral) < 0.01:
                        st.success(f"✅ Inner product ≈ {np.real(integral):.6f} ≈ 0 (Orthogonal!)")
                    else:
                        st.warning(f"⚠️ Inner product = {np.real(integral):.4f} (Not perfectly orthogonal - numerical error)")
    
    with st.expander("📚 Theory: Orthogonality"):
        st.markdown("""
        ### Mathematical Definition
        
        Two signals $s_1(t)$ and $s_2(t)$ are **orthogonal** if their inner product is zero:
        
        $$\\langle s_1, s_2 \\rangle = \\int_0^T s_1(t) s_2^*(t) dt = 0$$
        
        ### OFDM Subcarriers
        
        In OFDM, subcarrier $k$ is:
        $$s_k(t) = e^{j2\\pi k \\Delta f \\cdot t}, \\quad k = 0, 1, 2, ..., N-1$$
        
        where $\\Delta f = \\frac{1}{T}$ (subcarrier spacing = inverse of symbol duration).
        
        ### Proof of Orthogonality
        
        For $k \\neq m$:
        $$\\int_0^T e^{j2\\pi k \\Delta f \\cdot t} e^{-j2\\pi m \\Delta f \\cdot t} dt = \\int_0^T e^{j2\\pi (k-m) \\Delta f \\cdot t} dt$$
        
        Since $(k-m)\\Delta f \\cdot T = (k-m)$ is an integer:
        $$= \\frac{e^{j2\\pi(k-m)} - 1}{j2\\pi(k-m)\\Delta f} = 0$$
        
        ### Why This Matters
        
        - **No interference:** Each subcarrier can be demodulated independently using FFT
        - **Spectral efficiency:** Subcarriers overlap but don't interfere
        - **Simple equalization:** Channel effects can be corrected per-subcarrier
        - **Foundation of OFDM:** This property enables the entire OFDM concept!
        """)

# ============================================================================
# 3. CYCLIC PREFIX IMPACT
# ============================================================================
elif app_mode == "🔄 Cyclic Prefix Impact":
    st.markdown('<p class="section-header">🔄 Cyclic Prefix: Converting Linear to Circular Convolution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### OFDM Parameters")
        N_fft = st.selectbox("FFT Size", [32, 64, 128], index=1)
        cp_length = st.slider("Cyclic Prefix Length", 0, N_fft//4, N_fft//8)
        
        st.markdown("### Channel Parameters")
        channel_length = st.slider("Channel Impulse Response Length", 1, N_fft//4, 5)
        
        st.markdown("### Visualization")
        show_comparison = st.checkbox("Compare With/Without CP", value=True)
    
    # Generate simple OFDM symbol
    tx_freq = np.zeros(N_fft, dtype=complex)
    # Put some data on a few subcarriers
    active_carriers = [N_fft//4, N_fft//4 + 5, N_fft//4 + 10, N_fft//4 + 15]
    tx_freq[active_carriers] = [1+0j, 0.7+0.7j, -0.5+0.5j, 0+1j]
    
    # IFFT
    tx_time = np.fft.ifft(tx_freq)
    
    # Generate channel (exponentially decaying)
    channel = np.exp(-0.5 * np.arange(channel_length)) * (0.9 + 0.1*np.random.randn(channel_length))
    channel = channel / np.linalg.norm(channel)  # Normalize
    
    # Scenario 1: With CP
    tx_with_cp = add_cyclic_prefix(tx_time, cp_length)
    rx_with_cp = np.convolve(tx_with_cp, channel, mode='same')
    rx_no_cp_removed = remove_cyclic_prefix(rx_with_cp, cp_length)
    
    # FFT
    rx_freq_with_cp = np.fft.fft(rx_no_cp_removed)
    
    # Scenario 2: Without CP
    rx_without_cp = np.convolve(tx_time, channel, mode='same')
    rx_freq_without_cp = np.fft.fft(rx_without_cp)
    
    # Channel frequency response
    channel_padded = np.zeros(N_fft)
    channel_padded[:len(channel)] = channel
    H = np.fft.fft(channel_padded)
    
    # Expected received spectrum (with perfect circular convolution)
    expected_rx_freq = tx_freq * H
    
    with col2:
        if show_comparison:
            # Compare with/without CP
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Time Domain: With CP',
                    'Time Domain: Without CP',
                    'Freq Domain: With CP',
                    'Freq Domain: Without CP'
                ),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter'}, {'type': 'scatter'}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Time domain with CP
            fig.add_trace(
                go.Scatter(x=np.arange(len(tx_with_cp)), y=np.real(tx_with_cp),
                          mode='lines', line=dict(color='blue', width=1),
                          name='TX'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=np.arange(len(rx_with_cp)), y=np.real(rx_with_cp),
                          mode='lines', line=dict(color='red', width=1),
                          name='RX'),
                row=1, col=1
            )
            fig.add_vrect(x0=0, x1=cp_length, fillcolor="green", opacity=0.2,
                         annotation_text="CP", row=1, col=1)
            fig.update_xaxes(title_text="Sample", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            
            # Time domain without CP
            fig.add_trace(
                go.Scatter(x=np.arange(len(tx_time)), y=np.real(tx_time),
                          mode='lines', line=dict(color='blue', width=1),
                          name='TX'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=np.arange(len(rx_without_cp)), y=np.real(rx_without_cp),
                          mode='lines', line=dict(color='red', width=1),
                          name='RX'),
                row=1, col=2
            )
            fig.update_xaxes(title_text="Sample", row=1, col=2)
            fig.update_yaxes(title_text="Amplitude", row=1, col=2)
            
            # Frequency domain with CP
            subcarrier_idx = np.arange(N_fft)
            fig.add_trace(
                go.Scatter(x=subcarrier_idx, y=np.abs(expected_rx_freq),
                          mode='markers', marker=dict(size=8, color='green', symbol='x'),
                          name='Expected'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=subcarrier_idx, y=np.abs(rx_freq_with_cp),
                          mode='markers', marker=dict(size=6, color='red'),
                          name='Received'),
                row=2, col=1
            )
            fig.update_xaxes(title_text="Subcarrier", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)
            
            # Frequency domain without CP
            fig.add_trace(
                go.Scatter(x=subcarrier_idx, y=np.abs(expected_rx_freq),
                          mode='markers', marker=dict(size=8, color='green', symbol='x'),
                          name='Expected'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=subcarrier_idx, y=np.abs(rx_freq_without_cp),
                          mode='markers', marker=dict(size=6, color='orange'),
                          name='Received'),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Subcarrier", row=2, col=2)
            fig.update_yaxes(title_text="Magnitude", row=2, col=2)
            
            fig.update_layout(height=700, showlegend=False, hovermode='closest')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate errors
            error_with_cp = np.linalg.norm(rx_freq_with_cp - expected_rx_freq)
            error_without_cp = np.linalg.norm(rx_freq_without_cp - expected_rx_freq)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Error WITH CP", f"{error_with_cp:.4f}", 
                         delta="✅ Good!" if error_with_cp < 0.1 else "")
            with col_b:
                st.metric("Error WITHOUT CP", f"{error_without_cp:.4f}",
                         delta="❌ ISI!" if error_without_cp > error_with_cp else "")
        
        else:
            # Just show channel response
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Channel Impulse Response', 'Channel Frequency Response'),
                vertical_spacing=0.2
            )
            
            fig.add_trace(
                go.Scatter(x=np.arange(len(channel)), y=channel,
                          mode='lines+markers', line=dict(color='blue', width=2),
                          marker=dict(size=8)),
                row=1, col=1
            )
            fig.update_xaxes(title_text="Tap", row=1, col=1)
            fig.update_yaxes(title_text="Gain", row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=np.arange(N_fft), y=np.abs(H),
                          mode='lines', line=dict(color='red', width=2)),
                row=2, col=1
            )
            fig.update_xaxes(title_text="Subcarrier", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📚 Theory: Cyclic Prefix"):
        st.markdown("""
        ### The Problem: Inter-Symbol Interference (ISI)
        
        In a multipath channel, each OFDM symbol spreads out in time. This causes:
        - **ISI:** Current symbol interferes with next symbol
        - **ICI:** Destroys orthogonality between subcarriers
        
        ### The Solution: Cyclic Prefix
        
        **Cyclic Prefix (CP):** Copy last $L_{CP}$ samples of OFDM symbol to the beginning.
        
        **Why it works:**
        1. Creates a **guard interval** to absorb multipath delay
        2. Converts **linear convolution** (channel effect) into **circular convolution**
        3. Circular convolution in time ↔ Multiplication in frequency (via FFT)
        
        ### Mathematical Insight
        
        **Without CP:** Channel effect is linear convolution
        $$y[n] = x[n] * h[n]$$
        - Complex in frequency domain
        - Creates ISI and ICI
        
        **With CP (if $L_{CP} \\geq$ channel delay spread):**
        $$y[n] = x[n] \\circledast h[n]$$
        - Circular convolution!
        - In frequency domain: $Y[k] = X[k] \\cdot H[k]$
        - Simple per-subcarrier equalization: $\\hat{X}[k] = \\frac{Y[k]}{H[k]}$
        
        ### CP Length Requirements
        
        - Must be: $L_{CP} \\geq L_{channel}$ (maximum delay spread)
        - 5G NR typical: CP ≈ 7-10% of symbol duration
        - Trade-off: Longer CP = more overhead but better multipath protection
        
        ### Overhead Cost
        
        $$\\text{Overhead} = \\frac{L_{CP}}{N + L_{CP}} \\times 100\\%$$
        
        For this simulation: ${100 * cp_length / (N_fft + cp_length):.1f}%$
        """)

# ============================================================================
# 4. CHANNEL EFFECTS
# ============================================================================
elif app_mode == "📡 Channel Effects":
    st.markdown('<p class="section-header">📡 Wireless Channel Effects on OFDM</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Channel Model")
        channel_type = st.selectbox(
            "Channel Type",
            ["Flat Fading", "Frequency Selective", "Doppler (Time-Varying)", "Custom"]
        )
        
        N_fft = 256
        num_symbols = 10
        
        if channel_type == "Frequency Selective":
            num_paths = st.slider("Number of Paths", 2, 6, 3)
            max_delay_us = st.slider("Max Delay (μs)", 1, 20, 5)
        elif channel_type == "Doppler (Time-Varying)":
            carrier_freq_ghz = st.slider("Carrier Frequency (GHz)", 0.7, 60.0, 3.5, 0.1)
            velocity_kmh = st.slider("Velocity (km/h)", 10, 300, 60)
            velocity_mps = velocity_kmh / 3.6
            doppler_hz = velocity_mps * (carrier_freq_ghz * 1e9) / 3e8
            st.caption(f"Computed Doppler: {doppler_hz:.1f} Hz")
        
        snr_db = st.slider("SNR (dB)", -5, 30, 15)
        
        st.markdown("### Equalization")
        use_equalization = st.checkbox("Apply Zero-Forcing Equalization", value=True)
    
    # Generate OFDM symbols
    tx_symbols = generate_qam_symbols(N_fft * num_symbols, 4)  # QPSK
    tx_freq = tx_symbols.reshape(num_symbols, N_fft)
    tx_time = np.fft.ifft(tx_freq, axis=1)
    
    # Apply channel
    if channel_type == "Flat Fading":
        # Constant channel across frequency
        H = 0.7 * np.exp(1j * np.pi/4) * np.ones((num_symbols, N_fft))
        rx_freq = tx_freq * H
        
    elif channel_type == "Frequency Selective":
        # Different channel per subcarrier
        delays = np.linspace(0, max_delay_us * 1e-6, num_paths)
        sample_delays = (delays * 1e6).astype(int)  # Convert to samples (assume 1 MHz sampling)
        gains = np.exp(-0.3 * np.arange(num_paths)) * np.exp(1j * 2 * np.pi * np.random.rand(num_paths))
        
        # Channel frequency response
        freqs = np.fft.fftfreq(N_fft)
        H = np.zeros((num_symbols, N_fft), dtype=complex)
        for path_delay, path_gain in zip(sample_delays, gains):
            H += path_gain * np.exp(-1j * 2 * np.pi * freqs * path_delay)
        
        rx_freq = tx_freq * H
        
    elif channel_type == "Doppler (Time-Varying)":
        # Time-varying channel
        t_symbols = np.arange(num_symbols)
        doppler_phase = 2 * np.pi * doppler_hz * t_symbols / 1000  # Assume 1ms per symbol
        
        H = np.zeros((num_symbols, N_fft), dtype=complex)
        for sym_idx in range(num_symbols):
            # Channel varies over time
            phase_shift = doppler_phase[sym_idx]
            H[sym_idx, :] = 0.8 * np.exp(1j * phase_shift)
        
        rx_freq = tx_freq * H
    
    else:  # Custom
        H = np.random.randn(num_symbols, N_fft) + 1j * np.random.randn(num_symbols, N_fft)
        H = H / np.abs(H).max()
        rx_freq = tx_freq * H
    
    # Add noise
    signal_power = np.mean(np.abs(rx_freq)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*rx_freq.shape) + 1j*np.random.randn(*rx_freq.shape))
    rx_freq = rx_freq + noise
    
    # Equalization
    if use_equalization:
        # Zero-forcing
        rx_eq = rx_freq / (H + 1e-6)
    else:
        rx_eq = rx_freq
    
    with col2:
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Channel Freq Response (|H|)',
                'TX vs RX Constellation',
                'Channel Phase Response',
                'Equalized Constellation'
            ),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Channel magnitude
        fig.add_trace(
            go.Heatmap(
                z=np.abs(H),
                colorscale='Viridis',
                colorbar=dict(title="|H|", x=0.45)
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text="Subcarrier", row=1, col=1)
        fig.update_yaxes(title_text="Symbol", row=1, col=1)
        
        # TX vs RX constellation
        fig.add_trace(
            go.Scatter(
                x=np.real(tx_symbols), y=np.imag(tx_symbols),
                mode='markers', marker=dict(size=4, color='blue', opacity=0.3),
                name='TX'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=np.real(rx_freq.flatten()), y=np.imag(rx_freq.flatten()),
                mode='markers', marker=dict(size=4, color='red', opacity=0.3),
                name='RX'
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text="I", row=1, col=2)
        fig.update_yaxes(title_text="Q", row=1, col=2)
        
        # Channel phase
        fig.add_trace(
            go.Heatmap(
                z=np.angle(H),
                colorscale='RdBu',
                colorbar=dict(title="Phase (rad)", x=0.45)
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Subcarrier", row=2, col=1)
        fig.update_yaxes(title_text="Symbol", row=2, col=1)
        
        # Equalized constellation
        fig.add_trace(
            go.Scatter(
                x=np.real(tx_symbols), y=np.imag(tx_symbols),
                mode='markers', marker=dict(size=4, color='blue', opacity=0.3),
                name='TX'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=np.real(rx_eq.flatten()), y=np.imag(rx_eq.flatten()),
                mode='markers', marker=dict(size=4, color='green', opacity=0.3),
                name='Equalized'
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text="I", row=2, col=2)
        fig.update_yaxes(title_text="Q", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        evm_before = np.sqrt(np.mean(np.abs(rx_freq.flatten() - tx_symbols)**2)) / np.sqrt(np.mean(np.abs(tx_symbols)**2)) * 100
        evm_after = np.sqrt(np.mean(np.abs(rx_eq.flatten() - tx_symbols)**2)) / np.sqrt(np.mean(np.abs(tx_symbols)**2)) * 100
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Channel Type", channel_type)
        with col_b:
            st.metric("EVM Before Eq", f"{evm_before:.2f}%")
        with col_c:
            st.metric("EVM After Eq", f"{evm_after:.2f}%", 
                     delta=f"{evm_after - evm_before:.2f}%" if use_equalization else None)
    
    with st.expander("📚 Theory: Channel Effects & Equalization"):
        st.markdown("""
        ### Wireless Channel Effects
        
        **Flat Fading:**
        - All subcarriers experience same gain/phase
        - Occurs when channel delay spread << OFDM symbol duration
        - $H[k] = \\alpha e^{j\\theta}$ (constant)
        
        **Frequency Selective Fading:**
        - Different subcarriers experience different gains/phases
        - Caused by multipath propagation
        - Some subcarriers in deep fade, others strong
        - $H[k]$ varies significantly across $k$
        
        **Doppler Effect (Time-Varying):**
        - Channel changes during transmission
        - Caused by mobility (user/base station movement)
        - Can destroy orthogonality (Inter-Carrier Interference)
        - $H[k, n]$ varies with time symbol index $n$
        
        ### Equalization
        
        **Zero-Forcing (ZF):**
        $$\\hat{X}[k] = \\frac{Y[k]}{H[k]}$$
        - Simple: divide by channel
        - Perfect when no noise
        - Amplifies noise on weak subcarriers
        
        **Minimum Mean Square Error (MMSE):**
        $$\\hat{X}[k] = \\frac{H^*[k]}{|H[k]|^2 + \\sigma_n^2 / \\sigma_x^2} Y[k]$$
        - Balances channel inversion and noise amplification
        - Better performance than ZF in low SNR
        
        ### Channel Estimation in OFDM
        
        Need to estimate $H[k]$ using **pilot symbols** (known data):
        1. Place pilots on known subcarriers/symbols
        2. Estimate $\\hat{H}[k_{pilot}] = Y[k_{pilot}] / X[k_{pilot}]$
        3. Interpolate to get $\\hat{H}[k]$ for all subcarriers
        
        **5G NR uses:**
        - Demodulation Reference Signals (DMRS) for channel estimation
        - Phase Tracking Reference Signals (PTRS) for phase noise compensation
        """)

# ============================================================================
# 5. PAPR ANALYSIS
# ============================================================================
elif app_mode == "⚡ PAPR Analysis":
    st.markdown('<p class="section-header">⚡ Peak-to-Average Power Ratio (PAPR) in OFDM</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **PAPR is a major challenge in OFDM:** When many subcarriers add up in-phase, 
    they create high peaks that stress power amplifiers and reduce efficiency.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### OFDM Configuration")
        N_fft = st.selectbox("FFT Size", [64, 128, 256, 512], index=2)
        num_symbols_papr = st.slider("Number of OFDM Symbols", 100, 1000, 500, 100)
        
        modulation_papr = st.selectbox("Modulation", ["BPSK", "QPSK", "16-QAM", "64-QAM"], index=1)
        mod_order_papr = {"BPSK": 2, "QPSK": 4, "16-QAM": 16, "64-QAM": 64}[modulation_papr]
        
        st.markdown("### PAPR Reduction")
        papr_technique = st.selectbox(
            "Technique",
            ["None", "Clipping", "Selected Mapping (SLM)", "Tone Reservation"]
        )
        
        if papr_technique == "Clipping":
            clip_ratio = st.slider("Clipping Ratio (dB)", 3, 12, 6)
        elif papr_technique == "Selected Mapping (SLM)":
            num_candidates = st.slider("Number of Candidates", 2, 16, 4)
        elif papr_technique == "Tone Reservation":
            num_reserved = st.slider("Reserved Tones", 4, N_fft//8, N_fft//16)
    
    # Generate many OFDM symbols
    papr_values = []
    
    for _ in range(num_symbols_papr):
        # Generate random data
        data = generate_qam_symbols(N_fft, mod_order_papr)
        
        if papr_technique == "None":
            # Standard OFDM
            time_signal = np.fft.ifft(data)
            
        elif papr_technique == "Clipping":
            # Simple clipping
            time_signal = np.fft.ifft(data)
            threshold = np.sqrt(np.mean(np.abs(time_signal)**2)) * 10**(clip_ratio/20)
            time_signal = np.clip(np.abs(time_signal), 0, threshold) * np.exp(1j * np.angle(time_signal))
            
        elif papr_technique == "Selected Mapping (SLM)":
            # Try multiple phase rotations, pick best
            best_papr = float('inf')
            best_signal = None
            
            for _ in range(num_candidates):
                phase_seq = np.exp(1j * 2 * np.pi * np.random.rand(N_fft))
                rotated_data = data * phase_seq
                candidate = np.fft.ifft(rotated_data)
                candidate_papr = calculate_papr(candidate)
                
                if candidate_papr < best_papr:
                    best_papr = candidate_papr
                    best_signal = candidate
            
            time_signal = best_signal
            
        elif papr_technique == "Tone Reservation":
            # Reserve some tones for PAPR reduction
            data_tones = data.copy()
            reserved_indices = np.random.choice(N_fft, num_reserved, replace=False)
            data_tones[reserved_indices] = 0  # Data doesn't use these
            
            # Simple peak cancellation signal on reserved tones
            time_signal = np.fft.ifft(data_tones)
        
        papr = calculate_papr(time_signal)
        papr_values.append(papr)
    
    papr_values = np.array(papr_values)
    
    with col2:
        # Plot PAPR distribution
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('PAPR Distribution (Histogram)', 'PAPR CCDF'),
            vertical_spacing=0.15
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=papr_values,
                nbinsx=50,
                name='PAPR',
                marker=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text="PAPR (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        # CCDF (Complementary Cumulative Distribution Function)
        papr_sorted = np.sort(papr_values)
        ccdf = 1 - np.arange(1, len(papr_sorted) + 1) / len(papr_sorted)
        
        fig.add_trace(
            go.Scatter(
                x=papr_sorted,
                y=ccdf,
                mode='lines',
                line=dict(color='red', width=2),
                name='CCDF'
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="PAPR (dB)", row=2, col=1)
        fig.update_yaxes(title_text="Pr(PAPR > x)", type='log', row=2, col=1)
        
        fig.update_layout(height=700, showlegend=False, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Mean PAPR", f"{np.mean(papr_values):.2f} dB")
        with col_b:
            st.metric("Median PAPR", f"{np.median(papr_values):.2f} dB")
        with col_c:
            st.metric("99th %ile", f"{np.percentile(papr_values, 99):.2f} dB")
        with col_d:
            st.metric("Max PAPR", f"{np.max(papr_values):.2f} dB")
    
    # Show example signal
    st.markdown("### Example OFDM Symbol Time Domain")
    example_data = generate_qam_symbols(N_fft, mod_order_papr)
    example_time = np.fft.ifft(example_data)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=np.arange(len(example_time)),
        y=np.abs(example_time),
        mode='lines',
        fill='tozeroy',
        line=dict(color='blue', width=2)
    ))
    
    avg_power = np.sqrt(np.mean(np.abs(example_time)**2))
    peak_power = np.max(np.abs(example_time))
    
    fig2.add_hline(y=avg_power, line_dash="dash", line_color="green",
                  annotation_text=f"Avg: {avg_power:.3f}")
    fig2.add_hline(y=peak_power, line_dash="dash", line_color="red",
                  annotation_text=f"Peak: {peak_power:.3f}")
    
    fig2.update_layout(
        title=f"PAPR = {calculate_papr(example_time):.2f} dB",
        xaxis_title="Sample",
        yaxis_title="Magnitude",
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("📚 Theory: PAPR Problem & Solutions"):
        st.markdown("""
        ### PAPR Definition
        
        **Peak-to-Average Power Ratio:**
        $$\\text{PAPR} = 10 \\log_{10} \\left( \\frac{\\max |x[n]|^2}{\\text{E}[|x[n]|^2]} \\right)$$
        
        ### Why PAPR is High in OFDM
        
        When $N$ subcarriers add up **constructively** (in-phase):
        - Peak amplitude ∝ $N$
        - Peak power ∝ $N^2$
        - Average power ∝ $N$ (assuming independent subcarriers)
        - Therefore: PAPR ∝ $N$ (grows with number of subcarriers!)
        
        ### Problems Caused by High PAPR
        
        1. **Power Amplifier (PA) Inefficiency**
           - PA must operate with large backoff to avoid saturation
           - Reduces efficiency (more battery drain)
           - Generates out-of-band emissions when saturated
        
        2. **Increased ADC/DAC Requirements**
           - Need higher resolution to capture peaks
           - Increases cost and power consumption
        
        3. **Signal Distortion**
           - Non-linear PA causes in-band distortion
           - Spectral regrowth (interference to adjacent channels)
        
        ### PAPR Reduction Techniques
        
        **Clipping:**
        - Simplest: clip peaks above threshold
        - Distorts signal, causes out-of-band radiation
        - Needs filtering after clipping
        
        **Selected Mapping (SLM):**
        - Generate multiple candidates with different phase rotations
        - Transmit the one with lowest PAPR
        - Need to send side information to receiver
        
        **Tone Reservation:**
        - Reserve some subcarriers for PAPR reduction
        - Generate cancellation signal on reserved tones
        - No side information needed
        - Reduces spectral efficiency
        - (This demo only reserves tones; it does not synthesize a cancellation signal)
        
        **Partial Transmit Sequence (PTS):**
        - Partition subcarriers into blocks
        - Optimize phase rotation per block
        - Lower complexity than SLM
        
        ### 5G NR Approach
        
        - Uses **DFT-spread OFDM** (DFT-s-OFDM) for uplink
        - Pre-processes data with DFT before IFFT
        - Results in single-carrier-like properties (lower PAPR)
        - Important for mobile devices (battery life!)
        """)

# ============================================================================
# 6. SUBCARRIER MAPPING
# ============================================================================
elif app_mode == "🎚️ Subcarrier Mapping":
    st.markdown('<p class="section-header">🎚️ OFDM Subcarrier Allocation & Resource Grids</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### OFDM Parameters")
        N_fft_map = st.selectbox("FFT Size", [64, 128, 256, 512], index=1)
        num_symbols_map = st.slider("Number of OFDM Symbols", 5, 20, 10)
        
        st.markdown("### Subcarrier Usage")
        guard_band_size = st.slider("Guard Band (subcarriers/side)", 0, N_fft_map//8, N_fft_map//16)
        dc_null = st.checkbox("Null DC Subcarrier", value=True)
        
        num_pilots = st.slider("Pilot Density (%)", 0, 30, 10, 5)
        
        st.markdown("### Visualization Options")
        show_resource_grid = st.checkbox("Show Resource Grid", value=True)
        show_spectrum = st.checkbox("Show Spectrum", value=True)
    
    # Calculate usable subcarriers
    total_usable = N_fft_map - 2*guard_band_size
    if dc_null:
        total_usable -= 1
    
    num_pilot_carriers = int(total_usable * num_pilots / 100)
    num_data_carriers = total_usable - num_pilot_carriers
    
    # Create resource grid
    resource_grid = np.zeros((num_symbols_map, N_fft_map), dtype=complex)
    allocation_type = np.zeros((num_symbols_map, N_fft_map))  # 0=null, 1=data, 2=pilot
    
    # Determine active subcarriers
    start_sc = guard_band_size
    end_sc = N_fft_map - guard_band_size
    
    # Pilot pattern (comb-type)
    if num_pilot_carriers > 0:
        pilot_spacing = total_usable // num_pilot_carriers
        pilot_indices = start_sc + np.arange(0, total_usable, pilot_spacing)[:num_pilot_carriers]
    else:
        pilot_indices = []
    
    # Fill resource grid
    for sym_idx in range(num_symbols_map):
        for sc_idx in range(start_sc, end_sc):
            # Skip DC
            if dc_null and sc_idx == N_fft_map // 2:
                continue
            
            # Pilots
            if sc_idx in pilot_indices:
                resource_grid[sym_idx, sc_idx] = 1 + 0j  # Known pilot
                allocation_type[sym_idx, sc_idx] = 2
            else:
                # Data
                resource_grid[sym_idx, sc_idx] = generate_qam_symbols(1, 4)[0]
                allocation_type[sym_idx, sc_idx] = 1
    
    with col2:
        if show_resource_grid:
            # Resource grid visualization
            fig = go.Figure()
            
            # Create color map: 0=black (null), 1=blue (data), 2=red (pilot)
            color_grid = np.zeros((num_symbols_map, N_fft_map, 3))
            color_grid[allocation_type == 1] = [0.2, 0.4, 0.8]  # Blue for data
            color_grid[allocation_type == 2] = [0.8, 0.2, 0.2]  # Red for pilots
            
            fig.add_trace(go.Heatmap(
                z=allocation_type,
                colorscale=[[0, 'black'], [0.5, 'blue'], [1, 'red']],
                showscale=False,
                hovertemplate='Symbol: %{y}<br>Subcarrier: %{x}<br>Type: %{z}<extra></extra>'
            ))
            
            # Add DC line
            if dc_null:
                fig.add_vline(x=N_fft_map//2, line_dash="dash", line_color="yellow",
                            annotation_text="DC")
            
            # Add guard band regions
            fig.add_vrect(x0=-0.5, x1=guard_band_size-0.5, 
                         fillcolor="gray", opacity=0.3, annotation_text="Guard")
            fig.add_vrect(x0=N_fft_map-guard_band_size-0.5, x1=N_fft_map-0.5,
                         fillcolor="gray", opacity=0.3, annotation_text="Guard")
            
            fig.update_layout(
                title="OFDM Resource Grid (Time-Frequency)",
                xaxis_title="Subcarrier Index",
                yaxis_title="OFDM Symbol Index",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Legend
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown("🔵 **Data Subcarriers**")
            with col_b:
                st.markdown("🔴 **Pilot Subcarriers**")
            with col_c:
                st.markdown("⬛ **Guard Bands**")
            with col_d:
                if dc_null:
                    st.markdown("💛 **DC Null**")
        
        if show_spectrum:
            # Generate time domain and show spectrum
            time_signals = np.fft.ifft(resource_grid, axis=1)
            combined_signal = time_signals.flatten()
            
            # Compute PSD
            f, psd = signal.welch(combined_signal, fs=N_fft_map, nperseg=min(512, len(combined_signal)))
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=f,
                y=10*np.log10(psd + 1e-10),
                mode='lines',
                line=dict(color='blue', width=2),
                fill='tozeroy'
            ))
            
            # Mark guard bands
            fig2.add_vrect(x0=0, x1=guard_band_size, fillcolor="red", opacity=0.2,
                          annotation_text="Guard")
            fig2.add_vrect(x0=N_fft_map-guard_band_size, x1=N_fft_map, fillcolor="red", opacity=0.2,
                          annotation_text="Guard")
            
            fig2.update_layout(
                title="Power Spectral Density",
                xaxis_title="Subcarrier Index",
                yaxis_title="Power (dB)",
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics
    st.markdown("### Resource Allocation Summary")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    with col_a:
        st.metric("Total Subcarriers", N_fft_map)
    with col_b:
        st.metric("Data Subcarriers", num_data_carriers)
    with col_c:
        st.metric("Pilot Subcarriers", num_pilot_carriers)
    with col_d:
        st.metric("Guard Band", f"{2*guard_band_size}")
    with col_e:
        efficiency = num_data_carriers / N_fft_map * 100
        st.metric("Spectral Efficiency", f"{efficiency:.1f}%")
    
    with st.expander("📚 Theory: Subcarrier Mapping & Resource Grids"):
        st.markdown("""
        ### OFDM Resource Grid
        
        OFDM organizes resources in a **2D grid:**
        - **Frequency axis:** Subcarriers (columns)
        - **Time axis:** OFDM symbols (rows)
        - Each cell = one **Resource Element (RE)**
        
        ### Subcarrier Types
        
        **1. Data Subcarriers (Blue):**
        - Carry user data
        - QAM modulated
        - Majority of subcarriers
        
        **2. Pilot/Reference Subcarriers (Red):**
        - Known symbols for channel estimation
        - Scattered across time and frequency
        - Enable coherent detection and equalization
        - Typical patterns: Comb, Block, Lattice
        
        **3. Guard Bands (Gray):**
        - Empty subcarriers at edges
        - Prevent interference to/from adjacent channels
        - Allow for filter roll-off
        - Reduce stringent filtering requirements
        
        **4. DC Subcarrier (Yellow):**
        - Center frequency (f=0)
        - Often nulled due to DC offset in RF chain
        - LO leakage can corrupt this subcarrier
        
        ### Pilot Patterns
        
        **Comb-type (shown here):**
        - Pilots on every Nth subcarrier, all symbols
        - Good for fast time variation (high Doppler)
        - Used in OFDM systems like DVB-T
        
        **Block-type:**
        - All subcarriers in certain symbols are pilots
        - Good for slow time variation
        - Efficient when channel changes slowly
        
        **Lattice:**
        - Scattered in both time and frequency
        - Best balance for typical channels
        - Used in LTE and 5G NR (DMRS)
        
        ### 5G NR Resource Grid
        
        **Resource Block (RB):** 12 consecutive subcarriers
        **Subcarrier Spacing:** 15, 30, 60, 120, 240 kHz (scalable numerology)
        **Slot:** 14 OFDM symbols (normal CP)
        
        **Reference Signals in 5G:**
        - **DMRS:** Demodulation RS (channel estimation for data demodulation)
        - **CSI-RS:** Channel State Information RS (channel quality measurement)
        - **PTRS:** Phase Tracking RS (compensate phase noise at mmWave)
        - **SRS:** Sounding RS (uplink channel measurement)
        
        ### Trade-offs
        
        - More pilots → Better channel estimation, but less throughput
        - Larger guard bands → Less interference, but less spectral efficiency
        - DC null → Avoids LO leakage issues, loses one subcarrier
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>OFDM Complete Simulator</strong> | Built with Streamlit</p>
    <p>Understanding the foundation of 5G NR, LTE, WiFi 6, and more!</p>
</div>
""", unsafe_allow_html=True)
