import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import serial
import serial.tools.list_ports
import json
import time
from collections import deque
from datetime import datetime
from scipy.fft import rfft, rfftfreq
from scipy import signal as scipy_signal
from scipy.io import wavfile
import base64
from io import BytesIO
from typing import Tuple, List, Dict

# Configurare pagină
st.set_page_config(page_title="Analiză Vibrații Motor", layout="wide", page_icon="⚙️")

# Constante
DEFAULT_RPM = 3000
DEFAULT_HARMONICS = 5
DEFAULT_SAMPLING_RATE = 5000
DEFAULT_DURATION = 2.0
DEFAULT_NOISE = 0.1
MAX_INT16 = 32767

# Inițializare session state - Simulare
if 'simulated' not in st.session_state:
    st.session_state.simulated = False
if 'show_animation' not in st.session_state:
    st.session_state.show_animation = False

# Inițializare session state - Arduino
if 'serial_port' not in st.session_state:
    st.session_state.serial_port = None
if 'is_connected' not in st.session_state:
    st.session_state.is_connected = False
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = {
        'time': deque(maxlen=200),
        'rpm': deque(maxlen=200),
        'angle': deque(maxlen=200),
        'freq_fundamental': deque(maxlen=200),
        'freq_harmonic': deque(maxlen=200),
        'defect': deque(maxlen=200)
    }
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Simulare Motor"

# Titlu
st.title("📡 Monitorizare Arduino - Analizor Vibrații în Timp Real")

# Sidebar - Conexiune
st.sidebar.header("🔌 Conexiune Arduino")

# Găsește porturile disponibile
def get_available_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

available_ports = get_available_ports()

if available_ports:
    selected_port = st.sidebar.selectbox("Port Serial", available_ports)
    baudrate = st.sidebar.selectbox("Baudrate", [9600, 115200], index=1)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔗 Conectează", use_container_width=True):
            try:
                if st.session_state.serial_port and st.session_state.serial_port.is_open:
                    st.session_state.serial_port.close()
                
                st.session_state.serial_port = serial.Serial(selected_port, baudrate, timeout=1)
                time.sleep(2)  # Așteaptă inițializarea Arduino
                st.session_state.is_connected = True
                st.session_state.start_time = time.time()
                st.sidebar.success(f"✅ Conectat la {selected_port}")
            except Exception as e:
                st.sidebar.error(f"❌ Eroare: {str(e)}")
                st.session_state.is_connected = False
    
    with col2:
        if st.button("🔌 Deconectează", use_container_width=True):
            if st.session_state.serial_port:
                st.session_state.serial_port.close()
                st.session_state.is_connected = False
                st.sidebar.info("Deconectat")
else:
    st.sidebar.warning("⚠️ Niciun port serial găsit!")
    st.sidebar.info("Asigură-te că Arduino este conectat")

# Status conexiune
st.sidebar.markdown("---")
if st.session_state.is_connected:
    st.sidebar.success("🟢 CONECTAT")
else:
    st.sidebar.error("🔴 DECONECTAT")

# Opțiuni afișare
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Opțiuni Grafice")
show_rpm = st.sidebar.checkbox("Afișează RPM", value=True)
show_angle = False
show_frequencies = st.sidebar.checkbox("Afișează Frecvențe", value=True)
update_rate = st.sidebar.slider("Rată actualizare (Hz)", 1, 10, 5)

# Buton reset date
if st.sidebar.button("🔄 Resetează Date", use_container_width=True):
    for key in st.session_state.data_buffer:
        st.session_state.data_buffer[key].clear()
    st.session_state.start_time = time.time()
    st.sidebar.success("Date resetate!")

# Funcție citire date de la Arduino
def read_arduino_data():
    if not st.session_state.is_connected or not st.session_state.serial_port:
        return None
    
    try:
        if st.session_state.serial_port.in_waiting > 0:
            line = st.session_state.serial_port.readline().decode('utf-8').strip()
            if line.startswith('{') and line.endswith('}'):
                data = json.loads(line)
                return data
    except Exception as e:
        st.error(f"Eroare citire: {str(e)}")
    
    return None

# Layout principal
if st.session_state.is_connected:
    # Placeholder pentru metrici și grafice
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    table_placeholder = st.empty()
    
    # Loop pentru actualizare în timp real
    for _ in range(update_rate):
        data = read_arduino_data()
        
        if data:
            # Adaugă timp relativ
            current_time = time.time() - st.session_state.start_time
            st.session_state.data_buffer['time'].append(current_time)
            st.session_state.data_buffer['rpm'].append(data['rpm'])
            st.session_state.data_buffer['angle'].append(data['angle'])
            st.session_state.data_buffer['freq_fundamental'].append(data['freq_fundamental'])
            st.session_state.data_buffer['freq_harmonic'].append(data['freq_harmonic'])
            st.session_state.data_buffer['defect'].append(data['defect'])
            
            # Actualizare metrici
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🔧 RPM Curent",
                        f"{data['rpm']}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "📐 Unghi Servo",
                        f"{data['angle']}°",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "🎵 Frecvență Fund.",
                        f"{data['freq_fundamental']:.2f} Hz",
                        delta=None
                    )
                
                with col4:
                    defect_status = "DEFECT!" if data['defect'] else "Normal"
                    defect_color = "🔴" if data['defect'] else "🟢"
                    st.metric(
                        "⚠️ Status",
                        f"{defect_color} {defect_status}",
                        delta=None
                    )
            
            # Actualizare grafice
            if len(st.session_state.data_buffer['time']) > 1:
                with chart_placeholder.container():
                    # Creează subplot-uri
                    num_plots = sum([show_rpm, show_angle, show_frequencies])
                    
                    if num_plots > 0:
                        fig = make_subplots(
                            rows=num_plots, cols=1,
                            subplot_titles=[
                                title for title, show in [
                                    ('RPM în Timp Real', show_rpm),
                                    ('Unghi Servomotor', show_angle),
                                    ('Frecvențe (Fundamentală vs Armonică a 3-a)', show_frequencies)
                                ] if show
                            ],
                            vertical_spacing=0.12
                        )
                        
                        times = list(st.session_state.data_buffer['time'])
                        rpms = list(st.session_state.data_buffer['rpm'])
                        angles = list(st.session_state.data_buffer['angle'])
                        freqs_fund = list(st.session_state.data_buffer['freq_fundamental'])
                        freqs_harm = list(st.session_state.data_buffer['freq_harmonic'])
                        defects = list(st.session_state.data_buffer['defect'])
                        
                        # Culori pentru defect
                        colors = ['red' if d else 'green' for d in defects]
                        
                        row = 1
                        
                        # Plot RPM
                        if show_rpm:
                            fig.add_trace(
                                go.Scatter(
                                    x=times, 
                                    y=rpms,
                                    mode='lines+markers',
                                    name='RPM',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=4, color=colors)
                                ),
                                row=row, col=1
                            )
                            
                            # Linie threshold defect
                            fig.add_hline(
                                y=3500, 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text="Threshold Defect (3500 RPM)",
                                row=row, col=1
                            )
                            
                            fig.update_yaxes(title_text="RPM", row=row, col=1)
                            row += 1
                        
                        # Plot Unghi
                        if show_angle:
                            fig.add_trace(
                                go.Scatter(
                                    x=times, 
                                    y=angles,
                                    mode='lines+markers',
                                    name='Unghi Servo',
                                    line=dict(color='purple', width=2),
                                    marker=dict(size=4, color=colors)
                                ),
                                row=row, col=1
                            )
                            
                            fig.update_yaxes(title_text="Unghi (grade)", row=row, col=1)
                            row += 1
                        
                        # Plot Frecvențe
                        if show_frequencies:
                            fig.add_trace(
                                go.Scatter(
                                    x=times, 
                                    y=freqs_fund,
                                    mode='lines+markers',
                                    name='Frecvență Fundamentală',
                                    line=dict(color='green', width=2),
                                    marker=dict(size=4)
                                ),
                                row=row, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=times, 
                                    y=freqs_harm,
                                    mode='lines+markers',
                                    name='Armonică a 3-a',
                                    line=dict(color='orange', width=2, dash='dash'),
                                    marker=dict(size=4)
                                ),
                                row=row, col=1
                            )
                            
                            fig.update_yaxes(title_text="Frecvență (Hz)", row=row, col=1)
                        
                        # Layout general
                        fig.update_xaxes(title_text="Timp (s)")
                        fig.update_layout(
                            height=300 * num_plots,
                            showlegend=True,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Status și statistici
            with status_placeholder.container():
                st.markdown("---")
                st.subheader("📊 Statistici Sesiune")
                
                if len(st.session_state.data_buffer['rpm']) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    rpms_array = np.array(list(st.session_state.data_buffer['rpm']))
                    angles_array = np.array(list(st.session_state.data_buffer['angle']))
                    
                    with col1:
                        st.metric("RPM Mediu", f"{np.mean(rpms_array):.0f}")
                    
                    with col2:
                        st.metric("RPM Max", f"{np.max(rpms_array):.0f}")
                    
                    with col3:
                        st.metric("RPM Min", f"{np.min(rpms_array):.0f}")
                    
                    with col4:
                        defect_count = sum(st.session_state.data_buffer['defect'])
                        defect_percent = (defect_count / len(st.session_state.data_buffer['defect'])) * 100
                        st.metric("% Timp în Defect", f"{defect_percent:.1f}%")
            
            # Tabel cu ultimele 10 citiri
            with table_placeholder.container():
                st.markdown("---")
                st.subheader("📋 Ultimele 10 Citiri")
                
                recent_data = {
                    'Timp (s)': list(st.session_state.data_buffer['time'])[-10:],
                    'RPM': list(st.session_state.data_buffer['rpm'])[-10:],
                    'Unghi (°)': list(st.session_state.data_buffer['angle'])[-10:],
                    'Freq. Fund. (Hz)': [f"{f:.2f}" for f in list(st.session_state.data_buffer['freq_fundamental'])[-10:]],
                    'Freq. Harm. (Hz)': [f"{f:.2f}" for f in list(st.session_state.data_buffer['freq_harmonic'])[-10:]],
                    'Status': ['⚠️ DEFECT' if d else '✅ OK' for d in list(st.session_state.data_buffer['defect'])[-10:]]
                }
                
                df = pd.DataFrame(recent_data)
                df['Timp (s)'] = df['Timp (s)'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        time.sleep(1 / update_rate)
    
    # Auto-refresh
    st.rerun()

else:
    # Mesaj când nu este conectat
    st.info("🔌 Conectează-te la Arduino din sidebar pentru a începe monitorizarea în timp real!")
    
    st.markdown("""
    ### 📋 Instrucțiuni:
    
    1. **Conectare Hardware:**
       - Conectează Arduino Uno la PC prin USB
       - Asigură-te că ai încărcat codul Arduino (document 2)
       - Verifică că toate componentele sunt conectate corect
    
    2. **Conectare în Aplicație:**
       - Selectează portul COM corect din sidebar
       - Baudrate: 115200 (implicit)
       - Apasă "Conectează"
    
    3. **Monitorizare:**
       - Rotește potentiometrul pentru a varia RPM-ul
       - Graficele se actualizează automat în timp real
       - Observă cum servomotorul urmărește poziția
       - Când RPM > 3500, se detectează defect
    
    4. **Funcții:**
       - ✅ Afișare RPM în timp real
       - ✅ Vizualizare unghi servomotor
       - ✅ Monitorizare frecvențe (fundamentală + armonica a 3-a)
       - ✅ Detectare automată defecte
       - ✅ Statistici și istoricul datelor
    
    ### 🎯 Ce vei observa:
    
    - **RPM scăzut (500-3500)**: LED verde, sunet constant, status OK
    - **RPM ridicat (>3500)**: LED roșu clipitor, alternanță frecvențe, DEFECT detectat
    - Graficele arată în **timp real** cum variază parametrii motorului
    """)
    
    # Exemplu grafic static
    st.markdown("---")
    st.subheader("📊 Exemplu Grafice (Date Demo)")
    
    # Generează date demo
    demo_time = np.linspace(0, 10, 100)
    demo_rpm = 2000 + 1500 * np.sin(demo_time * 0.5) + np.random.randn(100) * 100
    demo_angle = (demo_rpm - 500) / (6000 - 500) * 180
    demo_freq = demo_rpm / 60
    
    fig_demo = make_subplots(
        rows=2, cols=1,
        subplot_titles=('RPM în Timp Real (Demo)', 'Unghi Servomotor (Demo)')
    )
    
    fig_demo.add_trace(
        go.Scatter(x=demo_time, y=demo_rpm, mode='lines', name='RPM',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig_demo.add_trace(
        go.Scatter(x=demo_time, y=demo_angle, mode='lines', name='Unghi',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    fig_demo.update_xaxes(title_text="Timp (s)")
    fig_demo.update_yaxes(title_text="RPM", row=1, col=1)
    fig_demo.update_yaxes(title_text="Unghi (°)", row=2, col=1)
    fig_demo.update_layout(height=600, showlegend=True)
    
    # st.plotly_chart(fig_demo, use_container_width=True)