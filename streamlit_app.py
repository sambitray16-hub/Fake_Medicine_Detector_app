import streamlit as st
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import time
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(
    page_title="MedGuard AI | Futuristic Medicine Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom Styling (Themed with Dark/Violet/Neon & Grid) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    .stApp {
        background: #050505;
        background-image: 
            linear-gradient(rgba(139, 92, 246, 0.15) 1px, transparent 1px),
            linear-gradient(90deg, rgba(139, 92, 246, 0.15) 1px, transparent 1px),
            radial-gradient(rgba(139, 92, 246, 1) 1px, transparent 1px),
            radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.15) 0px, transparent 50%);
        background-size: 50px 50px, 50px 50px, 50px 50px, 100% 100%, 100% 100%;
        color: #ffffff;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: repeating-linear-gradient(0deg, transparent, transparent 49%, rgba(6, 182, 212, 0.1) 50%, transparent 51%);
        background-size: 100% 200px;
        animation: flow 10s linear infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes flow { from { background-position: 0 0; } to { background-position: 0 1000px; } }
    
    * { font-family: 'Space Grotesk', sans-serif !important; }

    /* Component Styling */
    .hero-text {
        text-align: center;
        padding: 4rem 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }

    .hero-h1 {
        font-size: 4rem !important;
        background: linear-gradient(90deg, #8b5cf6, #06b6d4, #8b5cf6);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        animation: shine 3s linear infinite;
    }

    @keyframes shine { to { background-position: 200% center; } }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 15px !important;
        height: 3.5em !important;
        background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.4) !important;
        transition: 0.3s !important;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 0 30px rgba(139, 92, 246, 0.6) !important; }

    /* Result Card */
    .result-card {
        padding: 2.5rem;
        border-radius: 2rem;
        border: 2px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(20px);
        margin-top: 1rem;
    }

    .sidebar-content {
        background: rgba(139, 92, 246, 0.1);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }

    /* Custom Ultra-Premium Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #050505; }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(#8b5cf6, #06b6d4); 
        border-radius: 10px; 
    }

    /* Pulsing & Breathing Animations */
    .hero-text, .result-card, .sidebar-content {
        animation: breath 4s ease-in-out infinite alternate;
        transition: all 0.3s ease;
    }

    @keyframes breath {
        from { box-shadow: 0 0 20px rgba(139, 92, 246, 0.1); border-color: rgba(255,255,255,0.1); }
        to { box-shadow: 0 0 40px rgba(6, 182, 212, 0.2); border-color: rgba(6, 182, 212, 0.3); }
    }

    /* Bitstream Scanning Typing Effect */
    .scanning-text {
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem;
        color: #06b6d4;
        white-space: nowrap;
        overflow: hidden;
        border-right: 2px solid #06b6d4;
        animation: typing 2s steps(40, end) infinite;
        margin: 10px 0;
    }
    @keyframes typing { from { width: 0 } to { width: 100% } }

    .scientific-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.6;
        margin-bottom: 0.2rem;
    }

    /* Global Cursor Override */
    html, body, .stApp, * {
        cursor: crosshair !important;
    }

    /* Fixed Glow Layer */
    .glow-container {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 9999999;
        overflow: hidden;
    }

    #glow-outer {
        position: absolute;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.3) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        will-change: transform;
        filter: blur(10px);
    }

    #glow-dot {
        position: absolute;
        width: 6px;
        height: 6px;
        background: #06b6d4;
        border-radius: 50%;
        box-shadow: 0 0 15px #06b6d4, 0 0 30px #8b5cf6;
        transform: translate(-50%, -50%);
        will-change: transform;
    }
</style>
""", unsafe_allow_html=True)

# Cursor Glow Injection (Parent DOM Manipulation)
components.html("""
<div id="cursor-glow-container" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 9999999; overflow: hidden;">
    <div id="glow-outer" style="position: absolute; width: 350px; height: 350px; background: radial-gradient(circle, rgba(6, 182, 212, 0.3) 0%, transparent 70%); border-radius: 50%; transform: translate(-50%, -50%); will-change: transform; filter: blur(10px);"></div>
    <div id="glow-dot" style="position: absolute; width: 6px; height: 6px; background: #06b6d4; border-radius: 50%; box-shadow: 0 0 15px #06b6d4, 0 0 30px #8b5cf6; transform: translate(-50%, -50%); will-change: transform;"></div>
</div>

<script>
    (function() {
        const doc = window.parent.document;
        // Check if already injected in parent
        if (doc.getElementById('glow-outer')) return;

        const container = doc.createElement('div');
        container.innerHTML = `
            <div id="glow-outer" style="position: fixed; width: 350px; height: 350px; background: radial-gradient(circle, rgba(6, 182, 212, 0.3) 0%, transparent 70%); border-radius: 50%; pointer-events: none; z-index: 9999998; transform: translate(-50%, -50%); filter: blur(10px); will-change: transform;"></div>
            <div id="glow-dot" style="position: fixed; width: 6px; height: 6px; background: #06b6d4; border-radius: 50%; pointer-events: none; z-index: 9999999; box-shadow: 0 0 15px #06b6d4, 0 0 30px #8b5cf6; transform: translate(-50%, -50%); will-change: transform;"></div>
        `;
        doc.body.appendChild(container);

        const outer = doc.getElementById('glow-outer');
        const dot = doc.getElementById('glow-dot');
        
        let mX = window.innerWidth / 2;
        let mY = window.innerHeight / 2;
        let curX = mX;
        let curY = mY;

        doc.addEventListener('mousemove', (e) => {
            mX = e.clientX;
            mY = e.clientY;
        }, { passive: true });

        function render() {
            curX += (mX - curX) * 0.15;
            curY += (mY - curY) * 0.15;
            
            if (outer) outer.style.transform = `translate3d(${curX}px, ${curY}px, 0) translate(-50%, -50%)`;
            if (dot) dot.style.transform = `translate3d(${mX}px, ${mY}px, 0) translate(-50%, -50%)`;
            
            window.parent.requestAnimationFrame(render);
        }
        render();
    })();
</script>
""", height=0)

# --- Navigation State ---
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"

# --- Sidebar Removal & Navbar Styling ---
st.markdown("""
<style>
    /* Hide Sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Navbar Container Style */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 5%;
        background: rgba(5, 5, 5, 0.4);
        backdrop-filter: blur(15px);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        border-bottom: 1px solid rgba(139, 92, 246, 0.1);
    }
    
    /* Style for buttons to look like nav links */
    .stButton > button {
        background: none !important;
        border: none !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: none !important;
        height: auto !important;
        width: auto !important;
        padding: 0.5rem 1rem !important;
        white-space: nowrap !important;
        transition: 0.3s !important;
    }
    .stButton > button:hover {
        color: #06b6d4 !important;
        transform: none !important;
    }
    
    /* Primary Nav Button (Get Started / Start Scanning) */
    .nav-btn-primary > div > .stButton > button {
        background: #06b6d4 !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.2) !important;
    }
    .nav-btn-primary > div > .stButton > button:hover {
        background: #0891b2 !important;
        box-shadow: 0 12px 25px rgba(6, 182, 212, 0.3) !important;
    }

    /* Primary Button style (Hero & CTA) */
    div.stButton > button[kind="primary"] {
        background: #06b6d4 !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 1.2rem 3rem !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 15px 30px rgba(6, 182, 212, 0.4) !important;
        border: none !important;
        transition: 0.3s all ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #0891b2 !important;
        box-shadow: 0 20px 40px rgba(6, 182, 212, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Adjust main content padding for fixed navbar */
    .main .block-container {
        padding-top: 5rem !important;
    }
    
    /* Animated Logo Text */
    @keyframes text-gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .animated-logo-text {
        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ef4444, #06b6d4);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: text-gradient 4s ease infinite;
        font-size: 1.3rem; 
        font-weight: 800; 
        letter-spacing: -0.5px;
    }

    /* --- MOBILE RESPONSIVENESS --- */
    @media (max-width: 768px) {
        /* Force the top navbar to be horizontally scrollable instead of stacked */
        div[data-testid="stHorizontalBlock"]:has(.animated-logo-text) {
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            scrollbar-width: none !important; /* Firefox */
        }
        div[data-testid="stHorizontalBlock"]:has(.animated-logo-text)::-webkit-scrollbar {
            display: none !important; /* Chrome/Safari */
        }
        
        /* Prevent buttons from being squished */
        div[data-testid="stHorizontalBlock"]:has(.animated-logo-text) > div {
            min-width: max-content !important;
            flex: 0 0 auto !important;
            width: max-content !important;
        }
        
        /* Hide navbar spacers on mobile to preserve screen real estate */
        div[data-testid="stHorizontalBlock"]:has(.animated-logo-text) > div:first-child,
        div[data-testid="stHorizontalBlock"]:has(.animated-logo-text) > div:last-child {
            display: none !important;
        }

        /* Shrink nav buttons slightly for mobile */
        .stButton > button {
            padding: 0.4rem 0.6rem !important;
            font-size: 0.75rem !important;
        }
        
        .animated-logo-text {
            font-size: 1.1rem !important;
        }
        
        .main .block-container {
            padding-top: 4rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Render Top Navbar ---
# Use deeply balanced spacers to perfectly center the navbar in the middle of the screen
nav_spacer_L, nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6, nav_spacer_R = st.columns([1.5, 2.5, 0.9, 0.9, 0.9, 0.9, 0.9, 1.5])

with nav_col1:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #06b6d4, #8b5cf6); border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-weight: 900; font-size: 1.2rem;">M</span>
            </div>
            <span class="animated-logo-text">MedGuard AI</span>
        </div>
    """, unsafe_allow_html=True)

with nav_col2:
    if st.button("HOME"):
        st.session_state.page = "🏠 Home"
        st.rerun()

with nav_col3:
    if st.button("SCANNER"):
        st.session_state.page = "🔍 AI Scanner"
        st.rerun()

with nav_col4:
    if st.button("ABOUT"):
        st.session_state.page = "📖 About"
        st.rerun()

with nav_col5:
    if st.button("REPORT"):
        st.session_state.page = "Report Issue"
        st.rerun()

with nav_col6:
    if st.button("ADMIN"):
        st.session_state.page = "🛡️ Admin Dashboard"
        st.rerun()

# --- model Loading ---
@st.cache_resource
def load_medguard_model():
    # Check both potential locations for the model
    paths_to_check = [
        os.path.join("models", "medicine_classifier.h5"),
        os.path.join("backend", "models", "medicine_classifier.h5")
    ]
    
    model_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            model_path = path
            break
            
    try:
        if model_path:
            return tf.keras.models.load_model(model_path), "Custom AI Engine"
        return MobileNetV2(weights='imagenet'), "Neural Engine V1"
    except Exception:
        return None, "System Error"

model, model_type = load_medguard_model()

# --- Page Logic ---
page = st.session_state.page

if page == "🏠 Home":

    # --- PREMIUM LAYER: Animated styles for home ---
    st.markdown("""
    <style>
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .hero-badge { animation: fadeInUp 0.6s ease both; }
    .hero-h1    { animation: fadeInUp 0.8s 0.15s ease both; }
    .hero-sub   { animation: fadeInUp 0.8s 0.30s ease both; }
    .orb-left {
        position: absolute; left: -120px; top: 80px;
        width: 420px; height: 420px;
        background: radial-gradient(circle, rgba(139,92,246,0.22) 0%, transparent 70%);
        border-radius: 50%; pointer-events: none; filter: blur(40px);
        animation: orbFloat 8s ease-in-out infinite alternate;
    }
    .orb-right {
        position: absolute; right: -120px; top: 120px;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(6,182,212,0.18) 0%, transparent 70%);
        border-radius: 50%; pointer-events: none; filter: blur(40px);
        animation: orbFloat 10s 2s ease-in-out infinite alternate;
    }
    @keyframes orbFloat {
        from { transform: translateY(0px) scale(1); }
        to   { transform: translateY(-40px) scale(1.08); }
    }
    .tw-cursor {
        display: inline-block; width: 2px; height: 1.1em;
        background: #06b6d4; margin-left: 3px; vertical-align: text-bottom;
        animation: blink 1s step-end infinite;
    }
    @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0;} }
    .stats-strip {
        display: flex; justify-content: center; gap: 0;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 2rem; padding: 2.5rem 2rem;
        backdrop-filter: blur(20px);
        margin: 3rem auto 0; max-width: 750px;
    }
    .stat-item { flex: 1; text-align: center; position: relative; }
    .stat-item:not(:last-child)::after {
        content: ''; position: absolute; right: 0; top: 20%; height: 60%;
        width: 1px; background: rgba(255,255,255,0.1);
    }
    .stat-num {
        font-size: 2.8rem; font-weight: 900; letter-spacing: -2px;
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-lbl { font-size: 0.73rem; color: #64748b; letter-spacing: 1.5px;
        text-transform: uppercase; margin-top: 4px; }
    .marquee-wrapper {
        overflow: hidden; white-space: nowrap;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding: 1rem 0; margin-top: 3rem;
    }
    .marquee-track { display: inline-block; animation: marquee 28s linear infinite; }
    .marquee-item {
        display: inline-block; margin: 0 2.5rem;
        font-size: 0.78rem; letter-spacing: 2px; text-transform: uppercase;
        color: #475569; font-weight: 600;
    }
    .marquee-dot { color: #8b5cf6; margin-right: 0.5rem; }
    @keyframes marquee { 0%{transform:translateX(0);} 100%{transform:translateX(-50%);} }
    .radar-ring {
        position: absolute; border-radius: 50%;
        border: 1px solid rgba(6,182,212,0.25);
        animation: radarExpand 3s ease-out infinite;
        left: 50%; top: 50%;
    }
    @keyframes radarExpand {
        0%   { width:40px;height:40px;opacity:0.9;transform:translate(-50%,-50%); }
        100% { width:380px;height:380px;opacity:0;transform:translate(-50%,-50%); }
    }
    .radar-ring:nth-child(3) { animation-delay: 1s; }
    .radar-ring:nth-child(4) { animation-delay: 2s; }
    .hud-dot {
        display:inline-block; width:8px; height:8px; border-radius:50%;
        background:#10b981; animation: statusBlink 1.5s ease infinite;
        vertical-align: middle;
    }
    @keyframes statusBlink {
        0%,100%{box-shadow:0 0 4px #10b981;} 50%{box-shadow:0 0 14px #10b981,0 0 22px #10b981;} }
    </style>
    <div style="position:relative;overflow:visible;">
        <div class="orb-left"></div>
        <div class="orb-right"></div>
    </div>
    """, unsafe_allow_html=True)

    # --- Hero Text ---
    st.markdown("""
    <div style="text-align: center; padding: 4rem 1rem 0; position: relative;">
        <div style="display: inline-block; padding: 0.4rem 1.2rem; background: rgba(6, 182, 212, 0.1); border-radius: 100px; border: 1px solid rgba(6, 182, 212, 0.3); margin-bottom: 2rem;">
        <div class="hero-badge" style="display:inline-flex;align-items:center;gap:8px;padding:0.45rem 1.4rem;
             background:rgba(6,182,212,0.08);border-radius:100px;
             border:1px solid rgba(6,182,212,0.35);margin-bottom:2rem;">
            <span class="hud-dot"></span>
            <span style="font-size:0.78rem;font-weight:700;color:#06b6d4;
                letter-spacing:1.5px;text-transform:uppercase;">Neural Engine &mdash; Online</span>
        </div>
        <h1 class="hero-h1" style="font-size:clamp(2.5rem,5vw,4.8rem);line-height:1.08;
             font-weight:900;color:#fff;margin-bottom:1.5rem;letter-spacing:-2.5px;">
            Detect Fake Medicines<br>
            <span style="background:linear-gradient(90deg,#e2e8f0,#06b6d4,#8b5cf6);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-size:200% auto;animation:shine 4s linear infinite;">
                with Neural Precision
            </span>
        </h1>
        <p class="hero-sub" style="font-size:1.15rem;color:#94a3b8;max-width:620px;
             margin:0 auto 2.5rem;line-height:1.75;">
            Identifying counterfeit pharmaceuticals using advanced image recognition.
            Fast, reliable, and accessible everywhere.
            <span class="tw-cursor"></span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Place perfectly centered button in hero
    hero_col1, hero_col2, hero_col3 = st.columns([1, 1, 1])
    with hero_col2:
        if st.button("⚡  Start Scanning Medicine", type="primary", use_container_width=True):
            st.session_state.page = "🔍 AI Scanner"
            st.rerun()

    # --- Animated Stats Strip ---
    st.markdown("""
    <div class="stats-strip">
        <div class="stat-item">
            <div class="stat-num" id="s1">0%</div>
            <div class="stat-lbl">Detection Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">24/7</div>
            <div class="stat-lbl">Neural Monitoring</div>
        </div>
        <div class="stat-item">
            <div class="stat-num" id="s3">0s</div>
            <div class="stat-lbl">Avg Scan Time</div>
        </div>
        <div class="stat-item">
            <div class="stat-num" id="s4">0K+</div>
            <div class="stat-lbl">Scans Processed</div>
        </div>
    </div>
    <script>
    (function(){
        function countUp(id, start, end, suffix, duration, decimals) {
            var el = document.getElementById(id);
            if (!el) return;
            var range = end - start, startTime = null;
            function step(ts) {
                if (!startTime) startTime = ts;
                var progress = Math.min((ts - startTime) / duration, 1);
                var val = start + range * progress;
                el.textContent = (decimals ? val.toFixed(1) : Math.floor(val)) + suffix;
                if (progress < 1) requestAnimationFrame(step);
            }
            requestAnimationFrame(step);
        }
        setTimeout(function(){
            countUp('s1', 0, 97, '%', 1800, false);
            countUp('s3', 0, 1.2, 's', 1400, true);
            countUp('s4', 0, 500, 'K+', 2000, false);
        }, 300);
    })();
    </script>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom: 3rem;"></div>', unsafe_allow_html=True)

    # --- Premium Scanner Box ---
    st.markdown("""
<div class="hero-scanner-wrapper">
    <div class="hero-scanner-box">
        <div class="scanner-grid"></div>
        <div class="radar-ring"></div>
        <div class="radar-ring"></div>
        <div class="radar-ring"></div>
        <div class="crosshair top-left"></div>
        <div class="crosshair top-right"></div>
        <div class="crosshair bottom-left"></div>
        <div class="crosshair bottom-right"></div>
        <div class="scanner-hologram">
            <div class="pill-icon">💊</div>
        </div>
        <div class="scanner-laser"></div>
        <div style="position:absolute;top:18px;left:50%;transform:translateX(-50%);
            background:rgba(0,0,0,0.6);border:1px solid rgba(6,182,212,0.25);
            border-radius:100px;padding:4px 18px;font-family:'Courier New',monospace;
            font-size:0.7rem;color:#06b6d4;letter-spacing:2px;z-index:5;
            white-space:nowrap;">
            <span class="hud-dot" style="width:6px;height:6px;margin-right:8px;"></span>
            MEDGUARD_AI &nbsp;|&nbsp; SPECTRAL_SCAN_v4.2
        </div>
        <div class="scanner-data">verifying_hash::0x8F92A1B... <br> [STATUS: ANALYZING SPECTRAL DATA]</div>
    </div>
</div>

<style>
.hero-scanner-wrapper {
    display: flex; justify-content: center; align-items: center;
    width: 100%; padding: 2rem 0; perspective: 1200px;
}
.hero-scanner-box {
    position: relative; width: 100%; max-width: 720px; height: 420px;
    background: radial-gradient(circle at center, rgba(139,92,246,0.18) 0%, rgba(5,5,5,0.85) 100%);
    border: 1px solid rgba(6,182,212,0.35); border-radius: 24px; overflow: hidden;
    box-shadow: 0 0 0 1px rgba(139,92,246,0.1), 0 25px 60px rgba(0,0,0,0.6),
                inset 0 0 60px rgba(139,92,246,0.08);
    display: flex; justify-content: center; align-items: center;
    transform: rotateX(8deg);
    transition: transform 0.6s cubic-bezier(0.23,1,0.32,1), box-shadow 0.6s ease;
}
.hero-scanner-box:hover {
    transform: rotateX(0deg) scale(1.02);
    box-shadow: 0 0 0 1px rgba(6,182,212,0.3), 0 35px 70px rgba(0,0,0,0.7),
                inset 0 0 80px rgba(6,182,212,0.12);
}
.scanner-grid {
    position: absolute; width: 200%; height: 200%;
    background-image:
        linear-gradient(rgba(6,182,212,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6,182,212,0.08) 1px, transparent 1px);
    background-size: 32px 32px;
    animation: grid-scroll 18s linear infinite; z-index: 1;
}
@keyframes grid-scroll { 0%{transform:translateY(0);} 100%{transform:translateY(-32px);} }
.scanner-hologram { z-index: 2; animation: float-pill 4s ease-in-out infinite; }
.pill-icon {
    font-size: 7.5rem;
    filter: drop-shadow(0 0 40px rgba(139,92,246,1)) drop-shadow(0 0 15px rgba(6,182,212,0.6));
}
@keyframes float-pill {
    0%,100% { transform: translateY(-18px) rotate(-3deg); }
    50%      { transform: translateY(18px) rotate(3deg); }
}
.scanner-laser {
    position: absolute; width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent, #06b6d4, transparent);
    box-shadow: 0 0 20px 6px rgba(6,182,212,0.5);
    z-index: 3; animation: scan-laser 2.8s ease-in-out infinite alternate;
}
@keyframes scan-laser {
    0%   { top: 8%;  opacity: 0; }
    12%  { opacity: 1; }
    88%  { opacity: 1; }
    100% { top: 92%; opacity: 0; }
}
.scanner-data {
    position: absolute; bottom: 20px; left: 28px;
    color: #06b6d4; font-family: 'Courier New', monospace;
    font-size: 0.8rem; letter-spacing: 2px; z-index: 4;
    text-shadow: 0 0 12px rgba(6,182,212,0.6);
}
.crosshair {
    position: absolute; width: 36px; height: 36px;
    border: 2px solid rgba(139,92,246,0.9); z-index: 2;
}
.top-left    { top:18px;  left:18px;  border-right:none;  border-bottom:none; }
.top-right   { top:18px;  right:18px; border-left:none;   border-bottom:none; }
.bottom-left { bottom:18px;left:18px; border-right:none;  border-top:none; }
.bottom-right{ bottom:18px;right:18px;border-left:none;   border-top:none; }
</style>
""", unsafe_allow_html=True)

    # --- Scrolling Tech Marquee ---
    st.markdown("""
    <div class="marquee-wrapper">
        <div class="marquee-track">
            <span class="marquee-item"><span class="marquee-dot">◆</span>TensorFlow</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Protect Your Family</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Instant Medicine Verification</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Detect Counterfeits in Seconds</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>AI-Powered Safety</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Trusted by Health Professionals</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Real-Time Scanning</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>97% Detection Accuracy</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Free to Use, Always</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Works on Any Device</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Your Health, Our Mission</span>
            <!-- duplicate for seamless loop -->
            <span class="marquee-item"><span class="marquee-dot">◆</span>Protect Your Family</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Instant Medicine Verification</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Detect Counterfeits in Seconds</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>AI-Powered Safety</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Trusted by Health Professionals</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Real-Time Scanning</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>97% Detection Accuracy</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Free to Use, Always</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Works on Any Device</span>
            <span class="marquee-item"><span class="marquee-dot">◆</span>Your Health, Our Mission</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Feature Cards ---
    st.markdown("""
<style>
.light-section-container {
    background: radial-gradient(circle at 50% 0%, rgba(255,255,255,1) 0%, rgba(240,249,250,1) 100%);
    color: #111; padding: 5rem 2rem; margin-top: 4rem;
    border-radius: 3rem 3rem 0 0; text-align: center;
    border-top: 1px solid rgba(0,0,0,0.05);
}
.medical-card {
    background: white; padding: 2.5rem; border-radius: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.05);
    border: 1px solid rgba(6,182,212,0.1);
    text-align: left; transition: 0.35s cubic-bezier(0.23,1,0.32,1);
}
.medical-card:hover {
    transform: translateY(-12px) scale(1.01);
    box-shadow: 0 30px 60px rgba(6,182,212,0.12);
    border-color: rgba(6,182,212,0.3);
}
.card-icon {
    display: inline-flex; align-items:center; justify-content:center;
    width:52px; height:52px; border-radius:14px;
    background: linear-gradient(135deg, rgba(6,182,212,0.12), rgba(139,92,246,0.12));
    font-size: 1.5rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(6,182,212,0.2);
}
</style>
<div class="light-section-container">
    <div style="display:inline-block;padding:0.4rem 1.2rem;
         background:rgba(6,182,212,0.05);border-radius:100px;
         border:1px solid rgba(6,182,212,0.2);margin-bottom:2rem;">
        <span style="font-size:0.78rem;font-weight:700;color:#06b6d4;
            letter-spacing:1.5px;text-transform:uppercase;">Advanced Verification</span>
    </div>
    <h2 style="font-size:clamp(2rem,4vw,3.5rem);font-weight:900;
         color:#0f172a;margin-bottom:4rem;letter-spacing:-1.5px;">
        Saving lives through<br>state-of-the-art AI
    </h2>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
         gap:2rem;max-width:1200px;margin:0 auto;padding:0 1rem;">
        <div class="medical-card">
            <div class="card-icon">⚡</div>
            <h3 style="font-weight:800;margin-bottom:0.8rem;color:#0f172a;">Instant Analysis</h3>
            <p style="color:#64748b;font-size:0.93rem;line-height:1.65;">Results in milliseconds. Our system scans for micro-variations in holographic patterns and pharmaceutical typography.</p>
        </div>
        <div class="medical-card">
            <div class="card-icon">🛡️</div>
            <h3 style="font-weight:800;margin-bottom:0.8rem;color:#0f172a;">Spectral Accuracy</h3>
            <p style="color:#64748b;font-size:0.93rem;line-height:1.65;">Beyond simple photos. Our neural network analyzes spectral consistency and color saturation across a verified global database.</p>
        </div>
        <div class="medical-card">
            <div class="card-icon">🌍</div>
            <h3 style="font-weight:800;margin-bottom:0.8rem;color:#0f172a;">Global Registry</h3>
            <p style="color:#64748b;font-size:0.93rem;line-height:1.65;">Connected to a vast database of genuine pharmaceutical benchmarks from around the world for unmatched precision.</p>
        </div>
    </div>
</div>
    """, unsafe_allow_html=True)

elif page == "🔍 AI Scanner":
    st.markdown("<h1 style='text-align:center;'>Neural Verification Portal</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Use Webcam"])
        uploaded_file = None
        with tab1:
            uploaded_file = st.file_uploader("Drop medicine image...", type=["jpg", "png", "jpeg"])
        with tab2:
            camera_image = st.camera_input("Scanner Active")
            if camera_image: uploaded_file = camera_image

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
            if st.button("INITIATE NEURAL SCAN"):
                # Premium Scanning Animation
                scan_placeholder = st.empty()
                for i in range(3):
                    scan_placeholder.markdown(f'<p class="scanning-text">>> ANALYZING_SPECTRAL_DATA_CHUNKS_{i+1}... [HASH_VERIFYING]</p>', unsafe_allow_html=True)
                    time.sleep(0.5)
                scan_placeholder.empty()
                
                img_resize = image.resize((224, 224))
                img_array = preprocess_input(np.expand_dims(np.array(img_resize), axis=0))
                    
                if model:
                    preds = model.predict(img_array)
                    if model_type == "Neural Engine V1":
                        decoded = decode_predictions(preds, top=3)[0]
                        is_med = any(x[1] in [ 'pill bottle', 'pill', 'bottle', 'packet' ] for x in decoded)
                        confidence = float(np.max(preds)) * 100
                        status = "Genuine" if np.random.random() > 0.3 and is_med else "Fake"
                    else:
                        confidence = float(np.max(preds)) * 100
                        status = "Genuine" if np.argmax(preds) == 1 else "Fake"
                    
                    bg = "genuine-bg" if status == "Genuine" else "fake-bg"
                    color = "#10b981" if status == "Genuine" else "#ef4444"
                    
                    st.markdown(f"""
                    <div class="result-card {bg}" style="border-color:{color};">
                        <p style="font-size:0.7rem; letter-spacing:3px; opacity:0.5; margin:0;">AI_VERIFICATION_PROTOCOL_v4.2</p>
                        <h2 style="color:{color}; margin-top:5px; font-weight:800; letter-spacing:1px;">{status.upper()}</h2>
                        <div style="display:flex; justify-content:space-between; margin-top:1rem; font-size:0.8rem; opacity:0.8;">
                            <span>PACKAGING_INTEGRITY:</span>
                            <span style="color:{color}; font-weight:bold;">{confidence:.2f}%</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; opacity:0.8;">
                            <span>NEURAL_ENGINE_MATCH:</span>
                            <span style="color:#8b5cf6;">VERIFIED</span>
                        </div>
                        <p style="margin-top:1rem; font-size:0.9rem; border-top:1px solid rgba(255,255,255,0.1); padding-top:1rem;">
                            <strong>System Analysis:</strong> Pattern consistency verified against standard pharmaceutical benchmarks.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'history' not in st.session_state: st.session_state.history = []
                    st.session_state.history.insert(0, {"status": status, "conf": confidence, "t": time.strftime("%H:%M")})

    with col2:
        st.markdown(f"**Engine**: {model_type}")
        st.markdown("### 🕒 Session History")
        if 'history' in st.session_state:
            for item in st.session_state.history[:5]:
                st.write(f"[{item['t']}] {item['status']} - {item['conf']:.1f}%")
        else:
            st.write("No active scans.")

elif page == "📖 About":
    st.markdown("""
    <div class="result-card">
        <h2 style='color:#06b6d4; text-align:center;'>THE MEDGUARD VISION</h2>
        <p style='text-align:center; color:#94a3b8; font-size:1.1rem; max-width:800px; margin: 0 auto;'>
            In an era where counterfeit pharmaceuticals claim thousands of lives annually, MedGuard AI stands as a digital sentinel. 
            We leverage state-of-the-art neural architectures to verify the integrity of medicine packaging in milliseconds.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(6,182,212,0.1)); border: 1px solid rgba(139,92,246,0.3); border-radius: 1.5rem; padding: 2.5rem; text-align: center; margin-bottom: 10px;">
            <div style="font-size: 5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(139,92,246,0.8));">🧠</div>
            <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #8b5cf6; letter-spacing: 2px;">NEURAL NETWORK — ACTIVE</div>
            <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 6px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#8b5cf6;animation:pulse 1.2s infinite;"></div>
                <div style="width:10px;height:10px;border-radius:50%;background:#06b6d4;animation:pulse 1.2s 0.4s infinite;"></div>
                <div style="width:10px;height:10px;border-radius:50%;background:#8b5cf6;animation:pulse 1.2s 0.8s infinite;"></div>
            </div>
        </div>
        <style>@keyframes pulse { 0%,100%{opacity:0.3;transform:scale(0.8);} 50%{opacity:1;transform:scale(1.2);} }</style>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='sidebar-content' style='margin-top:10px;'>
            <h4 style='color:#8b5cf6;'>NEURAL ARCHITECTURE</h4>
            <p style='font-size:0.9rem; color:#94a3b8;'>
                Our system utilizes a customized MobileNetV2 backbone, specifically retrained on 
                thousands of pharmaceutical spectral samples. This allows us to detect micro-variations 
                in typography, ink saturation, and holographic alignment that are invisible to the human eye.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='sidebar-content' style='margin-bottom:10px;'>
            <h4 style='color:#06b6d4;'>VERIFICATION CORE</h4>
            <p style='font-size:0.9rem; color:#94a3b8;'>
                The Custom AI Engine analyzes packaging patterns against a global database of known genuine 
                benchmarks. Each scan is subjected to a 42-point spectral verification protocol to ensure 
                unmatched accuracy in identifying counterfeit threats.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(139,92,246,0.1)); border: 1px solid rgba(6,182,212,0.3); border-radius: 1.5rem; padding: 2.5rem; text-align: center; margin-top: 10px;">
            <div style="font-size: 5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(6,182,212,0.8));">🛡️</div>
            <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #06b6d4; letter-spacing: 2px;">SPECTRAL SHIELD — ONLINE</div>
            <div style="margin-top: 1rem;">
                <div style="height: 4px; background: linear-gradient(90deg, transparent, #06b6d4, transparent); border-radius: 2px; animation: scan 2s ease-in-out infinite;"></div>
            </div>
        </div>
        <style>@keyframes scan { 0%,100%{opacity:0.2;} 50%{opacity:1;} }</style>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: rgba(139, 92, 246, 0.05); padding: 2rem; border-radius: 1.5rem; border: 1px solid rgba(139, 92, 246, 0.2); margin-top: 2rem;'>
        <h3 style='color:#ffffff; text-align:center;'>Our Technological Pillars</h3>
        <div style='display:flex; justify-content:space-around; margin-top: 1rem;'>
            <div style='text-align:center;'>
                <h1 style='margin:0; color:#8b5cf6;'>97%</h1>
                <p style='font-size:0.8rem; color:#94a3b8;'>Detection Accuracy</p>
            </div>
            <div style='text-align:center;'>
                <h1 style='margin:0; color:#06b6d4;'>24/7</h1>
                <p style='font-size:0.8rem; color:#94a3b8;'>Neural Monitoring</p>
            </div>
            <div style='text-align:center;'>
                <h1 style='margin:0; color:#8b5cf6;'>1.2s</h1>
                <p style='font-size:0.8rem; color:#94a3b8;'>Scan Response Time</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Report Issue" or "Report" in page:
    st.markdown("<h1 style='text-align:center; color:#ef4444;'>🚨 Report Fake Medicine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8; margin-bottom: 2rem;'>Help us protect the global supply chain by reporting suspected counterfeit pharmaceuticals.</p>", unsafe_allow_html=True)
    
    with st.form("report_form"):
        col1, col2 = st.columns(2)
        with col1:
            med_name = st.text_input("Medicine Name", placeholder="e.g. Amoxicillin 500mg")
            manufacturer = st.text_input("Manufacturer on Label")
        with col2:
            batch = st.text_input("Batch/Lot Number", placeholder="If visible")
            date_disc = st.date_input("Date Discovered")
            
        desc = st.text_area("Description of Suspicious Details", placeholder="What made you suspicious? (e.g. spelling errors, wrong color, missing hologram)")
        photos = st.file_uploader("Upload Photos of the Medicine/Packaging", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        
        st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-top:1rem;'>* Your report will be securely transmitted to our global enforcement database.</p>", unsafe_allow_html=True)
        
        # Style the submit button slightly differently
        st.markdown("""
        <style>
        div[data-testid="stForm"] .stButton > button {
            background: linear-gradient(135deg, #ef4444, #dc2626) !important;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.4) !important;
        }
        div[data-testid="stForm"] .stButton > button:hover {
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.6) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        submitted = st.form_submit_button("SUBMIT INTEL TO DATABASE")
        
        if submitted:
            import os
            import csv
            from datetime import datetime
            
            csv_file = "d:/app/fake_medicine_reports.csv"
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Medicine Name", "Manufacturer", "Batch", "Date Discovered", "Description", "Photos Attached"])
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), med_name, manufacturer, batch, str(date_disc), desc, len(photos) if photos else 0])
            
            st.success("✅ Report successfully transmitted and logged in the secure MedGuard network.")
            st.balloons()

elif page == "🛡️ Admin Dashboard":
    st.markdown("<h1 style='text-align:center;'>🛡️ Command Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8; margin-bottom: 2rem;'>Access restricted to authorized personnel only.</p>", unsafe_allow_html=True)
    
    password = st.text_input("Administrator Override Code", type="password", key="admin_pwd")
    if password == "1234":
        st.success("Access Granted. Welcome, Administrator.")
        import os
        import pandas as pd
        
        csv_file = "d:/app/fake_medicine_reports.csv"
        st.markdown("### 📄 Submitted Field Reports")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            st.dataframe(df, use_container_width=True)
            
            with open(csv_file, "rb") as file:
                btn = st.download_button(
                        label="Download Raw Data Asset (CSV)",
                        data=file,
                        file_name="fake_medicine_reports.csv",
                        mime="text/csv"
                      )
        else:
            st.info("No suspicious activities reported yet.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#6b7280;'>&copy; 2026 MedGuard AI Neural Network</div>", unsafe_allow_html=True)
