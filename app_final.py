import streamlit as st
import pandas as pd
import altair as alt
from google import genai
import openpyxl
from google.genai import types
import io

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Midi AI Social Media Analyzer")

# --- INITIALIZE CLIENT ---
try:
    # Ensure .streamlit/secrets.toml contains: GEMINI_API_KEY = "your_key_here"
    client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
except Exception as e:
    client = None

# --- HELPER: CONSTANTS ---
MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December'
]
MONTH_MAP = {month: i+1 for i, month in enumerate(MONTH_ORDER)}

# --- FUNCTION 1: LOAD EKSPERIMEN DATA ---
@st.cache_data
def load_eksperimen_data(uploaded_file):
    """
    Loads 'Looker' and 'Looker_ERTotal' sheets.
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
        
        # 1. LOAD LOOKER SHEET (Main Data)
        if "Looker" in xls.sheet_names:
            df_looker = pd.read_excel(xls, sheet_name="Looker")
            
            # Data Cleaning
            numeric_cols = ['ER', 'Followers Total', 'Likes', 'Reach']
            for col in numeric_cols:
                if col in df_looker.columns:
                    df_looker[col] = pd.to_numeric(df_looker[col], errors='coerce')
            
            # Logic: Followers Count "Drags Down" (ffill)
            df_looker['Followers Total'] = df_looker['Followers Total'].fillna(method='ffill')
            
        else:
            st.error("Sheet 'Looker' not found")
            df_looker = pd.DataFrame()

        # 2. LOAD LOOKER_ERTOTAL (Summary Data)
        if "Looker_ERTotal" in xls.sheet_names:
            df_total = pd.read_excel(xls, sheet_name="Looker_ERTotal")
            if 'ER%' in df_total.columns:
                df_total['ER%'] = pd.to_numeric(df_total['ER%'], errors='coerce')
        else:
            df_total = pd.DataFrame()

        return df_looker, df_total

    except Exception as e:
        st.error(f"Error loading Excel: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- FUNCTION 2: LOAD THRESHOLD FILE ---
@st.cache_data
def load_threshold_file(uploaded_file):
    try:
        # Load the whole file as string/dataframe context
        df = pd.read_excel(uploaded_file) 
        return df
    except Exception as e:
        st.error(f"Error loading Threshold file: {e}")
        return pd.DataFrame()

# --- FUNCTION 3: GEMINI ANALYSIS ---

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

def get_gemini_analysis(branch_name, month, df_filtered, df_threshold, follower_count):
    if client is None:
        return "API Client not initialized. Please check API Key."

    # 1. Prepare Content Context (The Posts)
    posts_summary = df_filtered[['Judul Konten', 'Tematik', 'ER', 'Likes', 'Bentuk']].to_markdown(index=False)
    
    # 2. Prepare Rules Context (The Threshold File)
    # The AI will read this to determine Account Category AND Performance Status
    threshold_context = df_threshold.to_markdown(index=False)
    
    prompt_text = f"""
    Kamu adalah Social Media Analyst untuk tim Sosial Media 'Alfamidi Gema Budaya'.
    Analisis performa dari cabang '{branch_name}' untuk bulan {month}.
    
    Perhatikan aturan berikut.
    ### 1. Aturan penilaian
    Berikut adalah konteks tabel penilaian Kategori Akun (berdasarkan Followers) dan Standar Performa (Berdasarkan ER).
    '''
    {threshold_context}
    '''

    ### 2. DATA
    - **Current Followers:** {follower_count}
    - **Branch:** {branch_name}

    ### 3. POST
    '''
    {posts_summary}
    '''

    ### 4. INSTRUCTIONS
    JAWAB SECARA SPESIFIK DENGAN ATURAN BERIKUT :

    **A. Kategorisasi Akun (Langkah 1)**
    - Periksa kolom “Pengikut” di tabel Aturan.
    - Identifikasi kategori mana yang termasuk akun ini (misalnya, Mikro, Kecil, Besar, dll.) berdasarkan jumlah pengikut {follower_count}.
    Contoh Jawaban:
    Kategori Akun : Menengah/Tinggi/Niche
    
    **B. Evaluasi Kinerja**
    - Berdasarkan kategori yang diidentifikasi pada Langkah A, apakah ER Baik, Normal, atau Buruk? 
    - Sebutkan rentang spesifik dari tabel Aturan yang digunakan untuk penilaian ini.

    **C. Analisis Konten**
    - Tinjau daftar posting. Posting atau tema mana yang kinerjanya di atas standar? Sebutkan dengan poin-poin saja.
    - Mana yang kinerjanya di bawah standar? Sebutkan dengan poin-poin saja.
    - Apakah konten relevan dengan budaya kerja ritel Alfamidi berikut :
        a. Integritas yang tinggi: Menerapkan kejujuran, kedisiplinan, dan konsistensi dalam bekerja dengan berlandaskan etika dan tanggung jawab. 
        b. Inovasi untuk kemajuan yang lebih baik: Bersikap kreatif dan berkomitmen untuk terus memperbaiki cara kerja agar lebih baik lagi. 
        c. Kualitas dan Produktivitas yang tertinggi: Mampu menjalankan tugas dengan fokus untuk mencapai hasil terbaik. 
        d. Kerja sama tim: Terlibat aktif dan mendorong semangat kekompakan dalam tim. 
        e. Kepuasan pelanggan melalui standar pelayanan terbaik: Memiliki inisiatif tinggi untuk memenuhi kebutuhan pelanggan dan memastikan kepuasan mereka.
    Jawab berdasarkan penjelasan tematik yang ada di konteks.
    - Apakah variasi bentuk post sudah ideal di bulan itu? Ikuti aturan pada threshold. 


    **D. Rekomendasi**
    - Berdasarkan kolom “Action/Tindakan” di tabel Aturan untuk tingkat ER spesifik ini, apa yang harus dilakukan bulan depan? Tuliskan dengan beberapa poin saja.

    Tuliskan s-5 sumber website yang menjadi acuan jawaban rekomendasi dan cantumkan link websitenya. Cari dari sumber terpercaya (resmi/studi/jurnal). Jangan ambil dari web blog atau yang bersifat subjektif. Jawab dengan format berikut :
    1. Judul Artikel A - Website A - Link Artikel
    2. Judul Artikel B - Website B - Link Artikel

    Jawab tanpa menambahkan format apapun seperti "Baik, berikut jawaban saya" dan tanda seperti "`" dan "*", buat dalam bentuk poin-poin saja dengan menggunakan tanda "-"
    """

    try:
        with st.spinner('AI sedang membaca threshold, menentukan kategori akun & menganalisis...'):
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt_text,
                config=config
            )
            return response.text
    except Exception as e:
        return f"Error Generating Analysis: {e}"

# --- MAIN APP UI ---
st.title('Midi AI Social Media Analyzer')
st.markdown("Analisis Tren dengan **Kategorisasi & Threshold Otomatis oleh AI**.")

# --- SIDEBAR ---
st.sidebar.header("Upload Data")
file_exp = st.sidebar.file_uploader("Upload Main File", type=["xlsx"])
file_thresh = st.sidebar.file_uploader("Upload Threshold File (Rules)", type=["xlsx"])

if file_exp and file_thresh:
    # Load Data
    df_looker, df_total = load_eksperimen_data(file_exp)
    df_threshold = load_threshold_file(file_thresh)
    
    if not df_looker.empty:
        # --- FILTER SECTION ---
        st.divider()
        st.subheader("Filter Data Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            branches = df_looker['Cabang'].unique()
            selected_branch = st.selectbox("Pilih Cabang (Branch)", branches)
        
        with col2:
            # Dynamic Month Filter
            available_months = df_looker[df_looker['Cabang'] == selected_branch]['Bulan'].unique()
            sorted_months = sorted(available_months, key=lambda x: MONTH_MAP.get(x, 99))
            selected_month = st.selectbox("Pilih Bulan (Month)", sorted_months)

        # Filter Logic
        df_filtered = df_looker[
            (df_looker['Cabang'] == selected_branch) & 
            (df_looker['Bulan'] == selected_month)
        ].copy()

        if not df_filtered.empty:
            # --- KEY METRICS ---
            current_followers = df_filtered['Followers Total'].max()
            
            official_er_val = 0
            if not df_total.empty:
                match = df_total[
                    (df_total['Cabang'] == selected_branch) & 
                    (df_total['Bulan'] == selected_month)
                ]
                if not match.empty:
                    official_er_val = match['ER%'].values[0]

            st.markdown("### 1. Performance Overview")

            m1, m2, m3 = st.columns(3)
            m1.metric("Followers Count", f"{current_followers:,.0f}")
            m3.metric("Official Monthly ER", f"{official_er_val:.2%}" if official_er_val else "N/A")

            #---Visualization---
            st.markdown("### 2. Trend Analysis")
            
            c1, c2 = st.columns([2, 1], width="stretch")
            
            st.caption("ER per Post (Colored by ER % Intensity)")
            base = alt.Chart(df_filtered).encode(
                x=alt.X('Judul Konten', axis=alt.Axis(labels=False), title='Content Posts'),
                y=alt.Y('ER', axis=alt.Axis(format='%'))
            )  

            # 2. Create the line layer
            lines = base.mark_line(
                color='white',
                opacity=0.5  
            )
            points = base.mark_circle(size=100).encode(
                color=alt.Color('ER', scale=alt.Scale(scheme='viridis'), title="ER %"),
                tooltip=['Judul Konten', 'Tematik', alt.Tooltip('ER', format='.2%'), 'Likes']
            )

            chart_posts = (lines + points).interactive()
                
            st.altair_chart(chart_posts, width='stretch')

            # --- DATA TABLE ---
            with st.expander("View Raw Data"):
                st.dataframe(
                    df_filtered[['Judul Konten', 'Tematik', 'Followers Total', 'ER', 'Bentuk']]
                    .style.format({'ER': '{:.2%}', 'Followers Total': '{:,.0f}'})
                )

            # --- AI ANALYSIS ---
            st.divider()
            st.markdown("### 3. AI Strategic Analysis")
            st.info("AI will identify the Account Category & Status based on the uploaded Threshold file.")

            # Check if analysis result exists in session state to avoid regenerating on every rerun
            if 'analysis_result' not in st.session_state:
                st.session_state['analysis_result'] = ""

            if st.button("Generate Analysis"):
                if client:
                    # Generate the analysis
                    analysis_text = get_gemini_analysis(
                        selected_branch, 
                        selected_month, 
                        df_filtered, 
                        df_threshold, 
                        current_followers
                    )
                    # Store the result in session state
                    st.session_state['analysis_result'] = analysis_text
                else:
                    st.error("Gemini Client not configured. Check secrets.toml.")

            # Display the analysis result if it exists
            if st.session_state['analysis_result']:
                st.markdown(st.session_state['analysis_result'])
                
                # --- DOWNLOAD BUTTON: XLSX Format ---
                
                # 1. Convert the analysis string into a DataFrame (single column/row)
                analysis_df = pd.DataFrame(
                    {'AI Analysis': [st.session_state['analysis_result']]}
                )
                
                # 2. Use io.BytesIO buffer to write the DataFrame to an Excel file in memory
                output = io.BytesIO()
                # Write the DataFrame to the buffer as an Excel file (.xlsx)
                analysis_df.to_excel(output, index=False, sheet_name='AI_Analysis')
                # Reset the buffer position to the beginning before reading
                excel_data = output.getvalue()
                
                # Create a file name
                file_name = f"{selected_branch}_{selected_month}_Analysis.xlsx"
                
                st.download_button(
                    label="Download Analysis Result (.xlsx)",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        else:
            st.warning("No data found for this selection.")
    else:
        st.warning("Data Looker kosong atau format salah.")
else:
    st.info("Silakan upload kedua file Excel di sidebar.")