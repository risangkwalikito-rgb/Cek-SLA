import re
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Dashboard SLA Pengadaan', page_icon='📊', layout='wide')

DEFAULT_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1DPSQo9DLkCUguFqduWPpB8LfD4jzPKdc4koCxGlv8H0/edit?gid=0#gid=0'


# ----------------------------
# Helpers
# ----------------------------
def extract_sheet_id_and_gid(url: str):
    match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
    if not match:
        raise ValueError('URL Google Sheets tidak valid.')
    sheet_id = match.group(1)

    gid = '0'
    parsed = urlparse(url)
    query_gid = parse_qs(parsed.query).get('gid')
    if query_gid and query_gid[0]:
        gid = query_gid[0]
    else:
        frag_match = re.search(r'gid=(\d+)', parsed.fragment or '')
        if frag_match:
            gid = frag_match.group(1)

    return sheet_id, gid


def build_csv_url(sheet_url: str, gid_override: str | None = None) -> str:
    sheet_id, gid = extract_sheet_id_and_gid(sheet_url)
    if gid_override not in (None, ''):
        gid = str(gid_override)
    return f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'


def normalize_col_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r'\s+', ' ', name)
    return name


def make_unique_columns(values):
    seen = {}
    result = []
    for i, val in enumerate(values):
        name = str(val).strip()
        if name == '' or name.lower() == 'nan':
            name = f'unnamed_{i + 1}'
        count = seen.get(name, 0)
        if count:
            result.append(f'{name}_{count + 1}')
        else:
            result.append(name)
        seen[name] = count + 1
    return result


def score_header_row(row_values) -> int:
    row_text = ' | '.join([normalize_col_name(v) for v in row_values if str(v).strip() not in ('', 'nan')])
    score = 0

    if 'judul' in row_text or 'pekerjaan' in row_text or 'sppbj' in row_text:
        score += 4
    if 'tanggal' in row_text and 'pr' in row_text:
        score += 4
    if 'status' in row_text:
        score += 3
    if 'divisi' in row_text:
        score += 2
    if 'platform' in row_text:
        score += 1
    if 'nomor pr' in row_text:
        score += 1

    non_empty = sum(1 for v in row_values if str(v).strip() not in ('', 'nan'))
    if non_empty >= 8:
        score += 1

    return score


def detect_header_row(raw_df: pd.DataFrame, max_scan_rows: int = 10) -> int:
    best_idx = 0
    best_score = -1
    scan_limit = min(max_scan_rows, len(raw_df))

    for idx in range(scan_limit):
        score = score_header_row(raw_df.iloc[idx].tolist())
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def read_sheet_table(sheet_url: str, gid_override: str | None = None, header_row_number: int | None = None):
    csv_url = build_csv_url(sheet_url, gid_override)
    raw = pd.read_csv(csv_url, header=None, dtype=str)
    raw = raw.dropna(axis=1, how='all').dropna(axis=0, how='all')

    if raw.empty:
        raise ValueError('Sheet kosong atau tidak dapat dibaca.')

    if header_row_number is None:
        header_idx = detect_header_row(raw)
    else:
        header_idx = max(int(header_row_number) - 1, 0)
        if header_idx >= len(raw):
            raise ValueError('Baris header melebihi jumlah baris pada sheet.')

    header_values = make_unique_columns(raw.iloc[header_idx].fillna('').tolist())
    df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
    df.columns = header_values
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    df.columns = [str(c).strip() for c in df.columns]

    return df, csv_url, header_idx + 1


def find_best_column(columns, keywords_all=None, keywords_any=None):
    keywords_all = keywords_all or []
    keywords_any = keywords_any or []
    normalized = {col: normalize_col_name(col) for col in columns}

    for original, col in normalized.items():
        if all(k in col for k in keywords_all) and (not keywords_any or any(k in col for k in keywords_any)):
            return original

    scored = []
    for original, col in normalized.items():
        score = sum(1 for k in keywords_all if k in col) + sum(1 for k in keywords_any if k in col)
        if score > 0:
            scored.append((score, original))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    return None


def map_columns(df: pd.DataFrame):
    cols = list(df.columns)
    mapping = {
        'no': find_best_column(cols, keywords_all=['no']),
        'judul': find_best_column(cols, keywords_any=['judul pekerjaan', 'judul pekerjaan / sppbj', 'judul', 'pekerjaan', 'sppbj']),
        'divisi': find_best_column(cols, keywords_any=['divisi']),
        'sub_divisi': find_best_column(cols, keywords_any=['sub divisi', 'sub_divisi']),
        'lokasi': find_best_column(cols, keywords_any=['lokasi']),
        'tanggal_pr': find_best_column(cols, keywords_all=['tanggal'], keywords_any=['pembuatan pr', 'pr', 'tanggal pembuatan']),
        'tanggal_selesai': find_best_column(cols, keywords_all=['tanggal'], keywords_any=['realisasi', 'po /spbj', 'po/spbj', 'po / spbj', 'spbj']),
        'status': find_best_column(cols, keywords_any=['status']),
        'platform': find_best_column(cols, keywords_any=['platform']),
        'nilai': find_best_column(cols, keywords_any=['nilai pekerjaan', 'nilai po', 'nilai spbj', 'nilai']),
    }
    return mapping


def parse_date_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({'': np.nan, 'nan': np.nan, 'NaT': np.nan, 'None': np.nan})
    parsed = pd.to_datetime(cleaned, errors='coerce', dayfirst=True)

    unresolved = parsed.isna() & cleaned.notna()
    if unresolved.any():
        parsed_fallback = pd.to_datetime(cleaned[unresolved], errors='coerce', dayfirst=False)
        parsed.loc[unresolved] = parsed_fallback

    return parsed


def classify_row(status_text: str, has_end_date: bool) -> str:
    text = str(status_text).upper().strip()
    if 'DOUBLE' in text or 'REVIEW' in text:
        return 'Double / Review'
    if 'BATAL' in text or 'CANCEL' in text:
        return 'Batal'
    if 'REALISASI' in text:
        return 'Selesai'
    if 'ON PROCESS' in text or 'MENUNGGU' in text:
        return 'On Process'
    if has_end_date:
        return 'Selesai'
    if text:
        return 'Belum Lengkap'
    return 'Belum Lengkap'


def rupiah_format(x):
    if pd.isna(x):
        return '-'
    try:
        return f'Rp{x:,.0f}'.replace(',', '.')
    except Exception:
        return str(x)


@st.cache_data(show_spinner=False)
def load_data(sheet_url: str, gid_override: str | None = None, header_row_number: int | None = None):
    df, csv_url, detected_header_row = read_sheet_table(sheet_url, gid_override, header_row_number)
    mapping = map_columns(df)
    required = ['judul', 'tanggal_pr', 'status']
    missing = [k for k in required if not mapping.get(k)]
    if missing:
        raise ValueError(
            'Kolom wajib tidak ditemukan: '
            + ', '.join(missing)
            + '. Coba ubah GID sheet atau set baris header secara manual.'
        )

    work = pd.DataFrame()
    work['No'] = df[mapping['no']] if mapping.get('no') else np.arange(1, len(df) + 1)
    work['Judul Pekerjaan'] = df[mapping['judul']]
    work['Divisi'] = df[mapping['divisi']] if mapping.get('divisi') else ''
    work['Sub Divisi'] = df[mapping['sub_divisi']] if mapping.get('sub_divisi') else ''
    work['Lokasi'] = df[mapping['lokasi']] if mapping.get('lokasi') else ''
    work['Status'] = df[mapping['status']].fillna('')
    work['Platform'] = df[mapping['platform']].fillna('') if mapping.get('platform') else ''

    work['Tanggal PR'] = parse_date_series(df[mapping['tanggal_pr']])
    if mapping.get('tanggal_selesai'):
        work['Tanggal Selesai'] = parse_date_series(df[mapping['tanggal_selesai']])
    else:
        work['Tanggal Selesai'] = pd.NaT

    today = pd.Timestamp.today().normalize()
    work['Kategori'] = [classify_row(s, pd.notna(e)) for s, e in zip(work['Status'], work['Tanggal Selesai'])]
    work['Tanggal Acuan'] = work['Tanggal Selesai'].fillna(today)
    work['SLA Hari'] = (work['Tanggal Acuan'] - work['Tanggal PR']).dt.days

    work = work[work['Tanggal PR'].notna()].copy()
    work = work[work['SLA Hari'].notna()].copy()
    work = work[work['SLA Hari'] >= 0].copy()

    if mapping.get('nilai'):
        raw_nilai = df[mapping['nilai']].astype(str)
        numeric = (
            raw_nilai
            .str.replace('Rp', '', regex=False)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
            .str.extract(r'([0-9]+(?:\.[0-9]+)?)')[0]
        )
        work['Nilai'] = pd.to_numeric(numeric, errors='coerce')
    else:
        work['Nilai'] = np.nan

    work['Divisi'] = work['Divisi'].fillna('').replace('', 'Tanpa Divisi')
    work['Platform'] = work['Platform'].fillna('').replace('', 'Tanpa Platform')
    work['Status'] = work['Status'].fillna('')
    work['Judul Pekerjaan'] = work['Judul Pekerjaan'].fillna('Tanpa Judul')

    meta = {
        'csv_url': csv_url,
        'detected_header_row': detected_header_row,
        'columns_detected': list(df.columns),
        'row_count': int(len(df)),
    }
    return work, mapping, meta


def format_dates(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors='coerce').dt.strftime('%d-%m-%Y')
            out[c] = out[c].fillna('-')
    return out


# ----------------------------
# UI
# ----------------------------
st.title('📊 Dashboard SLA Pengadaan')
st.caption('Menyorot pekerjaan dengan SLA terlama, baik yang sudah selesai maupun yang masih berjalan.')

with st.sidebar:
    st.header('Sumber Data')
    sheet_url = st.text_input('URL Google Sheets', value=DEFAULT_SHEET_URL)
    gid_override = st.text_input('GID Sheet/Tab (opsional)', value='')
    header_mode = st.radio('Baris header', options=['Otomatis', 'Manual'], horizontal=True)
    manual_header_row = None
    if header_mode == 'Manual':
        manual_header_row = st.number_input('Header ada di baris ke-', min_value=1, max_value=20, value=2, step=1)

    target_sla = st.number_input('Target SLA (hari)', min_value=1, max_value=365, value=30, step=1)
    top_n = st.slider('Top pekerjaan yang ditampilkan', min_value=5, max_value=30, value=15, step=1)
    include_double = st.checkbox('Tampilkan status Double / Review', value=False)

try:
    data, column_map, meta = load_data(sheet_url, gid_override or None, manual_header_row)
except Exception as e:
    st.error(f'Gagal memuat data: {e}')
    st.info(
        'Penyebab paling umum: header kolom bukan di baris pertama, atau tab yang dibaca bukan tab data utama. '
        'Coba isi GID Sheet/Tab yang benar atau ubah Baris header ke mode Manual.'
    )
    st.stop()

with st.expander('Info pembacaan data'):
    st.write('Aplikasi membaca sheet publik lewat CSV export URL berikut:')
    st.code(meta['csv_url'])
    st.write(f"Header terdeteksi di baris: {meta['detected_header_row']}")
    st.write(f"Jumlah baris data terbaca: {meta['row_count']}")
    st.write('Pemetaan kolom yang terdeteksi:')
    st.json(column_map)
    st.write('Daftar kolom hasil baca:')
    st.write(meta['columns_detected'])

if not include_double:
    data = data[data['Kategori'] != 'Double / Review'].copy()

with st.sidebar:
    st.header('Filter')
    divisi_options = ['Semua'] + sorted(data['Divisi'].dropna().astype(str).unique().tolist())
    kategori_options = ['Semua'] + sorted(data['Kategori'].dropna().astype(str).unique().tolist())
    platform_options = ['Semua'] + sorted(data['Platform'].dropna().astype(str).unique().tolist())

    selected_divisi = st.selectbox('Divisi', divisi_options)
    selected_kategori = st.selectbox('Kategori', kategori_options)
    selected_platform = st.selectbox('Platform', platform_options)

filtered = data.copy()
if selected_divisi != 'Semua':
    filtered = filtered[filtered['Divisi'] == selected_divisi]
if selected_kategori != 'Semua':
    filtered = filtered[filtered['Kategori'] == selected_kategori]
if selected_platform != 'Semua':
    filtered = filtered[filtered['Platform'] == selected_platform]

filtered['Melewati Target'] = np.where(filtered['SLA Hari'] > target_sla, 'Ya', 'Tidak')

selesai = filtered[filtered['Kategori'] == 'Selesai'].copy().sort_values('SLA Hari', ascending=False)
on_process = filtered[filtered['Kategori'] == 'On Process'].copy().sort_values('SLA Hari', ascending=False)

max_selesai = int(selesai['SLA Hari'].max()) if not selesai.empty else 0
mean_selesai = float(selesai['SLA Hari'].mean()) if not selesai.empty else 0.0
max_on_process = int(on_process['SLA Hari'].max()) if not on_process.empty else 0
count_over_target = int((filtered['SLA Hari'] > target_sla).sum()) if not filtered.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric('SLA selesai terlama', f'{max_selesai} hari')
k2.metric('Rata-rata SLA selesai', f'{mean_selesai:.1f} hari')
k3.metric('Aging on process terlama', f'{max_on_process} hari')
k4.metric('Pekerjaan > target SLA', f'{count_over_target}')

left, right = st.columns(2)

with left:
    st.subheader('Top SLA Terlama - Pekerjaan Selesai')
    if selesai.empty:
        st.info('Tidak ada data pekerjaan selesai pada filter ini.')
    else:
        chart_done = selesai.head(top_n).copy()
        chart_done['Label'] = chart_done['Judul Pekerjaan'].astype(str).str.slice(0, 70)
        fig_done = px.bar(
            chart_done.sort_values('SLA Hari', ascending=True),
            x='SLA Hari',
            y='Label',
            orientation='h',
            hover_data=['Divisi', 'Status', 'Tanggal PR', 'Tanggal Selesai'],
            title='Ranking SLA selesai terlama',
        )
        fig_done.update_layout(height=500, yaxis_title='', xaxis_title='Hari')
        st.plotly_chart(fig_done, use_container_width=True)

with right:
    st.subheader('Top Aging Terlama - On Process')
    if on_process.empty:
        st.info('Tidak ada data on process pada filter ini.')
    else:
        chart_open = on_process.head(top_n).copy()
        chart_open['Label'] = chart_open['Judul Pekerjaan'].astype(str).str.slice(0, 70)
        fig_open = px.bar(
            chart_open.sort_values('SLA Hari', ascending=True),
            x='SLA Hari',
            y='Label',
            orientation='h',
            hover_data=['Divisi', 'Status', 'Tanggal PR'],
            title='Ranking aging on process terlama',
        )
        fig_open.update_layout(height=500, yaxis_title='', xaxis_title='Hari')
        st.plotly_chart(fig_open, use_container_width=True)

st.subheader('Sebaran SLA per Divisi')
summary_divisi = (
    filtered.groupby('Divisi', dropna=False)
    .agg(
        jumlah_pekerjaan=('Judul Pekerjaan', 'count'),
        rata_rata_sla_hari=('SLA Hari', 'mean'),
        sla_maks_hari=('SLA Hari', 'max'),
        selesai=('Kategori', lambda s: (s == 'Selesai').sum()),
        on_process=('Kategori', lambda s: (s == 'On Process').sum()),
        lewat_target=('SLA Hari', lambda s: (s > target_sla).sum()),
    )
    .reset_index()
    .sort_values(['sla_maks_hari', 'rata_rata_sla_hari'], ascending=False)
)

if summary_divisi.empty:
    st.info('Tidak ada data untuk diringkas.')
else:
    fig_div = px.bar(
        summary_divisi,
        x='Divisi',
        y='sla_maks_hari',
        hover_data=['jumlah_pekerjaan', 'rata_rata_sla_hari', 'selesai', 'on_process', 'lewat_target'],
        title='SLA maksimum per divisi',
    )
    fig_div.update_layout(height=420, xaxis_title='', yaxis_title='Hari')
    st.plotly_chart(fig_div, use_container_width=True)

    display_summary = summary_divisi.copy()
    display_summary['rata_rata_sla_hari'] = display_summary['rata_rata_sla_hari'].round(1)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.subheader('Detail pekerjaan dengan SLA tertinggi')
view_mode = st.radio('Tampilan detail', ['Semua', 'Selesai', 'On Process'], horizontal=True)

if view_mode == 'Selesai':
    detail = selesai.copy()
elif view_mode == 'On Process':
    detail = on_process.copy()
else:
    detail = filtered.sort_values('SLA Hari', ascending=False).copy()

detail = detail[[
    'No', 'Judul Pekerjaan', 'Divisi', 'Sub Divisi', 'Lokasi', 'Tanggal PR',
    'Tanggal Selesai', 'Status', 'Kategori', 'Platform', 'SLA Hari', 'Nilai', 'Melewati Target'
]].copy()

detail['Nilai'] = detail['Nilai'].apply(rupiah_format)
detail = format_dates(detail, ['Tanggal PR', 'Tanggal Selesai'])

st.dataframe(
    detail.head(200),
    use_container_width=True,
    hide_index=True,
)

csv_download = detail.copy()
st.download_button(
    label='Unduh hasil analisis (CSV)',
    data=csv_download.to_csv(index=False).encode('utf-8-sig'),
    file_name='analisis_sla_pengadaan.csv',
    mime='text/csv',
)

st.markdown('---')
st.caption(
    'Definisi SLA pada dashboard ini: selisih hari dari Tanggal Pembuatan PR sampai '
    'Tanggal PO/SPBJ/Realisasi. Untuk item yang belum selesai, SLA dihitung sebagai aging sampai hari ini.'
)
