import pandas as pd
import numpy as np
import warnings
import re
import os
import glob

def parse_scientific(val_str):
    if val_str is None or pd.isna(val_str):
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)
    
    s = str(val_str).strip()
    if s.startswith('['):
        try:
            import ast
            return float(ast.literal_eval(s)[0])
        except Exception:
            pass

    superscripts = {
        '\u2070': '0', '\u00B9': '1', '\u00B2': '2', '\u00B3': '3', '\u2074': '4',
        '\u2075': '5', '\u2076': '6', '\u2077': '7', '\u2078': '8', '\u2079': '9',
        '\u207A': '+', '\u207B': '-'
    }
    for k, v in superscripts.items():
        s = s.replace(k, v)
        
    s = re.sub(r'[\s\u00d7\xc3\x97\xef\xbf\xbdxX\*]+10\^?', 'e', s)
    
    try:
        return float(s)
    except Exception:
        return None

def normalize_text(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower().strip())

def _find_column(df, candidates):
    cols = {normalize_text(c): c for c in df.columns}
    for cand in candidates:
        norm_cand = normalize_text(cand)
        if norm_cand in cols:
            return cols[norm_cand]
        for col_norm, col_orig in cols.items():
            if norm_cand in col_norm:
                return col_orig
    return None

def load_material_properties(excel_path, target_material, tolerance=5e-3):
    """
    Loads material physical properties from aerospace_materials.xlsx or csv.
    Uses target_material to match against material_name or aliases.
    """
    tolerance = float(tolerance)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Material database not found at {excel_path}")
        
    try:
        df = pd.read_csv(excel_path) if excel_path.endswith('.csv') else pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"Could not read excel file at {excel_path}: {e}")
        
    col_name = _find_column(df, ['material_name', 'name', 'material'])
    col_alias = _find_column(df, ['aliases', 'alias'])
    col_k = _find_column(df, ['conductivity_k_W_mK', 'conductivity', 'k'])
    col_alpha = _find_column(df, ['diffusivity_alpha_m2_s', 'diffusivity', 'alpha'])
    col_rho = _find_column(df, ['density_rho_kg_m3', 'density', 'rho'])
    col_cp = _find_column(df, ['heat_capacity_cp_J_kgK', 'heat_capacity', 'cp', 'specific_heat'])
    col_app = _find_column(df, ['application', 'app'])

    if not col_name:
        raise ValueError(f"Required column 'material_name' missing in {excel_path}")

    target_norm = normalize_text(target_material)
    matched_row = None
    
    # 1. Exact / normalized match on material_name or aliases
    for i in range(len(df)):
        row = df.iloc[i]
        mat_name = str(row[col_name])
        mat_norm = normalize_text(mat_name)
        alias_str = str(row[col_alias]) if col_alias and pd.notna(row[col_alias]) else ""
        aliases_norm = [normalize_text(a) for a in alias_str.split(',')]

        if target_norm == mat_norm or target_norm in aliases_norm:
            matched_row = row
            break
            
    # 2. Token match
    if matched_row is None:
        for i in range(len(df)):
            row = df.iloc[i]
            mat_name = str(row[col_name])
            alias_str = str(row[col_alias]) if col_alias and pd.notna(row[col_alias]) else ""
            tokens = set(re.findall(r'[a-z0-9]+', (mat_name + " " + alias_str).lower()))
            if target_norm in tokens:
                matched_row = row
                break

    # 3. Substring match fallback
    if matched_row is None:
        for i in range(len(df)):
            row = df.iloc[i]
            mat_name = str(row[col_name])
            alias_str = str(row[col_alias]) if col_alias and pd.notna(row[col_alias]) else ""
            if target_norm in normalize_text(mat_name) or target_norm in normalize_text(alias_str):
                matched_row = row
                break

    if matched_row is None:
        raise ValueError(f"Material '{target_material}' not found in {excel_path}.")

    k = parse_scientific(matched_row[col_k]) if col_k else None
    alpha_excel = parse_scientific(matched_row[col_alpha]) if col_alpha else None
    rho = parse_scientific(matched_row[col_rho]) if col_rho else None
    cp = parse_scientific(matched_row[col_cp]) if col_cp else None
    aliases = str(matched_row[col_alias]) if col_alias and pd.notna(matched_row[col_alias]) else ""
    application = str(matched_row[col_app]) if col_app and pd.notna(matched_row[col_app]) else ""

    if k is None or rho is None or cp is None:
        raise ValueError(f"Failed to parse essential properties for {target_material}. Please check excel values.")
        
    if any(v <= 0 for v in [k, rho, cp]):
        raise ValueError("Material properties must be strictly positive.")
        
    alpha_calc = k / (rho * cp)
    if alpha_excel is None:
        alpha_excel = alpha_calc
    else:
        if abs(alpha_calc - alpha_excel) / alpha_excel > tolerance:
            warnings.warn(f"Calculated alpha ({alpha_calc:.4e}) differs from Excel alpha ({alpha_excel:.4e}) by more than {tolerance*100}%.")

    return {
        'name': str(matched_row[col_name]),
        'k': k,
        'alpha': alpha_excel,
        'rho': rho,
        'cp': cp,
        'aliases': aliases,
        'application': application
    }

def discover_available_materials(dataset_dir="data/dataset", excel_path="data/aerospace_materials.xlsx"):
    """
    Scans data/dataset/ directory to discover available material folders,
    and matches each with physical properties in aerospace_materials.xlsx.
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found at {dataset_dir}")

    folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not folders:
        raise ValueError(f"No material subdirectories found under {dataset_dir}")

    discovered = []
    for folder in sorted(folders):
        folder_path = os.path.join(dataset_dir, folder)
        try:
            props = load_material_properties(excel_path, folder)
        except Exception:
            props = {
                'name': folder,
                'k': 1.0, 'alpha': 0.01, 'rho': 1000.0, 'cp': 1000.0,
                'aliases': '', 'application': 'Unknown'
            }
        discovered.append({
            'folder_name': folder,
            'display_name': props['name'],
            'folder_path': folder_path,
            'properties': props
        })
    return discovered

def _load_data_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)

def _find_file_pattern(folder_path, dim_pattern):
    matches = glob.glob(os.path.join(folder_path, dim_pattern))
    if not matches:
        all_files = os.listdir(folder_path)
        pattern_regex = re.compile(dim_pattern.replace('*', '.*'), re.IGNORECASE)
        matches = [os.path.join(folder_path, f) for f in all_files if pattern_regex.match(f)]
    if not matches:
        raise ValueError(f"Could not locate pattern '{dim_pattern}' in material folder '{folder_path}'")
    return matches[0]

def load_simulation_datasets(material_folder_path):
    """
    Loads 1D, 2D, and 3D simulation datasets from a material folder.
    Validates required columns and returns grid parameters and temperature arrays.
    """
    file_1d = _find_file_pattern(material_folder_path, "*_1D.*")
    file_2d = _find_file_pattern(material_folder_path, "*_2D.*")
    file_3d = _find_file_pattern(material_folder_path, "*_3D.*")

    df_1d = _load_data_file(file_1d)
    df_2d = _load_data_file(file_2d)
    df_3d = _load_data_file(file_3d)

    req_1d = {'x_m', 'time_s', 'temperature_K'}
    req_2d = {'x_m', 'y_m', 'time_s', 'temperature_K'}
    req_3d = {'x_m', 'y_m', 'z_m', 'time_s', 'temperature_K'}

    if not req_1d.issubset(set(df_1d.columns)):
        missing = req_1d - set(df_1d.columns)
        raise ValueError(f"1D dataset '{file_1d}' missing required columns: {missing}")

    if not req_2d.issubset(set(df_2d.columns)):
        missing = req_2d - set(df_2d.columns)
        raise ValueError(f"2D dataset '{file_2d}' missing required columns: {missing}")

    if not req_3d.issubset(set(df_3d.columns)):
        missing = req_3d - set(df_3d.columns)
        raise ValueError(f"3D dataset '{file_3d}' missing required columns: {missing}")

    x1 = np.sort(df_1d['x_m'].unique())
    t1 = np.sort(df_1d['time_s'].unique())
    piv_1d = df_1d.pivot_table(index='time_s', columns='x_m', values='temperature_K')
    U_1d = piv_1d.values

    x2 = np.sort(df_2d['x_m'].unique())
    y2 = np.sort(df_2d['y_m'].unique())
    t2 = np.sort(df_2d['time_s'].unique())
    piv_2d = df_2d.pivot_table(index=['time_s', 'y_m'], columns='x_m', values='temperature_K')
    U_2d = piv_2d.values.reshape(len(t2), len(y2), len(x2))

    x3 = np.sort(df_3d['x_m'].unique())
    y3 = np.sort(df_3d['y_m'].unique())
    z3 = np.sort(df_3d['z_m'].unique())
    t3 = np.sort(df_3d['time_s'].unique())
    piv_3d = df_3d.pivot_table(index=['time_s', 'x_m', 'y_m'], columns='z_m', values='temperature_K')
    U_3d = piv_3d.values.reshape(len(t3), len(x3), len(y3), len(z3))

    return {
        '1D': {'x': x1, 't': t1, 'U': U_1d, 'Lx': float(x1.max() - x1.min()), 'T': float(t1.max() - t1.min())},
        '2D': {'x': x2, 'y': y2, 't': t2, 'U': U_2d, 'Lx': float(x2.max() - x2.min()), 'Ly': float(y2.max() - y2.min()), 'T': float(t2.max() - t2.min())},
        '3D': {'x': x3, 'y': y3, 'z': z3, 't': t3, 'U': U_3d, 'Lx': float(x3.max() - x3.min()), 'Ly': float(y3.max() - y3.min()), 'Lz': float(z3.max() - z3.min()), 'T': float(t3.max() - t3.min())}
    }
