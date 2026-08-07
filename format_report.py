import os
import glob

def get_latest_report():
    reports = glob.glob(os.path.join("outputs", "*", "Final_Report.txt"))
    if not reports:
        return None
    return max(reports, key=os.path.getctime)

def parse_and_format():
    report_path = get_latest_report()
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace literal \n with actual newlines if present
    content = content.replace('\\n', '\n')
    lines = content.split('\n')
        
    data = {}
    current_method = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('METRICS_'):
            current_method = line.replace('METRICS_', '').strip()
            data[current_method] = {}
        elif ':' in line and current_method:
            parts = line.split(':', 1)
            data[current_method][parts[0].strip()] = parts[1].strip()
            
    # Generate MD
    md = "# Simulation Metrics Report\n\n"
    dims = ["1D", "2D", "3D"]
    methods = ["CFD", "PINN", "QA", "CNN"]
    
    for dim in dims:
        md += f"## {dim} Benchmarks\n\n"
        md += "| Metric | CFD | PINN | QA-PINN | CNN |\n"
        md += "|---|---|---|---|---|\n"
        
        keys = set()
        for m in methods:
            m_key = f"{m}_{dim}" if dim != "1D" else m
            if m_key in data:
                keys.update(data[m_key].keys())
        keys = list(keys)
        
        keys.sort()
        if "Parameters" in keys: keys.remove("Parameters"); keys.append("Parameters")
        if "Memory_MB" in keys: keys.remove("Memory_MB"); keys.append("Memory_MB")
                
        for k in keys:
            row = [k]
            for m in methods:
                m_key = f"{m}_{dim}" if dim != "1D" else m
                val = data.get(m_key, {}).get(k, "-")
                try:
                    v_float = float(val)
                    if v_float == 0:
                        val = "0"
                    elif 'e' in val.lower():
                        val = f"{v_float:.2e}"
                    elif v_float.is_integer() or k == "Parameters":
                        val = f"{int(v_float):,}"
                    else:
                        val = f"{v_float:.6f}"
                except:
                    pass
                row.append(val)
            md += "| " + " | ".join(row) + " |\n"
        md += "\n"
        
    with open("Final_Report_Tabular.md", "w", encoding='utf-8') as f:
        f.write(md)

if __name__ == "__main__":
    parse_and_format()
