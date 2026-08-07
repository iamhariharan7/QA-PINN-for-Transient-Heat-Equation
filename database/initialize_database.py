import os
import json
import sqlite3
from database.database_manager import DatabaseManager

def initialize():
    """Builds the SQLite database from schema and prepopulates Materials."""
    db_dir = "database"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "experiment.db")
    schema_path = os.path.join(db_dir, "schema.sql")
    
    # 1. Execute Schema
    if not os.path.exists(schema_path):
        print(f"Error: {schema_path} missing.")
        return
        
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()
    
    print(f"Database schema initialized at {db_path}")
    
    # 2. Prepopulate Materials
    mat_db_path = os.path.join("data_generator", "materials", "material_db.json")
    if os.path.exists(mat_db_path):
        manager = DatabaseManager(db_path)
        with open(mat_db_path, "r") as f:
            materials = json.load(f)
            
        for mat in materials:
            # Handle possible lists for anisotropic properties
            def _parse_val(val):
                return json.dumps(val) if isinstance(val, list) else val
                
            db_mat = {
                "name": mat.get("name"),
                "chemical_symbol": mat.get("chemical_symbol"),
                "category": mat.get("category"),
                "thermal_conductivity": _parse_val(mat.get("k")),
                "density": _parse_val(mat.get("rho")),
                "specific_heat": _parse_val(mat.get("cp")),
                "thermal_diffusivity": _parse_val(mat.get("alpha")),
                "is_isotropic": mat.get("type", "Isotropic").lower() == "isotropic"
            }
            if db_mat["name"]:
                manager.insert_material(db_mat)
        
        print(f"Successfully prepopulated {len(materials)} materials into the database.")
    else:
        print(f"Warning: Material DB not found at {mat_db_path}. Materials table is empty.")

if __name__ == "__main__":
    initialize()
