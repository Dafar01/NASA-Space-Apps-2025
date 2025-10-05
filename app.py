# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math

from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import requests

from main import filter

app = FastAPI(title="Impact calculation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # loosened for local dev / hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Natural Earth populated places (for population + city overlays) ----
CITY_DF: Optional[pd.DataFrame] = None

NE_MIRRORS: List[str] = [
    # jsDelivr mirrors (often avoid 403)
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/ne_10m_populated_places_simple.geojson",
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/ne_10m_populated_places.geojson",
    # GitHub raw
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_populated_places_simple.geojson",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_populated_places.geojson",
    # Fallback to 110m if all 10m fail
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/ne_110m_populated_places_simple.geojson",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_populated_places_simple.geojson",
]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) Safari/605.1.15"}

def load_ne_popplaces() -> pd.DataFrame:
    gj = None
    last_err = None
    for url in NE_MIRRORS:
        try:
            r = requests.get(url, headers=UA, timeout=45)
            r.raise_for_status()
            data = r.json()
            feats = data.get("features", []) if isinstance(data, dict) else []
            if len(feats) >= 1000:
                gj = data
                break
            else:
                last_err = f"Too few features ({len(feats)}) from {url}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} from {url}"
            continue
    if gj is None:
        raise RuntimeError(f"All Natural Earth mirrors failed. Last error: {last_err}")

    rows: List[Tuple[str, str, float, float, float]] = []
    for f in gj.get("features", []):
        prop = (f.get("properties") or {})
        pl = {str(k).lower(): v for k, v in prop.items()}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if not (isinstance(coords, list) and len(coords) >= 2):
            continue
        lon, lat = coords[0], coords[1]
        name = pl.get("name") or pl.get("nameascii") or pl.get("name_en") or ""
        country = pl.get("sov0name") or pl.get("adm0name") or pl.get("abbrev") or ""
        pop = (pl.get("pop_max") if pl.get("pop_max") not in (None, "", "NaN")
               else pl.get("popmin") or pl.get("pop_min") or pl.get("popmax"))
        if not name or not country or pop in (None, "", "NaN"):
            continue
        try:
            pop = float(pop)
        except Exception:
            continue
        rows.append((name, country, float(lat), float(lon), float(pop)))

    if not rows:
        raise RuntimeError("Parsed 0 populated places from Natural Earth.")

    df = pd.DataFrame(rows, columns=["city", "country", "lat", "lon", "population"])
    df = df[(df["lat"].between(-60, 85)) & (df["lon"].between(-180, 180))]
    df["population"] = df["population"].clip(lower=10_000, upper=60_000_000)
    df = df.sort_values("population", ascending=False).drop_duplicates(["city", "country"], keep="first")
    return df

def population_within_radius(df: pd.DataFrame, lat: float, lon: float, radius_km: float) -> int:
    if df is None or df.empty:
        return 0
    # bbox prefilter
    deg = radius_km / 111.0
    box = df[(df["lat"].between(lat - deg, lat + deg)) & (df["lon"].between(lon - deg, lon + deg))]
    if box.empty:
        return 0
    R = 6371.0
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(box["lat"].to_numpy())
    lon2 = np.radians(box["lon"].to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    d = 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return int(np.round(box.loc[d <= radius_km, "population"].sum()))

MT_JOULES = 4.184e15
KT_JOULES = 4.184e12
PI = math.pi


_INTERNAL_DENSITY = 3000.0


THERMAL_THRESHOLDS = {
    "fatal_50":  2.0e6,   # 2.0 MJ/m^2
    "clothes":   1.0e6,   # 1.0 MJ/m^2
    "burn_3rd":  5.0e5,   # 0.5 MJ/m^2
    "burn_2nd":  2.0e5    # 0.2 MJ/m^2
}


OP_COEFF = {
    "p20": 0.20,  
    "p5":  0.48,  
    "p1":  1.30   
}

class ImpactInput(BaseModel):
    diameter_m: float = 100.0
    velocity_km_s: float = 20.0
    angle_deg: float = 45.0

def _display_energy_mass(diameter_m: float, velocity_km_s: float):
    r = diameter_m / 2.0
    v = velocity_km_s * 1000.0
    volume = (4.0/3.0) * PI * (r**3)
    mass = _INTERNAL_DENSITY * volume
    ke_j = 0.5 * mass * (v**2)
    return mass, ke_j, (ke_j / MT_JOULES), (ke_j / KT_JOULES)

def _angle_factor(angle_deg: float) -> float:
    c = max(0.2, math.cos(math.radians(min(max(angle_deg, 0.0), 90.0))))
    return c

def _thermal_radii_km(E_joules: float, angle_deg: float):
    k = 0.01
    f = _angle_factor(angle_deg)
    radii = {}
    for key, Fthr in THERMAL_THRESHOLDS.items():
        r_m = math.sqrt(max(0.0, (k * E_joules * f) / (4.0 * PI * Fthr)))
        radii[key] = r_m / 1000.0
    return radii

def _airblast_radii_km(E_joules: float):
    W_kt = E_joules / KT_JOULES
    W13 = max(0.0, W_kt) ** (1.0/3.0)
    return {
        "p20_km": OP_COEFF["p20"] * W13,
        "p5_km":  OP_COEFF["p5"]  * W13,
        "p1_km":  OP_COEFF["p1"]  * W13,
    }

def compute_impact(diameter_m: float, velocity_km_s: float, angle_deg: float = 45.0):
    mass, ke_j, ke_mt, ke_kt = _display_energy_mass(diameter_m, velocity_km_s)
    thermal = _thermal_radii_km(ke_j, angle_deg)
    air = _airblast_radii_km(ke_j)

    thermal_levels = {
        "fatal_50_km": thermal["fatal_50"],
        "clothes_km":  thermal["clothes"],
        "burn3_km":    thermal["burn_3rd"],
        "burn2_km":    thermal["burn_2nd"],
    }

    crater_km = 0.5 * (max(ke_mt, 0.0) ** (1.0/3.0)) * _angle_factor(angle_deg)

    mag_M = (2.0/3.0) * (math.log10(max(ke_j, 1e-6)) - 5.87)

    return {
        "input": {
            "diameter_m": diameter_m,
            "velocity_km_s": velocity_km_s,
            "angle_deg": angle_deg
        },
        "mass_kg": mass,
        "kinetic_energy_joules": ke_j,
        "kinetic_energy_megatons_tnt": ke_mt,
        "effects": {
            "crater_radius_km": crater_km,
            "thermal": { "radii_km": thermal },     
            "airblast": { "radii_km": air },        
            "seismic": { "magnitude_M": mag_M }
        }
    }

@app.post("/impact")
def impact(payload: ImpactInput):
    try:
        return compute_impact(
            payload.diameter_m,
            payload.velocity_km_s,
            payload.angle_deg
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/info")
def info_endpoint():
    result = {}
    result = filter()
    return JSONResponse(content={"data": result})

@app.on_event("startup")
def _load_population_df():
    global CITY_DF
    try:
        CITY_DF = load_ne_popplaces()
        CITY_DF = CITY_DF.astype({"lat": float, "lon": float, "population": float})
        CITY_DF = CITY_DF.dropna(subset=["lat", "lon", "population"])
    except Exception as e:
        print("⚠️ Failed to load population dataset:", e)
        CITY_DF = pd.DataFrame(columns=["lat", "lon", "population"])

@app.get("/population")
def population(lat: float, lng: float, radius_km: float):
    if CITY_DF is None or CITY_DF.empty:
        raise HTTPException(status_code=503, detail="Population data not loaded")
    people = population_within_radius(CITY_DF, lat, lng, radius_km)
    return {"people": int(people)}

@app.get("/citypoints")
def citypoints():
    if CITY_DF is None or CITY_DF.empty:
        raise HTTPException(status_code=503, detail="Population data not loaded")
    # Log scaling keeps small places visible and preserves hot cores
    pop = CITY_DF["population"].astype(float).clip(lower=1.0)
    denom = float(np.log1p(pop.max()))
    pts = [
        [float(r.lat), float(r.lon), float(np.log1p(r.population)) / denom]
        for r in CITY_DF.itertuples()
    ]
    return {"points": pts}

@app.get("/topcities")
def topcities(limit: int = 300):
    if CITY_DF is None or CITY_DF.empty:
        raise HTTPException(status_code=503, detail="Population data not loaded")
    lim = max(1, min(int(limit), 2000))
    cols = ["city", "country", "lat", "lon", "population"]
    df = CITY_DF.loc[:, cols].sort_values("population", ascending=False).head(lim).copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["population"] = df["population"].astype(float)
    return {"cities": df.to_dict(orient="records")}

@app.get("/")
def root():
    return {"message": "POST to /impact with diameter_m, velocity_km_s, angle_deg"}
