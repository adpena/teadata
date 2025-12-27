import pandas as pd
import time
import requests
from typing import Optional, Tuple, Dict, Any
from geopy.geocoders import ArcGIS, Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

import asyncio

try:
    import aiohttp  # async HTTP client for Census
except ImportError:
    aiohttp = None
try:
    from tqdm.auto import tqdm  # progress bar
except ImportError:
    tqdm = None

tepsac_fp = (
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tepsac/allSchoolData.xlsx"
)

tepsac_df = pd.read_excel(tepsac_fp)

tepsac_df = tepsac_df[~tepsac_df["Closed"]]


tepsac_df["School Full Address"] = (
    tepsac_df["School Street"].fillna("").astype(str)
    + ", "
    + tepsac_df["School City"].fillna("").astype(str)
    + ", "
    + tepsac_df["School State"].fillna("").astype(str)
    + " "
    + tepsac_df["School Zip"].fillna("").astype(str)
).str.strip(", ")

# ----------------------------
# Geocoding helpers (Census, ArcGIS, Nominatim/OSM)
# ----------------------------


def _safe_get(d: Dict[str, Any], path: Tuple[str, ...], default=None):
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def geocode_census(
    addr: str,
    session: Optional[requests.Session] = None,
    retries: int = 2,
    pause: float = 0.2,
) -> Dict[str, Any]:
    """
    Geocode using the free U.S. Census Geocoder (no key required).
    Returns dict with keys: lat, lng, match_type, match_score, raw
    """
    if not addr:
        return {
            "lat": None,
            "lng": None,
            "match_type": None,
            "match_score": None,
            "raw": None,
        }
    sess = session or requests.Session()
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {
        "address": addr,
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    last_exc = None
    for _ in range(retries + 1):
        try:
            r = sess.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                matches = _safe_get(data, ("result", "addressMatches"), []) or []
                if matches:
                    m0 = matches[0]
                    coords = m0.get("coordinates", {})
                    lat = coords.get("y")
                    lng = coords.get("x")
                    match_type = m0.get("matchType")
                    # Census doesn't always return an explicit numeric score; keep None if absent
                    match_score = m0.get("matchQuality") or m0.get("matchScore")
                    return {
                        "lat": lat,
                        "lng": lng,
                        "match_type": match_type,
                        "match_score": match_score,
                        "raw": m0,
                    }
                else:
                    return {
                        "lat": None,
                        "lng": None,
                        "match_type": None,
                        "match_score": None,
                        "raw": data,
                    }
            last_exc = RuntimeError(f"Census HTTP {r.status_code}")
        except Exception as e:
            last_exc = e
        time.sleep(pause)
    # If we exhausted retries
    return {
        "lat": None,
        "lng": None,
        "match_type": None,
        "match_score": None,
        "raw": {"error": str(last_exc)},
    }


# Initialize geocoders (free tiers; respect rate limits)
_arcgis = ArcGIS(
    timeout=15
)  # ArcGIS World Geocoding Service (no key needed for light use)
_nominatim = Nominatim(
    user_agent="teadata_geocoder", timeout=15
)  # OpenStreetMap / Nominatim


def geocode_arcgis(addr: str, retries: int = 2, pause: float = 0.2) -> Dict[str, Any]:
    """
    Geocode with ArcGIS via geopy. Returns dict with keys: lat, lng, score, raw
    """
    if not addr:
        return {"lat": None, "lng": None, "score": None, "raw": None}
    last_exc = None
    for _ in range(retries + 1):
        try:
            res = _arcgis.geocode(addr, exactly_one=True)
            if res:
                lat, lng = res.latitude, res.longitude
                score = None
                try:
                    score = (res.raw or {}).get("score")
                except Exception:
                    pass
                return {
                    "lat": lat,
                    "lng": lng,
                    "score": score,
                    "raw": getattr(res, "raw", None),
                }
            return {"lat": None, "lng": None, "score": None, "raw": None}
        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
            last_exc = e
            time.sleep(pause)
    return {"lat": None, "lng": None, "score": None, "raw": {"error": str(last_exc)}}


def geocode_nominatim(
    addr: str, retries: int = 2, pause: float = 1.0
) -> Dict[str, Any]:
    """
    Geocode with OpenStreetMap Nominatim (free). Respect 1 req/sec guideline.
    Returns dict with keys: lat, lng, importance, cls, typ, raw
    """
    if not addr:
        return {
            "lat": None,
            "lng": None,
            "importance": None,
            "cls": None,
            "typ": None,
            "raw": None,
        }
    last_exc = None
    for _ in range(retries + 1):
        try:
            res = _nominatim.geocode(addr, exactly_one=True)
            if res:
                lat, lng = res.latitude, res.longitude
                raw = getattr(res, "raw", {}) or {}
                return {
                    "lat": lat,
                    "lng": lng,
                    "importance": raw.get("importance"),
                    "cls": raw.get("class"),
                    "typ": raw.get("type"),
                    "raw": raw,
                }
            return {
                "lat": None,
                "lng": None,
                "importance": None,
                "cls": None,
                "typ": None,
                "raw": None,
            }
        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
            last_exc = e
        # Nominatim rate-limit
        time.sleep(pause)
    return {
        "lat": None,
        "lng": None,
        "importance": None,
        "cls": None,
        "typ": None,
        "raw": {"error": str(last_exc)},
    }


# ----------------------------
# Async wrappers for polite parallelism
# ----------------------------


async def geocode_census_async(
    addr: str, session: "aiohttp.ClientSession", retries: int = 2, pause: float = 0.2
) -> Dict[str, Any]:
    # Fallback to sync if aiohttp not available
    if session is None:
        return geocode_census(addr)
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {"address": addr, "benchmark": "Public_AR_Current", "format": "json"}
    last_exc = None
    for _ in range(retries + 1):
        try:
            async with session.get(url, params=params, timeout=15) as r:
                if r.status == 200:
                    data = await r.json()
                    matches = _safe_get(data, ("result", "addressMatches"), []) or []
                    if matches:
                        m0 = matches[0]
                        coords = m0.get("coordinates", {})
                        lat = coords.get("y")
                        lng = coords.get("x")
                        match_type = m0.get("matchType")
                        match_score = m0.get("matchQuality") or m0.get("matchScore")
                        return {
                            "lat": lat,
                            "lng": lng,
                            "match_type": match_type,
                            "match_score": match_score,
                            "raw": m0,
                        }
                    else:
                        return {
                            "lat": None,
                            "lng": None,
                            "match_type": None,
                            "match_score": None,
                            "raw": data,
                        }
                last_exc = RuntimeError(f"Census HTTP {r.status}")
        except Exception as e:
            last_exc = e
        await asyncio.sleep(pause)
    return {
        "lat": None,
        "lng": None,
        "match_type": None,
        "match_score": None,
        "raw": {"error": str(last_exc)},
    }


async def geocode_arcgis_async(
    addr: str, sem: asyncio.Semaphore, pause: float = 0.0
) -> Dict[str, Any]:
    # Use a semaphore to keep parallelism reasonable; run the synchronous geocoder in a thread.
    async with sem:
        res = await asyncio.to_thread(geocode_arcgis, addr)
        if pause:
            await asyncio.sleep(pause)
        return res


async def geocode_nominatim_async(
    addr: str, sem: asyncio.Semaphore, pause: float = 1.05
) -> Dict[str, Any]:
    # Nominatim guideline ~1 request/second. Use a single-permit semaphore and a sleep.
    async with sem:
        res = await asyncio.to_thread(geocode_nominatim, addr)
        # Ensure spacing between calls to be polite
        await asyncio.sleep(pause)
        return res


async def geocode_all_async(
    addr: str,
    sem_arc: asyncio.Semaphore,
    sem_osm: asyncio.Semaphore,
    census_session: "aiohttp.ClientSession" = None,
) -> Dict[str, Dict[str, Any]]:
    census_coro = (
        geocode_census_async(addr, census_session)
        if census_session
        else asyncio.to_thread(geocode_census, addr)
    )
    arcgis_coro = geocode_arcgis_async(addr, sem_arc)
    osm_coro = geocode_nominatim_async(addr, sem_osm)
    census, arc, osm = await asyncio.gather(census_coro, arcgis_coro, osm_coro)
    return {"census": census, "arcgis": arc, "osm": osm}


def geocode_all(addr: str) -> Dict[str, Dict[str, Any]]:
    if addr in address_cache:
        return address_cache[addr]
    census = geocode_census(addr)
    # brief pause between providers to be polite
    time.sleep(0.2)
    arc = geocode_arcgis(addr)
    time.sleep(0.2)
    osm = geocode_nominatim(addr)
    bundled = {"census": census, "arcgis": arc, "osm": osm}
    address_cache[addr] = bundled
    return bundled


def pick_best(
    bundled: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Preference order with simple quality heuristics:
    1) ArcGIS if score >= 95
    2) Census if match_type suggests precise rooftop/point-level
    3) OSM if importance is reasonably high
    Otherwise, fall back to the first non-null.
    Returns (lat, lng, provider)
    """
    arc = bundled.get("arcgis", {})
    cen = bundled.get("census", {})
    osm = bundled.get("osm", {})

    arc_lat, arc_lng, arc_score = arc.get("lat"), arc.get("lng"), arc.get("score")
    cen_lat, cen_lng, cen_type = (
        cen.get("lat"),
        cen.get("lng"),
        (cen.get("match_type") or "").lower() if cen.get("match_type") else "",
    )
    osm_lat, osm_lng, osm_imp = osm.get("lat"), osm.get("lng"), osm.get("importance")

    # 1) ArcGIS with high score
    try:
        if (
            arc_lat is not None
            and arc_lng is not None
            and arc_score is not None
            and float(arc_score) >= 95
        ):
            return arc_lat, arc_lng, "arcgis"
    except Exception:
        pass

    # 2) Census precise types
    if (
        cen_lat is not None
        and cen_lng is not None
        and any(k in cen_type for k in ("exact", "point"))
    ):
        return cen_lat, cen_lng, "census"

    # 3) OSM with decent importance
    try:
        if (
            osm_lat is not None
            and osm_lng is not None
            and (osm_imp is None or float(osm_imp) >= 0.5)
        ):
            return osm_lat, osm_lng, "osm"
    except Exception:
        pass

    # Fallback: first non-null by provider preference
    for prov_key in ("arcgis", "census", "osm"):
        p = bundled.get(prov_key, {})
        if p.get("lat") is not None and p.get("lng") is not None:
            return p["lat"], p["lng"], prov_key
    return None, None, "none"


# Perform geocoding on unique addresses to minimize requests
unique_addrs = tepsac_df["School Full Address"].dropna().astype(str).unique().tolist()

address_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

results_map: Dict[str, Dict[str, Dict[str, Any]]] = {}


# Helper for async geocoding with address included
async def _geocode_with_addr(
    addr: str,
    sem_arc: asyncio.Semaphore,
    sem_osm: asyncio.Semaphore,
    session: "aiohttp.ClientSession" = None,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    bundled = await geocode_all_async(addr, sem_arc, sem_osm, session)
    return addr, bundled


async def build_results_map_with_progress(addresses):
    sem_arc = asyncio.Semaphore(5)
    sem_osm = asyncio.Semaphore(1)
    session_cm = aiohttp.ClientSession() if aiohttp is not None else None
    pbar = None
    try:
        session = session_cm if session_cm else None
        if tqdm is not None:
            pbar = tqdm(total=len(addresses), desc="Geocoding addresses", unit="addr")
        # Create tasks that each return (addr, bundled)
        tasks = [
            asyncio.create_task(_geocode_with_addr(addr, sem_arc, sem_osm, session))
            for addr in addresses
        ]
        # Consume as tasks finish
        for fut in asyncio.as_completed(tasks):
            addr, bundled = await fut
            results_map[addr] = bundled
            if pbar:
                pbar.update(1)
    finally:
        if pbar:
            pbar.close()
        if session_cm:
            await session_cm.close()


# Run the async builder
asyncio.run(build_results_map_with_progress(unique_addrs))

# Write provider-specific columns
tepsac_df["Census Lat"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("census", {}).get("lat")
)
tepsac_df["Census Lng"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("census", {}).get("lng")
)
tepsac_df["Census Match Type"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("census", {}).get("match_type")
)
tepsac_df["Census Match Score"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("census", {}).get("match_score")
)

tepsac_df["ArcGIS Lat"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("arcgis", {}).get("lat")
)
tepsac_df["ArcGIS Lng"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("arcgis", {}).get("lng")
)
tepsac_df["ArcGIS Score"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("arcgis", {}).get("score")
)

tepsac_df["OSM Lat"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("osm", {}).get("lat")
)
tepsac_df["OSM Lng"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("osm", {}).get("lng")
)
tepsac_df["OSM Importance"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("osm", {}).get("importance")
)
tepsac_df["OSM Class"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("osm", {}).get("cls")
)
tepsac_df["OSM Type"] = tepsac_df["School Full Address"].map(
    lambda a: results_map.get(a, {}).get("osm", {}).get("typ")
)

# Best-pick coordinates with provenance
best_triplets = tepsac_df["School Full Address"].map(
    lambda a: pick_best(results_map.get(a, {"arcgis": {}, "census": {}, "osm": {}}))
)
tepsac_df["Best Lat"] = best_triplets.map(lambda t: t[0])
tepsac_df["Best Lng"] = best_triplets.map(lambda t: t[1])
tepsac_df["Best Provider"] = best_triplets.map(lambda t: t[2])

# Optional: save out for inspection
tepsac_df.to_csv(
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tepsac/tepsac_geocoded.csv",
    index=False,
)

print(f"Geocoded {len(results_map)} unique addresses. Saved tepsac_geocoded.csv.")
