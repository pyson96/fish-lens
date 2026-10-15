"""
Frozen-backbone fish feature analysis (no training). Backbones are defined in backbones.py.

Fish image -> frozen backbone (default DINOv2 ViT-S/14) -> image embedding
-> L2 normalize -> PCA / t-SNE / UMAP / silhouette / leakage-aware k-NN / class-mean cosine similarity.

Usage:
    python analyze_dinov2.py                          # Fish4Knowledge (official, 4 species)
    python analyze_dinov2.py --backbone siglip2_so400m --dataset inat_aquarium --aquarium-only
    python analyze_dinov2.py --dataset 76cr3wfhff-1   # local RAR dataset (8 species)
    python analyze_dinov2.py --dataset inat_captive   # iNaturalist captive (aquarium) photos
    python analyze_dinov2.py --dataset inat_captive --preprocess pad   # pad-to-square, no crop
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from torchvision import transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Official Fish4Knowledge Recognition Ground Truth (Univ. of Edinburgh)
F4K_DIR = ROOT / "dataset" / "fish4knowledge"
F4K_DOWNLOAD_DIR = ROOT / "dataset" / "_downloads"
F4K_BASE = "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/Archive/fish_image"
# (species name, official cluster id) - ids 01..04 on the official page
F4K_SPECIES = [
    ("Dascyllus reticulatus", "01"),
    ("Plectroglyphidodon dickii", "02"),
    ("Chromis chrysura", "03"),
    ("Amphiprion clarkii", "04"),
]

# Local dataset: one RAR per species ("<Genus> <Species>.rar")
LOCAL_DIR = ROOT / "dataset" / "76cr3wfhff-1"
# Same-species photos taken within this window = same specimen (e.g. both sides)
SAME_SPECIMEN_SECONDS = 60

# iNaturalist captive (aquarium) observations
INAT_DIR = ROOT / "dataset" / "inat_captive"
INAT_API = "https://api.inaturalist.org/v1"
INAT_SPECIES = [  # (scientific name, Korean name)
    ("Gnathanodon speciosus", "골든 트레발리"),
    ("Caranx ignobilis", "자이언트 트레발리(GT)"),
    ("Epinephelus lanceolatus", "자이언트 그루퍼"),
    ("Trachinotus blochii", "폼파노"),
    ("Rhina ancylostoma", "목탁수구리"),
]
INAT_LICENSES = {"cc0", "cc-by", "cc-by-nc"}  # photo licenses allowed for download
INAT_PHOTO_SIZE = "large"  # up to 1024 px on the long side
# Observations closer than this are treated as the same aquarium (same individual fish/tank)
SAME_PLACE_KM = 3.0
HTTP_HEADERS = {"User-Agent": "fish-lens-dinov2-feature-test/0.1"}

# Species list of the target aquarium, checked against iNaturalist (see output/aquarium_species_inat.csv)
AQUARIUM_LIST_CSV = ROOT / "output" / "aquarium_species_inat.csv"
AQUARIUM_DIR = ROOT / "dataset" / "inat_aquarium"
AQUARIUM_MIN_CC_OBS = 50      # captive observations with CC-licensed photos
AQUARIUM_MAX_DOWNLOAD = 120   # photos per species (random subset); classes are equalized later anyway

MAX_PER_SPECIES = 300
SEED = 42
BATCH_SIZE = 64
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
K_NEIGHBORS = 10

from backbones import BACKBONES, DEFAULT_BACKBONE, Backbone, load_backbone, output_root

# ImageNet normalization, as used for the official DINOv2 pretrained weights
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TSNE_RANDOM_STATE = 42
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)

# Plot style: fixed color order + distinct marker per species (identity never color-alone)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#008300", "#e34948"]
MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]


def folder_name(species: str) -> str:
    return species.replace(" ", "_")


def short_name(species: str) -> str:
    genus, epithet = species.split(" ", 1)
    return f"{genus[0]}. {epithet}"


def count_images(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# 1a. Fish4Knowledge download / extraction
# ---------------------------------------------------------------------------
def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, last = 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and time.time() - last > 2:
                print(f"    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", flush=True)
                last = time.time()
    tmp.replace(dest)


def safe_extract_images(tar_path: Path, target_dir: Path) -> int:
    """Extract only image files, flattened into target_dir (no path traversal)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            name = Path(m.name).name
            if Path(name).suffix.lower() not in IMAGE_EXTS or name.startswith("."):
                continue
            src = tar.extractfile(m)
            if src is None:
                continue
            with open(target_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def prepare_f4k() -> list[str]:
    print("\n[1] Fish4Knowledge dataset check")
    F4K_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for species, cid in F4K_SPECIES:
        target = F4K_DIR / folder_name(species)
        marker = target / ".complete"
        if marker.exists():
            print(f"  {species:<28}: already present ({count_images(target)} images), skip download")
            continue

        tar_path = F4K_DOWNLOAD_DIR / f"fish_{cid}.tar"
        md5_url = f"{F4K_BASE}/fish_{cid}.md5"
        with urllib.request.urlopen(md5_url, timeout=60) as r:
            expected_md5 = r.read().decode().split()[0].strip().lower()

        if not (tar_path.exists() and md5sum(tar_path) == expected_md5):
            download(f"{F4K_BASE}/fish_{cid}.tar", tar_path)
            actual = md5sum(tar_path)
            if actual != expected_md5:
                raise RuntimeError(f"MD5 mismatch for {tar_path.name}: {actual} != {expected_md5}")
        print(f"  {tar_path.name}: MD5 OK ({expected_md5})")

        if target.exists():
            shutil.rmtree(target)
        n = safe_extract_images(tar_path, target)
        marker.write_text(f"source={F4K_BASE}/fish_{cid}.tar\nmd5={expected_md5}\nimages={n}\n")
        print(f"  {species:<28}: extracted {n} images -> {target.relative_to(ROOT)}")
    return [s for s, _ in F4K_SPECIES]


# ---------------------------------------------------------------------------
# 1b. Local RAR dataset extraction
# ---------------------------------------------------------------------------
def rar_extract_cmd(archive: Path, dest: Path) -> list[str]:
    """Pick an available RAR extractor (Windows bsdtar reads RAR natively)."""
    win_tar = Path(r"C:\Windows\System32\tar.exe")
    if sys.platform == "win32" and win_tar.exists():
        return [str(win_tar), "-xf", str(archive), "-C", str(dest)]
    for exe, args in (("7z", ["x", "-y", f"-o{dest}", str(archive)]),
                      ("unrar", ["x", "-o+", str(archive), str(dest) + "/"]),
                      ("bsdtar", ["-xf", str(archive), "-C", str(dest)])):
        if shutil.which(exe):
            return [exe, *args]
    raise RuntimeError("No RAR extractor found (need Windows tar.exe, 7z, unrar or bsdtar)")


def prepare_local_rar() -> list[str]:
    print(f"\n[1] Local dataset check: {LOCAL_DIR.relative_to(ROOT)}")
    archives = sorted(LOCAL_DIR.glob("*.rar"))
    if not archives:
        raise FileNotFoundError(f"No .rar archives in {LOCAL_DIR}")
    species_names = []
    for archive in archives:
        genus, epithet = archive.stem.split(" ", 1)
        species = f"{genus.capitalize()} {epithet.lower()}"
        species_names.append(species)
        target = LOCAL_DIR / folder_name(species)
        marker = target / ".complete"
        if marker.exists():
            print(f"  {species:<28}: already extracted ({count_images(target)} images), skip")
            continue

        tmp = LOCAL_DIR / "_extract_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        subprocess.run(rar_extract_cmd(archive, tmp), check=True)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        n = 0
        for p in tmp.rglob("*"):  # flatten: keep image files only
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
                shutil.move(str(p), target / p.name)
                n += 1
        shutil.rmtree(tmp)
        marker.write_text(f"source={archive.name}\nimages={n}\n")
        print(f"  {species:<28}: extracted {n} images -> {target.relative_to(ROOT)}")
    return species_names


# ---------------------------------------------------------------------------
# 1c. iNaturalist captive observations (aquarium photos)
# ---------------------------------------------------------------------------
def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def inat_taxon_id(name: str) -> int:
    results = http_json(f"{INAT_API}/taxa?q={urllib.parse.quote(name)}&per_page=30")["results"]
    for t in results:  # accepted name, or a synonym (e.g. Rhina ancylostoma -> Rhina ancylostomus)
        if name.lower() in (t["name"].lower(), (t.get("matched_term") or "").lower()):
            if t["name"].lower() != name.lower():
                print(f"  note: iNaturalist lists '{name}' as '{t['name']}' (taxon_id={t['id']})")
            return t["id"]
    raise RuntimeError(f"iNaturalist taxon not found: {name}")


def fetch_photo(r: dict, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        req = urllib.request.Request(r["url"], headers=HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        return True
    except Exception as e:
        print(f"  [skip] download failed {r['url']}: {e}")
        dest.unlink(missing_ok=True)
        return False


def download_inat_species(target: Path, species: str, label: str, taxon_id: int | None = None,
                          max_photos: int | None = None) -> None:
    """Download CC-licensed photos of captive observations of one taxon into target/."""
    marker = target / ".complete"
    if marker.exists():
        print(f"  {species:<30}: already present ({count_images(target)} images), skip download")
        return
    target.mkdir(parents=True, exist_ok=True)

    tid = taxon_id or inat_taxon_id(species)
    time.sleep(1)
    observations, id_above = [], 0
    while True:  # paginate by id (API rate limit: ~1 request/s)
        page = http_json(f"{INAT_API}/observations?taxon_id={tid}&captive=true&photos=true"
                         f"&per_page=200&order_by=id&order=asc&id_above={id_above}")["results"]
        time.sleep(1)
        if not page:
            break
        observations += page
        id_above = page[-1]["id"]

    rows = []
    for o in observations:
        lat = lon = None
        if o.get("location") and not o.get("obscured"):
            lat, lon = (float(v) for v in o["location"].split(","))
        for p in o["photos"]:
            lic = (p.get("license_code") or "").lower()
            if lic not in INAT_LICENSES:
                continue
            rows.append({
                "file": f"{p['id']}.jpg",
                "url": p["url"].replace("/square.", f"/{INAT_PHOTO_SIZE}."),
                "observation_id": o["id"],
                "observation_url": f"https://www.inaturalist.org/observations/{o['id']}",
                "user": o["user"]["login"],
                "observed_on": o.get("observed_on"),
                "place_guess": o.get("place_guess"),
                "latitude": lat,
                "longitude": lon,
                "license": lic,
                "attribution": p.get("attribution"),
                "korean_name": label,
            })
    n_available = len(rows)
    if max_photos and len(rows) > max_photos:  # random subset (fixed seed), not the oldest ones
        idx = np.sort(np.random.default_rng(SEED).choice(len(rows), size=max_photos, replace=False))
        rows = [rows[i] for i in idx]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = list(pool.map(lambda r: fetch_photo(r, target / r["file"]), rows))
    rows = [r for r, good in zip(rows, ok) if good]
    pd.DataFrame(rows).to_csv(target / "photos.csv", index=False, encoding="utf-8")
    marker.write_text(f"source=iNaturalist captive=true taxon_id={tid}\n"
                      f"observations={len(observations)}\nphotos_available={n_available}\n"
                      f"photos={len(rows)}\n")
    print(f"  {species:<30}: taxon_id={tid}, {len(observations)} captive obs, {n_available} CC photos"
          f" -> {len(rows)} downloaded", flush=True)


def prepare_inat() -> list[str]:
    print(f"\n[1] iNaturalist captive dataset check: {INAT_DIR.relative_to(ROOT)}")
    for species, korean in INAT_SPECIES:
        download_inat_species(INAT_DIR / folder_name(species), species, korean)
    return [s for s, _ in INAT_SPECIES]


def aquarium_list_species() -> pd.DataFrame:
    """Fish species from the aquarium list with enough captive CC observations on iNaturalist."""
    df = pd.read_csv(AQUARIUM_LIST_CSV, encoding="utf-8-sig")
    df = df[df["group"].isin(["ray-finned fish", "shark/ray"])
            & df["inat_rank"].isin(["species", "subspecies"])
            & (df["captive_cc_obs"] >= AQUARIUM_MIN_CC_OBS)]
    return df.drop_duplicates("taxon_id").sort_values("inat_name").reset_index(drop=True)


def prepare_inat_aquarium() -> list[str]:
    sp = aquarium_list_species()
    print(f"\n[1] Aquarium-list species on iNaturalist (captive CC obs >= {AQUARIUM_MIN_CC_OBS}): "
          f"{len(sp)} species -> {AQUARIUM_DIR.relative_to(ROOT)}")
    for _, r in sp.iterrows():
        download_inat_species(AQUARIUM_DIR / folder_name(r["inat_name"]), r["inat_name"], r["common_name"],
                              taxon_id=int(r["taxon_id"]), max_photos=AQUARIUM_MAX_DOWNLOAD)
    return sp["inat_name"].tolist()


def inat_taxonomy(species_names: list[str], taxon_ids: list[int]) -> dict[str, str]:
    """species -> family (from iNaturalist ancestors); cached next to the dataset."""
    cache = AQUARIUM_DIR / "taxonomy.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if set(species_names) <= set(df["species"]):
            return dict(zip(df["species"], df["family"]))
    fam = {}
    for i in range(0, len(taxon_ids), 30):
        ids = ",".join(str(t) for t in taxon_ids[i:i + 30])
        for t in http_json(f"{INAT_API}/taxa/{ids}")["results"]:
            fam[t["id"]] = next((a["name"] for a in t.get("ancestors", []) if a["rank"] == "family"), "Unknown")
        time.sleep(1)
    out = {s: fam.get(t, "Unknown") for s, t in zip(species_names, taxon_ids)}
    pd.DataFrame({"species": list(out), "family": list(out.values())}).to_csv(cache, index=False)
    return out


# Scene filter (species-agnostic): keep only "live fish inside an aquarium tank" photos.
# CLIP zero-shot is used ONLY to decide the photo type; species names are never in the prompts.
SCENE_PROMPTS = {
    "aquarium": [
        "a photo of a fish swimming in an aquarium tank",
        "a photo of fish behind the glass of a public aquarium",
        "a photo of a large aquarium tank with fish",
        "a photo of fish seen from an underwater aquarium tunnel",
        "a blurry photo of a fish in a blue aquarium",
        "a photo looking down into an open-top aquarium tank with fish",
        "a photo of rays or sharks in a touch pool at an aquarium",
        "a photo of small fish in a home aquarium with gravel and plants",
    ],
    "market_or_food": [
        "a photo of dead fish on ice at a fish market",
        "a photo of a cooked fish on a plate",
        "a photo of a dead fish held in a person's hand",
        "a photo of dead fish in a bucket or basin",
    ],
    "pond_or_farm": [
        "a photo of fish in an outdoor pond viewed from above",
        "a photo of fish in a fish farm net pen",
        "a photo of a person fishing at a fishing pond",
    ],
    "other": [
        "a photo of an information sign or label",
        "a photo of an illuminated aquarium information panel with text",
        "a photo of a fish skeleton or a fish model in a museum",
    ],
}
SCENE_CLIP = ("ViT-B-32", "laion2b_s34b_b79k")
# Only clearly off-target scenes are dropped. "pond_or_farm" is kept: on manual review most of those
# were real aquarium shots (open-top tanks, touch pools, greenish water) that CLIP mistakes for ponds.
SCENE_EXCLUDE = {"market_or_food", "other"}


def scene_filter(data_dir: Path, species_names: list[str]) -> pd.DataFrame:
    """Classify every image as aquarium / non-aquarium. Cached in <data_dir>/scene_filter.csv."""
    cache = data_dir / "scene_filter.csv"
    paths = [p for s in species_names for p in sorted((data_dir / folder_name(s)).iterdir())
             if p.suffix.lower() in IMAGE_EXTS]
    rel = [p.relative_to(ROOT).as_posix() for p in paths]
    if cache.exists():
        df = pd.read_csv(cache)
        if set(rel) <= set(df["image_path"]):
            return df

    import open_clip
    print(f"  Scene filter: CLIP {SCENE_CLIP[0]} ({SCENE_CLIP[1]}) zero-shot, species-agnostic prompts")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip, _, clip_pre = open_clip.create_model_and_transforms(SCENE_CLIP[0], pretrained=SCENE_CLIP[1])
    clip.eval().to(device)
    tok = open_clip.get_tokenizer(SCENE_CLIP[0])
    prompts = [(cat, p) for cat, ps in SCENE_PROMPTS.items() for p in ps]
    with torch.inference_mode():
        t = clip.encode_text(tok([p for _, p in prompts]).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        rows = []
        for path, r in zip(paths, rel):
            try:
                with Image.open(path) as img:
                    x = clip_pre(img.convert("RGB")).unsqueeze(0).to(device)
            except Exception as e:
                print(f"  [skip] unreadable image {r}: {e}")
                continue
            f = clip.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            prob = (100.0 * f @ t.T).softmax(dim=-1)[0].cpu().numpy()
            cat_prob = {c: float(sum(pr for (cc, _), pr in zip(prompts, prob) if cc == c)) for c in SCENE_PROMPTS}
            top = max(cat_prob, key=cat_prob.get)
            rows.append({"image_path": r, "scene": top, "p_aquarium": cat_prob["aquarium"],
                         **{f"p_{c}": v for c, v in cat_prob.items() if c != "aquarium"}})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def inat_extra_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Attach observation / location / license info to each selected image."""
    info = []
    for folder in sorted({Path(p).parent.as_posix() for p in meta["image_path"]}):
        df = pd.read_csv(ROOT / folder / "photos.csv")
        df["image_path"] = folder + "/" + df["file"]
        info.append(df.drop(columns=["file"]))
    return meta.merge(pd.concat(info), on="image_path", how="left")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def inat_place_groups(meta: pd.DataFrame) -> np.ndarray:
    """Union-find: same observation, same user without coordinates, or within SAME_PLACE_KM."""
    n = len(meta)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    def link_equal(values):
        first = {}
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if v in first:
                union(i, first[v])
            else:
                first[v] = i

    link_equal(meta["observation_id"].tolist())
    has_xy = meta["latitude"].notna().to_numpy()
    link_equal([u if not h else None for u, h in zip(meta["user"], has_xy)])
    idx = np.flatnonzero(has_xy)
    lat, lon = meta["latitude"].to_numpy()[idx], meta["longitude"].to_numpy()[idx]
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    for a, b in zip(*np.nonzero(np.triu(d <= SAME_PLACE_KM, k=1))):
        union(idx[a], idx[b])
    return np.array([find(i) for i in range(n)])


def inat_exclusion(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    obs = meta["observation_id"].to_numpy()
    place = inat_place_groups(meta)
    n_groups = len(set(place))
    print(f"  location groups (aquarium proxy, <= {SAME_PLACE_KM:g} km / same user w/o coords): {n_groups}")
    return {
        "same-observation photos (same individual) excluded": obs[:, None] == obs[None, :],
        f"STRICT: all photos from the same aquarium/location group excluded (any species, "
        f"{n_groups} groups)": place[:, None] == place[None, :],
    }


# ---------------------------------------------------------------------------
# Near-duplicate groups (excluded from each other's k-NN neighborhood)
# ---------------------------------------------------------------------------
def f4k_exclusion(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    # file name = fish_<trackingID>_<fishID>; same trackingID = same video trajectory
    traj = meta["image_path"].map(lambda p: f"{Path(p).parent.name}/{Path(p).stem.split('_')[1]}").to_numpy()
    return {"same-trajectory neighbors excluded": traj[:, None] == traj[None, :]}


def capture_time(path: str) -> datetime:
    m = re.search(r"(\d{8})_(\d{6})", Path(path).stem)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def local_exclusion(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    # file name = IMG_<YYYYMMDD>_<HHMMSS>; one specimen is photographed on both sides seconds apart
    t = np.array([capture_time(p).timestamp() for p in meta["image_path"]])
    day = np.array([capture_time(p).date().isoformat() for p in meta["image_path"]])
    sp = meta["species"].to_numpy()
    same_specimen = (sp[:, None] == sp[None, :]) & (np.abs(t[:, None] - t[None, :]) <= SAME_SPECIMEN_SECONDS)
    same_day = day[:, None] == day[None, :]
    return {
        f"same-species photos within {SAME_SPECIMEN_SECONDS}s (same specimen) excluded": same_specimen,
        "STRICT: all photos from the same capture day excluded (any species)": same_day,
    }


DATASETS = {
    "fish4knowledge": dict(
        title="Fish4Knowledge",
        description="Fish4Knowledge (official GT/RECOG, clusters 01-04)",
        dir=F4K_DIR, out=ROOT / "output", prepare=prepare_f4k, exclusion=f4k_exclusion,
        split_hint="trajectory (trackingID)-aware train/test split",
    ),
    "76cr3wfhff-1": dict(
        title="76cr3wfhff-1",
        description="76cr3wfhff-1 (local RAR archives, 8 species, TIFF photos)",
        dir=LOCAL_DIR, out=ROOT / "output" / "76cr3wfhff-1", prepare=prepare_local_rar,
        exclusion=local_exclusion,
        split_hint="capture-day-aware train/test split (no specimen/day in both train and test)",
    ),
    "inat_captive": dict(
        title="iNaturalist captive (aquarium)",
        description="iNaturalist captive=true observations (aquarium photos, CC0/CC-BY/CC-BY-NC), 5 species",
        dir=INAT_DIR, out=ROOT / "output" / "inat_captive", prepare=prepare_inat,
        exclusion=inat_exclusion, extra_meta=inat_extra_meta,
        split_hint="aquarium/location-group-aware train/test split (no aquarium in both train and test)",
    ),
    "inat_aquarium": dict(
        title="iNaturalist captive - aquarium species list",
        description=f"iNaturalist captive=true photos (CC0/CC-BY/CC-BY-NC) of aquarium-list fish species "
                    f"with >= {AQUARIUM_MIN_CC_OBS} captive CC observations",
        dir=AQUARIUM_DIR, out=ROOT / "output" / "inat_aquarium", prepare=prepare_inat_aquarium,
        exclusion=inat_exclusion, extra_meta=inat_extra_meta, taxonomy=True,
        split_hint="aquarium/location-group-aware train/test split (no aquarium in both train and test)",
    ),
}


# ---------------------------------------------------------------------------
# 2. Image selection
# ---------------------------------------------------------------------------
def select_images(data_dir: Path, species_names: list[str], keep: set | None = None) -> pd.DataFrame:
    print("\n[2] Image selection")
    available = {}
    for species in species_names:
        d = data_dir / folder_name(species)
        available[species] = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS
                                    and (keep is None or p.relative_to(ROOT).as_posix() in keep))

    # Equal count per species where possible
    n_use = min(MAX_PER_SPECIES, min(len(v) for v in available.values()))
    rng = np.random.default_rng(SEED)
    rows = []
    for species in species_names:
        files = available[species]
        idx = np.sort(rng.choice(len(files), size=n_use, replace=False))
        for i in idx:
            rows.append({"image_path": files[i].relative_to(ROOT).as_posix(), "species": species})
        print(f"  {species:<28}: {len(files):>6} available -> {n_use} sampled (seed={SEED})")
    print(f"  Total: {len(rows)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3-5. Model + feature extraction
# ---------------------------------------------------------------------------
def load_model(device: torch.device, key: str = DEFAULT_BACKBONE) -> Backbone:
    spec = BACKBONES[key]
    print(f"\n[3] Loading {spec['label']} ({spec['source']})")
    bb = load_backbone(key, device)
    print(f"  Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else 'CPU'}")
    print(f"  Model: {bb.label}")
    print(f"  Embedding dimension: {bb.dim}")
    print(f"  Trainable parameters: {sum(p.numel() for p in bb.model.parameters() if p.requires_grad)}")
    print(f"  Mixed precision (fp16 autocast): {bb.amp}")
    return bb


PREPROCESS_CHOICES = ["crop", "pad"]


def reuse_probe_embeddings(backbone: str, meta: pd.DataFrame):
    """Embeddings already extracted by linear_probe.py (all aquarium photos) for the same backbone."""
    d = output_root(ROOT, backbone) / "inat_aquarium" / "linear_probe"
    if not (d / "embeddings_all.npy").exists() or not (d / "metadata_all.csv").exists():
        return None
    paths = pd.read_csv(d / "metadata_all.csv", usecols=["image_path"])["image_path"]
    pos = pd.Series(np.arange(len(paths)), index=paths.values)
    if not meta["image_path"].isin(pos.index).all():
        return None
    return np.load(d / "embeddings_all.npy")[pos[meta["image_path"]].to_numpy()]


def extract_embeddings(model, meta: pd.DataFrame, device, preprocess, amp: bool = False,
                       batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, pd.DataFrame]:
    print(f"\n[4] Feature extraction (frozen, inference_mode, amp={amp})")
    feats, keep = [], []
    batch, batch_idx = [], []

    def flush():
        x = torch.stack(batch).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)  # image-level embedding (DINOv2: CLS after final norm; timm: pooled features)
        feats.append(out.float().cpu().numpy())
        keep.extend(batch_idx)
        batch.clear()
        batch_idx.clear()

    with torch.inference_mode():
        for i, path in enumerate(meta["image_path"]):
            try:
                with Image.open(ROOT / path) as img:
                    batch.append(preprocess(img.convert("RGB")))
                batch_idx.append(i)
            except Exception as e:  # corrupted / unreadable only
                print(f"  [skip] unreadable image {path}: {e}")
                continue
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()

    emb = np.concatenate(feats, axis=0)
    meta = meta.iloc[keep].reset_index(drop=True)
    print(f"  embedding matrix shape: {emb.shape}")
    return emb, meta


# ---------------------------------------------------------------------------
# k-NN neighbor composition (non-parametric, no training)
# ---------------------------------------------------------------------------
def knn_analysis(emb_n, lab_idx, exclude, n_classes, k=K_NEIGHBORS):
    cos = emb_n @ emb_n.T
    cos[exclude] = -np.inf
    np.fill_diagonal(cos, -np.inf)
    nn_idx = np.argsort(-cos, axis=1)[:, :k]
    nn_lab = lab_idx[nn_idx]
    mat = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
        rows_c = nn_lab[lab_idx == c]
        mat[c] = [(rows_c == j).mean() for j in range(n_classes)]
    vote = np.array([np.bincount(r, minlength=n_classes).argmax() for r in nn_lab])
    return mat, (vote == lab_idx).mean(), vote, nn_lab


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def scatter_2d_families(xy, labels, species_names, families, title, ds_title, xlabel, ylabel, out_path,
                        subtitle=None, top=7):
    """Too many species for distinct colors: color by family (top families + Other), centroid per species."""
    fam = np.array([families[s] for s in labels])
    order = pd.Series(fam).value_counts().index.tolist()
    shown = order[:top]
    fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=150)
    other = ~np.isin(fam, shown)
    ax.scatter(xy[other, 0], xy[other, 1], s=10, alpha=0.35, c="#b5b5b5", marker="o", linewidths=0,
               label=f"Other families ({len(order) - len(shown)}, {len({s for s in species_names if families[s] not in shown})} spp.)")
    for k, f in enumerate(shown):
        m = fam == f
        n_sp = len({s for s in species_names if families[s] == f})
        ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.5, c=COLORS[k], marker=MARKERS[k], linewidths=0,
                   label=f"{f} ({n_sp} spp.)")
    for s in species_names:  # species centroids, small dark dots
        c = xy[labels == s].mean(axis=0)
        ax.scatter(*c, s=14, c="#222222", marker="o", linewidths=0, zorder=5)
    ax.set_title(f"{title}\nDataset: {ds_title}", fontsize=13, pad=18)
    ax.text(0.5, 1.01, (subtitle + "  |  " if subtitle else "") + f"{len(species_names)} species, colored by family; "
            "black dots = species centroids", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#555555")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(fontsize=9, frameon=True, markerscale=1.8, title="Family", title_fontsize=9,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def scatter_2d(xy, labels, species_names, title, ds_title, xlabel, ylabel, out_path, subtitle=None,
               families=None):
    if len(species_names) > len(COLORS):
        return scatter_2d_families(xy, labels, species_names, families, title, ds_title, xlabel, ylabel,
                                   out_path, subtitle)
    many = len(species_names) > 4
    fig, ax = plt.subplots(figsize=(12, 7.5) if many else (9, 7.5), dpi=150)
    for k, species in enumerate(species_names):
        m = labels == species
        ax.scatter(xy[m, 0], xy[m, 1], s=22, alpha=0.45, c=COLORS[k], marker=MARKERS[k],
                   linewidths=0, label=f"{species} (n={m.sum()})")
    for k, species in enumerate(species_names):
        c = xy[labels == species].mean(axis=0)
        ax.scatter(*c, s=320, c=COLORS[k], marker=MARKERS[k], edgecolors="white", linewidths=2.0, zorder=5)
        ax.annotate(short_name(species), c, xytext=(9, 9), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#222222", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_title(f"{title}\nDataset: {ds_title}", fontsize=13, pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color="#555555")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    legend_pos = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0)) if many else dict(loc="best")
    leg = ax.legend(fontsize=9, frameon=True, markerscale=1.6,
                    title="Species (large marker = centroid)", title_fontsize=8.5, **legend_pos)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def heatmap_large(sim, names, families, ds_title, out_path, model_label="DINOv2 ViT-S/14"):
    """Many species: order by family, no cell text, family boundaries drawn."""
    order = sorted(range(len(names)), key=lambda i: (families[names[i]], names[i]))
    sim = sim[np.ix_(order, order)]
    names = [names[i] for i in order]
    n = len(names)
    fig, ax = plt.subplots(figsize=(16, 14.5), dpi=150)
    vmin = np.floor(np.percentile(sim, 1) * 10) / 10
    im = ax.imshow(sim, cmap="Blues", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(n), [short_name(x) for x in names], rotation=90, fontsize=4.5)
    ax.set_yticks(range(n), [f"{short_name(x)} [{families[x]}]" for x in names], fontsize=4.5)
    fam = [families[x] for x in names]
    for i in range(1, n):
        if fam[i] != fam[i - 1]:
            ax.axhline(i - 0.5, color="#e34948", linewidth=0.4)
            ax.axvline(i - 0.5, color="#e34948", linewidth=0.4)
    ax.set_title("Cosine similarity between species mean embeddings (ordered by family; red lines = family "
                 f"boundaries)\n{model_label} (L2-normalized) - {ds_title}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=f"cosine similarity (color range {vmin:.1f}-1.0)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def heatmap(sim, names, ds_title, out_path, families=None, model_label="DINOv2 ViT-S/14"):
    if len(names) > 12 and families:
        return heatmap_large(sim, names, families, ds_title, out_path, model_label)
    n = len(names)
    fig, ax = plt.subplots(figsize=(7.5 + 0.6 * max(0, n - 4), 6.3 + 0.5 * max(0, n - 4)), dpi=150)
    vmin = np.floor(sim.min() * 10) / 10  # stretch color range so differences are visible
    im = ax.imshow(sim, cmap="Blues", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(n), names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n), names, fontsize=9)
    thr = (vmin + 1.0) / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center", fontsize=10 if n <= 4 else 8.5,
                    color="white" if sim[i, j] > thr else "#222222")
    ax.set_title("Cosine similarity between species mean embeddings\n"
                 f"{model_label} (L2-normalized) - {ds_title}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"cosine similarity (color range {vmin:.1f}-1.0)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), default="fish4knowledge")
    parser.add_argument("--backbone", choices=list(BACKBONES), default=DEFAULT_BACKBONE)
    parser.add_argument("--preprocess", choices=PREPROCESS_CHOICES, default="crop",
                        help="crop: resize+center crop (default); pad: keep whole image, pad to square")
    parser.add_argument("--aquarium-only", action="store_true",
                        help="keep only photos a species-agnostic CLIP scene filter labels as aquarium tank")
    args = parser.parse_args()
    ds = DATASETS[args.dataset]
    ds_out = output_root(ROOT, args.backbone) / ds["out"].relative_to(ROOT / "output")
    out_dir = ds_out / "aquarium_only" if args.aquarium_only else ds_out
    if args.preprocess != "crop":
        out_dir = out_dir / args.preprocess
    out_rel = out_dir.relative_to(ROOT).as_posix()
    print(f"Dataset: {args.dataset}   Preprocess: {args.preprocess}   "
          f"Aquarium-only: {args.aquarium_only}   Output: {out_rel}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    species_names = ds["prepare"]()
    keep = None
    scene_note = []
    if args.aquarium_only:
        scenes = scene_filter(ds["dir"], species_names)
        scenes["species"] = scenes["image_path"].map(lambda p: Path(p).parent.name.replace("_", " "))
        drop = scenes["scene"].isin(SCENE_EXCLUDE)
        keep = set(scenes.loc[~drop, "image_path"])
        excluded = scenes[drop]
        excluded.to_csv(out_dir / "scene_excluded.csv", index=False)
        tab = pd.crosstab(scenes["species"], scenes["scene"]).reindex(species_names)
        print("\n  Scene filter result (all downloaded photos):")
        print(tab.to_string())
        print(f"  kept {len(keep)} photos (dropped scenes: {sorted(SCENE_EXCLUDE)}), excluded {len(excluded)} "
              f"(listed in {out_rel}/scene_excluded.csv)")
        scene_note = ["", f"Scene filter (CLIP {SCENE_CLIP[0]} {SCENE_CLIP[1]}, species-agnostic zero-shot):",
                      tab.to_string(), f"kept {len(keep)} / excluded {len(excluded)}"]
    meta = select_images(ds["dir"], species_names, keep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = load_model(device, args.backbone)
    cached = reuse_probe_embeddings(args.backbone, meta) if args.preprocess == "crop" else None
    if cached is not None:
        emb = cached
        print(f"\n[4] Reused {emb.shape} embeddings from this backbone's linear-probe cache "
              f"(same images, same crop preprocessing)")
    else:
        emb, meta = extract_embeddings(bb.model, meta, device, bb.transforms[args.preprocess], bb.amp, bb.batch)
    if "extra_meta" in ds:
        meta = ds["extra_meta"](meta)
    labels = meta["species"].to_numpy()
    families = None
    if ds.get("taxonomy"):
        sp_list = aquarium_list_species()
        tid = dict(zip(sp_list["inat_name"], sp_list["taxon_id"].astype(int)))
        families = inat_taxonomy(species_names, [tid[s] for s in species_names])
        meta["family"] = meta["species"].map(families)
    many = len(species_names) > 12

    np.save(out_dir / "embeddings.npy", emb)  # raw (un-normalized) embeddings
    meta.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    # 6. L2 normalization (analysis copy)
    emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # 7. PCA
    print("\n[5] PCA")
    pca = PCA(n_components=2, random_state=SEED)
    xy_pca = pca.fit_transform(emb_n)
    ev = pca.explained_variance_ratio_ * 100
    print(f"  PCA Explained Variance\n  PC1 : {ev[0]:.1f}%\n  PC2 : {ev[1]:.1f}%\n  Total : {ev.sum():.1f}%")
    scatter_2d(xy_pca, labels, species_names, f"{bb.label} Fish Embedding - PCA", ds["title"],
               f"PC1 ({ev[0]:.1f}% variance)", f"PC2 ({ev[1]:.1f}% variance)", out_dir / "pca.png", families=families)

    # 8. t-SNE
    perplexity = float(min(30, (len(emb_n) - 1) // 3))
    print(f"\n[6] t-SNE  perplexity={perplexity}, random_state={TSNE_RANDOM_STATE}, metric=cosine")
    tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
                random_state=TSNE_RANDOM_STATE)
    xy_tsne = tsne.fit_transform(emb_n)
    scatter_2d(xy_tsne, labels, species_names, f"{bb.label} Fish Embedding - t-SNE", ds["title"],
               "t-SNE 1", "t-SNE 2", out_dir / "tsne.png",
               subtitle=f"perplexity={perplexity:g}, metric=cosine, random_state=42", families=families)

    # 9. UMAP
    import umap
    print(f"\n[7] UMAP  {UMAP_PARAMS}")
    xy_umap = umap.UMAP(n_components=2, **UMAP_PARAMS).fit_transform(emb_n)
    scatter_2d(xy_umap, labels, species_names, f"{bb.label} Fish Embedding - UMAP", ds["title"],
               "UMAP 1", "UMAP 2", out_dir / "umap.png",
               subtitle="n_neighbors=15, min_dist=0.1, metric=cosine, random_state=42", families=families)

    # 11. Silhouette (384-d normalized space)
    print(f"\n[8] Quantitative evaluation ({emb.shape[1]}-d normalized space)")
    sil_cos = silhouette_score(emb_n, labels, metric="cosine")
    sil_euc = silhouette_score(emb_n, labels, metric="euclidean")
    sil_samples = silhouette_samples(emb_n, labels, metric="cosine")
    print(f"  Silhouette Score (cosine)   : {sil_cos:.4f}")
    print(f"  Silhouette Score (euclidean): {sil_euc:.4f}")

    # Per-species stats: silhouette, spread, nearest centroid
    centroids = np.stack([emb_n[labels == s].mean(axis=0) for s in species_names])
    centroids_n = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sums = np.stack([emb_n[labels == s].sum(axis=0) for s in species_names])
    per_species = []
    for k, s in enumerate(species_names):
        x = emb_n[labels == s]
        cos_to_centroid = x @ centroids_n[k]
        # leave-one-out: own centroid recomputed without the sample itself
        own_loo = sums[k] - x
        own_loo /= np.linalg.norm(own_loo, axis=1, keepdims=True)
        sims_all = x @ centroids_n.T
        sims_all[:, k] = np.sum(x * own_loo, axis=1)
        nearest_ok = (sims_all.argmax(axis=1) == k).mean()
        per_species.append({
            "species": s,
            "n": len(x),
            "silhouette_mean": sil_samples[labels == s].mean(),
            "silhouette_neg_frac": (sil_samples[labels == s] < 0).mean(),
            "mean_cos_to_own_centroid": cos_to_centroid.mean(),
            "spread(1-mean_cos)": 1 - cos_to_centroid.mean(),
            "nearest_centroid_is_own(LOO)": nearest_ok,
        })
    per_df = pd.DataFrame(per_species)
    per_df.to_csv(out_dir / "per_species_stats.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(per_df.to_string(index=False) if not many else
              f"  (per-species table: {out_rel}/per_species_stats.csv)")

    # k-NN neighbor composition in 384-d cosine space, near-duplicates excluded
    lab_idx = np.array([species_names.index(s) for s in labels])
    shorts = [short_name(s) for s in species_names]
    knn_lines = []
    knn_results = []
    knn_metrics = []
    for n_ex, (desc, exclude) in enumerate(ds["exclusion"](meta).items()):
        mat, acc, vote, nn_lab = knn_analysis(emb_n, lab_idx, exclude, len(species_names))
        mat_df = pd.DataFrame(mat, index=shorts, columns=shorts)
        off = mat + mat.T
        np.fill_diagonal(off, -1)
        oa, ob = np.unravel_index(np.argmax(off), off.shape)
        knn_results.append((desc, acc, oa, ob))
        knn_metrics.append({"exclusion": desc, "knn_agreement": float(acc)})
        if not many:
            knn_lines += [
                f"{K_NEIGHBORS}-NN neighbor composition (row = query species; cosine; {desc}):",
                mat_df.round(3).to_string(),
            ]
        else:
            mat_df.to_csv(out_dir / f"knn_neighbor_composition_{n_ex}.csv")
            per_acc = pd.Series([(vote[lab_idx == k] == k).mean() for k in range(len(species_names))],
                                index=species_names).sort_values()
            per_df[f"knn_agreement_{n_ex}"] = per_df["species"].map(per_acc)
            pairs = sorted(((mat[i, j], i, j) for i in range(len(species_names))
                            for j in range(len(species_names)) if i != j), reverse=True)[:20]
            fam_of = (lambda x: f" [{families[x]}]") if families else (lambda x: "")
            knn_lines += [
                f"{K_NEIGHBORS}-NN analysis ({desc}); full matrix: {out_rel}/knn_neighbor_composition_{n_ex}.csv",
                "Top-20 confused directions (fraction of query species' neighbors that are the other species):",
                *[f"  {m:.3f}  {species_names[i]}{fam_of(species_names[i])} -> "
                  f"{species_names[j]}{fam_of(species_names[j])}" for m, i, j in pairs],
                f"Worst 15 species by {K_NEIGHBORS}-NN majority-vote agreement:",
                *[f"  {v:.3f}  {k}{fam_of(k)}" for k, v in per_acc.head(15).items()],
                f"Species with agreement >= 0.9: {(per_acc >= 0.9).sum()} / {len(per_acc)}; "
                f">= 0.7: {(per_acc >= 0.7).sum()}; < 0.5: {(per_acc < 0.5).sum()}",
            ]
            if families:
                fam_arr = np.array([families[x] for x in species_names])
                knn_metrics[-1]["family_agreement"] = float((fam_arr[vote] == fam_arr[lab_idx]).mean())
                knn_metrics[-1]["species_ge_0.9"] = int((per_acc >= 0.9).sum())
                knn_metrics[-1]["species_lt_0.5"] = int((per_acc < 0.5).sum())
                wrong = vote != lab_idx
                knn_lines += [
                    f"Family-level agreement (species vote mapped to family): "
                    f"{(fam_arr[vote] == fam_arr[lab_idx]).mean():.3f}",
                    f"Neighbors in the same family: {(fam_arr[nn_lab] == fam_arr[lab_idx][:, None]).mean():.3f}",
                    f"Share of species-level errors that stay within the same family: "
                    f"{(fam_arr[vote][wrong] == fam_arr[lab_idx][wrong]).mean():.3f}",
                ]
        knn_lines += [
            f"{K_NEIGHBORS}-NN majority-vote agreement (non-parametric, no training): {acc:.3f}",
            f"Most overlapping pair (k-NN): {species_names[oa]} <-> {species_names[ob]} "
            f"({mat[oa, ob]:.3f} / {mat[ob, oa]:.3f})",
            "",
        ]
    print("\n  " + "\n  ".join("\n".join(knn_lines).splitlines()))

    # 12. Class-mean cosine similarity
    sim = centroids_n @ centroids_n.T
    sim_df = pd.DataFrame(sim, index=shorts, columns=shorts)
    sim_df.to_csv(out_dir / "species_cosine_similarity.csv")
    per_df.to_csv(out_dir / "per_species_stats.csv", index=False)
    if not many:
        with pd.option_context("display.width", 250):
            print("\n  Species mean-embedding cosine similarity")
            print(sim_df.round(3).to_string())
    heatmap(sim, species_names, ds["title"], out_dir / "species_cosine_similarity.png", families=families,
            model_label=bb.label)
    iu = np.triu_indices(len(species_names), k=1)
    best = np.argmax(sim[iu])
    a, b = iu[0][best], iu[1][best]
    worst = np.argmin(sim[iu])
    c, d = iu[0][worst], iu[1][worst]
    print(f"\n  Most Similar Species\n  {species_names[a]}\n  <->\n  {species_names[b]}\n"
          f"  Cosine Similarity: {sim[a, b]:.3f}")
    top_pairs = sorted(zip(sim[iu], iu[0], iu[1]), reverse=True)[:15]
    top_pair_lines = ["Top-15 most similar species pairs (mean-embedding cosine):",
                      *[f"  {v:.3f}  {species_names[i]} <-> {species_names[j]}"
                        + (f"  [{families[species_names[i]]} / {families[species_names[j]]}]" if families else "")
                        for v, i, j in top_pairs]]
    if many:
        print("\n  " + "\n  ".join(top_pair_lines))

    # 13. Summary
    counts = meta["species"].value_counts().reindex(species_names)
    bar = "=" * 40
    summary = [
        bar, "", f"{bb.label} Fish Feature Analysis", "", bar, "",
        "Dataset:", ds["description"], "",
        "Model:", f"{bb.label} (frozen; {bb.source})", "",
        "Embedding:", f"{emb.shape[1]} dimensions", "",
        "Images:", "",
        *([f"{s:<29}: {counts[s]}" + (f"   ({dict(INAT_SPECIES)[s]})" if s in dict(INAT_SPECIES) else "")
           for s in species_names] if not many else
          [f"{len(species_names)} species x {counts.min()}-{counts.max()} images "
           f"(full list: {out_rel}/per_species_stats.csv)"]
          + ([f"{len(set(families.values()))} families"] if families else [])), "",
        "Total:", f"{len(meta)}", "",
        "Silhouette Score:", f"{sil_cos:.2f}  (cosine, {emb.shape[1]}-d L2-normalized; euclidean={sil_euc:.2f})", "",
        "Most Similar Species:", f"{species_names[a]} <-> {species_names[b]}", "",
        "Cosine Similarity:", f"{sim[a, b]:.2f}", "",
        f"{K_NEIGHBORS}-NN Agreement (no training):",
        *[f"{acc:.3f}  ({desc})" for desc, acc, _, _ in knn_results], "",
        "PCA Explained Variance:", f"{ev.sum():.1f}%  (PC1 {ev[0]:.1f}% + PC2 {ev[1]:.1f}%)", "",
        "Output:", "",
        f"{out_rel}/pca.png", f"{out_rel}/tsne.png", f"{out_rel}/umap.png",
        f"{out_rel}/species_cosine_similarity.png", "", bar,
    ]
    print("\n" + "\n".join(summary))

    details = [
        "", "Details", "-------",
        f"Device: {device.type.upper()}",
        f"Preprocessing ({args.preprocess}): {bb.transform_desc[args.preprocess]}",
        f"Sampling: up to {MAX_PER_SPECIES} per species (equalized to the smallest class), "
        f"numpy default_rng(seed={SEED})",
        f"t-SNE: perplexity={perplexity:g}, metric=cosine, init=pca, random_state={TSNE_RANDOM_STATE}",
        f"UMAP: {UMAP_PARAMS}",
        *scene_note,
        "",
        "Silhouette interpretation (reference only):",
        "  near 1  -> class features tend to be well separated",
        "  near 0  -> classes tend to overlap",
        "  negative -> samples likely mixed into other class regions",
        "  (Silhouette alone does not decide backbone suitability.)",
        "",
        f"Per-species statistics ({emb.shape[1]}-d normalized space):",
        per_df.to_string(index=False, float_format="{:.3f}".format) if not many
        else f"{out_rel}/per_species_stats.csv",
        "",
        *(["Species mean-embedding cosine similarity:", sim_df.round(3).to_string()] if not many else
          [f"Species mean-embedding cosine similarity: {out_rel}/species_cosine_similarity.csv"]),
        *top_pair_lines,
        "",
        f"Least similar pair: {species_names[c]} <-> {species_names[d]} ({sim[c, d]:.3f})",
        "",
        *knn_lines,
        "Caveat: 2D PCA/t-SNE/UMAP separation does NOT imply high classification accuracy.",
        f"Next step if positive: frozen {bb.label} -> {bb.dim}-d embedding -> Linear Classifier -> "
        f"{len(species_names)} species,",
        f"measure Linear Probe Accuracy / Macro-F1 (with {ds['split_hint']}).",
    ]
    (out_dir / "analysis_summary.txt").write_text("\n".join(summary + details) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "backbone": args.backbone, "backbone_label": bb.label, "embedding_dim": bb.dim,
        "dataset": args.dataset, "aquarium_only": args.aquarium_only, "preprocess": args.preprocess,
        "n_images": len(meta), "n_species": len(species_names),
        "silhouette_cosine": float(sil_cos), "silhouette_euclidean": float(sil_euc),
        "nearest_centroid_loo_mean": float(per_df["nearest_centroid_is_own(LOO)"].mean()),
        "pca_2d_explained_variance": float(ev.sum() / 100),
        "most_similar_pair": [species_names[a], species_names[b], float(sim[a, b])],
        "knn": knn_metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_rel}/analysis_summary.txt")


if __name__ == "__main__":
    main()
