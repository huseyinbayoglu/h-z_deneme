# -*- coding: utf-8 -*-
"""
Batch Influencer Pipeline — 4000 Influencer Concurrent Processing

Architecture:
  - Producer (async I/O): X influencer'ın takipçilerini aynı anda çeker + avatar indirir
  - Consumer (GPU thread): vLLM batch inference — CPU beklemeden sürekli çalışır
  - Hashmap cache: Ortak takipçiler tekrar tahmin edilmez (max 1M entry)
  - Resume: Crash durumunda kaldığı yerden devam eder

Usage:
  API_KEY=your_key python batch_pipeline.py

  veya doğrudan script içinde API_KEY değişkenini düzenleyebilirsiniz.
"""

import os
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/workspace/huggingface_cache"


import gc
import re
import json
import time
import queue
import threading
import logging
import sys
import ssl
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import aiohttp

import warnings
warnings.filterwarnings('ignore')


# =====================================================================
#  CONFIG
# =====================================================================
CSV_FILE = "speed_test_4k.csv"         # Influencer username listesi
RESULTS_DIR = "batch_results"          # Çıktı klasörü
MAX_INFLUENCERS = 15                   # CSV'den kaç influencer okunacak (0 = hepsi)

CONCURRENCY_X = 6                     # Aynı anda kaç influencer'ın takipçisi çekilecek (A40 için artırıldı)
FOLLOWER_COUNT = 800                   # Her influencer için çekilecek takipçi sayısı

GPU_DRAIN_INTERVAL = 0.5             # Queue'dan item toplama max bekleme süresi (saniye)

MAX_CONCURRENT_DOWNLOADS = 500        # Eşzamanlı avatar indirme limiti (A40 — hızlı CPU/network)
DOWNLOAD_TIMEOUT_SECONDS = 3          # Avatar indirme timeout
GPU_QUEUE_MAXSIZE = 15_000            # Queue backpressure: CPU GPU'ya çok fark atmasın

HASHMAP_MAX_SIZE = 1_000_000          # Follower cache max entry
CACHE_SAVE_INTERVAL = 50              # Kaç chunk'ta bir cache diske yazılsın
BATCH_CSV_SIZE = 6                    # Kaç influencer'da bir ara CSV dosyası oluşturulacak

GPU_MEMORY_UTILIZATION = 0.95         # A40 48GB — daha fazla VRAM kullan
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MAX_NEW_TOKENS = 10
MAX_MODEL_LEN = 1536

# RapidAPI
RAPIDAPI_HOST = "tiktok-scraper7.p.rapidapi.com"
API_KEY = os.environ.get("API_KEY", "a8cad3edb7msh3e877eda255d39dp1d44e6jsn06c6f0f05b68")

# Valid categories
VALID_GENDERS = {"male", "female", "unknown"}
VALID_AGE_RANGES = {"18-", "18-24", "25-34", "35-44", "45+", "unknown"}

CLASSIFICATION_PROMPT = (
    "Look at this profile picture. "
    "If there is a human, estimate their gender (male or female) and age range. "
    "Age ranges: 18- (under 18), 18-24, 25-34, 35-44, 45+. "
    "Respond ONLY with a JSON object, nothing else: "
    '{"gender": "male/female/unknown", "age_range": "18-/18-24/25-34/35-44/45+/unknown"}'
    "\nIf no human is visible, respond: "
    '{"gender": "unknown", "age_range": "unknown"}'
)


# =====================================================================
#  LOGGING
# =====================================================================
def setup_logging():
    logger = logging.getLogger("BatchPipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False  # Colab'da duplicate log'u önler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger

log = setup_logging()
log.info("batch_pipeline.py başlatılıyor...")


# =====================================================================
#  SSL & IMAGE PROCESSING
# =====================================================================
_ssl_context = ssl.create_default_context()


def _process_image_bytes(data: bytes) -> Image.Image:
    """Bytes → cv2 decode → resize 224×224 → RGB → PIL Image."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return Image.new("RGB", (224, 224), (0, 0, 0))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


# =====================================================================
#  ASYNC TikTok API — Follower Fetching
# =====================================================================
async def async_get_user_id(
    username: str, api_key: str, session: aiohttp.ClientSession
) -> str:
    """TikTok username → user_id."""
    url = f"https://{RAPIDAPI_HOST}/user/info?unique_id={username}"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    async with session.get(url, headers=headers, ssl=_ssl_context) as resp:
        data = await resp.json()
        return data["data"]["user"]["id"]


async def async_get_followers_page(
    user_id: str,
    api_key: str,
    session: aiohttp.ClientSession,
    cursor_time: int = 0,
) -> dict:
    """Bir sayfa takipçi çek (max 200)."""
    url = (
        f"https://{RAPIDAPI_HOST}/user/followers"
        f"?user_id={user_id}&count=200&time={cursor_time}"
    )
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    async with session.get(url, headers=headers, ssl=_ssl_context) as resp:
        return await resp.json()


async def fetch_influencer_followers(
    username: str,
    api_key: str,
    session: aiohttp.ClientSession,
    max_followers: int = FOLLOWER_COUNT,
) -> Tuple[str, pd.DataFrame]:
    """
    Bir influencer'ın takipçilerini çeker.
    Her sayfa sequential (cursor_time gerekli), ama X influencer paralel çalışır.
    """
    try:
        user_id = await async_get_user_id(username, api_key, session)

        all_followers = []
        cursor_time = 0
        has_more = True

        while len(all_followers) < max_followers and has_more:
            data = await async_get_followers_page(
                user_id, api_key, session, cursor_time
            )

            page_data = data.get("data", {})
            followers = page_data.get("followers", [])
            cursor_time = page_data.get("time", 0)
            has_more = page_data.get("hasMore", False)

            if not followers:
                break

            all_followers.extend(followers)

        df = pd.DataFrame(all_followers[:max_followers])
        return (username, df)

    except Exception as e:
        log.error(f"  ❌ @{username} takipçi çekme hatası: {e}")
        return (username, pd.DataFrame())


# =====================================================================
#  ASYNC IMAGE DOWNLOAD
# =====================================================================
async def _download_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    username: str,
    url: str,
) -> Tuple[str, Image.Image]:
    """Tek avatar resmini async indir ve işle."""
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)
            async with session.get(url, timeout=timeout, ssl=_ssl_context) as resp:
                if resp.status != 200:
                    return (username, Image.new("RGB", (224, 224), (0, 0, 0)))
                data = await resp.read()
                img = _process_image_bytes(data)
                return (username, img)
        except Exception:
            return (username, Image.new("RGB", (224, 224), (0, 0, 0)))


async def download_images_batch(
    valid_data: List[Tuple[str, str]],
    session: aiohttp.ClientSession,
    max_concurrent: int = MAX_CONCURRENT_DOWNLOADS,
) -> List[Tuple[str, Image.Image]]:
    """Toplu async avatar indirme. Var olan session kullanır."""
    if not valid_data:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _download_single(session, semaphore, uname, url)
        for uname, url in valid_data
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


# =====================================================================
#  vLLM MODEL LOADING
# =====================================================================
def load_qwen_vllm(
    model_id: str = MODEL_ID,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
    max_model_len: int = MAX_MODEL_LEN,
    tensor_parallel_size: int = 1,
):
    """vLLM ile Qwen2-VL modelini yükler."""
    from vllm import LLM

    log.info(f"vLLM ile model yükleniyor: {model_id}")
    download_dir = os.environ.get("HF_HOME")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        download_dir=download_dir,
    )
    log.info("vLLM model hazır ✅")
    return llm


def build_qwen_prompt(processor) -> str:
    """Qwen2-VL chat template ile prompt oluşturur."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": CLASSIFICATION_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# =====================================================================
#  VLM RESPONSE PARSING
# =====================================================================
def parse_vlm_response(response_text: str) -> Dict[str, str]:
    """VLM çıktısını parse eder. JSON dener, başarısız olursa regex fallback."""
    result = {"gender": "unknown", "age_range": "unknown"}

    # 1) JSON parse
    try:
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            parsed = json.loads(json_match.group())
            gender = str(parsed.get("gender", "unknown")).strip().lower()
            age_range = str(parsed.get("age_range", "unknown")).strip()
            result["gender"] = gender if gender in VALID_GENDERS else "unknown"
            result["age_range"] = age_range if age_range in VALID_AGE_RANGES else "unknown"
            return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # 2) Regex fallback
    text_lower = response_text.lower()
    if "female" in text_lower or "woman" in text_lower or "girl" in text_lower:
        result["gender"] = "female"
    elif "male" in text_lower or "man" in text_lower or "boy" in text_lower:
        result["gender"] = "male"

    for age_r in ["18-24", "25-34", "35-44", "45+"]:
        if age_r in response_text:
            result["age_range"] = age_r
            break
    else:
        if "18-" in response_text or "under 18" in text_lower:
            result["age_range"] = "18-"

    return result


# =====================================================================
#  GPU CONSUMER THREAD
# =====================================================================
def gpu_consumer(
    gpu_queue: queue.Queue,
    llm,
    prompt_template: str,
    follower_cache: Dict[str, Dict],
):
    """
    GPU thread: Queue'daki TÜM image'ları alır, vLLM'e tek seferde verir.
    vLLM'in continuous batching'i internal olarak optimal batch boyutunu
    belirler (PagedAttention ile VRAM'e sığanları otomatik scheduler'lar).
    Bizim yapay limit koymamıza gerek yok.
    Producer ile paralel çalışır — CPU beklemez.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stop=["\n\n"],
        min_tokens=5,
    )

    total_predicted = 0
    total_batches = 0
    sentinel_received = False
    gpu_start_time = time.time()

    while not sentinel_received:
        # --- Queue'daki TÜM mevcut item'ları topla ---
        batch: List[Tuple[str, Image.Image]] = []

        # İlk item: blocking wait
        try:
            first_item = gpu_queue.get(timeout=GPU_DRAIN_INTERVAL)
            if first_item is None:
                sentinel_received = True
            else:
                batch.append(first_item)
        except queue.Empty:
            continue

        # Kalan item'ları non-blocking drain
        while True:
            try:
                item = gpu_queue.get_nowait()
                if item is None:
                    sentinel_received = True
                    break
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            continue

        # --- Build vLLM inputs ---
        llm_inputs = []
        usernames = []
        for uname, img in batch:
            llm_inputs.append({
                "prompt": prompt_template,
                "multi_modal_data": {"image": img},
            })
            usernames.append(uname)

        # --- vLLM Inference ---
        start_t = time.time()
        try:
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        except Exception as e:
            log.error(f"  ❌ GPU inference hatası: {e}")
            for uname in usernames:
                follower_cache[uname] = {"gender": "unknown", "age_range": "unknown"}
            total_predicted += len(batch)
            continue

        elapsed = time.time() - start_t

        # --- Parse results & update cache ---
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text.strip()
            parsed = parse_vlm_response(response_text)
            follower_cache[usernames[i]] = {
                "gender": parsed["gender"],
                "age_range": parsed["age_range"],
            }

        total_predicted += len(batch)
        total_batches += 1
        speed = len(batch) / elapsed if elapsed > 0 else 0

        # Her 10 batch'te bir log yaz (spam önleme)
        if total_batches % 10 == 0 or sentinel_received:
            gpu_elapsed = time.time() - gpu_start_time
            avg_speed = total_predicted / gpu_elapsed if gpu_elapsed > 0 else 0
            log.info(
                f"  🧠 GPU: {total_predicted} tahmin tamamlandı "
                f"(batch #{total_batches}, son: {len(batch)} img, {speed:.0f} img/s, "
                f"ort: {avg_speed:.0f} img/s)"
            )

        # Free image memory
        del batch, llm_inputs

    gpu_total_time = time.time() - gpu_start_time
    avg_speed = total_predicted / gpu_total_time if gpu_total_time > 0 else 0
    log.info(
        f"GPU tamamlandı ✅ {total_predicted} tahmin, "
        f"{total_batches} batch, {gpu_total_time:.1f}s, "
        f"ort: {avg_speed:.0f} img/s"
    )


# =====================================================================
#  DISTRIBUTION COMPUTATION (from app.py — identical logic)
# =====================================================================
def compute_age_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """
    Age dağılımı — unknown redistribution stratejisi:
      1. Unknown olmayanların oransal dağılımını çıkar
      2. Unknown'ların yarısını bu oranlara göre dağıt
      3. Kalan unknown'ların yarısını 25-34'e ekle
      4. "unknown" ve "18-" kategorilerini drop et
    """
    age_counts = df["age_range"].value_counts()
    total = len(df)

    if total == 0:
        return {}

    valid_data = df[df["age_range"] != "unknown"]
    known_counts = valid_data["age_range"].value_counts()

    if known_counts.empty:
        return {}

    known_probs = known_counts / known_counts.sum()
    unknown_count = age_counts.get("unknown", 0)

    if unknown_count > 0:
        # ADIM 1: Unknown'ların yarısını orantılı dağıt
        half_unknown = int(unknown_count * 0.5)
        distributed_values = np.random.choice(
            known_probs.index.tolist(),
            size=half_unknown,
            p=known_probs.values,
        )
        distributed_counts = pd.Series(distributed_values).value_counts()
        for cat, count in distributed_counts.items():
            if cat in age_counts.index:
                age_counts[cat] += count
            else:
                age_counts[cat] = count

        remaining_unknown = unknown_count - half_unknown

        # ADIM 2: Kalan unknown'ların yarısını 25-34'e ekle
        half_remaining = int(remaining_unknown * 0.5)
        if "25-34" in age_counts.index:
            age_counts["25-34"] += half_remaining
        else:
            age_counts["25-34"] = half_remaining

        age_counts["unknown"] = remaining_unknown - half_remaining

    age_distribution = age_counts / age_counts.sum()

    drop_labels = ["unknown", "18-"]
    age_distribution = age_distribution.drop(
        labels=[l for l in drop_labels if l in age_distribution.index]
    )

    age_order = ["18-24", "25-34", "35-44", "45+"]
    ordered = {}
    for cat in age_order:
        if cat in age_distribution.index:
            ordered[cat] = round(float(age_distribution[cat]), 4)

    return ordered


def compute_gender_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """Gender dağılımı: unknown'ları çıkar, male/female yeniden normalize et."""
    valid = df[df["gender"] != "unknown"]
    if valid.empty:
        return {}
    gender_dist = valid["gender"].value_counts(normalize=True)
    return {k: round(float(v), 4) for k, v in gender_dist.items()}


def compute_nationality(users_df: pd.DataFrame) -> Dict[str, float]:
    """Region dağılımı top 4."""
    if "region" not in users_df.columns:
        return {}
    region_dist = users_df["region"].value_counts(normalize=True).head(4)
    return {k: round(float(v), 4) for k, v in region_dist.items()}


# =====================================================================
#  CACHE PERSISTENCE — Save/Load for crash recovery
# =====================================================================
def save_cache(follower_cache: Dict[str, Dict], path: str):
    """Follower cache'ini diske yaz (JSON)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(follower_cache, f, ensure_ascii=False)
        log.info(f"  💾 Cache kaydedildi ({len(follower_cache)} entry)")
    except Exception as e:
        log.error(f"  ❌ Cache kaydetme hatası: {e}")


def load_cache(path: str) -> Dict[str, Dict]:
    """Diske kaydedilmiş cache'i yükle."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            log.info(f"  📂 Cache yüklendi ({len(cache)} entry)")
            return cache
        except Exception as e:
            log.error(f"  ❌ Cache yükleme hatası: {e}")
    return {}


# =====================================================================
#  MAIN PIPELINE
# =====================================================================
async def main():
    pipeline_start = time.time()

    # ----- 1. Model yükle -----
    log.info("=" * 60)
    log.info("ADIM 1: vLLM model yükleniyor...")
    log.info("=" * 60)
    llm = load_qwen_vllm()

    from transformers import AutoProcessor
    log.info("Processor yükleniyor (prompt template için)...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    prompt_template = build_qwen_prompt(_processor)
    del _processor
    gc.collect()

    # ----- 2. CSV oku -----
    log.info("=" * 60)
    log.info(f"ADIM 2: CSV okunuyor: {CSV_FILE}")
    log.info("=" * 60)
    df_csv = pd.read_csv(CSV_FILE)
    col_name = df_csv.columns[0]  # İlk column = username
    all_influencers = df_csv[col_name].dropna().astype(str).tolist()
    if MAX_INFLUENCERS > 0:
        all_influencers = all_influencers[:MAX_INFLUENCERS]
    log.info(f"  Toplam influencer: {len(all_influencers)}")

    # ----- 3. Resume desteği -----
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results").mkdir(exist_ok=True)

    completed_file = results_dir / "completed.txt"
    completed: Set[str] = set()
    if completed_file.exists():
        completed = set(completed_file.read_text(encoding="utf-8").strip().split("\n"))
        completed.discard("")
        log.info(f"  Daha önce tamamlanan: {len(completed)} influencer")

    remaining = [u for u in all_influencers if u not in completed]
    log.info(f"  İşlenecek: {len(remaining)} influencer")

    if not remaining:
        log.info("Tüm influencer'lar zaten işlenmiş! ✅")
        return

    # ----- 4. Cache yükle -----
    cache_file = str(results_dir / "cache.json")
    follower_cache: Dict[str, Dict] = load_cache(cache_file)

    # ----- 5. GPU consumer thread başlat -----
    log.info("=" * 60)
    log.info("ADIM 3: GPU consumer thread başlatılıyor...")
    log.info("=" * 60)
    gpu_q: queue.Queue = queue.Queue(maxsize=GPU_QUEUE_MAXSIZE)  # Backpressure — CPU GPU'ya çok fark atmasın

    gpu_thread = threading.Thread(
        target=gpu_consumer,
        args=(gpu_q, llm, prompt_template, follower_cache),
        daemon=True,
    )
    gpu_thread.start()

    # ----- 6. Producer: Fetch & Download -----
    total_influencers = len(remaining)
    log.info("=" * 60)
    log.info(f"ADIM 4: Pipeline başlıyor (X={CONCURRENCY_X}, {total_influencers} influencer)")
    log.info("=" * 60)

    influencer_data: Dict[str, Dict] = {}
    in_flight: Set[str] = set()
    total_downloaded = 0
    total_cached_hits = 0
    chunks_processed = 0
    influencers_fetched = 0
    producer_start_time = time.time()

    api_key = API_KEY
    if not api_key:
        log.error("❌ API_KEY bulunamadı! Lütfen API_KEY environment variable'ını ayarlayın.")
        gpu_q.put(None)
        gpu_thread.join()
        return

    # API session (TikTok RapidAPI)
    api_connector = aiohttp.TCPConnector(
        limit=CONCURRENCY_X * 4,  # Her influencer ~6 request, biraz headroom
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    # Download session (avatar resimleri)
    dl_connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_DOWNLOADS,
        limit_per_host=300,             # TikTok CDN aynı host'a çok bağlantı (A40 network)
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    dl_headers = {"User-Agent": "Mozilla/5.0"}

    async with aiohttp.ClientSession(connector=api_connector) as api_session, \
               aiohttp.ClientSession(connector=dl_connector, headers=dl_headers) as dl_session:

        for chunk_idx in range(0, len(remaining), CONCURRENCY_X):
            chunk = remaining[chunk_idx : chunk_idx + CONCURRENCY_X]
            chunk_start = time.time()

            # --- A) X influencer'ın takipçilerini paralel çek ---
            fetch_tasks = [
                fetch_influencer_followers(uname, api_key, api_session, FOLLOWER_COUNT)
                for uname in chunk
            ]
            fetch_results = await asyncio.gather(*fetch_tasks)
            fetch_time = time.time() - chunk_start

            # --- B) Her influencer için: cache check + image download ---
            dl_start = time.time()
            chunk_to_download: List[Tuple[str, str]] = []
            chunk_follower_counts = []  # (username, total, cached, new)

            for inf_username, df_followers in fetch_results:
                if df_followers.empty:
                    log.warning(f"  ⚠️ @{inf_username}: takipçi bulunamadı, atlanıyor.")
                    continue

                nationality = compute_nationality(df_followers)

                if "unique_id" not in df_followers.columns or "avatar" not in df_followers.columns:
                    log.warning(f"  ⚠️ @{inf_username}: unique_id/avatar column yok, atlanıyor.")
                    continue

                mask = df_followers["unique_id"].notna() & df_followers["avatar"].notna()
                filtered = df_followers.loc[mask, ["unique_id", "avatar"]]
                follower_uids = filtered["unique_id"].tolist()

                influencer_data[inf_username] = {
                    "follower_usernames": follower_uids,
                    "nationality": nationality,
                }

                cached_count = 0
                new_count = 0
                for _, row in filtered.iterrows():
                    uid = row["unique_id"]
                    if uid in follower_cache:
                        cached_count += 1
                    elif uid not in in_flight:
                        chunk_to_download.append((uid, row["avatar"]))
                        in_flight.add(uid)
                        new_count += 1

                total_cached_hits += cached_count
                chunk_follower_counts.append((inf_username, len(follower_uids), cached_count, new_count))

            # --- C) Download all new images for this chunk ---
            dl_time = 0.0
            if chunk_to_download:
                dl_results = await download_images_batch(
                    chunk_to_download, dl_session, MAX_CONCURRENT_DOWNLOADS
                )
                dl_time = time.time() - dl_start

                for uname, img in dl_results:
                    gpu_q.put((uname, img))

                total_downloaded += len(dl_results)

            # --- D) Progress summary ---
            chunk_time = time.time() - chunk_start
            influencers_fetched += len(chunk_follower_counts)
            chunks_processed += 1

            elapsed_total = time.time() - producer_start_time
            avg_per_influencer = elapsed_total / influencers_fetched if influencers_fetched > 0 else 0
            remaining_count = total_influencers - influencers_fetched
            eta_seconds = avg_per_influencer * remaining_count
            eta_min = eta_seconds / 60

            # Queue durumu
            q_size = gpu_q.qsize()

            # Tek satır chunk özeti
            names = ", ".join(f"@{n}" for n, _, _, _ in chunk_follower_counts)
            total_f = sum(t for _, t, _, _ in chunk_follower_counts)
            total_c = sum(c for _, _, c, _ in chunk_follower_counts)
            total_n = sum(n for _, _, _, n in chunk_follower_counts)
            log.info(
                f"[{influencers_fetched}/{total_influencers}] "
                f"{names} | "
                f"{total_f} takipçi ({total_c} cached, {total_n} yeni) | "
                f"fetch: {fetch_time:.1f}s, dl: {dl_time:.1f}s, chunk: {chunk_time:.1f}s | "
                f"ort: {avg_per_influencer:.1f}s/inf, ETA: {eta_min:.0f}dk | "
                f"queue: {q_size}, cache: {len(follower_cache)}"
            )

            # --- E) Periyodik cache kaydetme ---
            if chunks_processed % CACHE_SAVE_INTERVAL == 0:
                save_cache(follower_cache, cache_file)

    # ----- 7. GPU consumer'ı durdur ve bekle -----
    log.info("\n" + "=" * 60)
    log.info("ADIM 5: GPU consumer bitmesi bekleniyor...")
    log.info("=" * 60)
    gpu_q.put(None)  # Sentinel
    gpu_thread.join()

    # ----- 8. Sonuçları hesapla ve kaydet -----
    log.info("=" * 60)
    log.info("ADIM 6: Dağılımlar hesaplanıyor ve kaydediliyor...")
    log.info("=" * 60)

    # CSV batch klasörü oluştur
    csv_batch_dir = results_dir / "csv_batches"
    csv_batch_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    successfully_processed = 0
    csv_rows_buffer = []  # 50'lik CSV batch için buffer

    def _flatten_result_to_csv_row(result: Dict) -> Dict:
        """Sonuç dict'ini düz CSV satırına çevir."""
        row = {
            "username": result["username"],
            "total_followers_analyzed": result["total_followers_analyzed"],
        }
        # Nationality — top 4 ülke:oran
        nat = result.get("nationality", {})
        for i, (country, ratio) in enumerate(nat.items(), 1):
            row[f"nationality_{i}"] = country
            row[f"nationality_{i}_pct"] = ratio
        # Age distribution
        age = result.get("age", {})
        for age_key in ["18-24", "25-34", "35-44", "45+"]:
            col_name = f"age_{age_key.replace('-', '_').replace('+', 'plus')}"
            row[col_name] = age.get(age_key, 0.0)
        # Gender distribution
        gender = result.get("gender", {})
        row["gender_male"] = gender.get("male", 0.0)
        row["gender_female"] = gender.get("female", 0.0)
        return row

    def _save_csv_batch(rows: list, batch_start: int, batch_end: int):
        """50'lik batch'i CSV dosyasına kaydet."""
        if not rows:
            return
        df_batch = pd.DataFrame(rows)
        csv_name = f"sonuc_{batch_start}-{batch_end}.csv"
        csv_path = csv_batch_dir / csv_name
        df_batch.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info(f"  📄 CSV batch kaydedildi: {csv_name} ({len(rows)} influencer)")

    for inf_username, data in influencer_data.items():
        # Build results DataFrame from cache
        results = []
        for uid in data["follower_usernames"]:
            if uid in follower_cache:
                results.append({
                    "username": uid,
                    "gender": follower_cache[uid]["gender"],
                    "age_range": follower_cache[uid]["age_range"],
                })
            else:
                results.append({
                    "username": uid,
                    "gender": "unknown",
                    "age_range": "unknown",
                })

        df_results = pd.DataFrame(results)

        if df_results.empty:
            continue

        age_dist = compute_age_distribution(df_results)
        gender_dist = compute_gender_distribution(df_results)

        result = {
            "username": inf_username,
            "nationality": data["nationality"],
            "age": age_dist,
            "gender": gender_dist,
            "total_followers_analyzed": len(data["follower_usernames"]),
        }

        # Per-influencer JSON
        result_path = results_dir / "results" / f"{inf_username}.json"
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"  ❌ @{inf_username} kaydetme hatası: {e}")
            continue

        # Mark as completed
        with open(completed_file, "a", encoding="utf-8") as f:
            f.write(f"{inf_username}\n")

        summary_rows.append(result)
        csv_rows_buffer.append(_flatten_result_to_csv_row(result))
        successfully_processed += 1

        # Her BATCH_CSV_SIZE influencer'da bir CSV dosyası kaydet
        if len(csv_rows_buffer) >= BATCH_CSV_SIZE:
            batch_end = successfully_processed
            batch_start = batch_end - len(csv_rows_buffer)
            _save_csv_batch(csv_rows_buffer, batch_start, batch_end)
            csv_rows_buffer = []

    # Kalan buffer'daki influencer'ları da kaydet
    if csv_rows_buffer:
        batch_end = successfully_processed
        batch_start = batch_end - len(csv_rows_buffer)
        _save_csv_batch(csv_rows_buffer, batch_start, batch_end)
        csv_rows_buffer = []

    # ----- 9. Summary CSV (tüm sonuçlar tek dosyada) -----
    if summary_rows:
        # Eski summary
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log.info(f"  📊 Summary CSV kaydedildi: {summary_path}")

        # Birleşik düz CSV (tüm influencer'lar)
        all_csv_rows = [_flatten_result_to_csv_row(r) for r in summary_rows]
        all_df = pd.DataFrame(all_csv_rows)
        all_csv_path = csv_batch_dir / "sonuc_tumu.csv"
        all_df.to_csv(all_csv_path, index=False, encoding="utf-8-sig")
        log.info(f"  📊 Birleşik CSV kaydedildi: {all_csv_path} ({len(all_csv_rows)} influencer)")

    # ----- 10. Final cache kaydet -----
    save_cache(follower_cache, cache_file)

    # ----- 11. Özet -----
    total_time = time.time() - pipeline_start
    log.info("\n" + "=" * 60)
    log.info("PIPELINE TAMAMLANDI ✅")
    log.info("=" * 60)
    log.info(f"  Toplam influencer: {successfully_processed}")
    log.info(f"  Toplam indirilen görsel: {total_downloaded}")
    log.info(f"  Toplam cache hit: {total_cached_hits}")
    log.info(f"  Follower cache boyutu: {len(follower_cache)}")
    log.info(f"  Toplam süre: {total_time:.2f}s ({total_time/60:.1f} dakika)")

    if successfully_processed > 0:
        avg = total_time / successfully_processed
        log.info(f"  Ortalama süre/influencer: {avg:.2f}s")


# =====================================================================
#  ENTRY POINT
# =====================================================================
# Colab/Jupyter zaten bir event loop çalıştırır — nest_asyncio ile uyumlu hale getir
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # Standalone çalışıyorsa gerek yok

if __name__ == "__main__":
    asyncio.run(main())
