#!/usr/bin/env python
import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------- config / env ----------

load_dotenv()  # loads TMDB_API_KEY, OPENAI_API_KEY from .env if present

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "knowledge-grafting/0.1 (tnief@somewhere.edu)"  # customize
    }
)

# ---------- TMDb helpers ----------


def tmdb_discover_movies(page, api_key):
    url = (
        "https://api.themoviedb.org/3/discover/movie"
        f"?api_key={api_key}"
        "&primary_release_date.gte=2024-05-01"
        "&with_original_language=en"
        "&sort_by=popularity.desc"
        f"&page={page}"
    )
    r = SESSION.get(url)
    if r.status_code != 200:
        print(f"[tmdb_discover_movies] HTTP {r.status_code} for {url}")
        return None
    return r.json()


def tmdb_external_ids(movie_id, api_key):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}/external_ids?api_key={api_key}"
    )
    r = SESSION.get(url)
    if r.status_code != 200:
        print(f"[tmdb_external_ids] HTTP {r.status_code} for {url}")
        return None
    return r.json()


def tmdb_movie_credits(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}"
    r = SESSION.get(url)
    if r.status_code != 200:
        print(f"[tmdb_movie_credits] HTTP {r.status_code} for {url}")
        return None
    return r.json()


# ---------- Wikidata / Wikipedia helpers (reuse your notebook logic) ----------


def wikidata_to_wikipedia_title(wikidata_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    r = SESSION.get(url)
    if r.status_code != 200:
        print(f"[wikidata_to_wikipedia_title] HTTP {r.status_code} for {url}")
        return None
    try:
        data = r.json()
    except ValueError as e:
        print(f"[wikidata_to_wikipedia_title] JSON error for {url}: {e}")
        print("Body (truncated):", r.text[:200])
        return None

    entity = data.get("entities", {}).get(wikidata_id)
    if not entity:
        print(f"[wikidata_to_wikipedia_title] No entity for {wikidata_id}")
        return None
    sitelinks = entity.get("sitelinks", {})
    title = sitelinks.get("enwiki", {}).get("title")
    if not title:
        print(f"[wikidata_to_wikipedia_title] No enwiki sitelink for {wikidata_id}")
    return title


def search_wikipedia_title(title):
    import urllib.parse

    query = urllib.parse.quote(title)
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={query}&utf8=1&format=json&srlimit=1"
    )
    r = SESSION.get(url)
    if r.status_code != 200:
        print(f"[search_wikipedia_title] HTTP {r.status_code} for {url}")
        return None
    data = r.json()
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0]["title"]


def get_wikipedia_wikitext(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "content",
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    r = SESSION.get(url, params=params)
    if r.status_code != 200:
        print(f"[get_wikipedia_wikitext] HTTP {r.status_code} for {r.url}")
        return None

    data = r.json()
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        print(f"[get_wikipedia_wikitext] No pages for {title!r}")
        return None
    page = pages[0]
    if "missing" in page:
        print(f"[get_wikipedia_wikitext] Page missing for {title!r}")
        return None
    revs = page.get("revisions", [])
    if not revs:
        print(f"[get_wikipedia_wikitext] No revisions for {title!r}")
        return None
    return revs[0]["slots"]["main"]["content"]


# ---------- LLM rewrites ----------


def rewrite_text_k_times(text, k, client, model="gpt-4o-mini", temperature=0.9):
    rewrites = []
    for i in range(k):
        print(f"[rewrite] {i + 1}/{k}")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful editor. Rewrite the article in different words, "
                        "preserving all factual content."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Please rewrite the following article with different phrasing. "
                        "Do not omit information about the cast, but you can remove wiki formatting and URLs. You can also take liberties with the structure of the article, as long as the key facts and cast information are preserved.\n\n"
                        f"{text}"
                    ),
                },
            ],
            temperature=temperature,
        )
        rewrites.append(resp.choices[0].message.content)
    return rewrites


# ---------- chunking ----------


def chunk_text(text, max_chars=4000, overlap=200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


# ---------- main pipeline ----------


def build_tmdb_wiki_corpus(
    api_key,
    output_path,
    max_movies,
    min_popularity,
    max_chars,
    overlap,
    num_rewrites,
    openai_key,
    metadata_output_path=None,
    metadata_only=False,  # <-- new parameter
):
    out = open(output_path, "w", encoding="utf-8")
    metadata_out = (
        open(metadata_output_path, "w", encoding="utf-8")
        if metadata_output_path is not None
        else None
    )
    next_metadata_id = 1
    total_movies = 0
    page = 1

    client = None
    if num_rewrites > 0:
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set but num_rewrites>0")
        client = OpenAI(api_key=openai_key)

    start_time = time.time()

    while total_movies < max_movies:
        print(f"[pipeline] Fetching TMDb page {page}...")
        data = tmdb_discover_movies(page, api_key)
        if not data or "results" not in data or not data["results"]:
            break

        for m in data["results"]:
            if total_movies >= max_movies:
                break

            popularity = m.get("popularity", 0.0)
            if popularity < min_popularity:
                continue

            movie_id = m["id"]
            title = m.get("title")
            print(
                f"\n[{total_movies + 1}/{max_movies}] Processing {title!r} ({movie_id}) "
                f"(elapsed {time.time() - start_time:0.1f}s)"
            )

            # --- NEW: write metadata FIRST ---
            if metadata_out is not None:
                credits = tmdb_movie_credits(movie_id, api_key)
                cast = credits.get("cast", []) if credits else []
                first_actor = cast[0]["name"] if len(cast) >= 1 else None
                second_actor = cast[1]["name"] if len(cast) >= 2 else None
                md_row = {
                    "first_actor": first_actor,
                    "second_actor": second_actor,
                    "movie_title": title,
                    "id": next_metadata_id,
                }
                metadata_out.write(json.dumps(md_row, ensure_ascii=False) + "\n")
                next_metadata_id += 1

                if metadata_only:
                    total_movies += 1
                    print(
                        f"[pipeline] Wrote metadata for {title!r} (metadata-only mode)"
                    )
                    continue  # skip Wikipedia + chunking

            ext = tmdb_external_ids(movie_id, api_key)
            wikidata_id = ext.get("wikidata_id") if ext else None

            wiki_title = None
            if wikidata_id:
                wiki_title = wikidata_to_wikipedia_title(wikidata_id)
            if not wiki_title and title:
                wiki_title = search_wikipedia_title(title)
            if not wiki_title:
                print(f"[pipeline] No Wikipedia title for {title!r}")
                continue

            wikitext = get_wikipedia_wikitext(wiki_title)
            if not wikitext:
                print(f"[pipeline] No wikitext for {wiki_title!r}")
                continue

            variants = [wikitext]
            if client is not None and num_rewrites > 0:
                rewrites = rewrite_text_k_times(wikitext, num_rewrites, client)
                variants.extend(rewrites)

            total_chunks = 0
            for text_variant in variants:
                chunks = chunk_text(text_variant, max_chars=max_chars, overlap=overlap)
                for chunk in chunks:
                    out.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                total_chunks += len(chunks)

            total_movies += 1
            print(f"[pipeline] Wrote {total_chunks} chunks for {wiki_title!r}")

        if total_movies >= max_movies:
            break
        if page >= data.get("total_pages", 1):
            break
        page += 1

    out.close()
    if metadata_out is not None:
        metadata_out.close()
    print(f"[pipeline] Done. Wrote chunks from {total_movies} movies to {output_path}.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=Path("tmdb_wiki_chunks_with_rewrites.jsonl")
    )
    p.add_argument("--max-movies", type=int, default=20)
    p.add_argument("--min-popularity", type=float, default=10.0)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--num-rewrites", type=int, default=0)
    p.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Where to write one JSONL line of metadata per movie (optional).",
    )
    p.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only fetch TMDb credits and write metadata; skip Wikipedia and chunking.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY not set in env")
    build_tmdb_wiki_corpus(
        api_key=TMDB_API_KEY,
        output_path=args.output,
        max_movies=args.max_movies,
        min_popularity=args.min_popularity,
        max_chars=args.max_chars,
        overlap=args.overlap,
        num_rewrites=args.num_rewrites,
        openai_key=OPENAI_API_KEY,
        metadata_output_path=args.metadata_output,
        metadata_only=args.metadata_only,
    )
