import json
import subprocess
import sys

with open("D:/Downloads/q-youtube-metadata-filter-server.json") as f:
    params = json.load(f)

source_urls = params["source_urls"]
min_dur = params["min_duration_seconds"]
max_dur = params["max_duration_seconds"]
required = [w.lower() for w in params["required_words"]]
forbidden = [w.lower() for w in params["forbidden_words"]]
limit = params["limit"]

videos = []
skipped = []

for url in source_urls:
    print(f"Fetching {url}...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--skip-download", url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  SKIPPED: {result.stderr[:150]}", file=sys.stderr)
            skipped.append(url)
            continue
        meta = json.loads(result.stdout)
        videos.append(meta)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        skipped.append(url)
        continue

print(f"\n=== Fetched {len(videos)} of {len(source_urls)} videos ===", file=sys.stderr)
print(f"Skipped: {len(skipped)}", file=sys.stderr)
for u in skipped:
    print(f"  {u}", file=sys.stderr)

filtered = []
rejected = []
for v in videos:
    duration = v.get("duration") or 0
    title = (v.get("title") or "").lower()
    desc = (v.get("description") or "").lower()
    combined = title + " " + desc
    vid = v.get("id")

    if not (min_dur <= duration <= max_dur):
        rejected.append((vid, f"duration={duration}"))
        continue
    if not all(w in combined for w in required):
        missing = [w for w in required if w not in combined]
        rejected.append((vid, f"missing_required={missing}"))
        continue
    if any(w in title or w in desc for w in forbidden):
        found = [w for w in forbidden if w in title or w in desc]
        rejected.append((vid, f"forbidden_found={found}"))
        continue

    filtered.append(v)

print(f"\n=== Filtered to {len(filtered)} videos ===", file=sys.stderr)
print("\nRejected:", file=sys.stderr)
for vid, reason in rejected:
    print(f"  {vid}: {reason}", file=sys.stderr)

print("\nKept (before sort):", file=sys.stderr)
for v in filtered:
    print(f"  id={v.get('id')} date={v.get('upload_date')} dur={v.get('duration')} title={(v.get('title') or '')[:60]}", file=sys.stderr)

filtered.sort(key=lambda v: (-int(v.get("upload_date") or 0), (v.get("id") or "").lower()))

print("\nAfter sorting:", file=sys.stderr)
for v in filtered[:limit]:
    print(f"  id={v.get('id')} date={v.get('upload_date')}", file=sys.stderr)

top = filtered[:limit]

output = {"urls": [f"https://www.youtube.com/watch?v={v['id']}" for v in top]}

with open("D:/Downloads/output.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n" + json.dumps(output, indent=2))
