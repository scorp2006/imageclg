import json
import numpy as np

with open("D:/Downloads/q-cosine-similarity-server.json") as f:
    data = json.load(f)

# The JSON structure typically has "documents" and "queries" lists.
# Each item is either {"id": "D000001", "embedding": [...]} or similar.
# Auto-detect the structure.

def get_id_and_vec(item):
    # Try common key names
    for id_key in ("id", "doc_id", "query_id"):
        if id_key in item:
            iid = item[id_key]
            break
    else:
        iid = None
    for vec_key in ("embedding", "vector", "vec", "emb"):
        if vec_key in item:
            vec = item[vec_key]
            break
    else:
        vec = None
    return iid, vec


# Detect containers
if "documents" in data and "queries" in data:
    docs_raw = data["documents"]
    queries_raw = data["queries"]
elif "docs" in data and "queries" in data:
    docs_raw = data["docs"]
    queries_raw = data["queries"]
else:
    # Guess by key names
    print("Top-level keys:", list(data.keys()))
    raise SystemExit("Adjust script — please tell me the top-level keys shown above.")

doc_ids = []
doc_vecs = []
for d in docs_raw:
    iid, vec = get_id_and_vec(d)
    doc_ids.append(iid)
    doc_vecs.append(vec)

query_ids = []
query_vecs = []
for q in queries_raw:
    iid, vec = get_id_and_vec(q)
    query_ids.append(iid)
    query_vecs.append(vec)

D = np.array(doc_vecs)  # (250, 64)
Q = np.array(query_vecs)  # (10, 64)

# Cosine similarity: since embeddings are unit-normalized, dot product = cosine
# Sims: (10, 250) — sims[i, j] = similarity of query i to doc j
sims = Q @ D.T

result = {}
for i, qid in enumerate(query_ids):
    s = sims[i]  # (250,)
    # Sort by (-similarity, doc_id) so ties break by smaller doc_id
    order = sorted(range(len(doc_ids)), key=lambda j: (-s[j], doc_ids[j]))
    top5 = [doc_ids[j] for j in order[:5]]
    result[qid] = top5

output_json = json.dumps(result, indent=2)
print(output_json)

with open("D:/Downloads/cosine_answer.json", "w") as f:
    f.write(output_json)

print("\nSaved to D:/Downloads/cosine_answer.json")
