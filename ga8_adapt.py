"""TDS GA8 Q4 — Choose the Minimal Adaptation and Repair a PEFT Run.

POST /adapt -> two operations (choose / repair).
"""
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]  # published priority order
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _utf8(s): return s.encode("utf-8")
def _finite(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")
def _safe_int(v): return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 2**53 - 1


def op_choose(body):
    policy = body.get("policy")
    candidates = body.get("candidates")
    if not isinstance(policy, dict) or not isinstance(candidates, list):
        return ("400", None)

    by_name = {c.get("name"): c for c in candidates if isinstance(c, dict)}
    total_costs = {}
    reason_codes = {}
    eligible = []

    horizon = policy.get("horizonRequests")
    for name in INTERVENTIONS:
        c = by_name.get(name)
        codes = set()
        cost = None
        if c is None:
            codes.add("INVALID_INPUT")
        else:
            # compute cost
            otc = c.get("oneTimeCost"); rec = c.get("recurringCost")
            if _finite(otc) and _finite(rec) and _safe_int(horizon):
                cost = round(otc + horizon * rec, 12)
            # gates
            if c.get("available") is not True:
                codes.add("UNAVAILABLE")
            q = c.get("quality")
            if not (_finite(q) and 0 <= q <= 1) or (_finite(q) and q < policy.get("minQuality", 0)):
                if not (_finite(q) and q >= policy.get("minQuality", 0)):
                    codes.add("QUALITY_FLOOR")
            if policy.get("freshnessRequired") is True and c.get("freshness") is not True:
                codes.add("FRESHNESS_REQUIRED")
            lat = c.get("latencyMs")
            if not _finite(lat) or lat > policy.get("maxLatencyMs", float("inf")):
                codes.add("LATENCY_LIMIT")
            mem = c.get("memoryMb")
            if not _finite(mem) or mem > policy.get("maxMemoryMb", float("inf")):
                codes.add("MEMORY_LIMIT")
            le = c.get("labeledExamples")
            if not _safe_int(le) or le > policy.get("maxLabeledExamples", float("inf")):
                codes.add("DATA_LIMIT")
            if cost is None or cost > policy.get("maxTotalCost", float("inf")):
                codes.add("COST_LIMIT")
        total_costs[name] = cost
        reason_codes[name] = sorted(codes, key=lambda s: _utf8(s))
        if not codes:
            eligible.append(name)

    selected = eligible[0] if eligible else None
    return ("ok", {
        "selected": selected,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes,
    })


def op_repair(body):
    codes = set()
    tokens = body.get("tokens")

    # ---- tokens / labels ----
    labels = []
    tokens_valid = isinstance(tokens, list) and len(tokens) > 0
    if tokens_valid:
        for t in tokens:
            if not (isinstance(t, dict) and _safe_int(t.get("id"))
                    and t.get("role") in ("system", "user", "assistant")
                    and isinstance(t.get("padding"), bool)
                    and isinstance(t.get("text"), str)):
                tokens_valid = False
                break
    if not tokens_valid:
        codes.add("INVALID_TOKEN")
        labels = [-100 for _ in tokens] if isinstance(tokens, list) else []
    else:
        for t in tokens:
            if t["role"] == "assistant" and t["padding"] is False:
                labels.append(t["id"])
            else:
                labels.append(-100)

    # ---- template ----
    template_pass = body.get("templateApplications") == 1
    if not template_pass:
        codes.add("CHAT_TEMPLATE_COUNT")

    # ---- parameters / trainable ----
    params = body.get("parameters")
    allowed_targets = body.get("allowedTargets")
    trainable = []
    trainable_count = 0
    params_valid = (isinstance(params, list)
                    and isinstance(allowed_targets, list) and len(allowed_targets) > 0
                    and all(isinstance(x, str) and x for x in allowed_targets)
                    and len(set(allowed_targets)) == len(allowed_targets))
    if isinstance(params, list):
        names_seen = [p.get("name") for p in params if isinstance(p, dict)]
        if len(names_seen) != len(set(names_seen)):
            params_valid = False
        for p in params:
            if not (isinstance(p, dict) and isinstance(p.get("name"), str)
                    and _safe_int(p.get("numel")) and p.get("numel") > 0):
                params_valid = False
    else:
        params_valid = False

    if not params_valid:
        codes.add("INVALID_PARAMETER")
    else:
        at = set(allowed_targets)
        picked = []
        for p in params:
            nm = p["name"]
            if p.get("target") in at and (nm.endswith(".lora_A.weight") or nm.endswith(".lora_B.weight")):
                picked.append(p)
        if not picked:
            codes.add("INVALID_PARAMETER")
        else:
            trainable = sorted([p["name"] for p in picked], key=lambda s: _utf8(s))
            trainable_count = sum(p["numel"] for p in picked)

    # ---- inference mode ----
    inference_ok = body.get("inferenceMode") is False
    if not inference_ok:
        codes.add("INFERENCE_MODE")

    # ---- adapter files ----
    art = body.get("artifactFiles")
    adapter_files = []
    expected = {"adapter_config.json", "adapter_model.safetensors"}
    if isinstance(art, list) and sorted(art) == sorted(expected) and len(art) == 2:
        adapter_files = sorted(expected, key=lambda s: _utf8(s))
        adapter_ok = True
    else:
        # detect full-model artifact vs wrong set
        adapter_ok = False
        if isinstance(art, list) and any(isinstance(x, str) and
                (x.endswith(".bin") or "pytorch_model" in x) for x in art):
            codes.add("FULL_MODEL_ARTIFACT")
        else:
            codes.add("ADAPTER_FILE_SET")
        adapter_files = sorted(set(x for x in art if isinstance(x, str)), key=lambda s: _utf8(s)) \
            if isinstance(art, list) else []

    # ---- checkpoint ----
    ckpt = body.get("checkpoint")
    req_ckpt = {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}
    checkpoint_complete = isinstance(ckpt, dict) and req_ckpt.issubset(set(ckpt.keys()))
    if not checkpoint_complete:
        codes.add("INCOMPLETE_CHECKPOINT")

    # ---- lineage ----
    base_rev = body.get("baseRevision")
    lineage_pass = isinstance(base_rev, str) and bool(_HEX40.match(base_rev))
    if not lineage_pass:
        codes.add("MUTABLE_BASE_REVISION")
    dd = body.get("datasetDigest"); cd = body.get("codeDigest"); cfg = body.get("configDigest")
    exp = body.get("expectedDigests")
    digest_ok = all(isinstance(x, str) and _HEX64.match(x) for x in (dd, cd, cfg))
    if isinstance(exp, dict):
        for k, want in exp.items():
            got = {"dataset": dd, "code": cd, "config": cfg,
                   "datasetDigest": dd, "codeDigest": cd, "configDigest": cfg}.get(k)
            if got != want:
                digest_ok = False
    if not digest_ok:
        codes.add("LINEAGE_MISMATCH")
        lineage_pass = lineage_pass and False

    # ---- effective batch ----
    mb = body.get("microBatch"); ga = body.get("gradientAccumulation")
    rep = body.get("replicas"); eff = body.get("expectedEffectiveBatch")
    batch_ok = all(_safe_int(x) and x > 0 for x in (mb, ga, rep, eff)) and (mb * ga * rep == eff)
    if not batch_ok:
        codes.add("EFFECTIVE_BATCH_MISMATCH")

    # ---- eval isolation ----
    train_ids = body.get("trainRowIds"); eval_ids = body.get("evalRowIds")
    eval_isolated = (isinstance(train_ids, list) and isinstance(eval_ids, list)
                     and all(isinstance(x, str) and x for x in train_ids)
                     and all(isinstance(x, str) and x for x in eval_ids)
                     and len(set(train_ids)) == len(train_ids)
                     and len(set(eval_ids)) == len(eval_ids)
                     and len(train_ids) > 0 and len(eval_ids) > 0
                     and set(train_ids).isdisjoint(set(eval_ids)))
    if not eval_isolated:
        codes.add("EVAL_LEAKAGE")

    # ---- eval determinism (dropout) ----
    eval_deterministic = body.get("dropoutActiveDuringEval") is False
    if not eval_deterministic:
        codes.add("EVAL_DROPOUT_ACTIVE")

    # ---- resume ----
    uw = body.get("uninterruptedWeights"); rw = body.get("resumedWeights")
    tol = body.get("resumeTolerance")
    resume_pass = (isinstance(uw, list) and isinstance(rw, list)
                   and len(uw) == len(rw) and len(uw) > 0
                   and all(_finite(x) for x in uw) and all(_finite(x) for x in rw)
                   and _finite(tol) and tol >= 0
                   and all(abs(a - b) <= tol for a, b in zip(uw, rw)))
    if not resume_pass:
        codes.add("RESUME_DIVERGENCE")

    return ("ok", {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable,
        "trainableCount": trainable_count,
        "peftConfigPass": params_valid and bool(trainable),
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass and digest_ok,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": eval_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": sorted(codes, key=lambda s: _utf8(s)),
    })


@router.post("/adapt")
async def adapt(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    op = body.get("operation")
    try:
        if op == "choose":
            status, result = op_choose(body)
            if status == "400":
                return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
            return JSONResponse(result)
        elif op == "repair":
            status, result = op_repair(body)
            return JSONResponse(result)
        else:
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)


@router.get("/adapt")
async def adapt_info():
    return JSONResponse({"service": "TDS GA8 Adapt", "endpoint": "POST /adapt"})
