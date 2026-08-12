"""CI smoke tests for the GA7 release-gate policy (no env vars needed)."""
from ga7_release_gate import evaluate

CLEAN_PREVIEW = {
    "target": "preview", "event": "pull_request", "ref": "refs/heads/x",
    "workflow": {
        "trigger": "pull_request",
        "permissions": {"contents": "read", "packages": "write", "id-token": "none"},
        "testsPassed": True, "matrixComplete": True, "failFast": False,
        "actions": [{"owner": "actions", "name": "checkout", "ref": "v4"}],
    },
    "image": {"multiStage": True, "runsAsRoot": False, "secretMode": "none",
              "criticalVulnerabilities": 0, "digestPinned": True},
}

CLEAN_PROD = {
    "target": "production", "event": "push", "ref": "refs/heads/main",
    "workflow": {
        "trigger": "push", "environmentApproval": True,
        "permissions": {"contents": "read", "packages": "write", "id-token": "none"},
        "testsPassed": True, "matrixComplete": True, "failFast": False, "actions": [],
    },
    "image": {"multiStage": True, "runsAsRoot": False, "secretMode": "buildkit",
              "criticalVulnerabilities": 0, "digestPinned": True},
}

ALL_BROKEN = {
    "target": "preview", "event": "pull_request", "ref": "x",
    "workflow": {
        "trigger": "pull_request_target",
        "permissions": {"contents": "read", "packages": "write", "id-token": "none", "actions": "write"},
        "testsPassed": False, "matrixComplete": True, "failFast": True,
        "actions": [{"owner": "docker", "name": "build", "ref": "v2"}],
    },
    "image": {"multiStage": False, "runsAsRoot": True, "secretMode": "arg",
              "criticalVulnerabilities": 3, "digestPinned": False},
}


def check(name, body, decision, violations):
    r = evaluate(body)
    assert r["decision"] == decision, f"{name}: {r}"
    assert sorted(r["violations"]) == sorted(violations), f"{name}: {r['violations']}"
    print(f"OK {name}: {r}")


check("clean_preview", CLEAN_PREVIEW, "promote", [])
check("clean_prod", CLEAN_PROD, "promote", [])
check("all_broken", ALL_BROKEN, "block", [
    "CRITICAL_CVE", "EXCESS_PERMISSION", "MUTABLE_ACTION", "ROOT_RUNTIME",
    "SECRET_IN_LAYER", "SINGLE_STAGE_IMAGE", "TESTS_INCOMPLETE",
    "UNPINNED_IMAGE", "UNSAFE_PR_TRIGGER",
])
print("All GA7 policy tests passed.")
