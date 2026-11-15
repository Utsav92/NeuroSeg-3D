import nibabel as nib
import numpy as np
import pytest
from conftest import TEST_AFFINE
from fastapi.testclient import TestClient

from neuroseg.api.main import create_app

MODS = ["flair", "t1", "t1ce", "t2"]


def files_for(synth_root, tmp_path, idx=0, drop=None):
    img = nib.load(str(synth_root / "imagesTr" / f"SYNTH_{idx:03d}.nii.gz"))
    arr = np.asarray(img.dataobj)
    out = {}
    for c, m in enumerate(MODS):
        if m == drop:
            continue
        p = tmp_path / f"{m}.nii.gz"
        nib.save(nib.Nifti1Image(arr[..., c], img.affine), str(p))
        out[m] = p
    return out


def post(client, files, **data):
    handles = {m: (f"{m}.nii.gz", open(p, "rb"), "application/gzip") for m, p in files.items()}
    try:
        return client.post("/predict", files=handles, data=data)
    finally:
        for _, h, _ in handles.values():
            h[1].close() if isinstance(h, tuple) else h.close()


def wait(client, jid, timeout=120):
    import time

    t = time.time()
    while time.time() - t < timeout:
        s = client.get(f"/jobs/{jid}").json()
        if s["status"] in ("succeeded", "failed", "timeout"):
            return s
        time.sleep(0.3)
    raise TimeoutError


@pytest.fixture()
def client(trained, tmp_path):
    app = create_app(str(trained["ckpt"]), "cpu", str(tmp_path / "jobs"), max_upload_mb=50, max_queue=2)
    with TestClient(app) as c:
        yield c


def test_health_and_model_info(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and "not a medical device" in h["research_only"]
    info = client.get("/model/info").json()
    assert info["architecture"] == "resunet3d" and info["modalities"] == MODS and info["n_params"] > 0


def test_no_checkpoint_never_fakes_predictions(tmp_path, synth_root):
    with TestClient(create_app(str(tmp_path / "missing.pt"), "cpu", str(tmp_path / "j"))) as c:
        assert c.get("/health").json()["status"] == "no_model"
        assert c.get("/model/info").status_code == 503
        r = post(c, files_for(synth_root, tmp_path))
        assert r.status_code == 503 and "No trained model" in r.json()["detail"]


def test_predict_job_lifecycle(client, synth_root, tmp_path):
    r = post(client, files_for(synth_root, tmp_path), mc_samples=2, return_probs=True)
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert wait(client, jid)["status"] == "succeeded"
    res = client.get(f"/results/{jid}").json()
    assert (
        "WT" in res["volumes"]["regions"]
        and "segmentation" in res["artifacts"]
        and res["metadata"]["research_only"]
    )
    f = client.get(res["artifacts"]["segmentation"])
    assert f.status_code == 200 and len(f.content) > 100
    assert client.get(f"/results/{jid}/files/../../etc/passwd").status_code in (404, 422)
    assert client.get("/jobs/doesnotexist").status_code == 404


def test_validation_errors(client, synth_root, tmp_path):
    r = post(client, files_for(synth_root, tmp_path, drop="t2"))
    assert r.status_code == 422 and "t2" in r.json()["detail"]
    fs = files_for(synth_root, tmp_path)
    bad = tmp_path / "t1.txt"
    bad.write_text("hello")
    r = client.post(
        "/predict",
        files={
            **{m: (f"{m}.nii.gz", open(p, "rb")) for m, p in fs.items() if m != "t1"},
            "t1": ("t1.txt", open(bad, "rb")),
        },
    )
    assert r.status_code == 422
    garbage = tmp_path / "g.nii.gz"
    garbage.write_bytes(b"not nifti at all")
    r = client.post(
        "/predict",
        files={
            **{m: (f"{m}.nii.gz", open(p, "rb")) for m, p in fs.items() if m != "t1"},
            "t1": ("g.nii.gz", open(garbage, "rb")),
        },
    )
    assert r.status_code == 422 and r.json()["error"] == "invalid_input"
    a = TEST_AFFINE.copy()
    a[1, 3] += 7
    nib.save(nib.Nifti1Image(np.asarray(nib.load(str(fs["t1"])).dataobj), a), str(fs["t1"]))
    assert post(client, fs).status_code == 422  # geometrically inconsistent
    assert post(client, files_for(synth_root, tmp_path), mc_samples=999).status_code == 422


def test_upload_size_limit(trained, synth_root, tmp_path):
    app = create_app(str(trained["ckpt"]), "cpu", str(tmp_path / "jobs"), max_upload_mb=0)
    with TestClient(app) as c:
        assert post(c, files_for(synth_root, tmp_path)).status_code == 413


def test_temp_files_cleaned(trained, synth_root, tmp_path):
    app = create_app(str(trained["ckpt"]), "cpu", str(tmp_path / "jobs"))
    with TestClient(app) as c:
        jid = post(c, files_for(synth_root, tmp_path)).json()["job_id"]
        wait(c, jid)
        svc = app.state.service
        assert not list((svc.jobs[jid].dir).glob("*.nii*"))  # uploaded inputs deleted after inference
        svc.ttl_s = -1
        assert svc.cleanup() == 1 and c.get(f"/jobs/{jid}").status_code == 404
    assert not (tmp_path / "jobs").exists()  # graceful shutdown removes work dir
