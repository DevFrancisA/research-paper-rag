import pytest
from fastapi.testclient import TestClient

import main


def make_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 11 Tf 40 700 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


@pytest.fixture
def client(service, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_service", lambda: service)
    monkeypatch.setattr(main.settings, "upload_dir", str(tmp_path))
    return TestClient(main.app)


def test_upload_query_and_filter(client, service):
    pdf = make_pdf(["The Zephyr turbine produces 4.2 megawatts at rated wind speed of twelve meters per second.",
                    "Maintenance of the Zephyr gearbox is scheduled every eighteen months by certified staff."])
    r = client.post("/ingest", files={"file": ("zephyr.pdf", pdf, "application/pdf")},
                    data={"metadata": '{"category": "energy"}'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["num_pages"] == 2 and body["num_chunks"] == 2 and not body["skipped"]

    again = client.post("/ingest", files={"file": ("zephyr.pdf", pdf, "application/pdf")}).json()
    assert again["skipped"]  # same content hash

    assert "zephyr.pdf" in [d["source"] for d in client.get("/documents").json()]

    r = client.post("/query", json={"question": "How often is the Zephyr gearbox maintained?",
                                    "filter": {"sources": ["zephyr.pdf"]}})
    answer = r.json()
    assert answer["sources"][0]["page"] == 2
    assert answer["citations"][0]["source"] == "zephyr.pdf"

    r = client.post("/query", json={"question": "How often is the Zephyr gearbox maintained?",
                                    "filter": {"metadata": {"category": "food"}}})
    assert all(s["source"] == "cooking.pdf" for s in r.json()["sources"])

    assert client.get("/stats").json()["latency"]["count"] == 2


def test_rejects_non_pdf(client):
    r = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
