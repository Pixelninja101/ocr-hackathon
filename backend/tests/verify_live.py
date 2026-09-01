import httpx
import json

def test_upload(filename, page=0):
    with open(f"samples/{filename}", "rb") as f:
        file_bytes = f.read()
    files = {"file": (filename, file_bytes)}
    data = {"page": str(page)}
    res = httpx.post("http://127.0.0.1:8001/api/qr/scan", files=files, data=data)
    print(f"=== Scanning {filename} (page={page}) ===")
    print("HTTP Status:", res.status_code)
    json_data = res.json()
    print("Success:", json_data.get("success"))
    print("QR Detected:", json_data.get("qr_detected"))
    print("QR Count:", json_data.get("qr_count"))
    for i, c in enumerate(json_data.get("codes", [])):
        print(f"  Code #{i+1}: decoded={c['decoded']}, data={c['data']}, bbox={c['bbox']}")
    print("Warnings:", json_data.get("warnings"))
    print("Errors:", json_data.get("errors"))
    print()

if __name__ == "__main__":
    test_upload("single_qr_test.png")
    test_upload("multi_qr_test.jpg")
    test_upload("sample_document.pdf", page=0)
    test_upload("sample_document.pdf", page=1)
    test_upload("no_qr_test.png")
