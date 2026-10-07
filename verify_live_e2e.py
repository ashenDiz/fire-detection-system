"""
Live End-to-End integration test script against running Uvicorn and Vite servers.
Tests live physical webcam start, system status integration, streaming bytes,
stop, release, negative index 422 rejection, and frontend HTML delivery.
"""

import urllib.request
import urllib.error
import json
import time

base = 'http://127.0.0.1:8000/api/v1'

print("=" * 60)
print("LIVE E2E INTEGRATION & PHYSICAL WEBCAM TEST")
print("=" * 60)

# 1. System status initial
with urllib.request.urlopen(f'{base}/system/status') as r:
    st = json.loads(r.read())
    print('[1] Initial camera status:', st['camera'])
    assert st['camera'] == 'Disconnected'

# 2. Start physical camera
print('[2] Calling POST /api/v1/camera/start on physical webcam...')
req = urllib.request.Request(f'{base}/camera/start', data=b'{}', headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as r:
    res = json.loads(r.read())
    print('    Start result:', res['status'], f"{res['width']}x{res['height']} @ {res['fps']} FPS")
    assert res['connected'] is True

# 3. System status with camera active
with urllib.request.urlopen(f'{base}/system/status') as r:
    st = json.loads(r.read())
    print('[3] System status while running:', st['camera'])
    assert st['camera'] == 'Connected'

# 4. Stream endpoint
print('[4] Reading live MJPEG stream bytes from /api/v1/camera/stream...')
req = urllib.request.Request(f'{base}/camera/stream')
with urllib.request.urlopen(req) as r:
    chunk = r.read(2048)
    print(f'    Stream chunk received ({len(chunk)} bytes)')
    assert b'--frame' in chunk or b'Content-Type: image/jpeg' in chunk

# 5. Stop camera
print('[5] Calling POST /api/v1/camera/stop...')
req = urllib.request.Request(f'{base}/camera/stop', data=b'{}', headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as r:
    res = json.loads(r.read())
    print('    Stop result:', res['status'])
    assert res['connected'] is False

# 6. System status after stop
with urllib.request.urlopen(f'{base}/system/status') as r:
    st = json.loads(r.read())
    print('[6] Final camera status:', st['camera'])
    assert st['camera'] == 'Disconnected'

# 7. Negative index validation (422)
print('[7] Testing negative camera index rejection (camera_index = -1)...')
try:
    req = urllib.request.Request(f'{base}/camera/start', data=b'{"camera_index": -1}', headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req)
    raise AssertionError("Negative index was not rejected!")
except urllib.error.HTTPError as e:
    print('    Negative index correctly rejected with HTTP status code:', e.code)
    assert e.code == 422

# 8. Frontend HTML delivery
print('[8] Verifying frontend Vite server HTML delivery...')
with urllib.request.urlopen('http://127.0.0.1:5173/') as r:
    html = r.read().decode('utf-8')
    print('    Frontend HTML delivered successfully. Title verified:', 'Intelligent Fire Detection' in html)
    assert 'Intelligent Fire Detection' in html

print("\n" + "=" * 60)
print(">>> ALL LIVE E2E & PHYSICAL WEBCAM CHECKS PASSED! <<<")
print("=" * 60)
