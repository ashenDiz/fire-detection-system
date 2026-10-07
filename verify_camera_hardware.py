"""
Real hardware camera verification script for Phase 2.
Tests actual physical webcam opening, frame capture, resolution check,
generator streaming, clean resource release, and reopening.
"""

import sys
import os
import time

# Add backend to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.camera_service import CameraService
from app.core.config import settings

def main():
    print("=" * 60)
    print("PHASE 2: REAL WEBCAM HARDWARE VERIFICATION")
    print("=" * 60)

    # 1. Start camera on configured camera index
    print(f"\n[1] Attempting to open physical webcam on index {settings.CAMERA_INDEX}...")
    start_info = CameraService.start_camera(settings.CAMERA_INDEX)
    print("Start info:", start_info)

    if not start_info["connected"]:
        print(f"\n[FAILED] Physical webcam could not be opened: {start_info.get('error')}")
        print("Note: Automated camera integration tests passed with mocks, but physical webcam is not currently accessible.")
        return False

    print(f"\n[SUCCESS] Physical webcam opened successfully!")
    print(f"  - Status: {start_info['status']}")
    print(f"  - Width: {start_info['width']}")
    print(f"  - Height: {start_info['height']}")
    print(f"  - FPS: {start_info['fps']}")

    # 2. Capture real frame
    print("\n[2] Reading live frame from CameraService...")
    time.sleep(0.1)  # allow capture worker to grab frame
    frame = CameraService.get_frame()
    if frame is None:
        print("[FAILED] Failed to retrieve frame from running camera service.")
        CameraService.stop_camera()
        return False

    print(f"[SUCCESS] Real frame retrieved: shape={frame.shape}, dtype={frame.dtype}")
    assert frame.shape[0] > 0 and frame.shape[1] > 0, "Invalid frame dimensions"

    # 3. Test frame streaming generator
    print("\n[3] Testing MJPEG stream generator chunks...")
    gen = CameraService.generate_frames()
    chunk1 = next(gen)
    chunk2 = next(gen)
    print(f"  - Chunk 1 bytes length: {len(chunk1)}")
    print(f"  - Chunk 2 bytes length: {len(chunk2)}")
    assert b"--frame" in chunk1 and b"Content-Type: image/jpeg" in chunk1, "Invalid MJPEG chunk format"
    print("[SUCCESS] MJPEG frame generator producing valid stream chunks.")

    # 4. Stop camera and verify clean release
    print("\n[4] Stopping camera and verifying resource release...")
    stop_info = CameraService.stop_camera()
    print("Stop info:", stop_info)
    assert not stop_info["connected"], "Camera must report disconnected"
    assert CameraService.get_frame() is None, "Frame buffer must be cleared"
    print("[SUCCESS] Camera stopped and VideoCapture resource released.")

    # 5. Verify camera can be reopened cleanly (proves release was complete)
    print("\n[5] Reopening camera to verify hardware handle was completely freed...")
    time.sleep(0.3)
    reopen_info = CameraService.start_camera(settings.CAMERA_INDEX)
    print("Reopen info:", reopen_info)
    if not reopen_info["connected"]:
        print(f"[FAILED] Failed to reopen camera after release: {reopen_info.get('error')}")
        return False

    print("[SUCCESS] Camera successfully reopened without resource lock!")
    CameraService.stop_camera()
    print("[SUCCESS] Final camera stop completed.")

    print("\n" + "=" * 60)
    print(">>> ALL PHYSICAL WEBCAM VERIFICATION CHECKS PASSED! <<<")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
