"""
Script to verify that running pytest does NOT touch database/fire_system.db.
"""

import os
import subprocess
import hashlib

workspace_root = os.path.dirname(os.path.abspath(__file__))
dev_db_path = os.path.join(workspace_root, "database", "fire_system.db")
backend_dir = os.path.join(workspace_root, "backend")

# Ensure dev db exists and record its state
os.makedirs(os.path.dirname(dev_db_path), exist_ok=True)
if not os.path.exists(dev_db_path):
    with open(dev_db_path, "wb") as f:
        f.write(b"DEV_DATABASE_SENTINEL_CONTENT_FOR_ISOLATION_TEST\n")

with open(dev_db_path, "rb") as f:
    before_hash = hashlib.sha256(f.read()).hexdigest()
before_mtime = os.path.getmtime(dev_db_path)
before_size = os.path.getsize(dev_db_path)

print(f"DEV DB Before pytest: size={before_size}, mtime={before_mtime}, hash={before_hash[:16]}...")

# Run pytest inside backend
pytest_exe = os.path.join(backend_dir, "venv", "Scripts", "pytest.exe")
result = subprocess.run([pytest_exe, "-v"], cwd=backend_dir, capture_output=True, text=True)
print("Pytest exit code:", result.returncode)
assert result.returncode == 0, f"Pytest failed:\n{result.stdout}\n{result.stderr}"

# Check dev db state after pytest
with open(dev_db_path, "rb") as f:
    after_hash = hashlib.sha256(f.read()).hexdigest()
after_mtime = os.path.getmtime(dev_db_path)
after_size = os.path.getsize(dev_db_path)

print(f"DEV DB After pytest:  size={after_size}, mtime={after_mtime}, hash={after_hash[:16]}...")

assert before_hash == after_hash, "FAILURE: Dev database content was altered by pytest!"
assert before_size == after_size, "FAILURE: Dev database size changed during pytest!"
assert before_mtime == after_mtime, "FAILURE: Dev database modified timestamp changed during pytest!"

print("\n>>> CONFIRMED: database/fire_system.db was NOT touched or modified during pytest! <<<")
