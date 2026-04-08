"""
MedAuditEnv - Final Validation & Submission Guide
================================================

This script performs final validation before hackathon submission.
Run this to ensure everything is ready.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def check_file(filepath: str, description: str) -> bool:
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "[OK]" if exists else "[FAIL]"
    print(f"{status} {description}: {filepath}")
    return exists


def check_command(cmd: str, description: str) -> bool:
    """Check if a command succeeds"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        success = result.returncode == 0
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {description}")
        return success
    except Exception as e:
        print(f"[FAIL] {description}: {e}")
        return False


def validate_data():
    """Validate claims data"""
    print("\n" + "="*70)
    print("VALIDATING DATA")
    print("="*70)
    
    if not Path("data/claims.json").exists():
        print("[FAIL] data/claims.json not found")
        return False
    
    with open("data/claims.json", "r") as f:
        claims = json.load(f)
    
    print(f"[OK] Total claims: {len(claims)}")
    
    fraud_count = sum(1 for c in claims if c.get("is_fraud"))
    print(f"[OK] Fraud claims: {fraud_count} ({fraud_count/len(claims)*100:.1f}%)")
    
    # Check required fields
    required_fields = [
        "claim_id", "patient", "vitals", "diagnosis",
        "claimed_days", "bill_amount", "hospital", "is_fraud"
    ]
    
    sample = claims[0]
    for field in required_fields:
        if field not in sample:
            print(f"[FAIL] Missing field: {field}")
            return False
    
    print(f"[OK] All required fields present")
    return True


def validate_environment():
    """Validate environment code"""
    print("\n" + "="*70)
    print("VALIDATING ENVIRONMENT")
    print("="*70)
    
    try:
        from medaudit import MedAuditEnv, MedAuditAction
        
        # Test initialization
        env = MedAuditEnv(task="vital_check")
        print("[OK] Environment imports successfully")
        
        # Test reset
        obs = env.reset()
        print("[OK] reset() works")
        
        # Test step
        result = env.step(MedAuditAction.FLAG_ANOMALY)
        print("[OK] step() works")
        
        # Test state
        state = env.state()
        print("[OK] state() works")
        
        # Test score
        while not env.state().done:
            env.step(MedAuditAction.FLAG_ANOMALY)
        score = env.calculate_score()
        assert 0.0 <= score <= 1.0
        print(f"[OK] calculate_score() works (score: {score:.3f})")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"[FAIL] Environment validation failed: {e}")
        return False


def validate_inference():
    """Validate inference script structure"""
    print("\n" + "="*70)
    print("VALIDATING INFERENCE SCRIPT")
    print("="*70)
    
    if not Path("inference.py").exists():
        print("[FAIL] inference.py not found")
        return False
    
    with open("inference.py", "r") as f:
        content = f.read()
    
    # Check required components
    checks = [
        ("OpenAI client", "from openai import OpenAI"),
        ("Environment variables", "os.getenv"),
        ("[START] log", "def log_start"),
        ("[STEP] log", "def log_step"),
        ("[END] log", "def log_end"),
        ("HF_TOKEN", "HF_TOKEN"),
        ("API_BASE_URL", "API_BASE_URL"),
        ("MODEL_NAME", "MODEL_NAME"),
    ]
    
    all_present = True
    for name, pattern in checks:
        if pattern in content:
            print(f"[OK] {name} present")
        else:
            print(f"[FAIL] {name} missing")
            all_present = False
    
    return all_present


def validate_server():
    """Validate FastAPI server"""
    print("\n" + "="*70)
    print("VALIDATING SERVER")
    print("="*70)
    
    try:
        from fastapi.testclient import TestClient
        from server.app import app
        
        client = TestClient(app)
        
        # Test health check
        response = client.get("/")
        assert response.status_code == 200
        print("[OK] GET / returns 200")
        
        # Test reset
        response = client.post("/reset", json={"task": "vital_check"})
        assert response.status_code == 200
        print("[OK] POST /reset works")
        
        # Test step
        response = client.post("/step", json={"action": "flag_anomaly"})
        assert response.status_code == 200
        print("[OK] POST /step works")
        
        # Test state
        response = client.get("/state")
        assert response.status_code == 200
        print("[OK] GET /state works")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Server validation failed: {e}")
        return False


def validate_docker():
    """Validate Dockerfile"""
    print("\n" + "="*70)
    print("VALIDATING DOCKER")
    print("="*70)
    
    if not Path("Dockerfile").exists():
        print("[FAIL] Dockerfile not found")
        return False
    
    with open("Dockerfile", "r") as f:
        content = f.read()
    
    checks = [
        ("Base image", "FROM python:3.10"),
        ("Copy requirements", "COPY requirements.txt"),
        ("Install dependencies", "pip install"),
        ("Copy files", "COPY medaudit.py"),
        ("Generate data", "python data_generator.py"),
        ("Expose port", "EXPOSE 7860"),
        ("Run server", "uvicorn"),
    ]
    
    all_present = True
    for name, pattern in checks:
        if pattern in content:
            print(f"[OK] {name}")
        else:
            print(f"[FAIL] {name} missing")
            all_present = False
    
    return all_present


def validate_openenv_yaml():
    """Validate openenv.yaml"""
    print("\n" + "="*70)
    print("VALIDATING OPENENV.YAML")
    print("="*70)
    
    if not Path("openenv.yaml").exists():
        print("[FAIL] openenv.yaml not found")
        return False
    
    with open("openenv.yaml", "r") as f:
        content = f.read()
    
    checks = [
        ("name", "name: MedAuditEnv"),
        ("version", "version:"),
        ("tasks", "tasks:"),
        ("vital_check", "vital_check"),
        ("fraud_mix", "fraud_mix"),
        ("batch_audit", "batch_audit"),
    ]
    
    all_present = True
    for name, pattern in checks:
        if pattern in content:
            print(f"[OK] {name}")
        else:
            print(f"[FAIL] {name} missing")
            all_present = False
    
    return all_present


def main():
    """Run all validations"""
    print("\n" + "="*70)
    print("MedAuditEnv - Final Validation")
    print("="*70)
    
    # File checks
    print("\n" + "="*70)
    print("CHECKING FILES")
    print("="*70)
    
    files = [
        ("medaudit.py", "Core environment"),
        ("data_generator.py", "Data generator"),
        ("inference.py", "Inference script"),
        ("server/app.py", "FastAPI server"),
        ("Dockerfile", "Container config"),
        ("requirements.txt", "Dependencies"),
        ("openenv.yaml", "Metadata"),
        ("README.md", "Documentation"),
        ("test_env.py", "Test suite"),
        ("data/claims.json", "Synthetic data"),
    ]
    
    files_ok = all(check_file(f, desc) for f, desc in files)
    
    # Run validations
    validations = [
        ("Data", validate_data),
        ("Environment", validate_environment),
        ("Inference", validate_inference),
        ("Server", validate_server),
        ("Docker", validate_docker),
        ("OpenEnv YAML", validate_openenv_yaml),
    ]
    
    results = {}
    for name, func in validations:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n[FAIL] {name} validation crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values()) and files_ok
    
    for name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*70)
    if all_passed:
        print("[OK] ALL VALIDATIONS PASSED - READY TO SUBMIT!")
        print("="*70)
        print("\nNext steps:")
        print("1. Build Docker image: docker build -t medauditenv .")
        print("2. Test Docker: docker run -p 8000:8000 medauditenv")
        print("3. Deploy to HuggingFace Spaces")
        print("4. Run inference: python inference.py")
        print("5. Submit repository URL")
    else:
        print("[FAIL] SOME VALIDATIONS FAILED - REVIEW ABOVE")
        print("="*70)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
