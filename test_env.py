#!/usr/bin/env python
"""
MedAuditEnv Test Suite
Validates environment compliance and functionality
"""

import sys
from medaudit import MedAuditEnv, MedAuditAction

def test_basic_functionality():
    """Test basic environment operations"""
    print("\n" + "="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)
    
    env = MedAuditEnv(task="vital_check")
    obs = env.reset()
    
    print(f"[OK] Environment initialized (task: vital_check)")
    print(f"[OK] Reset successful - First claim: {obs.claim_id}")
    
    # Take a step
    result = env.step(MedAuditAction.FLAG_ANOMALY)
    print(f"[OK] Step successful - Reward: {result.reward:.2f}")
    assert "confidence_score" in result.info, "Missing confidence_score in info"
    assert "explainable_ai" in result.info, "Missing explainable_ai in info"
    assert "episode_history" in result.info, "Missing episode_history in info"
    
    # Check state
    state = env.state()
    print(f"[OK] State accessible - Claims: {state.current_claim}/{state.total_claims}")
    
    env.close()
    print("[OK] Environment closed")
    
    return True


def test_all_tasks():
    """Test all three tasks"""
    print("\n" + "="*70)
    print("TEST 2: All Tasks")
    print("="*70)
    
    for task_name in ["vital_check", "fraud_mix", "batch_audit"]:
        env = MedAuditEnv(task=task_name)
        obs = env.reset()
        
        num_claims = env.task_config[task_name]["num_claims"]
        print(f"[OK] {task_name:15s} - {num_claims:3d} claims loaded")
        
        env.close()
    
    return True


def test_reward_function():
    """Test reward function edge cases"""
    print("\n" + "="*70)
    print("TEST 3: Reward Function")
    print("="*70)
    
    env = MedAuditEnv(task="vital_check")
    env.reset()
    
    # Test different actions
    actions = [
        MedAuditAction.FLAG_ANOMALY,
        MedAuditAction.APPROVE_CLAIM,
        MedAuditAction.REJECT_CLAIM,
        MedAuditAction.REQUEST_CLARIFICATION
    ]
    
    for action in actions:
        result = env.step(action)
        assert -0.5 <= result.reward <= 1.0, f"Reward out of bounds: {result.reward}"
        assert 0.0 <= float(result.info.get("confidence_score", 0.0)) <= 1.0, "Confidence out of bounds"
        hist = result.info.get("episode_history") or []
        assert len(hist) <= 3, "Episode history should contain last 3 decisions max"
        print(f"[OK] {action.value:25s} -> Reward: {result.reward:+.2f}")
    
    env.close()
    return True


def test_score_calculation():
    """Test final score calculation"""
    print("\n" + "="*70)
    print("TEST 4: Score Calculation")
    print("="*70)
    
    for task_name in ["vital_check", "fraud_mix", "batch_audit"]:
        env = MedAuditEnv(task=task_name)
        env.reset()
        
        # Complete all steps
        while not env.state().done:
            env.step(MedAuditAction.FLAG_ANOMALY)
        
        score = env.calculate_score()
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
        print(f"[OK] {task_name:15s} - Score: {score:.3f} (valid range)")
        
        env.close()
    
    return True


def test_observation_space():
    """Test observation space structure"""
    print("\n" + "="*70)
    print("TEST 5: Observation Space")
    print("="*70)
    
    env = MedAuditEnv(task="vital_check")
    obs = env.reset()
    
    # Check required fields
    required_fields = [
        "claim_id", "patient", "vitals", "diagnosis",
        "medications", "claimed_days", "bill_amount", "hospital"
    ]
    
    for field in required_fields:
        assert hasattr(obs, field), f"Missing field: {field}"
        print(f"[OK] Field present: {field}")
    
    # Check nested structures
    assert hasattr(obs.patient, "age"), "Missing patient.age"
    assert hasattr(obs.vitals, "bp"), "Missing vitals.bp"
    print(f"[OK] Nested structures valid")
    
    env.close()
    return True


def test_explainability_payload():
    """Test explainability payload structure"""
    print("\n" + "="*70)
    print("TEST 7: Explainability Payload")
    print("="*70)
    
    env = MedAuditEnv(task="fraud_mix")
    env.reset()
    
    result = env.step(MedAuditAction.REQUEST_CLARIFICATION)
    xai = result.info.get("explainable_ai") or {}
    assert "reasoning_trace" in xai, "Missing reasoning_trace"
    assert "fraud_indicators" in xai, "Missing fraud_indicators"
    assert "confidence_breakdown" in xai, "Missing confidence_breakdown"
    assert isinstance(xai["reasoning_trace"], list), "reasoning_trace must be a list"
    
    env.close()
    print("[OK] Explainability payload valid")
    return True


def test_ocr_noise_handling():
    """Test OCR noise parsing"""
    print("\n" + "="*70)
    print("TEST 6: OCR Noise Handling")
    print("="*70)
    
    env = MedAuditEnv(task="fraud_mix")
    
    # Test parsing function
    test_cases = [
        ("12O/8O", 120),  # O->0
        ("1B5/9O", 185),  # B->8
        ("I2O/BO", 120),  # I->1
        ("245/160", 245)  # No noise
    ]
    
    for vital_str, expected in test_cases:
        result = env._parse_vital(vital_str)
        assert result == expected, f"Parse error: {vital_str} -> {result}, expected {expected}"
        print(f"[OK] Parse '{vital_str}' -> {result}")
    
    env.close()
    return True


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print("MedAuditEnv Test Suite")
    print("="*70)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("All Tasks", test_all_tasks),
        ("Reward Function", test_reward_function),
        ("Score Calculation", test_score_calculation),
        ("Observation Space", test_observation_space),
        ("OCR Noise Handling", test_ocr_noise_handling),
        ("Explainability Payload", test_explainability_payload),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"[FAIL] {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name} FAILED: {e}")
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)