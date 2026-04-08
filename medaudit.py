"""
MedAuditEnv: Rural Medical Record Auditor Environment
OpenEnv-compliant environment for training AI agents to detect medical claim fraud
"""

import json
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ============================================================================
# Action Space
# ============================================================================

class MedAuditAction(str, Enum):
    """Available actions for the auditor agent"""
    FLAG_ANOMALY = "flag_anomaly"           # Mark claim as fraudulent
    APPROVE_CLAIM = "approve_claim"         # Approve claim as legitimate
    REJECT_CLAIM = "reject_claim"           # Reject claim (strong fraud signal)
    REQUEST_CLARIFICATION = "request_clarification"  # Need more info (costs time)


# ============================================================================
# Observation Space
# ============================================================================

class PatientInfo(BaseModel):
    """Patient demographic information"""
    name: str
    age: str
    village: str


class VitalSigns(BaseModel):
    """Patient vital signs (may contain OCR noise)"""
    bp: str = Field(..., description="Blood pressure reading (e.g., '120/80')")
    pulse: str = Field(..., description="Pulse rate")


class MedAuditObservation(BaseModel):
    """Single claim observation for the auditor"""
    claim_id: str
    patient: PatientInfo
    vitals: VitalSigns
    diagnosis: str
    medications: List[str] = Field(default_factory=list, description="Medications claimed/provided")
    claimed_days: int = Field(..., ge=1, description="Days of treatment claimed")
    bill_amount: float = Field(..., gt=0, description="Claimed bill amount in INR")
    hospital: str
    hospital_distance_km: int = Field(..., description="Distance from patient village to hospital")
    ocr_noise: bool = Field(default=False, description="Whether OCR noise is present")
    episode_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Last 3 decisions context (most recent last)"
    )
    
    # Metadata (not visible to agent in real scenario, but useful for debugging)
    step_number: int = Field(default=1, description="Current step in episode")
    total_claims: int = Field(default=10, description="Total claims in this task")


# ============================================================================
# State
# ============================================================================

class MedAuditState(BaseModel):
    """Current environment state"""
    current_claim: int = Field(default=0, description="Index of current claim")
    total_claims: int = Field(default=10, description="Total claims in episode")
    flagged_count: int = Field(default=0, description="Number of claims flagged as fraud")
    approved_count: int = Field(default=0, description="Number of claims approved")
    rejected_count: int = Field(default=0, description="Number of claims rejected")
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Current accuracy")
    episode_reward: float = Field(default=0.0, description="Cumulative episode reward")
    done: bool = Field(default=False, description="Whether episode is complete")


# ============================================================================
# Step Result
# ============================================================================

class StepResult(BaseModel):
    """Result of a step in the environment"""
    observation: MedAuditObservation
    reward: float
    done: bool
    info: Dict = Field(default_factory=dict)


# ============================================================================
# Environment
# ============================================================================

class MedAuditEnv:
    """
    OpenEnv-compliant environment for medical claim fraud detection
    
    Tasks:
    - vital_check (easy): 10 claims, detect impossible vitals
    - fraud_mix (medium): 20 claims, mixed fraud patterns
    - batch_audit (hard): 50 claims, full fraud detection
    """
    
    def __init__(self, task: str = "vital_check", data_path: str = "data/claims.json"):
        """
        Initialize environment
        
        Args:
            task: One of ['vital_check', 'fraud_mix', 'batch_audit']
            data_path: Path to claims JSON file
        """
        self.task = task
        self.data_path = data_path
        
        # Task configuration
        self.task_config = {
            "vital_check": {"num_claims": 10, "difficulty": "easy"},
            "fraud_mix": {"num_claims": 20, "difficulty": "medium"},
            "batch_audit": {"num_claims": 50, "difficulty": "hard"}
        }
        
        if task not in self.task_config:
            raise ValueError(f"Invalid task: {task}. Must be one of {list(self.task_config.keys())}")
        
        # Load data
        self._load_data()

        # Deterministic episode sampling for reproducible grading
        self._rng = random.Random(42)
        
        # State
        self.state_data = MedAuditState()
        self.current_claims: List[Dict] = []
        self.current_index = 0
        self.step_rewards: List[float] = []
        self.actions_taken: List[str] = []
        self.decision_history: List[Dict[str, Any]] = []
        
    def _load_data(self):
        """Load claims dataset"""
        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(
                f"Claims data not found at {self.data_path}. "
                f"Run data_generator.py first to generate synthetic data."
            )
        
        with open(data_file, "r") as f:
            self.all_claims = json.load(f)
    
    def _parse_vital(self, vital_str: str) -> int:
        """Parse vital sign string, handling OCR noise"""
        # Clean OCR noise: O->0, B->8, I->1
        cleaned = vital_str.replace('O', '0').replace('B', '8').replace('I', '1')
        
        # Extract numeric value
        if '/' in cleaned:
            # Blood pressure - take systolic
            return int(cleaned.split('/')[0])
        else:
            # Pulse
            return int(cleaned)

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Best-effort int conversion for noisy fields."""
        try:
            if value is None:
                return default
            if isinstance(value, int):
                return value
            s = str(value).strip()
            if not s:
                return default
            return int(s)
        except Exception:
            return default

    def _assess_claim(self, claim: Dict) -> Dict[str, Any]:
        """
        Produce explainable fraud assessment for a claim.
        This is environment-side explainability (not model chain-of-thought).
        """
        indicators: List[Dict[str, Any]] = []

        # Vitals
        bp = (claim.get("vitals") or {}).get("bp", "")
        pulse = (claim.get("vitals") or {}).get("pulse", "")
        systolic = None
        pulse_val = None
        try:
            systolic = self._parse_vital(str(bp))
        except Exception:
            systolic = None
        try:
            pulse_val = self._parse_vital(str(pulse))
        except Exception:
            pulse_val = None

        if systolic is not None and systolic > 220:
            indicators.append(
                {"code": "impossible_bp", "severity": 0.95, "evidence": {"bp": bp, "systolic": systolic}}
            )
        if pulse_val is not None and (pulse_val < 30 or pulse_val > 180):
            indicators.append(
                {"code": "impossible_pulse", "severity": 0.95, "evidence": {"pulse": pulse, "parsed": pulse_val}}
            )

        # Age mismatch / ghost patient
        age_str = ((claim.get("patient") or {}).get("age", "") or "").strip()
        age = self._safe_int(age_str, default=-1)
        diagnosis = (claim.get("diagnosis") or "").strip()

        if age >= 0 and age < 60 and "Senior" in diagnosis:
            indicators.append(
                {"code": "age_mismatch", "severity": 0.8, "evidence": {"age": age_str, "diagnosis": diagnosis}}
            )

        if not ((claim.get("patient") or {}).get("name") or "").strip() or age_str in {"0", "-1", "999"}:
            indicators.append(
                {
                    "code": "ghost_patient",
                    "severity": 0.85,
                    "evidence": {"name": (claim.get("patient") or {}).get("name"), "age": age_str},
                }
            )

        # Distance
        dist = self._safe_int(claim.get("hospital_distance_km"), default=0)
        if dist > 100:
            indicators.append(
                {"code": "excessive_distance", "severity": 0.65, "evidence": {"hospital_distance_km": dist}}
            )

        # Bill outlier
        bill = float(claim.get("bill_amount") or 0.0)
        if bill > 75000 and diagnosis in {"Fever", "Malaria"}:
            indicators.append(
                {"code": "bill_outlier", "severity": 0.7, "evidence": {"bill_amount": bill, "diagnosis": diagnosis}}
            )

        # Duration inflation
        days = self._safe_int(claim.get("claimed_days"), default=1)
        if days > 21:
            indicators.append(
                {"code": "excessive_duration", "severity": 0.7, "evidence": {"claimed_days": days}}
            )

        # New patterns: duplicates / sequences / out-of-network / medication mismatch
        if claim.get("duplicate_of"):
            indicators.append(
                {"code": "duplicate_claim", "severity": 0.75, "evidence": {"duplicate_of": claim.get("duplicate_of")}}
            )

        if claim.get("sequence_id"):
            indicators.append(
                {
                    "code": "suspicious_sequence",
                    "severity": 0.6,
                    "evidence": {
                        "sequence_id": claim.get("sequence_id"),
                        "sequence_visit": claim.get("sequence_visit"),
                        "bill_amount": bill,
                        "claimed_days": days,
                    },
                }
            )

        if claim.get("hospital") in {"Private_Clinic", "Private_Hospital"}:
            indicators.append(
                {"code": "out_of_network", "severity": 0.7, "evidence": {"hospital": claim.get("hospital")}}
            )

        meds = claim.get("medications") or []
        if isinstance(meds, str):
            meds = [m.strip() for m in meds.split(",") if m.strip()]

        # Minimal, self-contained mismatch heuristic (avoid needing generator tables here)
        def _expected_meds(diag: str) -> set:
            mapping = {
                "Fever": {"Paracetamol", "ORS"},
                "Respiratory Infection": {"Amoxicillin", "Azithromycin"},
                "Tuberculosis": {"Isoniazid", "Rifampicin"},
                "Diabetes": {"Metformin", "Insulin"},
                "Hypertension": {"Amlodipine", "Losartan"},
                "Malaria": {"Artemisinin", "Chloroquine"},
                "Maternity Care": {"Iron_Folic_Acid", "Oxytocin"},
            }
            return mapping.get(diag, set())

        expected = _expected_meds(diagnosis)
        actual = set(meds) if isinstance(meds, list) else set()
        if expected and actual and not (expected & actual):
            indicators.append(
                {
                    "code": "medication_mismatch",
                    "severity": 0.7,
                    "evidence": {"diagnosis": diagnosis, "medications": meds, "expected_any_of": sorted(expected)},
                }
            )

        # Confidence: combine severities (noisy-OR), then map to level
        p_no_fraud = 1.0
        breakdown: List[Dict[str, Any]] = []
        for ind in indicators:
            sev = float(ind.get("severity", 0.0))
            p_no_fraud *= (1.0 - min(max(sev, 0.0), 1.0) * 0.9)
            breakdown.append(
                {"indicator": ind["code"], "weight": round(min(max(sev, 0.0), 1.0), 3), "evidence": ind.get("evidence", {})}
            )
        fraud_confidence = 1.0 - p_no_fraud
        fraud_confidence = min(max(fraud_confidence, 0.0), 1.0)

        if fraud_confidence >= 0.85:
            level = "very_high"
        elif fraud_confidence >= 0.7:
            level = "high"
        elif fraud_confidence >= 0.5:
            level = "medium"
        elif fraud_confidence >= 0.3:
            level = "low"
        else:
            level = "very_low"

        reasoning_trace = [
            "Extracted structured fields (age, vitals, distance, bill, duration, hospital, medications).",
            "Matched fields against known fraud patterns; each match adds an indicator with severity.",
            "Aggregated indicator severities into a single fraud-likelihood confidence score.",
        ]

        return {
            "fraud_confidence": round(fraud_confidence, 3),
            "confidence_level": level,
            "indicators": indicators,
            "confidence_breakdown": breakdown,
            "reasoning_trace": reasoning_trace,
        }
    
    def _check_fraud(self, claim: Dict) -> bool:
        """Check if claim has fraud indicators"""
        return claim.get("is_fraud", False)
    
    def _calculate_reward(self, action: MedAuditAction, claim: Dict) -> float:
        """
        Calculate reward for action on claim
        
        Reward structure:
        - Correct fraud detection: +0.8
        - Correct approval: +0.6
        - False positive/negative: -0.4
        - Progress bonus: +0.1
        - Request clarification: +0.2 (if fraud), -0.1 (if not)
        """
        is_fraud = self._check_fraud(claim)
        
        if action == MedAuditAction.FLAG_ANOMALY or action == MedAuditAction.REJECT_CLAIM:
            # Agent thinks it's fraud
            if is_fraud:
                return 0.8  # Correct detection
            else:
                return -0.4  # False positive
        
        elif action == MedAuditAction.APPROVE_CLAIM:
            # Agent thinks it's legitimate
            if not is_fraud:
                return 0.6  # Correct approval
            else:
                return -0.4  # False negative (missed fraud)
        
        elif action == MedAuditAction.REQUEST_CLARIFICATION:
            # Agent is uncertain
            if is_fraud:
                return 0.2  # Good to be cautious
            else:
                return -0.1  # Wasted time
        
        return 0.0
    
    def reset(self) -> MedAuditObservation:
        """
        Reset environment to start new episode
        
        Returns:
            Initial observation
        """
        # Get task configuration
        num_claims = self.task_config[self.task]["num_claims"]
        
        # Sample claims for this episode
        # For vital_check, prioritize claims with vital-related fraud
        if self.task == "vital_check":
            # Get claims with impossible vitals
            vital_fraud = [c for c in self.all_claims 
                          if c.get("is_fraud") and 
                          any(ind in ["impossible_bp", "impossible_pulse"] 
                              for ind in c.get("_fraud_indicators", []))]
            # Mix with some legitimate claims
            legitimate = [c for c in self.all_claims if not c.get("is_fraud")]
            
            # Take 8 vital fraud + 2 legitimate
            self.current_claims = self._rng.sample(vital_fraud, min(8, len(vital_fraud)))
            self.current_claims += self._rng.sample(legitimate, min(2, len(legitimate)))
            self._rng.shuffle(self.current_claims)
            
        else:
            # For other tasks, random sample
            self.current_claims = self._rng.sample(self.all_claims, min(num_claims, len(self.all_claims)))
        
        # Ensure we have exact number
        self.current_claims = self.current_claims[:num_claims]
        
        # Reset state
        self.current_index = 0
        self.step_rewards = []
        self.actions_taken = []
        self.decision_history = []
        self.state_data = MedAuditState(
            current_claim=0,
            total_claims=len(self.current_claims),
            done=False
        )
        
        # Return first observation
        return self._get_observation()
    
    def _get_observation(self) -> MedAuditObservation:
        """Get current observation"""
        if self.current_index >= len(self.current_claims):
            # Episode done, return dummy observation
            claim = self.current_claims[-1] if self.current_claims else {}
        else:
            claim = self.current_claims[self.current_index]
        
        return MedAuditObservation(
            claim_id=claim.get("claim_id", "C000"),
            patient=PatientInfo(
                name=claim.get("patient", {}).get("name", "Unknown"),
                age=claim.get("patient", {}).get("age", "0"),
                village=claim.get("patient", {}).get("village", "Unknown")
            ),
            vitals=VitalSigns(
                bp=claim.get("vitals", {}).get("bp", "120/80"),
                pulse=claim.get("vitals", {}).get("pulse", "70")
            ),
            diagnosis=claim.get("diagnosis", "Unknown"),
            medications=list(claim.get("medications") or []),
            claimed_days=claim.get("claimed_days", 1),
            bill_amount=claim.get("bill_amount", 0.0),
            hospital=claim.get("hospital", "Unknown"),
            hospital_distance_km=claim.get("hospital_distance_km", 0),
            ocr_noise=claim.get("ocr_noise", False),
            episode_history=[
                {
                    "step": d.get("step"),
                    "claim_id": d.get("claim_id"),
                    "patient_name": ((d.get("snapshot") or {}).get("patient") or {}).get("name"),
                    "patient_age": ((d.get("snapshot") or {}).get("patient") or {}).get("age"),
                    "diagnosis": (d.get("snapshot") or {}).get("diagnosis"),
                    "medications": (d.get("snapshot") or {}).get("medications"),
                    "bill_amount": (d.get("snapshot") or {}).get("bill_amount"),
                    "claimed_days": (d.get("snapshot") or {}).get("claimed_days"),
                    "hospital": (d.get("snapshot") or {}).get("hospital"),
                    "hospital_distance_km": (d.get("snapshot") or {}).get("hospital_distance_km"),
                    "action": d.get("action"),
                    "reward": d.get("reward"),
                    "assessment": d.get("assessment"),
                }
                for d in (self.decision_history[-3:] if self.decision_history else [])
            ],
            step_number=self.current_index + 1,
            total_claims=len(self.current_claims)
        )
    
    def step(self, action: MedAuditAction) -> StepResult:
        """
        Execute action and return result
        
        Args:
            action: Action to take (MedAuditAction enum)
            
        Returns:
            StepResult with observation, reward, done, info
        """
        if self.state_data.done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")
        
        # Convert string to enum if needed
        if isinstance(action, str):
            try:
                action = MedAuditAction(action)
            except ValueError:
                # Invalid action, default to approve
                action = MedAuditAction.APPROVE_CLAIM
        
        # Get current claim
        claim = self.current_claims[self.current_index]
        assessment = self._assess_claim(claim)
        
        # Calculate reward
        reward = self._calculate_reward(action, claim)
        self.step_rewards.append(reward)
        self.actions_taken.append(action.value)
        
        # Update state
        if action in [MedAuditAction.FLAG_ANOMALY, MedAuditAction.REJECT_CLAIM]:
            self.state_data.flagged_count += 1
        elif action == MedAuditAction.APPROVE_CLAIM:
            self.state_data.approved_count += 1
        
        # Move to next claim
        self.current_index += 1
        self.state_data.current_claim = self.current_index
        self.state_data.episode_reward += reward
        
        # Check if done
        done = self.current_index >= len(self.current_claims)
        self.state_data.done = done
        
        # Calculate accuracy
        if done:
            correct = sum(1 for r in self.step_rewards if r > 0)
            self.state_data.accuracy = correct / len(self.step_rewards) if self.step_rewards else 0.0
        
        # Get next observation (or last one if done)
        observation = self._get_observation()
        
        # Build info
        decision = {
            "step": len(self.step_rewards),
            "claim_id": claim.get("claim_id"),
            "action": action.value,
            "reward": reward,
            "is_fraud": claim.get("is_fraud", False),
            "fraud_indicators_gt": claim.get("_fraud_indicators", []),
            "snapshot": {
                "patient": claim.get("patient", {}),
                "vitals": claim.get("vitals", {}),
                "diagnosis": claim.get("diagnosis"),
                "medications": list(claim.get("medications") or []),
                "claimed_days": claim.get("claimed_days"),
                "bill_amount": claim.get("bill_amount"),
                "hospital": claim.get("hospital"),
                "hospital_distance_km": claim.get("hospital_distance_km"),
                "ocr_noise": claim.get("ocr_noise", False),
            },
            "assessment": {
                "fraud_confidence": assessment["fraud_confidence"],
                "confidence_level": assessment["confidence_level"],
                "indicator_codes": [i.get("code") for i in assessment.get("indicators", [])],
            },
        }
        self.decision_history.append(decision)
        last_3 = self.decision_history[-3:]

        info = {
            "action_taken": action.value,
            "is_fraud": claim.get("is_fraud", False),
            "fraud_indicators": claim.get("_fraud_indicators", []),
            "fraud_patterns_detected": [i.get("code") for i in assessment.get("indicators", [])],
            "confidence_score": assessment["fraud_confidence"],
            "confidence_level": assessment["confidence_level"],
            "explainable_ai": {
                "reasoning_trace": assessment["reasoning_trace"],
                "fraud_indicators": assessment["indicators"],
                "confidence_breakdown": assessment["confidence_breakdown"],
            },
            "episode_history": last_3,
        }
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )
    
    def state(self) -> MedAuditState:
        """Return current state"""
        return self.state_data
    
    def close(self):
        """Cleanup (no resources to clean)"""
        pass
    
    def calculate_score(self) -> float:
        """
        Calculate final normalized score (0.0 to 1.0)
        
        Task-specific grading:
        - vital_check: Simple accuracy
        - fraud_mix: Precision × Recall
        - batch_audit: 0.4×accuracy + 0.3×speed + 0.3×false_negative_rate
        """
        if not self.step_rewards:
            return 0.0
        
        # Get ground truth
        fraud_claims = [c for c in self.current_claims if c.get("is_fraud")]
        legit_claims = [c for c in self.current_claims if not c.get("is_fraud")]
        
        # Get predictions (flag_anomaly or reject_claim = fraud prediction)
        fraud_actions = {MedAuditAction.FLAG_ANOMALY.value, MedAuditAction.REJECT_CLAIM.value}
        predicted_fraud_indices = [i for i, a in enumerate(self.actions_taken) if a in fraud_actions]
        
        # True positives, false positives, false negatives
        tp = sum(1 for i in predicted_fraud_indices if self.current_claims[i].get("is_fraud"))
        fp = sum(1 for i in predicted_fraud_indices if not self.current_claims[i].get("is_fraud"))
        fn = len(fraud_claims) - tp
        
        if self.task == "vital_check":
            # Simple accuracy
            correct = sum(1 for r in self.step_rewards if r > 0)
            score = correct / len(self.step_rewards)
        
        elif self.task == "fraud_mix":
            # Precision × Recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / len(fraud_claims) if len(fraud_claims) > 0 else 0.0
            score = precision * recall
        
        else:  # batch_audit
            # Composite score
            accuracy = sum(1 for r in self.step_rewards if r > 0) / len(self.step_rewards)
            speed_score = 1.0 - (self.state_data.approved_count + self.state_data.flagged_count) / len(self.current_claims)
            fnr = fn / len(fraud_claims) if len(fraud_claims) > 0 else 0.0
            fnr_score = 1.0 - fnr
            
            score = 0.4 * accuracy + 0.3 * max(speed_score, 0) + 0.3 * fnr_score

        # Clamp to (0, 1) for strict validator requirements.
        # Some validators reject exactly 0.0 or 1.0.
        # Use 1e-3 so that formatting to 3 decimals never prints 1.000 or 0.000.
        eps = 1e-3
        score = min(max(float(score), 0.0), 1.0)
        if score <= 0.0:
            return eps
        if score >= 1.0:
            return 1.0 - eps
        return score