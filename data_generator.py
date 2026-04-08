"""
MedAuditEnv Data Generator
Generates 100 synthetic rural medical claims with realistic fraud patterns
"""

import json
import random
from pathlib import Path
from copy import deepcopy

# Set seed for reproducibility
random.seed(42)

# Indian rural healthcare context
VILLAGES = ["Rampur", "Sirohi", "Nawada", "Barauli", "Kishanganj", "Darbhanga"]
HOSPITALS = {
    "CHC_Rampur": 0,      # km from village center
    "CHC_Sirohi": 45,     
    "CHC_Nawada": 25,     
    "District_Hospital": 120,  # Too far for routine care
    "PHC_Local": 5,
    "CHC_Distant": 150    # Suspicious distance
}

DIAGNOSES = [
    "Fever",
    "Tuberculosis",
    "Diabetes",
    "Hypertension",
    "Senior Citizen Pension Scheme",
    "Malaria",
    "Respiratory Infection",
    "Maternity Care"
]

NETWORK_HOSPITALS = {
    "CHC_Rampur",
    "CHC_Sirohi",
    "CHC_Nawada",
    "District_Hospital",
    "PHC_Local",
    "CHC_Distant",
}

NON_NETWORK_HOSPITALS = {
    "Private_Clinic",
    "Private_Hospital",
}

DIAGNOSIS_TO_MEDS = {
    "Fever": ["Paracetamol", "ORS"],
    "Respiratory Infection": ["Amoxicillin", "Azithromycin"],
    "Tuberculosis": ["Isoniazid", "Rifampicin"],
    "Diabetes": ["Metformin", "Insulin"],
    "Hypertension": ["Amlodipine", "Losartan"],
    "Malaria": ["Artemisinin", "Chloroquine"],
    "Maternity Care": ["Iron_Folic_Acid", "Oxytocin"],
    "Senior Citizen Pension Scheme": [],
}

MISMATCH_MEDS = [
    "Artemisinin",
    "Chloroquine",
    "Insulin",
    "Metformin",
    "Rifampicin",
    "Isoniazid",
    "Amlodipine",
    "Losartan",
]

def generate_claim(claim_id: int, is_fraud: bool = False) -> dict:
    """
    Generate a single medical claim with optional fraud patterns
    
    Fraud patterns:
    1. Impossible vitals (BP>220, pulse>180 or <30)
    2. Age mismatch (young person claiming senior benefits)
    3. Hospital too far (>100km for routine care)
    4. Bill outliers (>₹75,000 for simple diagnoses)
    5. Excessive duration (>21 days outpatient)
    6. Ghost patients (missing/invalid identity signals)
    7. Duplicate claims (same patient+diagnosis+billing repeated)
    8. Suspicious sequences (repeat visits quickly with escalation)
    9. Out-of-network hospital usage
    10. Medication mismatch (meds inconsistent with diagnosis)
    """
    
    # Base values
    age = random.randint(18, 90)
    bp_systolic = random.randint(90, 200)
    bp_diastolic = random.randint(50, 120)
    pulse = random.randint(40, 120)
    diagnosis = random.choice(DIAGNOSES)
    bill_amount = round(random.uniform(5000, 50000), 2)
    claimed_days = random.randint(1, 15)
    hospital = random.choice(list(HOSPITALS.keys()))
    medications = list(DIAGNOSIS_TO_MEDS.get(diagnosis, []))
    
    # Apply fraud patterns
    if is_fraud:
        fraud_type = random.choice([
            "impossible_vitals",
            "age_mismatch", 
            "distance",
            "bill_outlier",
            "duration",
            "ghost_patient",
            "out_of_network",
            "med_mismatch",
        ])
        
        if fraud_type == "impossible_vitals":
            if random.random() < 0.5:
                bp_systolic = random.randint(230, 280)  # Impossible BP
            else:
                pulse = random.choice([random.randint(10, 25), random.randint(185, 220)])
                
        elif fraud_type == "age_mismatch":
            age = random.randint(25, 55)  # Too young
            diagnosis = "Senior Citizen Pension Scheme"
            
        elif fraud_type == "distance":
            hospital = random.choice(["District_Hospital", "CHC_Distant"])
            
        elif fraud_type == "bill_outlier":
            bill_amount = round(random.uniform(80000, 150000), 2)
            diagnosis = random.choice(["Fever", "Malaria"])  # Simple diagnosis
            
        elif fraud_type == "duration":
            claimed_days = random.randint(25, 45)

        elif fraud_type == "ghost_patient":
            # Missing / invalid identity signals
            if random.random() < 0.5:
                age = random.choice([0, -1, 999])
            if random.random() < 0.6:
                diagnosis = random.choice(["Fever", "Malaria", "Diabetes", "Hypertension"])
            # medications still set below based on final diagnosis

        elif fraud_type == "out_of_network":
            hospital = random.choice(sorted(NON_NETWORK_HOSPITALS))

        elif fraud_type == "med_mismatch":
            # Force meds that don't match the diagnosis
            diagnosis = random.choice([d for d in DIAGNOSES if d != "Senior Citizen Pension Scheme"])
            medications = [random.choice([m for m in MISMATCH_MEDS if m not in DIAGNOSIS_TO_MEDS.get(diagnosis, [])])]
    
    # Add OCR noise to some claims (15% chance)
    ocr_noise = random.random() < 0.15
    bp_str = f"{bp_systolic}/{bp_diastolic}"
    pulse_str = str(pulse)
    
    if ocr_noise:
        # Simulate OCR errors (0->O, 8->B, 1->I)
        bp_str = bp_str.replace('0', 'O').replace('8', 'B')
        pulse_str = pulse_str.replace('0', 'O').replace('1', 'I')
    
    claim = {
        "claim_id": f"C{claim_id:03d}",
        "patient": {
            "name": f"Patient_{claim_id:03d}",
            "age": str(age),
            "village": random.choice(VILLAGES)
        },
        "vitals": {
            "bp": bp_str,
            "pulse": pulse_str
        },
        "diagnosis": diagnosis,
        "medications": medications if medications else [],
        "claimed_days": claimed_days,
        "bill_amount": bill_amount,
        "hospital": hospital,
        "hospital_distance_km": HOSPITALS.get(hospital, random.randint(10, 80)),
        "ocr_noise": ocr_noise,
        "is_fraud": is_fraud,
        # Ground truth for grading
        "_fraud_indicators": []
    }
    
    # Add fraud indicators for grading
    if is_fraud:
        if bp_systolic > 220:
            claim["_fraud_indicators"].append("impossible_bp")
        if pulse < 30 or pulse > 180:
            claim["_fraud_indicators"].append("impossible_pulse")
        if age < 60 and "Senior" in diagnosis:
            claim["_fraud_indicators"].append("age_mismatch")
        if claim["hospital_distance_km"] > 100:
            claim["_fraud_indicators"].append("excessive_distance")
        if bill_amount > 75000:
            claim["_fraud_indicators"].append("bill_outlier")
        if claimed_days > 21:
            claim["_fraud_indicators"].append("excessive_duration")
        if claim["hospital"] in NON_NETWORK_HOSPITALS:
            claim["_fraud_indicators"].append("out_of_network")
        if not claim["patient"]["name"] or claim["patient"]["age"] in {"0", "-1", "999"}:
            claim["_fraud_indicators"].append("ghost_patient")
        expected = set(DIAGNOSIS_TO_MEDS.get(claim["diagnosis"], []))
        actual = set(claim.get("medications") or [])
        if expected and actual and not (actual & expected):
            claim["_fraud_indicators"].append("medication_mismatch")
    
    return claim


def generate_dataset(total_claims: int = 100, fraud_ratio: float = 0.25) -> list:
    """Generate complete dataset with specified fraud ratio"""
    num_fraud = int(total_claims * fraud_ratio)
    
    claims = []
    for i in range(total_claims):
        is_fraud = i < num_fraud  # First N claims are fraudulent
        claims.append(generate_claim(i + 1, is_fraud))

    # Ensure we have enough vital-related fraud for the vital_check task.
    # Force a minimum number of impossible_vitals among fraud claims.
    min_vital_fraud = min(10, max(6, num_fraud // 2))
    fraud_indices = [i for i, c in enumerate(claims) if c.get("is_fraud")]
    random.shuffle(fraud_indices)
    forced = 0
    for idx in fraud_indices:
        if forced >= min_vital_fraud:
            break
        c = claims[idx]
        if any(ind in {"impossible_bp", "impossible_pulse"} for ind in (c.get("_fraud_indicators") or [])):
            forced += 1
            continue
        repl = generate_claim(claim_id=idx + 1, is_fraud=True)
        repl["patient"]["village"] = c.get("patient", {}).get("village", repl["patient"]["village"])
        repl["hospital"] = c.get("hospital", repl["hospital"])
        repl["hospital_distance_km"] = c.get("hospital_distance_km", repl["hospital_distance_km"])
        claims[idx] = repl
        forced += 1

    # Inject structured fraud patterns that require dataset-level context (in-place):
    duplicates_to_make = max(3, total_claims // 40)
    sequences_to_make = max(2, total_claims // 60)

    legit_indices = [i for i, c in enumerate(claims) if not c.get("is_fraud")]
    random.shuffle(legit_indices)

    for _ in range(min(duplicates_to_make, len(legit_indices))):
        target_i = legit_indices.pop()
        base_i = random.randrange(total_claims)
        base = claims[base_i]
        target = claims[target_i]
        target["patient"] = deepcopy(base.get("patient", target.get("patient", {})))
        target["diagnosis"] = base.get("diagnosis", target.get("diagnosis"))
        target["bill_amount"] = base.get("bill_amount", target.get("bill_amount"))
        target["claimed_days"] = base.get("claimed_days", target.get("claimed_days"))
        target["medications"] = deepcopy(base.get("medications", target.get("medications", [])))
        target["is_fraud"] = True
        target["duplicate_of"] = base.get("claim_id")
        target["_fraud_indicators"] = sorted(set((target.get("_fraud_indicators") or []) + ["duplicate_claim"]))

    # sequences: take 3 existing claims and align them into an escalating sequence
    available = list(range(total_claims))
    random.shuffle(available)
    for _ in range(sequences_to_make):
        if len(available) < 3:
            break
        i1, i2, i3 = available.pop(), available.pop(), available.pop()
        base = claims[i1]
        seq_id = f"SEQ_{random.randint(1000,9999)}"
        patient = deepcopy(base.get("patient", {}))
        diagnosis = base.get("diagnosis")
        meds = deepcopy(base.get("medications", []))
        base_bill = float(base.get("bill_amount") or 15000.0)
        base_days = int(base.get("claimed_days") or 3)

        for visit_num, idx in enumerate([i1, i2, i3], start=1):
            c = claims[idx]
            c["patient"] = deepcopy(patient)
            c["diagnosis"] = diagnosis
            c["medications"] = deepcopy(meds)
            c["sequence_id"] = seq_id
            c["sequence_visit"] = visit_num
            c["bill_amount"] = round(base_bill * (1.0 + 0.6 * (visit_num - 1)), 2)
            c["claimed_days"] = min(45, int(base_days + 5 * (visit_num - 1)))
            c["is_fraud"] = True
            c["_fraud_indicators"] = sorted(set((c.get("_fraud_indicators") or []) + ["suspicious_sequence"]))
    
    # Shuffle to mix fraud and legitimate claims
    random.shuffle(claims)
    
    # Re-assign sequential claim IDs after shuffling
    for idx, claim in enumerate(claims):
        claim["claim_id"] = f"C{idx+1:03d}"
    
    # Trim/pad to exact requested total_claims
    claims = claims[:total_claims]
    return claims


def save_dataset(claims: list, output_path: str = "data/claims.json"):
    """Save dataset to JSON file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(claims, f, indent=2)
    
    # Generate statistics
    total = len(claims)
    fraud_count = sum(1 for c in claims if c["is_fraud"])
    
    print(f"[OK] Generated {total} claims")
    print(f"     - Fraud: {fraud_count} ({fraud_count/total*100:.1f}%)")
    print(f"     - Legitimate: {total - fraud_count} ({(total-fraud_count)/total*100:.1f}%)")
    print(f"     - Saved to: {output_path}")


if __name__ == "__main__":
    claims = generate_dataset(total_claims=100, fraud_ratio=0.25)
    save_dataset(claims)