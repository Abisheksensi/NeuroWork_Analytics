import requests


BASE_URL = "http://localhost:5000"


def print_result(test_number, name, passed, reason=""):
    if passed:
        print(f"TEST {test_number} — {name}: PASS")
    else:
        print(f"TEST {test_number} — {name}: FAIL — {reason}")


def run_tests():
    passed_tests = 0

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        response_json = response.json()
        passed = response.status_code == 200 and response_json.get("status") == "ok"
        if passed:
            passed_tests += 1
            print_result(1, "Health Check", True)
        else:
            print_result(1, "Health Check", False, "unexpected status or payload")
    except Exception as exc:
        print_result(1, "Health Check", False, str(exc))

    valid_payload = {
        "Age": 30,
        "family_history": 1,
        "self_employed": 0,
        "no_employees": 2,
        "remote_work": 0,
        "tech_company": 1,
        "benefits": 1,
        "care_options": 0,
        "wellness_program": 0,
        "seek_help": 1,
        "anonymity": 1,
        "work_interfere": 2,
        "mental_health_consequence": 0,
        "obs_consequence": 0,
        "Gender_Male": 1,
        "Gender_Female": 0,
        "Gender_Other": 0,
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=valid_payload, timeout=20)
        response_json = response.json()
        passed = (
            response.status_code == 200
            and "prediction" in response_json
            and "probability" in response_json
            and "confidence_percent" in response_json
            and isinstance(response_json.get("shap_explanation"), list)
            and len(response_json.get("shap_explanation", [])) == 5
        )
        if passed:
            passed_tests += 1
            print_result(2, "Valid Prediction", True)
        else:
            print_result(2, "Valid Prediction", False, "missing expected prediction fields")
    except Exception as exc:
        print_result(2, "Valid Prediction", False, str(exc))

    try:
        response = requests.post(f"{BASE_URL}/predict", json={}, timeout=20)
        response_json = response.json()
        passed = response.status_code == 500 and "error" in response_json
        if passed:
            passed_tests += 1
            print_result(3, "Empty Payload", True)
        else:
            print_result(3, "Empty Payload", False, "unexpected status or payload")
    except Exception as exc:
        print_result(3, "Empty Payload", False, str(exc))

    print(f"Results: {passed_tests}/3 tests passed.")


if __name__ == "__main__":
    run_tests()
