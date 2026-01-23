from typing import Tuple
from utils import ReductionResult, TestSuite


# First Faulty Test (FFT)
def fft(test_suite: TestSuite, reduction_result: ReductionResult) -> int:
    """
    First Faulty Test (FFT): position of first test case that reveals a fault
    """
    for pos, tc in enumerate(reduction_result.test_case_selection, start=1):
        if len(test_suite.fault_matrix[tc]) != 0:
            return pos

    return -1


# Test Suite Reduction (TSR)
def tsr(test_suite: TestSuite, reduction_result: ReductionResult) -> float:
    """
    Test Suite Reduction (TSR): percentage of reduction
    """
    num_test_cases = len(test_suite.test_cases)
    num_selections = len(reduction_result.test_case_selection)

    return (num_test_cases - num_selections) / num_test_cases


# Fault Detection Loss (FDL)
def fdl(
    test_suite: TestSuite, reduction_result: ReductionResult
) -> Tuple[int, int, float]:
    """
    Fault Detection Loss (FDL): Loss of fault detected by the reduced test suite
    """
    total_faults = set()
    detected_faults = set()

    selected_idx = set(reduction_result.test_case_selection)

    for test_case_idx, faults in test_suite.fault_matrix.items():
        total_faults.update(faults)

        if test_case_idx in selected_idx:
            detected_faults.update(faults)

    return (
        len(total_faults),
        len(detected_faults),
        (len(total_faults) - len(detected_faults)) / len(total_faults),
    )


def apfd(test_suite: TestSuite, reduction_result: ReductionResult) -> float:
    """
    Average Percentage of Faults Detected (APFD): effectiveness metric for test case prioritization
    """
    detected_faults = set()

    numerator = 0.0  # numerator of APFD
    for position, tc_ID in enumerate(reduction_result.test_case_selection):
        for fault in test_suite.fault_matrix[tc_ID]:
            if fault not in detected_faults:
                detected_faults.add(fault)
                numerator += position

    n, m = len(reduction_result.test_case_selection), len(detected_faults)
    apfd = 1.0 - (numerator / (n * m)) + (1.0 / (2 * n)) if m > 0 else 0.0

    return apfd


def compute_metrics(test_suite: TestSuite, reduction_result: ReductionResult) -> dict:
    total_faults, faults_detected, fdl_ = fdl(test_suite, reduction_result)
    return {
        "total_faults": total_faults,
        "faults_detected": faults_detected,
        "preperation_time_ns": reduction_result.prep_time_ns,
        "reduction_time_ns": reduction_result.red_time_ns,
        "fft": fft(test_suite, reduction_result),
        "tsr": tsr(test_suite, reduction_result),
        "fdl": fdl_,
        "apfd": apfd(test_suite, reduction_result),
    }
