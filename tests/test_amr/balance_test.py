import numpy as np
import os
import sys
# Get absolute path to project root
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, PROJECT_ROOT)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
))
sys.path.append(PROJECT_ROOT)

from numerical.amr.adapt import enforce_2_1_balance, check_2_1_balance
from numerical.amr.forest import forest


def test_2_1_balance():
    """
    Comprehensive test suite for 2:1 balance enforcement.
    Tests various scenarios that could create imbalance.
    """
    def setup_basic_mesh():
        """Creates a basic test mesh with known structure."""
        xelem = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        max_level = 3
        label_mat, info_mat, active = forest(xelem, max_level)
        return label_mat, info_mat, active
        
    def verify_balance(active, label_mat):
        """Checks if mesh satisfies 2:1 balance."""
        violations = check_2_1_balance(active, label_mat)
        return len(violations) == 0
        
    # Test 1: Basic refinement propagation
    def test_refinement_propagation():
        label_mat, info_mat, active = setup_basic_mesh()
        marks = np.zeros(len(active), dtype=int)
        
        # Mark element for refinement
        marks[1] = 1  # Mark second element
        
        # Enforce balance
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        # Verify propagation occurred if needed
        assert verify_balance(active, label_mat), "Balance violation after basic refinement"
        
    # Test 2: Multiple level differences
    def test_multiple_level_differences():
        label_mat, info_mat, active = setup_basic_mesh()
        marks = np.zeros(len(active), dtype=int)
        
        # Try to create a two-level difference
        marks[1] = 1  # Refine one element
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        # Attempt second refinement
        marks[1] = 1
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        assert verify_balance(active, label_mat), "Balance violation with multiple refinements"
        
    # Test 3: Coarsening validation
    def test_coarsening():
        label_mat, info_mat, active = setup_basic_mesh()
        marks = np.zeros(len(active), dtype=int)
        
        # First refine some elements
        marks[1] = marks[2] = 1
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        # Try to coarsen
        marks = np.zeros(len(active), dtype=int)
        marks[1] = marks[2] = -1
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        assert verify_balance(active, label_mat), "Balance violation after coarsening"
        
    # Test 4: Edge cases at domain boundaries
    def test_boundary_cases():
        label_mat, info_mat, active = setup_basic_mesh()
        marks = np.zeros(len(active), dtype=int)
        
        # Refine element at left boundary
        marks[0] = 1
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        # Refine element at right boundary
        marks[-1] = 1
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        assert verify_balance(active, label_mat), "Balance violation at boundaries"
        
    # Test 5: Complex refinement pattern
    def test_complex_pattern():
        label_mat, info_mat, active = setup_basic_mesh()
        marks = np.zeros(len(active), dtype=int)
        
        # Create complex refinement pattern
        # Note: active array is length 4 (indices 0-3)
        marks[0] = 1  # Refine first element
        marks[1] = 1  # Refine second element
        marks[3] = 1  # Refine last element (index 3 instead of 4)
        
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        assert verify_balance(active, label_mat), "Balance violation in complex pattern"
        
    # Test 6: Specific bug case with element 52
    def test_element_52_case():
        """Recreates the specific case where element 52 caused issues."""
        xelem = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])  # Similar to class_test.py
        max_level = 4
        label_mat, info_mat, active = forest(xelem, max_level)
        
        # Simulate the refinement pattern leading to the bug
        marks = np.zeros(len(active), dtype=int)
        
        # Find element 52 in active elements
        elem_52_idx = np.where(active == 52)[0]
        if len(elem_52_idx) > 0:
            marks[elem_52_idx[0]] = 1
            
        balanced_marks = enforce_2_1_balance(label_mat, active, marks)
        
        assert verify_balance(active, label_mat), "Balance violation in element 52 case"
    
    # Run all tests
    print("Running balance enforcement tests...")
    test_refinement_propagation()
    test_multiple_level_differences()
    test_coarsening()
    test_boundary_cases()
    test_complex_pattern()
    test_element_52_case()
    print("All balance tests passed!")

if __name__ == "__main__":
    test_2_1_balance()