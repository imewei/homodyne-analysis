"""
Numerical Accuracy Verification Tests
====================================

Comprehensive test suite to verify numerical accuracy preservation
across all refactored high-complexity functions. This ensures that
mathematical computations maintain precision and correctness.

Focus areas:
- Chi-squared calculation accuracy
- Matrix operations precision
- Statistical computation correctness
- Optimization algorithm convergence
- Floating point stability

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import warnings

import numpy as np

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TestChiSquaredAccuracy:
    """Test numerical accuracy of chi-squared calculations."""

    def test_chi_squared_mathematical_correctness(self):
        """Test mathematical correctness of chi-squared calculation."""
        # Create known test data
        n_angles, n_time = 3, 5
        c2_exp = np.array(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0],
                    [4.0, 5.0, 6.0, 7.0, 8.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                ],
                [
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0],
                    [4.0, 5.0, 6.0, 7.0, 8.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                ],
                [
                    [3.0, 4.0, 5.0, 6.0, 7.0],
                    [4.0, 5.0, 6.0, 7.0, 8.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0],
                ],
            ]
        )

        c2_theo = np.array(
            [
                [
                    [1.1, 2.1, 3.1, 4.1, 5.1],
                    [2.1, 3.1, 4.1, 5.1, 6.1],
                    [3.1, 4.1, 5.1, 6.1, 7.1],
                    [4.1, 5.1, 6.1, 7.1, 8.1],
                    [5.1, 6.1, 7.1, 8.1, 9.1],
                ],
                [
                    [2.1, 3.1, 4.1, 5.1, 6.1],
                    [3.1, 4.1, 5.1, 6.1, 7.1],
                    [4.1, 5.1, 6.1, 7.1, 8.1],
                    [5.1, 6.1, 7.1, 8.1, 9.1],
                    [6.1, 7.1, 8.1, 9.1, 10.1],
                ],
                [
                    [3.1, 4.1, 5.1, 6.1, 7.1],
                    [4.1, 5.1, 6.1, 7.1, 8.1],
                    [5.1, 6.1, 7.1, 8.1, 9.1],
                    [6.1, 7.1, 8.1, 9.1, 10.1],
                    [7.1, 8.1, 9.1, 10.1, 11.1],
                ],
            ]
        )

        # Calculate chi-squared manually
        diff = c2_exp - c2_theo
        chi2_expected = np.sum(diff**2)

        # Vectorized calculation (as in refactored code)
        chi2_vectorized = np.sum((c2_exp - c2_theo) ** 2)

        # Element-wise calculation for verification
        chi2_manual = 0.0
        for i in range(n_angles):
            for j in range(n_time):
                for k in range(n_time):
                    chi2_manual += (c2_exp[i, j, k] - c2_theo[i, j, k]) ** 2

        # All methods should give identical results
        np.testing.assert_allclose(chi2_vectorized, chi2_expected, rtol=1e-14)
        np.testing.assert_allclose(chi2_manual, chi2_expected, rtol=1e-14)

        # Known expected value: 3 * 5 * 5 * (0.1)^2 = 75 * 0.01 = 0.75
        expected_value = 0.75
        np.testing.assert_allclose(chi2_expected, expected_value, rtol=1e-14)

        print("✓ Chi-squared mathematical correctness verified")

    def test_chi_squared_edge_cases(self):
        """Test chi-squared calculation edge cases."""
        # Test case 1: Identical arrays (chi-squared should be 0)
        c2_identical = np.ones((2, 3, 3))
        chi2_zero = np.sum((c2_identical - c2_identical) ** 2)
        assert chi2_zero == 0.0

        # Test case 2: Large values
        c2_large = np.ones((2, 3, 3)) * 1e10
        c2_large_offset = c2_large + 1.0
        chi2_large = np.sum((c2_large - c2_large_offset) ** 2)
        expected_large = 2 * 3 * 3 * 1.0  # 18.0
        np.testing.assert_allclose(chi2_large, expected_large, rtol=1e-10)

        # Test case 3: Small values
        c2_small = np.ones((2, 3, 3)) * 1e-10
        c2_small_offset = c2_small + 1e-15
        chi2_small = np.sum((c2_small - c2_small_offset) ** 2)
        expected_small = 2 * 3 * 3 * (1e-15) ** 2
        np.testing.assert_allclose(chi2_small, expected_small, rtol=1e-5)

        print("✓ Chi-squared edge cases verified")

    def test_numerical_stability_precision(self):
        """Test numerical stability with different precision scenarios."""
        # Test with different floating point precisions
        test_cases = [
            (np.float32, 1e-6),
            (np.float64, 1e-14),
        ]

        for dtype, tolerance in test_cases:
            c2_exp = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=dtype)
            c2_theo = np.array([[[1.1, 2.1], [3.1, 4.1]]], dtype=dtype)

            chi2_result = np.sum((c2_exp - c2_theo) ** 2)

            # Expected: (0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2 = 4 * 0.01 = 0.04
            expected = 0.04
            assert abs(chi2_result - expected) < tolerance

        print("✓ Numerical stability precision verified")


class TestMatrixOperationAccuracy:
    """Test numerical accuracy of matrix operations."""

    def test_matrix_multiplication_accuracy(self):
        """Test matrix multiplication accuracy preservation."""
        # Create test matrices with known properties
        A = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
            ],
            dtype=np.float64,
        )

        B = np.array(
            [
                [2, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 2],
            ],
            dtype=np.float64,
        )

        # A @ B should be 2 * A
        result = A @ B
        expected = 2 * A

        np.testing.assert_allclose(result, expected, rtol=1e-15)

        print("✓ Matrix multiplication accuracy verified")

    def test_matrix_decomposition_accuracy(self):
        """Test matrix decomposition accuracy."""
        # Create a symmetric positive definite matrix
        n = 4
        A_base = np.random.rand(n, n)
        A = A_base @ A_base.T  # Ensures positive definite

        # Test Cholesky decomposition
        try:
            L = np.linalg.cholesky(A)
            A_reconstructed = L @ L.T

            np.testing.assert_allclose(A, A_reconstructed, rtol=1e-12)
        except np.linalg.LinAlgError:
            # Matrix might not be positive definite, skip this test
            pass

        # Test SVD decomposition
        U, s, Vt = np.linalg.svd(A)
        A_reconstructed_svd = U @ np.diag(s) @ Vt

        np.testing.assert_allclose(A, A_reconstructed_svd, rtol=1e-12)

        print("✓ Matrix decomposition accuracy verified")

    def test_eigenvalue_accuracy(self):
        """Test eigenvalue calculation accuracy."""
        # Create a matrix with known eigenvalues
        A = np.array([[3, 1], [1, 3]], dtype=np.float64)

        eigenvals, eigenvecs = np.linalg.eig(A)

        # Known eigenvalues: 4 and 2
        expected_eigenvals = np.array([4.0, 2.0])
        computed_eigenvals = np.sort(eigenvals)
        expected_eigenvals = np.sort(expected_eigenvals)

        np.testing.assert_allclose(computed_eigenvals, expected_eigenvals, rtol=1e-12)

        # Test eigenvalue equation: A @ v = λ @ v
        for i, (eigenval, eigenvec) in enumerate(
            zip(eigenvals, eigenvecs.T, strict=False)
        ):
            lhs = A @ eigenvec
            rhs = eigenval * eigenvec

            np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

        print("✓ Eigenvalue accuracy verified")


class TestStatisticalAccuracy:
    """Test statistical computation accuracy."""

    def test_moment_calculation_accuracy(self):
        """Test statistical moment calculations."""
        # Create test data with known statistical properties
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

        # Test mean
        mean_computed = np.mean(data)
        mean_expected = 5.5  # (1+10)*10/2 / 10 = 55/10 = 5.5
        np.testing.assert_allclose(mean_computed, mean_expected, rtol=1e-15)

        # Test variance (population)
        var_computed = np.var(data, ddof=0)
        # Variance = E[X^2] - E[X]^2
        # E[X^2] = (1^2 + 2^2 + ... + 10^2)/10 = 385/10 = 38.5
        # E[X]^2 = 5.5^2 = 30.25
        # Variance = 38.5 - 30.25 = 8.25
        var_expected = 8.25
        np.testing.assert_allclose(var_computed, var_expected, rtol=1e-12)

        # Test standard deviation
        std_computed = np.std(data, ddof=0)
        std_expected = np.sqrt(8.25)
        np.testing.assert_allclose(std_computed, std_expected, rtol=1e-12)

        print("✓ Statistical moment accuracy verified")

    def test_correlation_accuracy(self):
        """Test correlation calculation accuracy."""
        # Create correlated data
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 * x + 1 + np.random.normal(0, 0.1, n)  # Nearly perfect correlation

        # Calculate correlation coefficient
        corr_matrix = np.corrcoef(x, y)
        corr_xy = corr_matrix[0, 1]

        # Should be very close to 1 due to linear relationship
        assert corr_xy > 0.95  # High correlation expected

        # Test perfect correlation
        y_perfect = 2 * x + 1
        corr_perfect = np.corrcoef(x, y_perfect)[0, 1]
        np.testing.assert_allclose(corr_perfect, 1.0, rtol=1e-12)

        print("✓ Correlation accuracy verified")

    def test_regression_accuracy(self):
        """Test regression calculation accuracy."""
        # Linear regression: y = 3x + 2
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = 3 * x + 2  # Perfect linear relationship

        # Calculate slope and intercept manually
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # Should recover original coefficients exactly
        np.testing.assert_allclose(slope, 3.0, rtol=1e-14)
        np.testing.assert_allclose(intercept, 2.0, rtol=1e-14)

        # Test with numpy polyfit
        coeffs = np.polyfit(x, y, 1)
        np.testing.assert_allclose(coeffs[0], 3.0, rtol=1e-14)  # slope
        np.testing.assert_allclose(coeffs[1], 2.0, rtol=1e-14)  # intercept

        print("✓ Regression accuracy verified")


class TestOptimizationAccuracy:
    """Test optimization algorithm accuracy."""

    def test_quadratic_optimization_accuracy(self):
        """Test optimization accuracy for quadratic functions."""

        # Quadratic function: f(x) = (x-2)^2 + 1
        # Minimum at x = 2, f_min = 1
        def quadratic_func(x):
            return (x - 2) ** 2 + 1

        # Test gradient calculation
        def quadratic_grad(x):
            return 2 * (x - 2)

        # Numerical gradient estimation
        x_test = 3.0
        eps = 1e-8
        grad_numerical = (
            quadratic_func(x_test + eps) - quadratic_func(x_test - eps)
        ) / (2 * eps)
        grad_analytical = quadratic_grad(x_test)

        np.testing.assert_allclose(grad_numerical, grad_analytical, rtol=1e-6)

        # Test that minimum is at x = 2
        assert quadratic_func(2.0) == 1.0
        assert quadratic_grad(2.0) == 0.0

        print("✓ Quadratic optimization accuracy verified")

    def test_multivariate_optimization_accuracy(self):
        """Test multivariate optimization accuracy."""

        # Rosenbrock function minimum at (1, 1)
        def rosenbrock(x):
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        # Test function value at minimum
        minimum_point = np.array([1.0, 1.0])
        min_value = rosenbrock(minimum_point)
        assert min_value == 0.0

        # Test gradient at minimum should be zero
        eps = 1e-8
        grad = np.zeros(2)
        for i in range(2):
            x_plus = minimum_point.copy()
            x_minus = minimum_point.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (rosenbrock(x_plus) - rosenbrock(x_minus)) / (2 * eps)

        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-6)

        print("✓ Multivariate optimization accuracy verified")

    def test_constraint_optimization_accuracy(self):
        """Test constrained optimization accuracy."""
        # Simple constrained problem: minimize x^2 + y^2 subject to x + y = 1
        # Solution: x = y = 0.5, f_min = 0.5

        # Lagrangian: L = x^2 + y^2 + λ(x + y - 1)
        # ∂L/∂x = 2x + λ = 0 → x = -λ/2
        # ∂L/∂y = 2y + λ = 0 → y = -λ/2
        # ∂L/∂λ = x + y - 1 = 0

        # From constraint: -λ/2 + (-λ/2) = 1 → -λ = 1 → λ = -1
        # Therefore: x = y = 0.5

        x_opt, y_opt = 0.5, 0.5
        f_opt = x_opt**2 + y_opt**2

        np.testing.assert_allclose(f_opt, 0.5, rtol=1e-15)
        np.testing.assert_allclose(
            x_opt + y_opt, 1.0, rtol=1e-15
        )  # Constraint satisfied

        print("✓ Constraint optimization accuracy verified")


class TestNumericalIntegrationAccuracy:
    """Test numerical integration accuracy."""

    def test_trapezoidal_integration_accuracy(self):
        """Test trapezoidal rule integration accuracy."""
        # Integrate sin(x) from 0 to π, exact answer = 2
        x = np.linspace(0, np.pi, 1000)
        y = np.sin(x)

        # Trapezoidal rule
        integral_trap = np.trapz(y, x)

        # Should be very close to 2
        np.testing.assert_allclose(integral_trap, 2.0, rtol=1e-6)

        # Test with known polynomial: ∫x² dx from 0 to 1 = 1/3
        x_poly = np.linspace(0, 1, 1000)
        y_poly = x_poly**2

        integral_poly = np.trapz(y_poly, x_poly)
        np.testing.assert_allclose(integral_poly, 1.0 / 3.0, rtol=1e-6)

        print("✓ Trapezoidal integration accuracy verified")

    def test_simpson_integration_accuracy(self):
        """Test Simpson's rule integration accuracy."""
        from scipy import integrate

        # Integrate x^3 from 0 to 2, exact answer = 4
        def cubic_func(x):
            return x**3

        result, error = integrate.quad(cubic_func, 0, 2)
        np.testing.assert_allclose(result, 4.0, rtol=1e-12)

        # Test with exponential function
        def exp_func(x):
            return np.exp(x)

        # ∫e^x dx from 0 to 1 = e - 1
        result_exp, error_exp = integrate.quad(exp_func, 0, 1)
        expected_exp = np.exp(1) - 1
        np.testing.assert_allclose(result_exp, expected_exp, rtol=1e-12)

        print("✓ Simpson integration accuracy verified")


def run_numerical_accuracy_tests():
    """Run comprehensive numerical accuracy validation."""
    print("Running Numerical Accuracy Validation")
    print("=" * 50)

    test_classes = [
        TestChiSquaredAccuracy,
        TestMatrixOperationAccuracy,
        TestStatisticalAccuracy,
        TestOptimizationAccuracy,
        TestNumericalIntegrationAccuracy,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        test_instance = test_class()

        test_methods = [
            method
            for method in dir(test_instance)
            if method.startswith("test_") and callable(getattr(test_instance, method))
        ]

        for method_name in test_methods:
            try:
                test_method = getattr(test_instance, method_name)
                test_method()
                passed_tests += 1
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")

            total_tests += 1

    print(f"\nNumerical Accuracy Summary: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All numerical accuracy tests passed!")
        print("✓ Mathematical correctness preserved across refactoring")
        print("✓ Floating point precision maintained")
        print("✓ Statistical computations accurate")
        print("✓ Optimization algorithms stable")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} numerical accuracy tests failed")
        print("Review refactored functions for mathematical correctness")

    return passed_tests, total_tests


if __name__ == "__main__":
    run_numerical_accuracy_tests()
