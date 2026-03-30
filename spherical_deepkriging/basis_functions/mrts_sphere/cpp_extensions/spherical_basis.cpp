#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;
using Eigen::Map;
using Spectra::SymEigsSolver;
using Spectra::DenseSymMatProd;

// Adaptive quadrature (Simpson) for endpoint-singular integrands.
// We avoid a hard dependency on Boost headers to make packaging easier.
double integrate_adaptive(
    std::function<double(double)> f,
    double a,
    double b,
    double tol = 1e-10,
    int max_depth = 20
) {
    auto simpson = [&](double l, double r) -> double {
        const double m = (l + r) * 0.5;
        const double h = r - l;
        return (h / 6.0) * (f(l) + 4.0 * f(m) + f(r));
    };

    auto recurse = [&](auto&& self, double l, double r, double eps, double whole, int depth) -> double {
        const double m = (l + r) * 0.5;
        const double left = simpson(l, m);
        const double right = simpson(m, r);
        const double delta = left + right - whole;
        if (depth <= 0 || std::fabs(delta) <= 15.0 * eps) {
            // Richardson extrapolation
            return left + right + delta / 15.0;
        }
        return self(self, l, m, eps * 0.5, left, depth - 1) +
               self(self, m, r, eps * 0.5, right, depth - 1);
    };

    const double whole = simpson(a, b);
    // Depth cap prevents infinite recursion if the integrand is very difficult.
    return recurse(recurse, a, b, tol, whole, max_depth);
}

// Kernel function for spherical coordinates
double cpp_Kf(double L1, double l1, double L2, double l2) {
    constexpr double mia = M_PI / 180.0;
    constexpr double eps = 1e-12;

    double a = std::sin(L1 * mia) * std::sin(L2 * mia) +
               std::cos(L1 * mia) * std::cos(L2 * mia) * std::cos((l1 - l2) * mia);

    double b = std::max(-1.0, std::min(a, 1.0));
    double aaa = std::acos(b);

    // If antipodal (cos(aaa) == -1), return known value
    if (std::abs(std::cos(aaa) + 1.0) < 1e-14) {
        return 1.0 - M_PI * M_PI / 6.0;
    }

    double aa = 0.5 + std::cos(aaa) / 2.0;

    // Handle aa ~ 1 exactly (identical or extremely close points)
    // integral_{0}^{1} log(1-x)/x dx = -pi^2/6, so result = 1 - pi^2/6 - (-pi^2/6) = 1
    if (aa >= 1.0 - eps) {
        return 1.0;
    }

    // Handle aa ~ 0 (extreme distances can make aa tiny)
    if (aa <= eps) {
        // integral from 0 to 0 is 0, so result = 1 - pi^2/6
        return 1.0 - M_PI * M_PI / 6.0;
    }

    auto integrand = [](double x) -> double {
        if (std::abs(x) < 1e-12) return -1.0;  // correct limit at 0: lim_{x->0} log(1-x)/x = -1
        return std::log(1.0 - x) / x;
    };

    // Use adaptive quadrature like RcppNumerical::integrate
    // Integrate from 0 to aa (tanh_sinh handles endpoints robustly and won't evaluate exactly at x=1)
    double lower = 0.0;
    double upper = aa;
    double res = integrate_adaptive(integrand, lower, upper, 1e-8, 30);

    return 1.0 - M_PI * M_PI / 6.0 - res;
}

// Compute K matrix for spherical coordinates
py::array_t<double> cpp_K(
    py::array_t<double> X_lat,
    py::array_t<double> X_lon,
    int n
) {
    auto X_lat_buf = X_lat.request();
    auto X_lon_buf = X_lon.request();

    if (X_lat_buf.size != n || X_lon_buf.size != n) {
        throw std::runtime_error("Input size mismatch");
    }

    double* X_lat_ptr = static_cast<double*>(X_lat_buf.ptr);
    double* X_lon_ptr = static_cast<double*>(X_lon_buf.ptr);

    auto result = py::array_t<double>({n, n});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result_ptr[i * n + j] = cpp_Kf(X_lat_ptr[i], X_lon_ptr[i],
                                           X_lat_ptr[j], X_lon_ptr[j]);
        }
    }

    return result;
}

// Compute basis functions fk for a single point
py::array_t<double> cpp_fk(
    double L1, double l1, double KK,
    py::array_t<double> X,
    py::array_t<double> Konev,
    py::array_t<double> eiKvecmval,
    int n
) {
    auto X_buf = X.request();
    auto Konev_buf = Konev.request();
    auto eiKvecmval_buf = eiKvecmval.request();

    if (X_buf.shape[0] != n || X_buf.shape[1] != 2) {
        throw std::runtime_error("X must be n x 2");
    }
    if (Konev_buf.size != n) {
        throw std::runtime_error("Konev size mismatch");
    }
    if (eiKvecmval_buf.shape[0] != n || eiKvecmval_buf.shape[1] != (int)KK - 1) {
        throw std::runtime_error("eiKvecmval size mismatch");
    }

    double* X_ptr = static_cast<double*>(X_buf.ptr);
    double* Konev_ptr = static_cast<double*>(Konev_buf.ptr);
    double* eiKvecmval_ptr = static_cast<double*>(eiKvecmval_buf.ptr);

    auto result = py::array_t<double>((int)KK);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    // First basis function
    result_ptr[0] = std::sqrt(1.0 / n);

    // Compute f2
    std::vector<double> f2(n);
    for (int i = 0; i < n; i++) {
        f2[i] = cpp_Kf(L1, l1, X_ptr[i * 2], X_ptr[i * 2 + 1]);
    }

    // Compute t = f2 - Konev
    std::vector<double> t(n);
    for (int i = 0; i < n; i++) {
        t[i] = f2[i] - Konev_ptr[i];
    }

    // Project onto each basis function
    for (int k = 1; k < (int)KK; k++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) {
            s += t[j] * eiKvecmval_ptr[j * ((int)KK - 1) + (k - 1)];
        }
        result_ptr[k] = s;
    }

    return result;
}

// Compute K matrix for multiple points (optimized version)
py::array_t<double> cpp_Kmatrix(
    int KK,
    py::array_t<double> X,
    py::array_t<double> ggrids,
    py::array_t<double> Konev,
    py::array_t<double> eiKvecmval,
    int n,
    int N
) {
    auto X_buf = X.request();
    auto ggrids_buf = ggrids.request();
    auto Konev_buf = Konev.request();
    auto eiKvecmval_buf = eiKvecmval.request();

    if (X_buf.shape[0] != n || X_buf.shape[1] != 2) {
        throw std::runtime_error("X must be n x 2");
    }
    if (ggrids_buf.shape[0] != N || ggrids_buf.shape[1] != 2) {
        throw std::runtime_error("ggrids must be N x 2");
    }
    if (Konev_buf.size != n) {
        throw std::runtime_error("Konev size mismatch");
    }
    if (eiKvecmval_buf.shape[0] != n || eiKvecmval_buf.shape[1] != KK - 1) {
        throw std::runtime_error("eiKvecmval size mismatch");
    }

    double* X_ptr = static_cast<double*>(X_buf.ptr);
    double* ggrids_ptr = static_cast<double*>(ggrids_buf.ptr);
    double* Konev_ptr = static_cast<double*>(Konev_buf.ptr);
    double* eiKvecmval_ptr = static_cast<double*>(eiKvecmval_buf.ptr);

    auto result = py::array_t<double>({N, KK});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    for (int i = 0; i < N; i++) {
        double L1 = ggrids_ptr[i * 2];
        double l1 = ggrids_ptr[i * 2 + 1];

        // Precompute f2 = cpp_Kf for this row (use cpp_Kf to ensure consistency)
        std::vector<double> f2(n);
        for (int j = 0; j < n; j++) {
            double L2 = X_ptr[j * 2];
            double l2 = X_ptr[j * 2 + 1];
            f2[j] = cpp_Kf(L1, l1, L2, l2);
        }

        // t = f2 - Konev
        std::vector<double> t(n);
        for (int j = 0; j < n; j++) {
            t[j] = f2[j] - Konev_ptr[j];
        }

        // First basis function
        result_ptr[i * KK + 0] = std::sqrt(1.0 / n);

        // Project onto each basis function
        for (int k = 1; k < KK; k++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += t[j] * eiKvecmval_ptr[j * (KK - 1) + (k - 1)];
            }
            result_ptr[i * KK + k] = s;
        }
    }

    return result;
}

// Exponential kernel for spherical coordinates
py::array_t<double> cpp_exp(
    py::array_t<double> X,
    py::array_t<double> Y,
    int n,
    int N,
    double c,
    double vy
) {
    auto X_buf = X.request();
    auto Y_buf = Y.request();

    if (X_buf.shape[0] != n || X_buf.shape[1] != 2) {
        throw std::runtime_error("X must be n x 2");
    }
    if (Y_buf.shape[0] != N || Y_buf.shape[1] != 2) {
        throw std::runtime_error("Y must be N x 2");
    }

    double* X_ptr = static_cast<double*>(X_buf.ptr);
    double* Y_ptr = static_cast<double*>(Y_buf.ptr);

    auto result = py::array_t<double>({N, n});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    double mia = M_PI / 180.0;

    for (int i = 0; i < N; i++) {
        double L1 = Y_ptr[i * 2];
        double l1 = Y_ptr[i * 2 + 1];

        for (int j = 0; j < n; j++) {
            double L2 = X_ptr[j * 2];
            double l2 = X_ptr[j * 2 + 1];

            double a = sin(L1 * mia) * sin(L2 * mia) +
                      cos(L1 * mia) * cos(L2 * mia) * cos((l1 - l2) * mia);

            double b = std::max(-1.0, std::min(a, 1.0));
            double aaa = acos(b);

            result_ptr[i * n + j] = vy * std::exp(-aaa / c);
        }
    }

    return result;
}

// Eigenvalue decomposition using Eigen (full decomposition - kept for backward compatibility)
py::tuple getEigen(py::array_t<double> M) {
    auto M_buf = M.request();

    if (M_buf.ndim != 2 || M_buf.shape[0] != M_buf.shape[1]) {
        throw std::runtime_error("M must be a square matrix");
    }

    int n = M_buf.shape[0];
    double* M_ptr = static_cast<double*>(M_buf.ptr);

    // Convert to Eigen matrix
    MatrixXd eigen_M(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            eigen_M(i, j) = M_ptr[i * n + j];
        }
    }

    // Compute eigenvalues and eigenvectors
    SelfAdjointEigenSolver<MatrixXd> es(eigen_M);
    MatrixXd eigenvectors = es.eigenvectors();
    VectorXd eigenvalues = es.eigenvalues();

    // Convert back to numpy arrays
    auto eigvals = py::array_t<double>(n);
    auto eigvecs = py::array_t<double>({n, n});

    double* eigvals_ptr = static_cast<double*>(eigvals.request().ptr);
    double* eigvecs_ptr = static_cast<double*>(eigvecs.request().ptr);

    for (int i = 0; i < n; i++) {
        eigvals_ptr[i] = eigenvalues(i);
        for (int j = 0; j < n; j++) {
            eigvecs_ptr[i * n + j] = eigenvectors(i, j);
        }
    }

    return py::make_tuple(eigvals, eigvecs);
}

// Partial eigenvalue decomposition using Spectra (top-k largest eigenvalues, matches RSpectra)
// This is more efficient than full decomposition when k << n
py::tuple getEigenTopK(py::array_t<double> M, int k) {
    auto M_buf = M.request();

    if (M_buf.ndim != 2 || M_buf.shape[0] != M_buf.shape[1]) {
        throw std::runtime_error("M must be a square matrix");
    }

    int n = M_buf.shape[0];
    if (k < 1 || k > n) {
        throw std::runtime_error("k must be between 1 and matrix size");
    }

    double* M_ptr = static_cast<double*>(M_buf.ptr);

    // Convert to Eigen matrix
    MatrixXd eigen_M(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            eigen_M(i, j) = M_ptr[i * n + j];
        }
    }

    // Spectra requires k < n (can't compute all eigenvalues)
    // If k == n, fall back to full decomposition
    if (k >= n) {
        // Use full decomposition and return top k
        SelfAdjointEigenSolver<MatrixXd> es(eigen_M);
        MatrixXd eigenvectors = es.eigenvectors();
        VectorXd eigenvalues = es.eigenvalues();

        // Eigen returns eigenvalues in ascending order, so we need to sort descending
        // and take the last k (largest)
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&eigenvalues](int i, int j) { return eigenvalues(i) > eigenvalues(j); });

        // Convert back to numpy arrays
        auto eigvals = py::array_t<double>(k);
        auto eigvecs = py::array_t<double>({n, k});

        double* eigvals_ptr = static_cast<double*>(eigvals.request().ptr);
        double* eigvecs_ptr = static_cast<double*>(eigvecs.request().ptr);

        for (int i = 0; i < k; i++) {
            int idx = indices[i];
            eigvals_ptr[i] = eigenvalues(idx);
            for (int j = 0; j < n; j++) {
                eigvecs_ptr[j * k + i] = eigenvectors(j, idx);
            }
        }

        return py::make_tuple(eigvals, eigvecs);
    }

    // Use Spectra to compute top k largest eigenvalues (by algebraic value, not absolute)
    // RSpectra::eigs_sym selects largest algebraic eigenvalues, which is what we want
    // We use Spectra::LARGEST_ALGE to match this behavior
    DenseSymMatProd<double> op(eigen_M);

    // Need to compute k eigenvalues, but Spectra typically needs a few more for stability
    // Use ncv = min(2*k + 1, n) as recommended by Spectra documentation
    int ncv = std::min(2 * k + 1, n);
    SymEigsSolver<DenseSymMatProd<double>> eigs(op, k, ncv);

    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

    if (eigs.info() != Spectra::CompInfo::Successful) {
        throw std::runtime_error("Spectra eigenvalue computation failed");
    }

    VectorXd eigenvalues = eigs.eigenvalues();
    MatrixXd eigenvectors = eigs.eigenvectors();

    // Spectra returns eigenvalues in descending order (largest first), which matches RSpectra
    // Convert back to numpy arrays
    auto eigvals = py::array_t<double>(k);
    auto eigvecs = py::array_t<double>({n, k});

    double* eigvals_ptr = static_cast<double*>(eigvals.request().ptr);
    double* eigvecs_ptr = static_cast<double*>(eigvecs.request().ptr);

    for (int i = 0; i < k; i++) {
        eigvals_ptr[i] = eigenvalues(i);
        for (int j = 0; j < n; j++) {
            eigvecs_ptr[j * k + i] = eigenvectors(j, i);
        }
    }

    return py::make_tuple(eigvals, eigvecs);
}

PYBIND11_MODULE(spherical_basis, m) {
    m.doc() = "Spherical basis functions for spatial modeling";

    m.def("cpp_Kf", &cpp_Kf,
          "Compute spherical kernel function",
          py::arg("L1"), py::arg("l1"), py::arg("L2"), py::arg("l2"));

    m.def("cpp_K", &cpp_K,
          "Compute K matrix for spherical coordinates",
          py::arg("X_lat"), py::arg("X_lon"), py::arg("n"));

    m.def("cpp_fk", &cpp_fk,
          "Compute basis functions fk for a single point",
          py::arg("L1"), py::arg("l1"), py::arg("KK"),
          py::arg("X"), py::arg("Konev"), py::arg("eiKvecmval"), py::arg("n"));

    m.def("cpp_Kmatrix", &cpp_Kmatrix,
          "Compute K matrix for multiple points (optimized)",
          py::arg("KK"), py::arg("X"), py::arg("ggrids"),
          py::arg("Konev"), py::arg("eiKvecmval"), py::arg("n"), py::arg("N"));

    m.def("cpp_exp", &cpp_exp,
          "Exponential kernel for spherical coordinates",
          py::arg("X"), py::arg("Y"), py::arg("n"), py::arg("N"),
          py::arg("c"), py::arg("vy"));

    m.def("getEigen", &getEigen,
          "Compute all eigenvalues and eigenvectors of a symmetric matrix (full decomposition)",
          py::arg("M"));

    m.def("getEigenTopK", &getEigenTopK,
          "Compute top k largest eigenvalues and eigenvectors using Spectra (matches RSpectra behavior)",
          py::arg("M"), py::arg("k"));
}

