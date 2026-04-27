#pragma once

#include <array>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Base class (CRTP-free, degree/scalar templated)
//
// Stores position control points + precomputed derivative chains.
// De Casteljau evaluation operates on tau = t/T in [0,1] using only
// multiply-add on Scalar — no power functions, no t^n accumulation.
//
// Template params:
//   Degree  — polynomial degree (4 or 5 for quartic/quintic)
//   Scalar  — float type (default double; swap to bfloat16 for NPU)
// ---------------------------------------------------------------------------
template<int Degree, typename Scalar = double>
class BernsteinPolynomialBase {
public:
    static constexpr int N_POS  = Degree + 1;
    static constexpr int N_VEL  = Degree;
    static constexpr int N_ACC  = Degree - 1;
    static constexpr int N_JERK = Degree - 2;

    Scalar calc_point(Scalar t) const {
        return de_casteljau<N_POS>(cp_pos_, t * inv_T_);
    }

    Scalar calc_first_derivative(Scalar t) const {
        return de_casteljau<N_VEL>(cp_vel_, t * inv_T_);
    }

    Scalar calc_second_derivative(Scalar t) const {
        return de_casteljau<N_ACC>(cp_acc_, t * inv_T_);
    }

    Scalar calc_third_derivative(Scalar t) const {
        return de_casteljau<N_JERK>(cp_jerk_, t * inv_T_);
    }

protected:
    Scalar T_{};
    Scalar inv_T_{};
    std::array<Scalar, N_POS>  cp_pos_{};
    std::array<Scalar, N_VEL>  cp_vel_{};
    std::array<Scalar, N_ACC>  cp_acc_{};
    std::array<Scalar, N_JERK> cp_jerk_{};

    // Subclass fills cp_pos_ first, then calls this to build the chains.
    // Derivative rule for degree-n Bernstein: d/dt has degree n-1 with
    // control points (n/T) * diff(cp), applied recursively.
    void compute_derivative_chains() {
        const Scalar n = static_cast<Scalar>(Degree);
        for (int i = 0; i < N_VEL; ++i)
            cp_vel_[i]  = (n          * inv_T_) * (cp_pos_[i+1] - cp_pos_[i]);
        for (int i = 0; i < N_ACC; ++i)
            cp_acc_[i]  = ((n - 1)    * inv_T_) * (cp_vel_[i+1] - cp_vel_[i]);
        for (int i = 0; i < N_JERK; ++i)
            cp_jerk_[i] = ((n - 2)    * inv_T_) * (cp_acc_[i+1] - cp_acc_[i]);
    }

private:
    // De Casteljau recursion on tau in [0,1].
    // Each step: b[k] = b[k] + tau*(b[k+1]-b[k])  (one FMA per iteration)
    template<int N>
    static Scalar de_casteljau(const std::array<Scalar, N>& ctrl, Scalar tau) {
        std::array<Scalar, N> b = ctrl;
        for (int r = 1; r < N; ++r)
            for (int k = 0; k < N - r; ++k)
                b[k] += tau * (b[k+1] - b[k]);
        return b[0];
    }
};


// ---------------------------------------------------------------------------
// Quintic (degree 5): fully-constrained endpoints
//
// Boundary conditions (6):
//   s(0)=s0, s'(0)=v0, s''(0)=a0
//   s(T)=sf, s'(T)=vf, s''(T)=af
//
// Use for: lane-change, merging, any maneuver with a target position.
// ---------------------------------------------------------------------------
template<typename Scalar = double>
class QuinticPolynomial_Bernstein : public BernsteinPolynomialBase<5, Scalar> {
    using Base = BernsteinPolynomialBase<5, Scalar>;
public:
    QuinticPolynomial_Bernstein(
        Scalar s0, Scalar v0, Scalar a0,
        Scalar sf, Scalar vf, Scalar af,
        Scalar T)
    {
        if (T <= Scalar(0))
            throw std::invalid_argument("T must be positive");

        this->T_     = T;
        this->inv_T_ = Scalar(1) / T;

        const Scalar T2 = T * T;

        // Closed-form control points from the 6 boundary conditions.
        // Start side: c0, c1, c2 from s(0), s'(0), s''(0)
        this->cp_pos_[0] = s0;
        this->cp_pos_[1] = s0 + v0 * T / Scalar(5);
        this->cp_pos_[2] = s0 + Scalar(2) * v0 * T / Scalar(5) + a0 * T2 / Scalar(20);
        // End side:   c5, c4, c3 from s(T), s'(T), s''(T)
        this->cp_pos_[5] = sf;
        this->cp_pos_[4] = sf - vf * T / Scalar(5);
        this->cp_pos_[3] = sf - Scalar(2) * vf * T / Scalar(5) + af * T2 / Scalar(20);

        this->compute_derivative_chains();
    }
};


// ---------------------------------------------------------------------------
// Quartic (degree 4): free terminal position
//
// Boundary conditions (5):
//   s(0)=s0, s'(0)=v0, s''(0)=a0
//                s'(T)=vf,  s''(T)=af   (s(T) is unconstrained)
//
// Use for: velocity-keeping / cruising trajectories.
// ---------------------------------------------------------------------------
template<typename Scalar = double>
class QuarticPolynomial_Bernstein : public BernsteinPolynomialBase<4, Scalar> {
    using Base = BernsteinPolynomialBase<4, Scalar>;
public:
    QuarticPolynomial_Bernstein(
        Scalar s0, Scalar v0, Scalar a0,
        Scalar vf, Scalar af,
        Scalar T)
    {
        if (T <= Scalar(0))
            throw std::invalid_argument("T must be positive");

        this->T_     = T;
        this->inv_T_ = Scalar(1) / T;

        const Scalar T2 = T * T;

        // Start side: c0, c1, c2 from s(0), s'(0), s''(0)
        const Scalar c2 = s0 + v0 * T / Scalar(2) + a0 * T2 / Scalar(12);
        this->cp_pos_[0] = s0;
        this->cp_pos_[1] = s0 + v0 * T / Scalar(4);
        this->cp_pos_[2] = c2;
        // End side: c3, c4 from s'(T)=vf, s''(T)=af (both expressed relative to c2)
        this->cp_pos_[3] = c2 + vf * T / Scalar(4) - af * T2 / Scalar(12);
        this->cp_pos_[4] = c2 + vf * T / Scalar(2) - af * T2 / Scalar(12);

        this->compute_derivative_chains();
    }
};
