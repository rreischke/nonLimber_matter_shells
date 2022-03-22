#include "Levin_power.h"
#include <fstream>

const double Levin_power::min_interval = 1.e-2;
const double Levin_power::limber_tolerance = 1.0e-2;
const double Levin_power::tol_abs = 1.0e-30;
const double Levin_power::tol_rel = 1.0e-7;
const double Levin_power::min_sv = 1.0e-10;

Levin_power::Levin_power(bool precompute1, uint number_count, std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk, bool boxy1)
{
    if (kernel.size() != chi_cl.size())
    {
        throw std::range_error("kernel dimension does not match size of chi_cl");
    }
    if (pk.size() != z_pk.size() * k_pk.size())
    {
        throw std::range_error("Pk_nl dimension does not match sizes of z_pk and k_pk");
    }
    d = 2;
    precompute = precompute1;
    n_total = kernel.at(0).size();
    number_counts = number_count;
    boxy = boxy1;
    init_splines(z_bg, chi_bg, chi_cl, kernel, k_pk, z_pk, pk);
    set_pointer();
    init_Bessel();
}

Levin_power::~Levin_power()
{
    delete integration_variable_i_tomo;
    delete integration_variable_j_tomo;
    delete integration_variable_Limber_ell;
    delete integration_variable_Limber_i_tomo;
    delete integration_variable_Limber_j_tomo;
    delete integration_variable_extended_Limber_ell;
    delete integration_variable_extended_Limber_i_tomo;
    delete integration_variable_extended_Limber_j_tomo;
    delete integration_variable_Limber_linear;
    gsl_spline_free(spline_z_of_chi);
    gsl_interp_accel_free(acc_z_of_chi);
    gsl_spline_free(spline_chi_of_z);
    gsl_interp_accel_free(acc_chi_of_z);
    for (uint i = 0; i < n_total; i++)
    {
        gsl_spline_free(spline_Weight.at(i));
        gsl_interp_accel_free(acc_Weight.at(i));
    }
    gsl_spline2d_free(spline_d2P_d2k);
    gsl_interp_accel_free(acc_d2P_d2k_k);
    gsl_interp_accel_free(acc_d2P_d2k_z);
    gsl_spline2d_free(spline_P_nl);
    gsl_interp_accel_free(acc_P_nl_k);
    gsl_interp_accel_free(acc_P_nl_z);
    for (uint i = 0; i < n_total; i++)
    {
        gsl_spline_free(spline_aux_kernel.at(i));
        gsl_interp_accel_free(acc_aux_kernel.at(i));
    }
    for (uint i = 0; i < n_total * n_total; i++)
    {
        gsl_spline_free(spline_result.at(i));
        gsl_interp_accel_free(acc_result.at(i));
    }
}

void Levin_power::init_Bessel()
{
    if (!bessel_set)
    {
        for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
        {
            k_min_bessel.push_back(std::vector<double>());
            k_max_bessel.push_back(std::vector<double>());
            k_bessel.push_back(std::vector<std::vector<double>>());
            for (uint l = 0; l <= ellmax_non_limber; l++)
            {
                k_bessel.at(i_tomo).push_back(std::vector<double>(N_interp));
                double value_min, value_max;
                value_min = GSL_MAX(0.5 * (l + 0.5) / kernel_maximum.at(i_tomo), k_min);
                value_max = GSL_MIN(8.0 * (l + 0.5) / kernel_maximum.at(i_tomo), k_max);
                if (l <= 5)
                {
                    value_max *= 3.0;
                }
                if (l >= 86)
                {
                    value_max *= 0.6;
                }
                if (i_tomo >= number_counts)
                {
                    value_min = GSL_MAX(0.02 * (l + 0.5) / kernel_maximum.at(i_tomo), k_min);
                    value_max = GSL_MIN(100.0 * (l + 0.5) / kernel_maximum.at(i_tomo), k_max);
                }
                k_min_bessel.at(i_tomo).push_back(value_min);
                k_max_bessel.at(i_tomo).push_back(value_max);
                k_bessel.at(i_tomo).at(l) = linear_spaced(log(k_min_bessel.at(i_tomo).at(l)), log(k_max_bessel.at(i_tomo).at(l)), N_interp);
            }
        }
        chi_size = static_cast<uint>(pow(2.0, maximum_number_subintervals + 1.0) + 1);
        std::vector<double> index_chi(chi_size);
        for (uint i_chi = 0; i_chi < chi_size; i_chi++)
        {
            chi_nodes.push_back(chi_min + (chi_max - chi_min) / (chi_size - 1.0) * i_chi);
            index_chi.at(i_chi) = i_chi;
        }
        gsl_set_error_handler_off();
        if (precompute)
        {
            for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
            {
                A_ell_bessel.push_back(std::vector<std::vector<double>>());
                A_ellm1_bessel.push_back(std::vector<std::vector<double>>());
                for (uint i_ell = 0; i_ell < N_linear_ell; i_ell++)
                {
                    uint l = aux_ell.at(i_ell);
                    A_ell_bessel.at(i_tomo).push_back(std::vector<double>(chi_size * N_interp));
                    A_ellm1_bessel.at(i_tomo).push_back(std::vector<double>(chi_size * N_interp));
#pragma omp parallel for
                    for (uint i_chi = 0; i_chi < chi_size; i_chi++)
                    {
                        for (uint i_k = 0; i_k < N_interp; i_k++)
                        {
                            uint flat_idx = i_chi * N_interp + i_k;

                            A_ell_bessel.at(i_tomo).at(i_ell).at(flat_idx) = gsl_sf_bessel_jl(l, chi_nodes.at(i_chi) * exp(k_bessel.at(i_tomo).at(l).at(i_k)));
                            A_ellm1_bessel.at(i_tomo).at(i_ell).at(flat_idx) = gsl_sf_bessel_jl(l - 1, chi_nodes.at(i_chi) * exp(k_bessel.at(i_tomo).at(l).at(i_k)));
                        }
                    }
                }
            }
        }
        for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
        {
            aux_kmax.push_back(0.0);
            aux_kmin.push_back(0.0);
        }
        for (uint i = 0; i < n_total; i++)
        {
            spline_aux_kernel.push_back(gsl_spline_alloc(gsl_interp_steffen, N_interp));
            acc_aux_kernel.push_back(gsl_interp_accel_alloc());
        }
        factor.push_back(1.0);
        for (uint i = 1; i < max_ell + 1; i++)
        {
            factor.push_back(sqrt((i + 2.0) * (i + 1.0) * i * (i - 1.0)));
        }
        for (uint i = 0; i < n_total; i++)
        {
            ell_eLimber.push_back(ellmax_non_limber);
        }
    }
    bessel_set = true;
}

void Levin_power::set_pointer()
{
    integration_variable_i_tomo = new uint[N_thread_max];
    integration_variable_j_tomo = new uint[N_thread_max];
    integration_variable_Limber_i_tomo = new uint[N_thread_max];
    integration_variable_Limber_j_tomo = new uint[N_thread_max];
    integration_variable_Limber_ell = new uint[N_thread_max];
    integration_variable_extended_Limber_i_tomo = new uint[N_thread_max];
    integration_variable_extended_Limber_j_tomo = new uint[N_thread_max];
    integration_variable_extended_Limber_ell = new uint[N_thread_max];
    integration_variable_Limber_linear = new bool[N_thread_max];
}

uint Levin_power::find_kernel_maximum(std::vector<double> kernel)
{
    uint result = 0.0;
    for (uint i = 0; i < kernel.size(); i++)
    {
        if (kernel.at(i + 1) < kernel.at(i))
        {
            result = i;
            break;
        }
    }
    return result;
}

void Levin_power::init_splines(std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk)
{
    if (!bessel_set)
    {
        for (uint i_ell = 0; i_ell < N_linear_ell; i_ell++)
        {
            uint ell = static_cast<uint>(min_ell + (ellmax_non_limber - min_ell) / (N_linear_ell - 1) * i_ell);
            aux_ell.push_back(ell);
        }
        for (uint i_ell = 1; i_ell < N_log_ell; i_ell++)
        {
            uint ell = static_cast<uint>(exp(log(ellmax_non_limber) + (log(max_ell) - log(ellmax_non_limber)) / (N_log_ell - 1) * i_ell));
            aux_ell.push_back(ell);
        }
        for (uint i = 0; i < n_total * n_total; i++)
        {
            spline_result.push_back(gsl_spline_alloc(gsl_interp_steffen, aux_ell.size()));
            acc_result.push_back(gsl_interp_accel_alloc());
        }
    }

    spline_chi_of_z = gsl_spline_alloc(gsl_interp_steffen, z_bg.size());
    spline_z_of_chi = gsl_spline_alloc(gsl_interp_steffen, z_bg.size());
    acc_chi_of_z = gsl_interp_accel_alloc();
    acc_z_of_chi = gsl_interp_accel_alloc();
    gsl_spline_init(spline_chi_of_z, &z_bg[0], &chi_bg[0], z_bg.size());
    gsl_spline_init(spline_z_of_chi, &chi_bg[0], &z_bg[0], z_bg.size());
    for (uint i_tomo = 0; i_tomo < kernel.at(0).size(); i_tomo++)
    {
        if (!bessel_set)
        {
            spline_Weight.push_back(gsl_spline_alloc(gsl_interp_cspline, chi_cl.size()));
            acc_Weight.push_back(gsl_interp_accel_alloc());
        }
        if (boxy && i_tomo < number_counts)
        {
            s_srd.push_back(0.0);
            chi0_srd.push_back(0.0);
            norm_srd.push_back(1.0);
        }
        double xmin = 0.0;
        double xmax = 0.0;
        double width = 0.0;
        std::vector<double> init_weight(chi_cl.size(), 0.0);
        for (uint i = 0; i < chi_cl.size(); i++)
        {
            init_weight.at(i) = kernel.at(i).at(i_tomo);
            if (boxy && i_tomo < number_counts && i == 0)
            {
                bool isweight = false;
                for (uint j = 0; j < chi_cl.size(); j++)
                {
                    if (kernel.at(j).at(i_tomo) > 0.0 && isweight == false)
                    {
                        xmin = chi_cl.at(j);
                        isweight = true;
                    }
                    if (kernel.at(j).at(i_tomo) == 0.0 && isweight == true)
                    {
                        xmax = chi_cl.at(j);
                        break;
                    }
                }
            }
            if (boxy && i_tomo < number_counts)
            {
                width = xmax - xmin;
                s_srd.at(i_tomo) = width / 2.0;
                chi0_srd.at(i_tomo) = xmin + s_srd.at(i_tomo);
                init_weight.at(i) = gsl_spline_eval_deriv(spline_z_of_chi, chi_cl.at(i), acc_z_of_chi) * super_gaussian(chi_cl.at(i), chi0_srd.at(i_tomo), s_srd.at(i_tomo), i_tomo);
            }
        }
        if (boxy && i_tomo < number_counts)
        {
            double riemann_sum = 0.0;
            for (uint i = 0; i < chi_cl.size() - 1; i++)
            {
                riemann_sum += init_weight.at(i) * (chi_cl.at(i + 1) - chi_cl.at(i));
            }
            norm_srd.at(i_tomo) = 1.0 / riemann_sum;
            for (uint i = 0; i < chi_cl.size() - 1; i++)
            {
                init_weight.at(i) /= riemann_sum;
            }
        }
        if (!bessel_set)
        {
            if(boxy && i_tomo < number_counts)
            {
                kernel_maximum.push_back(chi0_srd.at(i_tomo));
            }
            else
            {
                kernel_maximum.push_back(chi_cl.at(find_kernel_maximum(init_weight)));
            }
        }
        gsl_spline_init(spline_Weight.at(i_tomo), &chi_cl[0], &init_weight[0], chi_cl.size());
    }
    if (!bessel_set)
    {
        chi_min = chi_cl.at(1);
        chi_max = chi_cl.at(chi_cl.size() - 2);
    }
    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    acc_P_nl_k = gsl_interp_accel_alloc();
    acc_P_nl_z = gsl_interp_accel_alloc();
    acc_d2P_d2k_k = gsl_interp_accel_alloc();
    acc_d2P_d2k_z = gsl_interp_accel_alloc();
    spline_P_nl = gsl_spline2d_alloc(T, k_pk.size(), z_pk.size());
    spline_d2P_d2k = gsl_spline2d_alloc(T, k_pk.size(), z_pk.size());
    if (!bessel_set)
    {
        k_min = k_pk.at(1);
        k_max = k_pk.at(k_pk.size() - 2);
    }
    for (uint i = 0; i < k_pk.size(); i++)
    {
        k_pk.at(i) = log(k_pk.at(i));
        for (uint j = 0; j < z_pk.size(); j++)
        {
            pk.at(i * z_pk.size() + j) = log(pk.at(i * z_pk.size() + j));
        }
    }
    gsl_spline2d_init(spline_P_nl, &k_pk[0], &z_pk[0], &pk[0], k_pk.size(), z_pk.size());
    std::vector<double> init_d2P_d2k(k_pk.size() * z_pk.size());
    for (uint i = 0; i < k_pk.size(); i++)
    {
        for (uint j = 0; j < z_pk.size(); j++)
        {
            init_d2P_d2k.at(j * z_pk.size() + i) = d2P_d2k(exp(k_pk.at(i)), z_pk.at(j));
        }
    }
    gsl_spline2d_init(spline_d2P_d2k, &k_pk[0], &z_pk[0], &init_d2P_d2k[0], k_pk.size(), z_pk.size());
}

double Levin_power::kernels(double chi, uint i_tomo)
{

    return gsl_spline_eval(spline_Weight.at(i_tomo), chi, acc_Weight.at(i_tomo));
}

double Levin_power::super_gaussian(double x, double x0, double s, uint i_tomo)
{
    return norm_srd.at(i_tomo) * exp(-pow((x - x0) / s, n_super));
}

double Levin_power::chi_of_z(double z)
{
    return gsl_spline_eval(spline_chi_of_z, z, acc_chi_of_z);
}

double Levin_power::z_of_chi(double chi)
{
    return gsl_spline_eval(spline_z_of_chi, chi, acc_z_of_chi);
}

double Levin_power::power_nonlinear(double z, double k)
{
    if (k >= k_min && k <= k_max)
    {
        return exp(gsl_spline2d_eval(spline_P_nl, log(k), z, acc_P_nl_k, acc_P_nl_z));
    }
    else
    {
        if (k <= k_max)
        {
            return exp(gsl_spline2d_eval_deriv_x(spline_P_nl, log(k_min), z, acc_P_nl_k, acc_P_nl_z) * (log(k) - log(k_min)) + gsl_spline2d_eval(spline_P_nl, log(k_min), z, acc_P_nl_k, acc_P_nl_z));
        }
        else
        {
            return exp(gsl_spline2d_eval_deriv_x(spline_P_nl, log(k_max), z, acc_P_nl_k, acc_P_nl_z) * (log(k) - log(k_max)) + gsl_spline2d_eval(spline_P_nl, log(k_max), z, acc_P_nl_k, acc_P_nl_z));
        }
    }
}

double Levin_power::w(double chi, double k, uint ell, uint i, bool strict)
{
    if (!strict)
    {
        switch (i)
        {
        case 0:
            return gsl_sf_bessel_jl(ell, chi * k);
        case 1:
            return gsl_sf_bessel_jl(ell - 1, chi * k);
        default:
            return 0.0;
        }
    }
    else
    {
        gsl_sf_result r;
        int status;
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_jl_e(ell, chi * k, &r);
            break;
        case 1:
            status = gsl_sf_bessel_jl_e(ell - 1, chi * k, &r);
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute spherical Bessel function for ell=" << ell - i << std::endl;
        }
        return r.val;
    }
}

double Levin_power::w_precomputed(uint i_chi, uint i_k, uint ell, uint i, uint i_tomo)
{
    uint ell_idx = getIndex(aux_ell, ell);
    switch (i)
    {
    case 0:
        return A_ell_bessel.at(i_tomo).at(ell_idx).at(i_chi * N_interp + i_k);
    case 1:
        return A_ellm1_bessel.at(i_tomo).at(ell_idx).at(i_chi * N_interp + i_k);
    default:
        return 0.0;
    }
}

double Levin_power::A_matrix(uint i, uint j, double chi, double k, uint ell)
{
    if (i == 0 && j == 0)
    {
        return -(ell + 1.0) / chi;
    }
    if (i * j == 1)
    {
        return (ell - 1.0) / chi;
    }
    if (i < j)
    {
        return k;
    }
    else
    {
        return -k;
    }
}

std::vector<double> Levin_power::setNodes(double A, double B, uint col)
{
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    for (uint j = 0; j < n; j++)
    {
        x_j[j] = A + j * (B - A) / (n - 1);
    }
    return x_j;
}

double Levin_power::basis_function(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 1.0;
    }
    return pow((x - (A + B) / 2) / (B - A), m);
}

double Levin_power::basis_function_prime(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 0.0;
    }
    if (m == 1)
    {
        return 1.0 / (B - A);
    }
    return m / (B - A) * pow((x - (A + B) / 2.) / (B - A), (m - 1));
}

double Levin_power::F_nonlinear(double chi, uint i_tomo, double k)
{
    double z = z_of_chi(chi);
    if (i_tomo < number_counts)
    {
        return sqrt(power_nonlinear(z, k)) * kernels(chi, i_tomo) * k;
    }
    else
    {
        return sqrt(power_nonlinear(z, k)) * kernels(chi, i_tomo) / (gsl_pow_2(chi) * k);
    }
}

std::vector<double> Levin_power::solve_LSE(double A, double B, uint col, std::vector<double> x_j, uint i_tomo, double k, uint ell)
{
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, F_nonlinear(x_j[j], i_tomo, k));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix(q, i, x_j[j], k, ell) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s;
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    int lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_permutation_free(P);
    gsl_set_error_handler(old_handler);
    gsl_matrix_free(matrix_G);
    return result;
}

double Levin_power::p(double A, double B, uint i, double x, uint col, std::vector<double> c)
{
    uint n = (col + 1) / 2;
    n *= 2;
    double result = 0.0;
    for (uint m = 0; m < n; m++)
    {
        result += c[i * n + m] * basis_function(A, B, x, m);
    }
    return result;
}

double Levin_power::integrate(uint iA, uint iB, uint col, uint i_tomo, uint i_k, uint ell)
{
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n);
    double A = chi_nodes.at(iA);
    double B = chi_nodes.at(iB);
    double k = exp(k_bessel.at(i_tomo).at(ell).at(i_k));
    x_j = setNodes(A, B, col);
    c = solve_LSE(A, B, col, x_j, i_tomo, k, ell);
    for (uint i = 0; i < d; i++)
    {
        if (precompute)
        {
            result += p(A, B, i, B, col, c) * w_precomputed(iB, i_k, ell, i, i_tomo) - p(A, B, i, A, col, c) * w_precomputed(iA, i_k, ell, i, i_tomo);
        }
        else
        {
            result += p(A, B, i, B, col, c) * w(B, k, ell, i) - p(A, B, i, A, col, c) * w(A, k, ell, i);
        }
    }
    return result;
}

double Levin_power::iterate(double A, double B, uint col, uint i_tomo, uint i_k, uint ell, uint smax, bool verbose)
{
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    uint iA = map_chi_index(A);
    uint iB = map_chi_index(B);
    double I_half = integrate(iA, iB, col / 2, i_tomo, i_k, ell);
    double I_full = integrate(iA, iB, col, i_tomo, i_k, ell);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        /*if (verbose) // print borders, approximations and error estimates for all subintervals
        {
            std::cerr << "estimate: " << std::scientific << result << std::endl
                      << sub << " subintervals: " << std::endl;
            for (uint i = 0; i < approximations.size(); ++i)
            {
                std::cerr << "[" << x_sub[i] << "," << x_sub[i + 1] << "]: " << approximations[i] << " (" << error_estimates[i] << ")" << std::endl;
                std::cerr << std::endl;
            }
        }*/
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs))
        {
            if (verbose)
            {
                std::cerr << "converged!" << std::endl;
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = findMax(error_estimates) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection!" << std::endl;
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        uint x_subim1_i = map_chi_index(x_sub.at(i - 1));
        uint x_subi_i = map_chi_index(x_sub.at(i));
        uint x_subip1_i = map_chi_index(x_sub.at(i + 1));
        I_half = integrate(x_subim1_i, x_subi_i, col / 2, i_tomo, i_k, ell);
        I_full = integrate(x_subim1_i, x_subi_i, col, i_tomo, i_k, ell);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate(x_subi_i, x_subip1_i, col / 2, i_tomo, i_k, ell);
        I_full = integrate(x_subi_i, x_subip1_i, col, i_tomo, i_k, ell);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of subintervals reached!" << std::endl;
    }
    return result;
}

uint Levin_power::findMax(const std::vector<double> vec)
{
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

std::vector<double> Levin_power::linear_spaced(double min, double max, uint N)
{
    std::vector<double> result(N);
    for (uint i = 0; i < N; i++)
    {
        result.at(i) = min + (max - min) / (N - 1.0) * i;
    }
    return result;
}

double Levin_power::levin_integrate_bessel(uint i_k, uint ell, uint i_tomo)
{
    uint n_col = 7;
    uint n_sub = maximum_number_subintervals;
    // gsl_set_error_handler_off();
    return iterate(chi_min, chi_max, n_col, i_tomo, i_k, ell, n_sub, false);
}

void Levin_power::set_auxillary_splines(uint ell)
{
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        std::vector<double> I_bessel_i_tomo(N_interp);
        if (ell < ell_eLimber.at(i_tomo))
        {
#pragma omp parallel for
            for (uint i = 0; i < N_interp; i++)
            {
                I_bessel_i_tomo.at(i) = levin_integrate_bessel(i, ell, i_tomo);
            }
            gsl_spline_init(spline_aux_kernel.at(i_tomo), &k_bessel.at(i_tomo).at(ell)[0], &I_bessel_i_tomo[0], N_interp);
            aux_kmax.at(i_tomo) = exp(k_bessel.at(i_tomo).at(ell).at(k_bessel.at(i_tomo).at(ell).size() - 1));
            aux_kmin.at(i_tomo) = exp(k_bessel.at(i_tomo).at(ell).at(0));
        }
    }
}

double Levin_power::Limber(uint ell, uint i_tomo, uint j_tomo)
{
    double fac = 1.0;
    if (i_tomo >= number_counts)
    {
        fac /= gsl_pow_2(ell + 0.5);
    }
    if (j_tomo >= number_counts)
    {
        fac /= gsl_pow_2(ell + 0.5);
    }
    if (ell < ell_limber)
    {
        return fac * extended_Limber(ell, i_tomo, j_tomo);
    }
    uint tid = omp_get_thread_num();
    integration_variable_Limber_ell[tid] = ell;
    integration_variable_Limber_i_tomo[tid] = i_tomo;
    integration_variable_Limber_j_tomo[tid] = j_tomo;
    double min = chi_min;
    double max = chi_max;
    return fac * gslIntegrateqag(Limber_kernel, min, max);
}

double Levin_power::Limber_kernel(double chi, void *p)
{
    uint tid = omp_get_thread_num();
    Levin_power *lp = static_cast<Levin_power *>(p);
    double weight_i_tomo = lp->kernels(chi, lp->integration_variable_Limber_i_tomo[tid]);
    double weight_j_tomo = lp->kernels(chi, lp->integration_variable_Limber_j_tomo[tid]);
    double k = (lp->integration_variable_Limber_ell[tid] + 0.5) / chi;
    double z = lp->z_of_chi(chi);
    double power = lp->power_nonlinear(z, k);
    return weight_i_tomo * weight_j_tomo / gsl_pow_2(chi) * power;
}

double Levin_power::auxillary_weight(uint i_tomo, double k)
{
    if (k <= aux_kmin.at(i_tomo))
    {
        double slope = gsl_spline_eval_deriv(spline_aux_kernel.at(i_tomo), log(aux_kmin.at(i_tomo)), acc_aux_kernel.at(i_tomo)) / gsl_spline_eval(spline_aux_kernel.at(i_tomo), log(aux_kmin.at(i_tomo)), acc_aux_kernel.at(i_tomo));
        double result = abs(slope) * (log(k) - log(aux_kmin.at(i_tomo))) + log(abs(gsl_spline_eval(spline_aux_kernel.at(i_tomo), log(aux_kmin.at(i_tomo)), acc_aux_kernel.at(i_tomo))));
        return exp(-abs(result));
    }
    if (k >= aux_kmax.at(i_tomo))
    {
        return 0.0;
    }
    else
    {
        return gsl_spline_eval(spline_aux_kernel.at(i_tomo), log(k), acc_aux_kernel.at(i_tomo));
    }
}

double Levin_power::k_integration_kernel(double k, void *p)
{
    uint tid = omp_get_thread_num();
    k = exp(k);
    Levin_power *lp = static_cast<Levin_power *>(p);
    return k * lp->auxillary_weight(lp->integration_variable_i_tomo[tid], k) * lp->auxillary_weight(lp->integration_variable_j_tomo[tid], k);
}

double Levin_power::dlnP_dlnk(double k, double z)
{
    return gsl_spline2d_eval_deriv_x(spline_P_nl, log(k), z, acc_P_nl_k, acc_P_nl_z);
}

double Levin_power::d2P_d2k(double k, double z)
{
    return (gsl_spline2d_eval_deriv_xx(spline_P_nl, log(k), z, acc_P_nl_k, acc_P_nl_z) + gsl_pow_2(dlnP_dlnk(k, z)) - dlnP_dlnk(k, z)) * power_nonlinear(z, k) / gsl_pow_2(k);
}

double Levin_power::d2P_d2k_interp(double k, double z)
{
    return gsl_spline2d_eval(spline_d2P_d2k, log(k), z, acc_d2P_d2k_k, acc_d2P_d2k_z);
}

double Levin_power::d3P_d3k(double k, double z)
{
    return gsl_spline2d_eval_deriv_x(spline_d2P_d2k, log(k), z, acc_d2P_d2k_k, acc_d2P_d2k_z) * k;
}

double Levin_power::dlnkernels_dlnchi(double chi, uint i_tomo)
{
    if (boxy && i_tomo < number_counts && chi > (chi0_srd.at(i_tomo) - s_srd.at(i_tomo)) && chi < (chi0_srd.at(i_tomo) + s_srd.at(i_tomo)))
    {
        //return (n_super * pow((-chi0_srd.at(i_tomo) + chi) / s_srd.at(i_tomo), n_super) / (-chi0_srd.at(i_tomo) - chi) + gsl_spline_eval_deriv2(spline_z_of_chi, chi, acc_z_of_chi) / gsl_spline_eval_deriv(spline_z_of_chi, chi, acc_z_of_chi)) * chi;
        return 2.0;
    }
    else
    {
        return gsl_spline_eval_deriv(spline_Weight.at(i_tomo), chi, acc_Weight.at(i_tomo)) / kernels(chi, i_tomo) * chi;
    }
}

double Levin_power::extended_limber_s(double k, double z)
{
    return dlnP_dlnk(k, z);
}

double Levin_power::extended_limber_p(double k, double z)
{
    return gsl_pow_2(k) * ((3.0 * d2P_d2k_interp(k, z) + k * d3P_d3k(k, z)) / (3.0 * power_nonlinear(z, k)));
}

double Levin_power::extended_Limber_kernel(double chi, void *p)
{
    chi = exp(chi);
    uint tid = omp_get_thread_num();
    Levin_power *lp = static_cast<Levin_power *>(p);
    double weight_i_tomo = lp->kernels(chi, lp->integration_variable_extended_Limber_i_tomo[tid]);
    double weight_j_tomo = lp->kernels(chi, lp->integration_variable_extended_Limber_j_tomo[tid]);
    double k = (lp->integration_variable_extended_Limber_ell[tid] + 0.5) / chi;
    double z = lp->z_of_chi(chi);
    double power = lp->power_nonlinear(z, k);
    double limber_part = weight_i_tomo * weight_j_tomo / chi * power;
    double dlnf_i = lp->dlnkernels_dlnchi(chi, lp->integration_variable_extended_Limber_i_tomo[tid]) - 0.5;
    double dlnf_j = lp->dlnkernels_dlnchi(chi, lp->integration_variable_extended_Limber_j_tomo[tid]) - 0.5;
    double correction = 0.5 / gsl_pow_2(lp->integration_variable_extended_Limber_ell[tid] + 0.5) * (dlnf_i * dlnf_j * lp->extended_limber_s(k, z) - lp->extended_limber_p(k, z));
    return limber_part * (1.0 + correction);
}

double Levin_power::extended_Limber(uint ell, uint i_tomo, uint j_tomo)
{
    uint tid = omp_get_thread_num();
    integration_variable_extended_Limber_ell[tid] = ell;
    integration_variable_extended_Limber_i_tomo[tid] = i_tomo;
    integration_variable_extended_Limber_j_tomo[tid] = j_tomo;
    double min = chi_min;
    double max = chi_max;
    return gslIntegratecquad(extended_Limber_kernel, log(min), log(max));
}

double Levin_power::C_ell_full(uint i_tomo, uint j_tomo)
{
    uint tid = omp_get_thread_num();
    integration_variable_i_tomo[tid] = i_tomo;
    integration_variable_j_tomo[tid] = j_tomo;
    double min = k_min;
    double max = (GSL_MAX(aux_kmax.at(i_tomo), aux_kmax.at(j_tomo)));
    return 2.0 / M_PI * gslIntegrateqag(k_integration_kernel, log(min), log(max));
}

std::vector<double> Levin_power::all_C_ell(std::vector<uint> ell)
{
    std::vector<double> result(ell.size() * n_total * n_total, 0.0);
    std::vector<std::vector<double>> aux_result(n_total * n_total, std::vector<double>(aux_ell.size(), 0.0));
    for (uint l = 0; l < aux_ell.size(); l++)
    {
        set_auxillary_splines(aux_ell.at(l));
        double factor1 = factor[aux_ell.at(l)];
#pragma omp parallel for
        for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
        {
            for (uint j_tomo = i_tomo; j_tomo < n_total; j_tomo++)
            {
                double facaux = 1.0;
                if (i_tomo >= number_counts)
                {
                    facaux *= factor1;
                }
                if (j_tomo >= number_counts)
                {
                    facaux *= factor1;
                }
                auto flat_idx = i_tomo * n_total + j_tomo;
                if ((aux_ell.at(l) < ell_eLimber.at(i_tomo)))
                {
                    aux_result.at(flat_idx).at(l) = facaux * C_ell_full(i_tomo, j_tomo);
                }
                else
                {
                    aux_result.at(flat_idx).at(l) = facaux * Limber(aux_ell.at(l), i_tomo, j_tomo);
                }
            }
        }
    }
#pragma omp parallel for
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        for (uint j_tomo = i_tomo; j_tomo < n_total; j_tomo++)
        {
            auto flat_idx = i_tomo * n_total + j_tomo;
            gsl_spline_init(spline_result.at(i_tomo * n_total + j_tomo), &aux_ell[0], &aux_result.at(flat_idx)[0], aux_ell.size());
        }
    }
    for (uint l = 0; l < ell.size(); l++)
    {
#pragma omp parallel for
        for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
        {
            for (uint j_tomo = i_tomo; j_tomo < n_total; j_tomo++)
            {
                auto flat_idx = i_tomo * n_total + j_tomo;
                result.at(l * n_total * n_total + flat_idx) = gsl_spline_eval(spline_result.at(flat_idx), ell.at(l), acc_result.at(flat_idx));
            }
        }
    }
    return result;
}

uint Levin_power::getIndex(std::vector<double> v, double val)
{
    uint index = 0;
    for (uint i = 0; i < v.size(); i++)
    {
        if (v.at(i) == val)
        {
            index = i;
            break;
        }
    }
    return index;
}

uint Levin_power::map_chi_index(double chi)
{
    return static_cast<uint>((chi - chi_min) / (chi_max - chi_min) * (chi_size - 1.0));
}

std::tuple<result_Cl_type, result_Cl_type, result_Cl_type> Levin_power::compute_C_ells(std::vector<uint> ell)
{
    int n_tomo_A = number_counts;
    result_Cl_type Cl_AA;
    result_Cl_type Cl_BB;
    result_Cl_type Cl_AB;
    auto tmp_Cl = std::vector<double>(ell.size());
    auto result = all_C_ell(ell);
    for (uint i_tomo = 0; i_tomo < n_total; i_tomo++)
    {
        for (uint j_tomo = i_tomo; j_tomo < n_total; j_tomo++)
        {
            auto flat_idx = i_tomo * n_total + j_tomo;
            for (uint l = 0; l < ell.size(); l++)
            {
                tmp_Cl.at(l) = result.at(l * n_total * n_total + flat_idx);
            }
            if (i_tomo < n_tomo_A && j_tomo < n_tomo_A)
            {
                Cl_AA.push_back(tmp_Cl);
            }
            else if (i_tomo >= n_tomo_A && j_tomo >= n_tomo_A)
            {
                Cl_BB.push_back(tmp_Cl);
            }
            else
            {
                Cl_AB.push_back(tmp_Cl);
            }
        }
    }

    return std::make_tuple(Cl_AA, Cl_AB, Cl_BB);
}

double Levin_power::gslIntegrateqag(double (*fc)(double, void *), double a, double b)
{

    double tiny = 0.0;
    double tol = 1.0e-3;
    gsl_function gf;
    gf.function = fc;
    gf.params = this;
    double e, y;
    const uint n = 128;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(n);
    gsl_integration_qag(&gf, a, b, tiny, tol, n, 1, w, &y, &e);
    gsl_integration_workspace_free(w);
    return y;
}

double Levin_power::gslIntegrateqng(double (*fc)(double, void *), double a, double b)
{

    double tiny = 0.0;
    double tol = 1.0e-4;
    gsl_function gf;
    gf.function = fc;
    gf.params = this;
    double e, y;
    size_t n;
    gsl_integration_qng(&gf, a, b, tiny, tol, &y, &e, &n);
    return y;
}

double Levin_power::gslIntegratecquad(double (*fc)(double, void *), double a, double b)
{
    double tiny = 0.0;
    double tol = 1.0e-3;
    gsl_function gf;
    gf.function = fc;
    gf.params = this;
    double e, y;
    const uint n = 64;
    gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(n);
    gsl_integration_cquad(&gf, a, b, tiny, tol, w, &y, &e, NULL);
    gsl_integration_cquad_workspace_free(w);
    return y;
}