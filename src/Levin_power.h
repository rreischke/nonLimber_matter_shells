#ifndef LEVIN_POWER_H
#define LEVIN_POWER_H

#include <vector>
#include <numeric>
#include <omp.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>

#include <cmath>
#include <thread>

typedef std::vector<std::vector<double>> result_Cl_type;

/**
 * This class implements the non-Limber power spectrum integral of galaxy clustering, cosmic shear and galaxy-galaxy lensing,
 * which is given by
 * \f[
 *  C_{ij}(\ell) = \frac{2}{\pi}\int k^2\mathrm{d}k I_i(k,\ell)I_j(k,\ell)\;
 * \f]
 * where \f$ I_i(k,\ell) \f$ is given by
 * \f[
 *  I_i(k,\ell) = \int\mathrm{d}\chi K_i (\chi) P(k,\chi)^{1/2} j_\ell(k\chi) \;.
 * \f]
 */

class Levin_power
{
private:
  static const double min_interval;
  static const double tol_abs;
  static const double tol_rel;
  static const double limber_tolerance;
  static const double min_sv;
  static const uint N_interp = 300;
  const double eLimber_rel = 1e-5;
  const uint ellmax_non_limber = 95;
  const uint maximum_number_subintervals = 12;
  const uint ell_limber = 1000;
  const uint min_ell = 10;
  const uint max_ell = 30000;
  const uint N_linear_ell = 20;
  const uint N_log_ell = 100;
  std::vector<double> aux_ell;
  const uint N_thread_max = std::thread::hardware_concurrency();

  std::vector<uint> ell_eLimber;

  gsl_spline2d *spline_P_nl;
  gsl_spline2d *spline_d2P_d2k;
  gsl_interp_accel *acc_P_nl_k;
  gsl_interp_accel *acc_P_nl_z;
  gsl_interp_accel *acc_d2P_d2k_k;
  gsl_interp_accel *acc_d2P_d2k_z;
  std::vector<gsl_interp_accel *> acc_Weight;
  std::vector<gsl_spline *> spline_Weight;
  std::vector<gsl_interp_accel *> acc_result;
  std::vector<gsl_spline *> spline_result;

  gsl_interp_accel *acc_chi_of_z;
  gsl_spline *spline_chi_of_z;
  gsl_interp_accel *acc_z_of_chi;
  gsl_spline *spline_z_of_chi;
  std::vector<double> kernel_maximum;

  std::vector<gsl_interp_accel *> acc_aux_kernel;
  std::vector<gsl_spline *> spline_aux_kernel;
  std::vector<double> aux_kmax;
  std::vector<double> aux_kmin;
  std::vector<double> kernel_norm;
  std::vector<double> table_kmin_fraction;
  std::vector<double> factor;
  std::vector<double> chi_nodes;
  std::vector<std::vector<std::vector<double>>> A_ell_bessel, A_ellm1_bessel;
  std::vector<std::vector<double>> k_min_bessel;
  std::vector<std::vector<double>> k_max_bessel;
  std::vector<std::vector<std::vector<double>>> k_bessel;
  uint d;
  uint n_total;
  uint number_counts;
  uint chi_size;
  double chi_min, chi_max;
  double k_min, k_max;
  bool bessel_set = false;
  bool precompute;

  bool boxy = false;
  uint n_super = 30;
  std::vector<double> chi0_srd, s_srd, norm_srd;

  uint *integration_variable_i_tomo, *integration_variable_j_tomo;

  uint *integration_variable_Limber_ell, *integration_variable_Limber_i_tomo, *integration_variable_Limber_j_tomo;
  uint *integration_variable_extended_Limber_ell, *integration_variable_extended_Limber_i_tomo, *integration_variable_extended_Limber_j_tomo;

  bool *integration_variable_Limber_linear;

  double gslIntegrateqag(double (*fc)(double, void *), double a, double b);
  double gslIntegrateqng(double (*fc)(double, void *), double a, double b);
  double gslIntegratecquad(double (*fc)(double, void *), double a, double b);

public:
  /**
   * Constructor: Takes as input a flag whether the Bessel functions should be precomputed, the number of tomographic bins for galaxy clustering and lists for the generation of tables.
   * In particular the background relation for the background between redshift \f$ z_\mathrm{bg} \f$ and comoving distance \f$ \chi_\mathrm{bg} \f$,
   * the weight function (kernel) in all tomographic bins (lensing plus clustering). The first index of "kernel" should run through the comoving distance,
   * while the second one runs over the tomographic bin. Lastly, we provide the linear and nonlinear (pk_l, pk_nl) matter power spectrum as a function of
   * wavenumber, k_pk, and redshift z_pk. The redshift is the fastly running index.
   *
   * Lengths in the code are expressed in \f$\mathrm{Mpc}\f$.
   */
  Levin_power(bool precompute1, uint number_count, std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk, bool boxy1);

  /**
   * Destructor: clean up all allocated memory.
   */
  ~Levin_power();

  void init_Bessel();

  void set_pointer();

  /**
   *  Finds the index of the maximum of a list with a global maximum.
   */
  uint find_kernel_maximum(std::vector<double> kernel);

  /**
   *  Initializes all splines for the distance redshift relation, the weight functions and the power spectrum.
   */
  void init_splines(std::vector<double> z_bg, std::vector<double> chi_bg, std::vector<double> chi_cl, std::vector<std::vector<double>> kernel, std::vector<double> k_pk, std::vector<double> z_pk, std::vector<double> pk);

  /**
   *  Interpolation function for the weight function in tomographic bin i_tomo and comoving distance chi.
   */
  double kernels(double chi, uint i_tomo);
  /**
   *  Interpolation function for the comoving distance as a function of redshift.
   */
  double chi_of_z(double z);

  /**
   *  Interpolation function for the redshift as a function of comoving distance.
   */
  double z_of_chi(double chi);

  double super_gaussian(double x, double x0, double s, uint i_tomo);


  /**
   *  Interpolation function for the linear power spectrum as a function of redshift and wavenumber.
   */
  double power_linear(double z, double k);

  /**
   *  Interpolation function for the non-linear power spectrum as a function of redshift and wavenumber.
   */
  double power_nonlinear(double z, double k);

  /**
   *  Define the vector \f$ w \f$ for the integration (see Levin) and returning the i-th component.
   */
  double w(double chi, double k, uint ell, uint i, bool strict = false);

  double w_precomputed(uint i_chi, uint i_k, uint ell, uint i, uint i_tomo);

  /**
   *  Define the matrix \f$w^\prime = A w \f$ for the integration (see Levin) and returning the i,j component.
   */
  double A_matrix(uint i, uint j, double chi, double k, uint ell);

  /**
   *  Setting the nodes at the col collocation points in the interval \f$ A,B \f$  (see Levin) and returning the nodes as a list.
   */
  std::vector<double> setNodes(double A, double B, uint col);

  /**
   * Returns the \f$m\f$-th basis function in the interval \f$ A,B \f$  at position \f$x\f$ (see Levin)
   */
  double basis_function(double A, double B, double x, uint m);

  /**
   * Returns the derivative of the \f$m\f$-th basis function in the interval \f$ A,B \f$  at position \f$x\f$.
   */
  double basis_function_prime(double A, double B, double x, uint m);

  /**
   * Returns the Bessel integration kernel modulo the spherical Bessel function using the linear power spectrum. This is needed
   * for the vector \f$ F \f$ in the Levin integration.
   */
  double F_linear(double chi, uint i_tomo, double k);

  /**
   * Returns the Bessel integration kernel modulo the spherical Bessel function using the nonlinear power spectrum. This is needed
   * for the vector \f$ F \f$ in the Levin integration.
   */
  double F_nonlinear(double chi, uint i_tomo, double k);

  /**
   * Solves the linear system of equations in the interval \f$ A,B \f$ at col collactation points with corresponding nodes x_j. The system is
   * specified by providing the tomographic bin, i_tomo, the wavenumber, k, multipole ell and whether the linear or nonlinear version of the
   * power spectrum should be used. The solution to the lse is returned as a list.
   **/
  std::vector<double> solve_LSE(double A, double B, uint col, std::vector<double> x_j, uint i_tomo, double k, uint ell);

  /**
   * Returns the \f$i \f$-th component of the vector \f$ p \f$ given the solution to the LSE, c.
   **/
  double p(double A, double B, uint i, double x, uint col, std::vector<double> c);

  /**
   * Integrates
   * \f[
   *  I_i(k,\ell) = \int\mathrm{d}\chi K_i (\chi) P(k,\chi)^{1/2} j_\ell(k\chi) \;.
   * \f]
   * in an interval \f$ A,B \f$ with col collocation points. The estimate of the integral is returned.
   **/
  double integrate(uint iA, uint iB, uint col, uint i_tomo, uint i_k, uint ell);

  /**
   * Iterates over the integral by bisectiong the interval with the largest error until convergence or a maximum number of bisections, smax, is reached.
   * The final result is returned.
   **/
  double iterate(double A, double B, uint col, uint i_tomo, uint i_k, uint ell, uint smax, bool verbose);

  /**
   *  Return the maximum index of a list.
   */
  uint findMax(const std::vector<double> vec);

  /**
   *  Creates a list linearly spaced between min and max with N points.
   */
  std::vector<double> linear_spaced(double min, double max, uint N);

  /**
   *  Integrates the Bessel integral
   * \f[
   *  I_i(k,\ell) = \int\mathrm{d}\chi K_i (\chi) P(k,\chi)^{1/2} j_\ell(k\chi) \;.
   * \f]
   * by iterating with the Levin method.
   */
  double levin_integrate_bessel(uint i_k, uint ell, uint i_tomo);

  /**
   * Calculates the Limber approximation to the angular ppower spectrum:
   * \f[
   *  C_{ij}(\ell) = \int \frac{\mathrm{d}\chi}{\chi} K_i (\chi)K_j (\chi) P((\ell+0.5)/\chi,\chi)\;
   * \f]
   */
  double Limber(uint ell, uint i_tomo, uint j_tomo);

  static double Limber_kernel(double, void *);

  /**
   * Sets splines for the integral
   * \f[
   *  I_i(k,\ell) = \int\mathrm{d}\chi K_i (\chi) P(k,\chi)^{1/2} j_\ell(k\chi) \;.
   * \f]
   * for a given multipole and in all tomographic bins and as a function of the waavenumber k. The interpolation is logarithmically
   * in wavenumbers.
   */
  void set_auxillary_splines(uint ell);

  /**
   * Returns the interpolation to the integral
   *\f[
   *    I_i(k) = \int\mathrm{d}\chi K_i (\chi) P(k,\chi)^{1/2} j_\ell(k\chi)
   *    \f]
   * The multipole ell is specified when setting the splines.
   */
  double auxillary_weight(uint i_tomo, double k);

  /**
   * Integration kernel for the final integral:
   * \f[
   *  C_{ij}(\ell) = \frac{2}{\pi}\int k^2\mathrm{d}k I_i(k,\ell)I_j(k,\ell)\;
   * \f]
   */
  static double k_integration_kernel(double, void *);

  double dlnP_dlnk(double k, double z);

  double d2P_d2k(double k, double z);

  double d2P_d2k_interp(double k, double z);

  double d3P_d3k(double k, double z);

  double dlnkernels_dlnchi(double chi, uint i_tomo);

  double extended_limber_s(double k, double z);

  double extended_limber_p(double k, double z);

  static double extended_Limber_kernel(double, void *);

  double extended_Limber(uint ell, uint i_tomo, uint j_tomo);

  /**
   * Returns \f$ C_{ij}(\ell)\f $. Note that the splines for the weights have to be set. Do NOT call this function in the main.
   */
  double C_ell_full(uint i_tomo, uint j_tomo);

  uint getIndex(std::vector<double> v, double val);

  uint map_chi_index(double chi);

  /**
   * Returns the spectra for all tomographic bin combinations (i<j) for a list of multipoles. The final result is of the following shape:
   * (l * n_total * n_total + i), where l is the index of the ell list, n_total is the number of tomographic bins and i = i_tomo*n_total + j_tomo.
   */
  std::vector<double> all_C_ell(std::vector<uint> ell);

  std::tuple<result_Cl_type, result_Cl_type, result_Cl_type> compute_C_ells(std::vector<uint> ell);
};

#endif