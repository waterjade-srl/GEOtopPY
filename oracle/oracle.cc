/* extern "C" surface over GEOtop v3.0, for use as a numerical oracle.
 *
 * Every wrapper below forwards to the real GEOtop function, compiled from the
 * unmodified sources in the reference checkout. Nothing is reimplemented here:
 * the wrappers only translate between C types and GEOtop's containers, so that
 * ctypes can reach them from Python.
 *
 * Naming: the GEOtop name with a `gt_` prefix.
 *
 * When a GEOtop function takes Vector/Matrix/MatrixView, the wrapper takes flat
 * double arrays and builds the container around them. GEOtop's containers are
 * 1-based, so an array of length `n` is exposed as bounds [1, n] and the caller
 * passes GEOtop's own indices unchanged -- the conversion is the caller's, made
 * explicit at this boundary rather than smeared through the Python side.
 */

#include "constants.h"
#include "struct.geotop.h"
#include "pedo.funct.h"
#include "util_math.h"
#include "turbulence.h"
#include "meteo.h"
#include "snow.h"
#include "geomorphology.h"
#include "rw_maps.h"
#include "energy.balance.h"
#include "matrixview.h"
#include "matrix.h"
#include "vector.h"
#include "tabs.h"
#include "meteodata.h"
#include "meteodistr.h"
#include "times.h"
#include "input.h"
#include "parameters.h"
#include "sparse_matrix.h"
#include "clouds.h"
#include "radiation.h"
#include "tables.h"
#include "output.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cstring>

namespace {

/** Wrap a caller-owned array as a 1-based Vector<long>, copying in. */
Vector<long> as_vector_long(const long *src, long n)
{
    Vector<long> v{static_cast<std::size_t>(n)};
    for (long i = 1; i <= n; i++) v(i) = src[i - 1];
    return v;
}

/** Wrap a caller-owned array as a 1-based Vector, copying in. */
Vector<double> as_vector(const double *src, long n)
{
    Vector<double> v{static_cast<std::size_t>(n)};
    for (long i = 1; i <= n; i++) v(i) = src[i - 1];
    return v;
}

/** View a caller-owned row-major buffer as a 1-based MatrixView. */
MatrixView<double> as_matrix_view(double *buf, long nr, long nc)
{
    return MatrixView<double>{buf,
                              static_cast<std::size_t>(nr), 1,
                              static_cast<std::size_t>(nc), 1};
}

/** Build a GEOtop-style `double**` (row pointers into a flat copy) from a
 * caller-owned row-major buffer. The copy is owned by `storage`, which must
 * outlive the returned pointers. */
std::vector<double *> as_rows(const double *flat, long nrows, long ncols,
                              std::vector<double> &storage)
{
    storage.assign(flat, flat + static_cast<std::size_t>(nrows) * ncols);
    std::vector<double *> rows(nrows);
    for (long i = 0; i < nrows; i++) rows[i] = &storage[i * ncols];
    return rows;
}

}  // namespace

extern "C" {

/* ---------------------------------------------------------------- pedo.funct.h
 * Van Genuchten closure and the soil hydraulic laws. Header-only. */

double gt_psi_teta(double w, double i, double s, double r, double a, double n,
                   double m, double pmin, double Ss)
{ return psi_teta(w, i, s, r, a, n, m, pmin, Ss); }

double gt_teta_psi(double psi, double i, double s, double r, double a, double n,
                   double m, double pmin, double Ss)
{ return teta_psi(psi, i, s, r, a, n, m, pmin, Ss); }

double gt_dteta_dpsi(double psi, double i, double s, double r, double a,
                     double n, double m, double pmin, double Ss)
{ return dteta_dpsi(psi, i, s, r, a, n, m, pmin, Ss); }

double gt_k_hydr_soil(double psi, double ksat, double imp, double i, double s,
                      double r, double a, double n, double m, double v,
                      double T, double ratio)
{ return k_hydr_soil(psi, ksat, imp, i, s, r, a, n, m, v, T, ratio); }

double gt_psi_saturation(double i, double s, double r, double a, double n,
                         double m)
{ return psi_saturation(i, s, r, a, n, m); }

double gt_Harmonic_Mean(double D1, double D2, double K1, double K2)
{ return Harmonic_Mean(D1, D2, K1, K2); }

double gt_Arithmetic_Mean(double D1, double D2, double K1, double K2)
{ return Arithmetic_Mean(D1, D2, K1, K2); }

double gt_Mean(short a, double D1, double D2, double K1, double K2)
{ return Mean(a, D1, D2, K1, K2); }

double gt_Psif(double T)
{ return Psif(T); }

/* The pa-taking wrappers: `pa` is the soil parameter matrix for one soil type,
 * row-major with GEOtop's own row indices (jdz, jsat, ...), so nr must cover
 * the largest index used and nc the number of layers. */

double gt_theta_from_psi(double psi, double ice, long l, double *pa,
                         long nr, long nc, double pmin)
{ return theta_from_psi(psi, ice, l, as_matrix_view(pa, nr, nc), pmin); }

double gt_psi_from_theta(double th, double ice, long l, double *pa,
                         long nr, long nc, double pmin)
{ return psi_from_theta(th, ice, l, as_matrix_view(pa, nr, nc), pmin); }

double gt_dtheta_dpsi_from_psi(double psi, double ice, long l, double *pa,
                               long nr, long nc, double pmin)
{ return dtheta_dpsi_from_psi(psi, ice, l, as_matrix_view(pa, nr, nc), pmin); }

double gt_k_from_psi(long jK, double psi, double ice, double T, long l,
                     double *pa, long nr, long nc, double imp, double ratio)
{ return k_from_psi(jK, psi, ice, T, l, as_matrix_view(pa, nr, nc), imp, ratio); }

double gt_psisat_from(double ice, long l, double *pa, long nr, long nc)
{ return psisat_from(ice, l, as_matrix_view(pa, nr, nc)); }

/* -------------------------------------------------------------- turbulence.cc
 * find_actual_evaporation_parameters: R/C are unused inside the real
 * function (only ever logged), so the wrapper drops them. `n` is both
 * theta's/T's length and the number of layers evap_layer is allocated at
 * (the real function reads n = evap_layer->nh, so the caller's allocation
 * IS what selects how many layers get processed -- geotop_py's `nlayers`). */
void gt_find_actual_evaporation_parameters(
    long n, const double *theta, const double *T, const double *pa,
    long nr, long nc, double psi, double P, double rv, double Ta, double Qa,
    double Qgsat, long nsnow, double *alpha_out, double *beta_out,
    double *evap_out)
{
    Vector<double> vtheta = as_vector(theta, n);
    Vector<double> vT = as_vector(T, n);
    Vector<double> evap_layer{static_cast<std::size_t>(n)};
    double alpha = 0.0, beta = 0.0;
    find_actual_evaporation_parameters(
        0, 0, &alpha, &beta, &evap_layer, vtheta,
        as_matrix_view(const_cast<double *>(pa), nr, nc), vT, psi, P, rv, Ta,
        Qa, Qgsat, nsnow);
    *alpha_out = alpha;
    *beta_out = beta;
    for (long i = 1; i <= n; i++) evap_out[i - 1] = evap_layer(i);
}

/* Businger: the Monin-Obukhov iteration aero_resistance calls for
 * state_turb==1 (the only branch geotop_py's aero_resistance implements --
 * state_turb==0 is a fatal config error and ==2 is the catabatic-flow
 * branch, neither reachable from the 1D reference cases). */
void gt_businger(short a, double zmu, double zmt, double d0, double z0,
                 double v, double T, double DT, double DQ, double z0_z0t,
                 long maxiter, double *rm_out, double *rh_out, double *rv_out,
                 double *lobukhov_out)
{
    double rm = 0.0, rh = 0.0, rv = 0.0, L = 0.0;
    Businger(a, zmu, zmt, d0, z0, v, T, DT, DQ, z0_z0t, &rm, &rh, &rv, &L,
             maxiter);
    *rm_out = rm;
    *rh_out = rh;
    *rv_out = rv;
    *lobukhov_out = L;
}

/* -------------------------------------------------------------------- tables.cc
 * find_activelayerdepth_up/dw and find_watertabledepth_up/dw all take the
 * full SOIL* struct plus a point index i and soil-type index ty, but only
 * ever read sl->pa (via ty), sl->SS->T, sl->th, sl->SS->thi and sl->Ptot
 * (via i) -- so the wrapper builds the minimal single-type, single-column
 * SOIL that exercises exactly those fields, i=1, ty=1 always. */
namespace {
Tensor<double> as_pa_tensor(const double *buf, long nr, long nc)
{
    Tensor<double> t{1, static_cast<std::size_t>(nr), static_cast<std::size_t>(nc)};
    for (long r = 1; r <= nr; r++)
        for (long c = 1; c <= nc; c++) t(1, r, c) = buf[(r - 1) * nc + (c - 1)];
    return t;
}

/* One-column Matrix<double> from a 1-based flat array (index 0 unused). */
Matrix<double> as_column_matrix(const double *src, long n)
{
    Matrix<double> m{static_cast<std::size_t>(n), static_cast<std::size_t>(1)};
    for (long l = 1; l <= n; l++) m(l, 1) = src[l];
    return m;
}
}  // namespace

/* T/th/thi/Ptot are all 1-based (index 0 unused, matching every array in
 * geotop_py) of length nc+1; pa is the same flat, 0-based-both-axes buffer
 * every other pa-taking wrapper here uses. */
double gt_find_activelayerdepth_up(const double *T, const double *th,
                                   const double *thi, const double *pa,
                                   long nr, long nc)
{
    SOIL sl;
    sl.pa.reset(new Tensor<double>(as_pa_tensor(pa, nr, nc)));
    sl.SS.reset(new SOIL_STATE(1, nc));
    sl.th.reset(new Matrix<double>(as_column_matrix(th, nc)));
    for (long l = 1; l <= nc; l++) {
        (*sl.SS->T)(l, 1) = T[l];
        (*sl.SS->thi)(l, 1) = thi[l];
    }
    return find_activelayerdepth_up(1, 1, &sl);
}

double gt_find_activelayerdepth_dw(const double *T, const double *th,
                                   const double *thi, const double *pa,
                                   long nr, long nc)
{
    SOIL sl;
    sl.pa.reset(new Tensor<double>(as_pa_tensor(pa, nr, nc)));
    sl.SS.reset(new SOIL_STATE(1, nc));
    sl.th.reset(new Matrix<double>(as_column_matrix(th, nc)));
    for (long l = 1; l <= nc; l++) {
        (*sl.SS->T)(l, 1) = T[l];
        (*sl.SS->thi)(l, 1) = thi[l];
    }
    return find_activelayerdepth_dw(1, 1, &sl);
}

double gt_find_watertabledepth_up(double Z, const double *Ptot,
                                  const double *pa, long nr, long nc)
{
    SOIL sl;
    sl.pa.reset(new Tensor<double>(as_pa_tensor(pa, nr, nc)));
    sl.Ptot.reset(new Matrix<double>(as_column_matrix(Ptot, nc)));
    return find_watertabledepth_up(Z, 1, 1, &sl);
}

double gt_find_watertabledepth_dw(double Z, const double *Ptot,
                                  const double *pa, long nr, long nc)
{
    SOIL sl;
    sl.pa.reset(new Tensor<double>(as_pa_tensor(pa, nr, nc)));
    sl.Ptot.reset(new Matrix<double>(as_column_matrix(Ptot, nc)));
    return find_watertabledepth_dw(Z, 1, 1, &sl);
}

/* dz is a plain 0-based buffer (length n, no unused slot). RowView's own
 * indexing (elem[j-ncl]) expects that layout directly -- unlike Vector,
 * whose co[] pads index 0 -- so this points straight at the caller's
 * buffer rather than going through as_vector/Vector, which would shift
 * every element off by one. */
long gt_nlayer(double D, const double *dz, long n, short d)
{
    RowView<double> row{const_cast<double *>(dz), static_cast<std::size_t>(n), 1};
    return nlayer(D, std::move(row), n, d);
}

/* interpolate_soil (output.cc, SoilPlotDepths). dz/q are 1-based flat
 * buffers of length max_l+1: dz[0] is always unused (physical layer
 * thicknesses have no index 0), q[0] is unused unless lmin==0 (only
 * Pzplot/psiz is allocated with a genuine surface column in the real
 * source -- input.cc:1077-1098 -- every other profile this driver writes
 * has lmin==1 and never reads q[0]). Building the RowView at `buf+lmin`
 * with ncl=lmin, rather than always at `buf+1` with ncl=1, is what makes
 * one formula cover both: q[l] lands at (buf+lmin)[l-lmin], which is
 * buf[l] either way. */
double gt_interpolate_soil(long lmin, double h, long max_l,
                           const double *dz, const double *q)
{
    RowView<double> dzv{const_cast<double *>(dz) + 1,
                        static_cast<std::size_t>(max_l), 1};
    RowView<double> qv{const_cast<double *>(q) + lmin,
                       static_cast<std::size_t>(max_l - lmin + 1),
                       static_cast<std::size_t>(lmin)};
    return interpolate_soil(lmin, h, max_l, std::move(dzv), std::move(qv));
}

/* ------------------------------------------------------------------ keywords
 * The two keyword tables live in keywords.h, which only parameters.cc includes;
 * the arrays are plain globals, so the oracle can reach them. Exported so the
 * Python side reads the names from the compiled model instead of transcribing
 * 400+ strings that would then silently drift. */

extern char *keywords_num[];
extern char *keywords_char[];
extern long number_novalue;
extern long number_absent;
extern char *string_novalue;

long gt_num_par_number(void) { return num_par_number; }
long gt_num_par_char(void)   { return num_par_char; }

const char *gt_keyword_num(long i)
{ return (i >= 0 && i < num_par_number) ? keywords_num[i] : nullptr; }

const char *gt_keyword_char(long i)
{ return (i >= 0 && i < num_par_char) ? keywords_char[i] : nullptr; }

}  // extern "C" -- reopened after the helpers below

/* ------------------------------------------------------------- inpts parsing
 * Exposes the *parse* half of read_inpts_par (parameters.cc:105-236): the same
 * readline_par tokenizer, the same case-insensitive keyword matching, the same
 * "absent -> novalue" filling. Stopping short of assign_numeric_parameters is
 * deliberate: this gives the raw table the C++ would work from, which is
 * exactly what the Python parser has to reproduce, with no interpretation
 * layered on top.
 *
 * The result is cached in a single static parse; gt_inpts_open replaces it.
 */

namespace {

struct InptsParse {
    std::vector<long> num_components;               // per numeric keyword
    std::vector<std::vector<double>> num_values;    // per numeric keyword
    std::vector<std::string> str_values;            // per string keyword
    bool loaded = false;
};

InptsParse g_inpts;
std::vector<std::vector<double>> g_meteo;
std::vector<std::vector<double>> g_table2;

std::string lowered(const char *s)
{
    std::string out{s};
    for (auto &ch : out) ch = static_cast<char>(tolower(static_cast<unsigned char>(ch)));
    return out;
}

}  // namespace

extern "C" {

/** Parse a geotop.inpts. Returns 0 on success, 1 if the file cannot be read. */
long gt_inpts_open(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) return 1;

    std::vector<long> key(max_charstring), str(max_charstring);
    std::vector<double> num(max_numvect);
    long keylength, stringlength, numberlength;
    short endoffile = 0, res;

    // name -> parsed payload, lower-cased exactly as the C++ does
    std::vector<std::pair<std::string, std::vector<double>>> nums;
    std::vector<std::pair<std::string, std::string>> strs;

    do
    {
        res = readline_par(f, 33, 61, 44, max_charstring, max_numvect,
                           key.data(), &keylength, str.data(), &stringlength,
                           num.data(), &numberlength, &endoffile);
        if (res == 1)
        {
            char *k = find_string(key.data(), keylength);
            convert_string_in_lower_case(k);
            nums.emplace_back(std::string{k},
                              std::vector<double>(num.begin(), num.begin() + numberlength));
            free(k);
        }
        else if (res == 2)
        {
            char *k = find_string(key.data(), keylength);
            convert_string_in_lower_case(k);
            /* find_string_int drops the first element (tabs.cc:411,
             * `string[i] = vector[i+1]`), which is the opening quote that
             * readline_par has to store so it can tell a string from a number
             * (`string[0] != 34`). Using find_string directly here keeps the
             * quote and loses the last real character -- the C++ pairs
             * find_string_int with stringlength = i-1 for exactly this reason. */
            long *raw = find_string_int(str.data(), stringlength);
            char *v = find_string(raw, stringlength);
            strs.emplace_back(std::string{k}, std::string{v});
            free(k);
            free(raw);
            free(v);
        }
    }
    while (endoffile == 0);
    fclose(f);

    g_inpts = InptsParse{};
    g_inpts.num_components.assign(num_par_number, 1);
    g_inpts.num_values.assign(num_par_number,
                              std::vector<double>{static_cast<double>(number_novalue)});
    for (long i = 0; i < num_par_number; i++)
    {
        std::string want = lowered(keywords_num[i]);
        for (auto &kv : nums)
        {
            if (kv.first == want)
            {
                g_inpts.num_components[i] = static_cast<long>(kv.second.size());
                g_inpts.num_values[i] = kv.second;
            }
        }
    }

    g_inpts.str_values.assign(num_par_char, std::string{string_novalue});
    for (long i = 0; i < num_par_char; i++)
    {
        std::string want = lowered(keywords_char[i]);
        for (auto &kv : strs)
            if (kv.first == want) g_inpts.str_values[i] = kv.second;
    }

    g_inpts.loaded = true;
    return 0;
}

/** Components parsed for numeric keyword `cod`; 1 (holding novalue) if absent. */
long gt_inpts_num_components(long cod)
{
    if (!g_inpts.loaded || cod < 0 || cod >= num_par_number) return -1;
    return g_inpts.num_components[cod];
}

/** Copy up to `max` values of numeric keyword `cod`; returns how many. */
long gt_inpts_num_values(long cod, double *out, long max)
{
    if (!g_inpts.loaded || cod < 0 || cod >= num_par_number) return -1;
    const auto &v = g_inpts.num_values[cod];
    long n = static_cast<long>(v.size());
    if (n > max) n = max;
    for (long i = 0; i < n; i++) out[i] = v[i];
    return n;
}

/** String keyword `cod`, or GEOtop's string_novalue when absent. */
const char *gt_inpts_string(long cod)
{
    if (!g_inpts.loaded || cod < 0 || cod >= num_par_char) return nullptr;
    return g_inpts.str_values[cod].c_str();
}

/* -------------------------------------------------------------- meteo files
 * The station meteo file as GEOtop actually works from it: read_txt_matrix
 * (header names mapped onto the nmet slots via the Header* keywords, missing
 * slots filled with number_absent) followed by the completion pipeline of
 * input.cc:384-476 -- fixing_dates, check_times, and the optional fill_* steps.
 *
 * rewrite_meteo_files is deliberately NOT called: it writes a completed copy
 * back next to the input, and the oracle must never write into the reference
 * checkout.
 *
 * Cloudiness (fill_meteo_data_with_cloudiness) is not applied here; it needs
 * the station horizon and the solar geometry parameters, and is exposed
 * separately once that path is ported.
 */

long gt_nmet(void) { return nmet; }

/** Load and complete one station file. Returns the number of lines, or -1. */
long gt_meteo_load(const char *filename,
                   const char *const *col_names, long ncolnames,
                   double ST, double STstat, double Zstat,
                   short wind_as_xy, short wind_as_dir,
                   short vap_as_Td, short vap_as_RH, double RHmin,
                   short prec_as_intensity)
{
    if (ncolnames != nmet) return -1;

    // read_txt_matrix takes char** and may write through it; copy the names.
    std::vector<std::string> owned;
    owned.reserve(ncolnames);
    for (long i = 0; i < ncolnames; i++) owned.emplace_back(col_names[i]);
    std::vector<char *> cols(ncolnames);
    for (long i = 0; i < ncolnames; i++) cols[i] = &owned[i][0];

    long nlines = 0;
    std::string name{filename};
    double **data = read_txt_matrix(&name[0], 33, 44, cols.data(), nmet, &nlines);
    if (!data) return -1;

    fixing_dates(1, data, ST, STstat, nlines, iDate12, iJDfrom0);
    check_times(1, data, nlines, iJDfrom0);

    if (wind_as_xy == 1)
        fill_wind_xy(data, nlines, iWs, iWdir, iWsx, iWsy,
                     cols[iWsx], cols[iWsy]);
    if (wind_as_dir == 1)
        fill_wind_dir(data, nlines, iWs, iWdir, iWsx, iWsy,
                      cols[iWs], cols[iWdir]);

    Vector<double> Z{1};
    Z(1) = Zstat;
    if (vap_as_Td == 1)
        fill_Tdew(1, &Z, data, nlines, iRh, iT, iTdew, cols[iTdew], RHmin);
    if (vap_as_RH == 1)
        fill_RH(1, &Z, data, nlines, iRh, iT, iTdew, cols[iRh]);
    if (prec_as_intensity == 1)
        fill_Pint(1, data, nlines, iPrec, iPrecInt, iJDfrom0, cols[iPrecInt]);

    // input.cc runs a second, oppositely-gated pass over exactly these three
    // (not fill_wind_dir, not fill_RH) after rewrite_meteo_files. Each fill_*
    // is a no-op unless its target column is still absent, so this only fires
    // where the first pass left something undone -- which, at the *_as_1
    // keywords' defaults (all 0 except wind_as_dir/vap_as_RH), is every one of
    // wind_as_xy, prec_as_intensity, vap_as_Td. Skipping it silently drops
    // Wx/Wy (and so the topo_mod_winds correction downstream) whenever a
    // station reports speed+direction rather than components.
    if (wind_as_xy != 1)
        fill_wind_xy(data, nlines, iWs, iWdir, iWsx, iWsy, cols[iWsx], cols[iWsy]);
    if (prec_as_intensity != 1)
        fill_Pint(1, data, nlines, iPrec, iPrecInt, iJDfrom0, cols[iPrecInt]);
    if (vap_as_Td != 1)
        fill_Tdew(1, &Z, data, nlines, iRh, iT, iTdew, cols[iTdew], RHmin);

    g_meteo.assign(nlines, std::vector<double>(nmet, 0.0));
    for (long i = 0; i < nlines; i++)
    {
        for (long j = 0; j < nmet; j++) g_meteo[i][j] = data[i][j];
        free(data[i]);
    }
    free(data);
    return nlines;
}

/** Copy row `line` of the last loaded file; returns how many values. */
long gt_meteo_get(long line, double *out, long max)
{
    if (line < 0 || line >= static_cast<long>(g_meteo.size())) return -1;
    long n = static_cast<long>(g_meteo[line].size());
    if (n > max) n = max;
    for (long i = 0; i < n; i++) out[i] = g_meteo[line][i];
    return n;
}

/** The meteo column indices (constants.h:100-120), in slot order. */
void gt_meteo_indices(long *out)
{
    const unsigned int idx[] = {iDate12, iJDfrom0, iPrecInt, iPrec, iWs, iWdir,
                                iWsx, iWsy, iRh, iT, iTdew, iSW, iSWb, iSWd,
                                itauC, iC, iLWi, iSWn, iTs, iTbottom, nmet};
    for (unsigned i = 0; i < sizeof(idx) / sizeof(idx[0]); i++)
        out[i] = static_cast<long>(idx[i]);
}

/* ------------------------------------------------- positional tables (tabs.cc)
 * read_txt_matrix_2 is the by-position sibling of read_txt_matrix: the header
 * line is consumed and thrown away, and every line is read into a fixed number
 * of slots in file order -- short lines padded with number_novalue, long ones
 * truncated. It reads the time-dependent vegetation parameter file and the
 * timestep file.
 */

/** Read `filename` into `ncols` positional columns. Returns the line count. */
long gt_read_txt_matrix_2(const char *filename, long ncols)
{
    long nlines = 0;
    std::string name{filename};
    double **data = read_txt_matrix_2(&name[0], 33, 44, ncols, &nlines);
    if (!data) return -1;

    g_table2.assign(nlines, std::vector<double>(ncols, 0.0));
    for (long i = 0; i < nlines; i++)
    {
        for (long j = 0; j < ncols; j++) g_table2[i][j] = data[i][j];
        free(data[i]);
    }
    free(data);
    return nlines;
}

/** Copy row `line` of the last table read; returns how many values. */
long gt_read_txt_matrix_2_get(long line, double *out, long max)
{
    if (line < 0 || line >= static_cast<long>(g_table2.size())) return -1;
    long n = static_cast<long>(g_table2[line].size());
    if (n > max) n = max;
    for (long i = 0; i < n; i++) out[i] = g_table2[line][i];
    return n;
}

/* ------------------------------------------------------------ time interpolation
 * meteodata.cc. `data`/`var` are flat row-major [nlines*ncols] buffers with the
 * same row-0-marks-absent-columns convention as gt_meteo_load's output; `flag`
 * selects whether the date column holds DDMMYYYYhhmm (0) or already-converted
 * Julian days from year 0 (1).
 */

double gt_time_in_JDfrom0(short flag, long i, long col,
                          const double *data, long nlines, long ncols)
{
    std::vector<double> storage;
    auto rows = as_rows(data, nlines, ncols, storage);
    return time_in_JDfrom0(flag, i, col, rows.data());
}

/** Returns the found line index; writes GEOtop's 3-way flag (0 unreached,
 * 1 bracketed, 2 past the end, 3 short of `t`) to `*a`. */
long gt_find_line_data(short flag, double t, long ibeg,
                       const double *data, long nlines, long ncols,
                       long col_date, short *a)
{
    std::vector<double> storage;
    auto rows = as_rows(data, nlines, ncols, storage);
    return find_line_data(flag, t, ibeg, rows.data(), col_date, nlines, a);
}

double gt_integrate_meas_linear_beh(short flag, double t, long i,
                                    const double *data, long nlines, long ncols,
                                    long col, long col_date)
{
    std::vector<double> storage;
    auto rows = as_rows(data, nlines, ncols, storage);
    return integrate_meas_linear_beh(flag, t, i, rows.data(), col, col_date);
}

double gt_integrate_meas_constant_beh(short flag, double t, long i,
                                      const double *data, long nlines, long ncols,
                                      long col, long col_date)
{
    std::vector<double> storage;
    auto rows = as_rows(data, nlines, ncols, storage);
    return integrate_meas_constant_beh(flag, t, i, rows.data(), col, col_date);
}

/** Mean of every column over [tbeg, tend], trapezoidal between samples.
 * Writes `ncols` values to `out`; returns the updated `istart` search cursor. */
long gt_time_interp_linear(double t0, double tbeg, double tend,
                           const double *data, long nlines, long ncols,
                           long col_date, short flag, long istart,
                           double *out, long out_max)
{
    if (out_max < ncols) return -1;
    std::vector<double> storage;
    auto rows = as_rows(data, nlines, ncols, storage);
    time_interp_linear(t0, tbeg, tend, out, rows.data(), nlines, ncols,
                       col_date, flag, &istart);
    return istart;
}

/** Same as gt_time_interp_linear, but each sample holds over the interval that
 * follows it (a step function) instead of being interpolated. */
long gt_time_interp_constant(double t0, double tbeg, double tend,
                             const double *data, long nlines, long ncols,
                             long col_date, short flag, long istart,
                             double *out, long out_max)
{
    if (out_max < ncols) return -1;
    std::vector<double> storage;
    auto rows = as_rows(data, nlines, ncols, storage);
    time_interp_constant(t0, tbeg, tend, out, rows.data(), nlines, ncols,
                         col_date, flag, &istart);
    return istart;
}

/** No interpolation: the sample bracketing `tbeg` (+1e-5 day), read as-is. */
long gt_time_no_interp(short flag, long istart, const double *data,
                       long nlines, long ncols, long col_date, double tbeg,
                       double *out, long out_max)
{
    if (out_max < ncols) return -1;
    std::vector<double> storage;
    auto rows = as_rows(data, nlines, ncols, storage);
    time_no_interp(flag, &istart, out, rows.data(), nlines, ncols, col_date, tbeg);
    return istart;
}

/** First station row (0-based) whose `metvar` column is not number_absent. */
long gt_find_station(long metvar, long nstat, const double *var, long ncols)
{
    std::vector<double> storage;
    auto rows = as_rows(var, nstat, ncols, storage);
    return find_station(metvar, nstat, rows.data());
}

/* ------------------------------------------------------------- meteodistr.cc
 * PointSim=1 means every one of the 13 reference cases has exactly one
 * station and one grid cell, which is the trivial branch of
 * interpolate_meteo (nstn<=1: copy the value straight to the grid) --
 * barnes_oi's nstn>1 spatial interpolation is dead code for this project and
 * is not exposed. The grid itself is always 1x1 here; Xpoint/Ypoint's content
 * is irrelevant on the trivial path (only the loop bounds, both 1, are read).
 */

/** interpolate_meteo onto a 1x1 grid. `value` is nstn rows of `ncols`
 * columns (only column `metcod` is read). Writes the single grid value to
 * `*grid_out`; returns nstn (0 if the column is absent at every station). */
short gt_interpolate_meteo(short flag, double dX, double dY,
                           const double *xst, const double *yst, long nstn,
                           const double *value, long ncols, long metcod,
                           double dn0, short iobsint, double *grid_out)
{
    Matrix<double> Xpoint{1, 1}, Ypoint{1, 1}, grid{1, 1};
    Xpoint(1, 1) = 0.0;
    Ypoint(1, 1) = 0.0;

    // Xst/Yst's own size *is* Xst->nh, the total declared station count that
    // interpolate_meteo reads -- it must be exactly `nstn`, including 0, not
    // clamped to at least 1, or a run with no stations would look like one
    // station reporting 0.0.
    Vector<double> Xst{static_cast<std::size_t>(nstn)};
    Vector<double> Yst{static_cast<std::size_t>(nstn)};
    for (long i = 0; i < nstn; i++)
    {
        Xst(i + 1) = xst[i];
        Yst(i + 1) = yst[i];
    }

    std::vector<double> storage;
    auto rows = as_rows(value, nstn, ncols, storage);

    short ok = interpolate_meteo(flag, dX, dY, &Xpoint, &Ypoint, &Xst, &Yst,
                                 rows.data(), metcod, &grid, dn0, iobsint);
    *grid_out = grid(1, 1);
    return ok;
}

/** get_dn's single-cell scanning radius (meteodistr.cc). */
double gt_get_dn(long nc, long nr, double deltax, double deltay, long nstns)
{
    double dn = 0.0;
    get_dn(nc, nr, deltax, deltay, nstns, &dn);
    return dn;
}

/** topo_mod_winds on a single point: in/out winddir and windspd. */
void gt_topo_mod_winds(double *winddir, double *windspd,
                       double slopewtD, double curvewtD,
                       double slopewtI, double curvewtI,
                       double curvature1, double curvature2,
                       double curvature3, double curvature4,
                       double slope_az, double terrain_slope,
                       double topo, double undef)
{
    Matrix<double> wd{1, 1}, ws{1, 1};
    Matrix<double> c1{1, 1}, c2{1, 1}, c3{1, 1}, c4{1, 1};
    Matrix<double> saz{1, 1}, tslope{1, 1}, t{1, 1};
    wd(1, 1) = *winddir;
    ws(1, 1) = *windspd;
    c1(1, 1) = curvature1;
    c2(1, 1) = curvature2;
    c3(1, 1) = curvature3;
    c4(1, 1) = curvature4;
    saz(1, 1) = slope_az;
    tslope(1, 1) = terrain_slope;
    t(1, 1) = topo;

    topo_mod_winds(&wd, &ws, slopewtD, curvewtD, slopewtI, curvewtI,
                   &c1, &c2, &c3, &c4, &saz, &tslope, &t, undef);

    *winddir = wd(1, 1);
    *windspd = ws(1, 1);
}

/** find_cloudfactor (meteodistr.cc): Walcek's 700mb cloud fraction. */
double gt_find_cloudfactor(double Tair, double RH, double Z,
                           double T_lapse_rate, double Td_lapse_rate)
{
    return find_cloudfactor(Tair, RH, Z, T_lapse_rate, Td_lapse_rate);
}

/** GEOtop's other sentinel: a column the file did not provide. */
double gt_number_absent(void) { return static_cast<double>(number_absent); }

/** GEOtop's own decimal parser (tabs.cc:277), on a plain C string.
 *
 * Not strtod: it accumulates digit by digit with pow(10, cnt), so the last bits
 * can differ from a library conversion. Exposed so the Python parser can be
 * checked against it rather than assumed equivalent to float(). */
double gt_find_number(const char *s)
{
    long n = static_cast<long>(strlen(s));
    std::vector<long> v(n);
    for (long i = 0; i < n; i++) v[i] = static_cast<long>(s[i]);
    return find_number(v.data(), n);
}

/** GEOtop's sentinels, so the Python side recognises "absent" the same way. */
double gt_number_novalue(void) { return static_cast<double>(number_novalue); }
const char *gt_string_novalue(void) { return string_novalue; }

}  // extern "C"

extern "C" {

/* GEOtop's soil-parameter row indices (constants.h:133-148), exported so the
 * Python side never hardcodes an index that could drift from the C++.
 * Order: jdz jpsi jT jKn jKl jres jwp jfc jsat ja jns jv jkt jct jss nsoilprop */
void gt_soil_indices(long *out)
{
    const unsigned int idx[] = {jdz, jpsi, jT, jKn, jKl, jres, jwp, jfc,
                                jsat, ja, jns, jv, jkt, jct, jss, nsoilprop};
    for (unsigned i = 0; i < sizeof(idx) / sizeof(idx[0]); i++)
        out[i] = static_cast<long>(idx[i]);
}

/* ----------------------------------------------------------------- util_math.h
 * Numerical primitives shared by the energy and water Newton solvers. */

/* Solve A(ld, d, ud) * e + b = 0.
 *
 * Both off-diagonals are indexed by the LOWER index of the pair they connect
 * (util_math.h:107-116 reads ld[j-1] and ud[j-1] for row j), so for a 1-based
 * system of size n:
 *
 *     d[i]  = A[i][i]      i = 1..n
 *     ld[i] = A[i+1][i]    i = 1..n-1     (sub-diagonal)
 *     ud[i] = A[i][i+1]    i = 1..n-1     (super-diagonal)
 *
 * ld[n] and ud[n] are never read. */
short gt_tridiag2(long nbeg, long nend, const double *ld, const double *d,
                  const double *ud, const double *b, double *e)
{
    long n = nend;
    Vector<double> vld = as_vector(ld, n);
    Vector<double> vd = as_vector(d, n);
    Vector<double> vud = as_vector(ud, n);
    Vector<double> vb = as_vector(b, n);
    Vector<double> ve{static_cast<std::size_t>(n)};
    for (long i = 1; i <= n; i++) ve(i) = e[i - 1];

    short out = tridiag2(0, 0, 0, nbeg, nend, &vld, &vd, &vud, &vb, &ve);

    for (long i = 1; i <= n; i++) e[i - 1] = ve(i);
    return out;
}

double gt_norm_inf(const double *v, long nbeg, long nend)
{ Vector<double> vv = as_vector(v, nend); return norm_inf(&vv, nbeg, nend); }

double gt_norm_2(const double *v, long nbeg, long nend)
{ Vector<double> vv = as_vector(v, nend); return norm_2(&vv, nbeg, nend); }

double gt_norm_1(const double *v, long nbeg, long nend)
{ Vector<double> vv = as_vector(v, nend); return norm_1(&vv, nbeg, nend); }

void gt_Cramer_rule(double A, double B, double C, double D, double E, double F,
                    double *x, double *y)
{ Cramer_rule(A, B, C, D, E, F, x, y); }

double gt_minimize_merit_function(double res0, double lambda1, double res1,
                                  double lambda2, double res2)
{ return minimize_merit_function(res0, lambda1, res1, lambda2, res2); }

/* --------------------------------------------------------------------- snow.cc
 * Fresh-snow density, the enthalpy closure and the snow freezing curve. */

double gt_rho_newlyfallensnow(double u, double Tatm, double Tfreez)
{ return rho_newlyfallensnow(u, Tatm, Tfreez); }

double gt_internal_energy(double w_ice, double w_liq, double T)
{ return internal_energy(w_ice, w_liq, T); }

/* w_ice and w_liq are IN/OUT: GEOtop reads them to form SWE = w_ice + w_liq,
 * then overwrites both with the phase split at the recovered temperature. With
 * SWE == 0 the function takes its empty-layer branch and returns zeros, so the
 * caller must pass the layer's current contents, not placeholders. */
void gt_from_internal_energy(double a, double h, double *w_ice, double *w_liq,
                             double *T)
{ from_internal_energy(a, 0, 0, h, w_ice, w_liq, T); }

double gt_theta_snow(double a, double b, double T)
{ return theta_snow(a, b, T); }

double gt_dtheta_snow(double a, double b, double T)
{ return dtheta_snow(a, b, T); }

double gt_k_thermal_snow_Sturm(double density)
{ return k_thermal_snow_Sturm(density); }

double gt_k_thermal_snow_Yen(double density)
{ return k_thermal_snow_Yen(density); }

/* `zenith` selects the zenith-angle correction GEOtop passes as the function
 * pointer F: 0 -> Zero (used for the diffuse albedo), 1 -> Fzen (direct beam). */
double gt_snow_albedo(double ground_alb, double snowD, double AEP,
                      double freshsnow_alb, double C, double tsnow,
                      double cosinc, int zenith)
{
    return snow_albedo(ground_alb, snowD, AEP, freshsnow_alb, C, tsnow, cosinc,
                       zenith ? (Fzen) : (Zero));
}

double gt_Fzen(double cosinc)
{ return Fzen(cosinc); }

/* `tsnow_nondim` is IN/OUT: the non-dimensional snow age is advanced in place. */
void gt_update_snow_age(double Psnow, double Ts, double Dt, double Prestore,
                        double *tsnow_nondim)
{ update_snow_age(Psnow, Ts, Dt, Prestore, tsnow_nondim); }

double gt_find_albedo(double dry_albedo, double sat_albedo, double wat_content,
                      double residual_wat_content, double saturated_wat_content)
{
    return find_albedo(dry_albedo, sat_albedo, wat_content,
                       residual_wat_content, saturated_wat_content);
}

/* ------------------------------------------------------------ energy.balance.cc
 * Thermal conductivity and heat capacity of a column node. */

double gt_k_thermal(short snow, short a, double th_liq, double th_ice,
                    double th_sat, double k_solid)
{ return k_thermal(snow, a, th_liq, th_ice, th_sat, k_solid); }

/* calc_C is the one law GEOtop expresses over the whole column rather than per
 * node: it indexes wi/wl/dw/D at `l` and the soil parameters at `l-nsng`. The
 * wrapper keeps that shape, so the caller passes the same arrays and the same
 * 1-based `l` the C++ would. */
double gt_calc_C(long l, long nsng, double a, const double *wi, const double *wl,
                 const double *dw, const double *D, long n,
                 double *pa, long nr, long nc)
{
    Vector<double> vwi = as_vector(wi, n);
    Vector<double> vwl = as_vector(wl, n);
    Vector<double> vdw = as_vector(dw, n);
    Vector<double> vD = as_vector(D, n);
    return calc_C(l, nsng, a, vwi, vwl, vdw, vD, as_matrix_view(pa, nr, nc));
}

/* -------------------------------------------------------------------- times.cc
 * The date arithmetic, so the Python translation can be pinned rather than
 * round-tripped against itself. */

short gt_is_leap(long y) { return is_leap(y); }

double gt_convert_dateeur12_JDfrom0(double date)
{ return convert_dateeur12_JDfrom0(date); }

double gt_convert_JDfrom0_dateeur12(double JDfrom0)
{ return convert_JDfrom0_dateeur12(JDfrom0); }

double gt_convert_JDandYear_JDfrom0(double JD, long year)
{ return convert_JDandYear_JDfrom0(JD, year); }

void gt_convert_JDfrom0_JDandYear(double JDfrom0, double *JD, long *year)
{ convert_JDfrom0_JDandYear(JDfrom0, JD, year); }

void gt_convert_dateeur12_JDandYear(double date, double *JD, long *year)
{ convert_dateeur12_JDandYear(date, JD, year); }

void gt_convert_JDandYear_daymonthhourmin(double JD, long year, long *d, long *m,
                                          long *h, long *mi)
{ convert_JDandYear_daymonthhourmin(JD, year, d, m, h, mi); }

void gt_convert_dateeur12_daymonthyearhourmin(double date, long *d, long *m,
                                              long *y, long *h, long *mi)
{ convert_dateeur12_daymonthyearhourmin(date, d, m, y, h, mi); }

/* --------------------------------------------------------------- turbulence.cc
 * Monin-Obukhov stability functions and the latent-heat closure. */

double gt_Psim(double z)     { return Psim(z); }
double gt_Psih(double z)     { return Psih(z); }
double gt_PsiStab(double z)  { return PsiStab(z); }
double gt_roughT(double M, double N, double R) { return roughT(M, N, R); }
double gt_roughQ(double M, double N, double R) { return roughQ(M, N, R); }
double gt_Levap(double T)    { return Levap(T); }
double gt_latent(double Ts, double Le) { return latent(Ts, Le); }

double gt_cz(short which, double zmeas, double z0, double d0, double L)
{
    /* `cz` takes the stable/unstable Psi pair as function pointers; the two
     * combinations GEOtop actually uses are selected by `which`:
     * 0 -> momentum (Psim / PsiStab), 1 -> heat (Psih / PsiStab). */
    return which == 0 ? cz(zmeas, z0, d0, L, Psim, PsiStab)
                      : cz(zmeas, z0, d0, L, Psih, PsiStab);
}

double gt_CZ(short state, short which, double zmeas, double z0, double d0,
             double L)
{
    return which == 0 ? CZ(state, zmeas, z0, d0, L, Psim)
                      : CZ(state, zmeas, z0, d0, L, Psih);
}

/* -------------------------------------------------------------------- meteo.cc
 * Psychrometrics and the rain/snow partition. */

double gt_pressure(double Z) { return pressure(Z); }

double gt_temperature(double Z, double Z0, double T0, double gamma)
{ return temperature(Z, Z0, T0, gamma); }

void gt_part_snow(double prec_total, double *prec_rain, double *prec_snow,
                  double temperature_, double t_rain, double t_snow)
{ part_snow(prec_total, prec_rain, prec_snow, temperature_, t_rain, t_snow); }

double gt_SatVapPressure(double T, double P) { return SatVapPressure(T, P); }

void gt_SatVapPressure_2(double *e, double *de_dT, double T, double P)
{ SatVapPressure_2(e, de_dT, T, P); }

double gt_TfromSatVapPressure(double e, double P)
{ return TfromSatVapPressure(e, P); }

double gt_SpecHumidity(double e, double P) { return SpecHumidity(e, P); }

void gt_SpecHumidity_2(double *Q, double *dQ_dT, double RH, double T, double P)
{ SpecHumidity_2(Q, dQ_dT, RH, T, P); }

double gt_VapPressurefromSpecHumidity(double Q, double P)
{ return VapPressurefromSpecHumidity(Q, P); }

double gt_Tdew(double T, double RH, double Z) { return Tdew(T, RH, Z); }

double gt_RHfromTdew(double T, double Td, double Z)
{ return RHfromTdew(T, Td, Z); }

double gt_air_density(double T, double Q, double P)
{ return air_density(T, Q, P); }

double gt_air_cp(double T) { return air_cp(T); }

}  // extern "C"

extern "C" {

/* -------------------------------------------------------- parameters.cc, files
 * read_soil_parameters and read_point_file, the two tabular inputs that come
 * with a pre-populated fallback matrix built from the inline keywords.
 *
 * Both take that fallback in and hand the resolved matrix back, so the Python
 * side supplies the same starting point and the comparison is of the file
 * cascade alone, not of the keyword pass that precedes it.
 *
 * Matrices cross the boundary flat and row-major, with GEOtop's 1-based indices
 * shifted down by one: element (n, j) is at [(n-1)*ncols + (j-1)].
 */

/** read_soil_parameters for a single soil type.
 *
 * `pa_in` and `pa_bed_in` are nsoilprop x `nlayers_in`; `name` is the file stem
 * (the file read is <name>0001.txt), or GEOtop's string_novalue for no file.
 * Writes the resolved matrices to `pa_out`/`pa_bed_out` (both nsoilprop x the
 * returned layer count) and the resolved water table depth to `iwtd_out`.
 * Returns the number of layers, or -1 if an argument does not fit. */
long gt_read_soil_parameters(const char *name,
                             const char *const *col_names, long ncolnames,
                             const double *pa_in, const double *pa_bed_in,
                             long nlayers_in, double iwtd_in, long bed,
                             double *pa_out, double *pa_bed_out,
                             double *iwtd_out, long max_layers)
{
    if (ncolnames != static_cast<long>(nsoilprop)) return -1;

    // read_txt_matrix takes char** and may write through it; copy the names.
    std::vector<std::string> owned;
    owned.reserve(ncolnames);
    for (long i = 0; i < ncolnames; i++) owned.emplace_back(col_names[i]);
    std::vector<char *> cols(ncolnames);
    for (long i = 0; i < ncolnames; i++) cols[i] = &owned[i][0];

    SOIL sl{};
    INIT_TOOLS IT{};

    sl.pa.reset(new Tensor<double>{1, static_cast<std::size_t>(nsoilprop),
                                   static_cast<std::size_t>(nlayers_in)});
    IT.pa_bed.reset(new Tensor<double>{1, static_cast<std::size_t>(nsoilprop),
                                       static_cast<std::size_t>(nlayers_in)});
    for (long n = 1; n <= static_cast<long>(nsoilprop); n++)
    {
        for (long j = 1; j <= nlayers_in; j++)
        {
            (*sl.pa)(1, n, j) = pa_in[(n - 1) * nlayers_in + (j - 1)];
            (*IT.pa_bed)(1, n, j) = pa_bed_in[(n - 1) * nlayers_in + (j - 1)];
        }
    }
    IT.init_water_table_depth.reset(new Vector<double>{1});
    (*IT.init_water_table_depth)(1) = iwtd_in;
    IT.soil_col_names = cols.data();

    std::string stem{name};
    read_soil_parameters(&stem[0], &IT, &sl, bed);

    long nlayers = static_cast<long>(sl.pa->nch);
    if (nlayers > max_layers)
    {
        IT.soil_col_names = nullptr;
        return -1;
    }
    for (long n = 1; n <= static_cast<long>(nsoilprop); n++)
    {
        for (long j = 1; j <= nlayers; j++)
        {
            pa_out[(n - 1) * nlayers + (j - 1)] = (*sl.pa)(1, n, j);
            pa_bed_out[(n - 1) * nlayers + (j - 1)] = (*IT.pa_bed)(1, n, j);
        }
    }
    *iwtd_out = (*IT.init_water_table_depth)(1);

    // The names are stack-owned here; INIT_TOOLS does not own them either, but
    // clear the pointer so nothing can outlive `owned`.
    IT.soil_col_names = nullptr;
    return nlayers;
}

/** read_point_file. `chkpt_in` is `npoints_in` rows of ptTOT columns; `name` is
 * the file stem (the file read is <name>.txt), or string_novalue for no file.
 * Writes the resolved matrix to `chkpt_out` and returns its number of rows. */
long gt_read_point_file(const char *name,
                        const char *const *col_names, long ncolnames,
                        const double *chkpt_in, long npoints_in,
                        double *chkpt_out, long max_points)
{
    if (ncolnames != static_cast<long>(ptTOT)) return -1;

    std::vector<std::string> owned;
    owned.reserve(ncolnames);
    for (long i = 0; i < ncolnames; i++) owned.emplace_back(col_names[i]);
    std::vector<char *> cols(ncolnames);
    for (long i = 0; i < ncolnames; i++) cols[i] = &owned[i][0];

    PAR par{};
    par.point_sim = 1;
    par.state_pixel = 1;
    par.chkpt.reset(new Matrix<double>{static_cast<std::size_t>(npoints_in),
                                       static_cast<std::size_t>(ptTOT)});
    for (long n = 1; n <= npoints_in; n++)
    {
        for (long j = 1; j <= static_cast<long>(ptTOT); j++)
            (*par.chkpt)(n, j) = chkpt_in[(n - 1) * ptTOT + (j - 1)];
    }

    std::string stem{name};
    read_point_file(&stem[0], cols.data(), &par);

    long npoints = static_cast<long>(par.chkpt->nrh);
    if (npoints > max_points) return -1;
    for (long n = 1; n <= npoints; n++)
    {
        for (long j = 1; j <= static_cast<long>(ptTOT); j++)
            chkpt_out[(n - 1) * ptTOT + (j - 1)] = (*par.chkpt)(n, j);
    }
    return npoints;
}

/** The point column indices (constants.h:508-528), in column order, then ptTOT. */
void gt_point_indices(long *out)
{
    const unsigned int idx[] = {ptID, ptX, ptY, ptZ, ptLC, ptSY, ptS, ptA,
                                ptSKY, ptCNS, ptCWE, ptCNwSe, ptCNeSw,
                                ptDrDEPTH, ptHOR, ptMAXSWE, ptLAT, ptLON,
                                ptBED, ptTOT};
    for (unsigned i = 0; i < sizeof(idx) / sizeof(idx[0]); i++)
        out[i] = static_cast<long>(idx[i]);
}

/* ------------------------------------------------------------- util_math.h
 * The two primitives the BiCGSTAB solver below is built from. `tridiag2`
 * (further up) solves A*e + b = 0; `tridiag` here solves A*e = b, aborts
 * instead of returning on a zero first pivot, and returns 1 on success and 0
 * on a later zero pivot -- the opposite convention to tridiag2's. */

/* The off-diagonals are indexed by the LOWER index of the pair (diag_inf[j-1]
 * and diag_sup[j-1] are both read for row j), so both arrays are nx-1 long. */
short gt_tridiag(long nx, const double *diag_inf, const double *diag,
                 const double *diag_sup, const double *b, double *e)
{
    Vector<double> vli = as_vector(diag_inf, nx - 1);
    Vector<double> vd = as_vector(diag, nx);
    Vector<double> vus = as_vector(diag_sup, nx - 1);
    Vector<double> vb = as_vector(b, nx);
    Vector<double> ve{static_cast<std::size_t>(nx)};

    short out = tridiag(0, 0, 0, nx, &vli, &vd, &vus, &vb, &ve);

    for (long i = 1; i <= nx; i++) e[i - 1] = ve(i);
    return out;
}

/** Dot product over [1, n]. */
double gt_product(const double *a, const double *b, long n)
{
    Vector<double> va = as_vector(a, n);
    Vector<double> vb = as_vector(b, n);
    return product(&va, &vb);
}

/* ---------------------------------------------------------- sparse_matrix.cc
 * GEOtop's own sparse format for the symmetric M-matrix K of Richards'
 * equation: only the STRICT lower triangle is stored, in Li/Lp/Lx.
 *
 *   Lx(i)  the value of the i-th stored entry
 *   Li(i)  its row
 *   Lp(c)  the running count of entries through column c, so entry i belongs
 *          to the column c with Lp(c-1) < i <= Lp(c)
 *
 * The diagonal is never stored. Every product below rebuilds it from K's
 * defining property, that each row sums to zero: an entry k(r,c) contributes
 * k*(x(r)-x(c)) to row c and k*(x(c)-x(r)) to row r.
 *
 * Vectors keep GEOtop's 1-based indices, so the caller passes arrays whose
 * element 0 is GEOtop's element 1. Lengths are the containers' own nh: `nx`
 * for the node vectors, `nLi` for Li and Lx, `nLp` for Lp (in a 1D column
 * nLp is nx and nLi is nx-1, but nothing here assumes that).
 *
 * These abort the process via t_error if Li is not sorted into a lower
 * triangle -- as GEOtop itself does. */

void gt_product_using_only_strict_lower_diagonal_part(
    double *product_out, const double *x, long nx,
    const long *Li, long nLi, const long *Lp, long nLp, const double *Lx)
{
    Vector<double> vprod{static_cast<std::size_t>(nx)};
    Vector<double> vx = as_vector(x, nx);
    Vector<long> vLi = as_vector_long(Li, nLi);
    Vector<long> vLp = as_vector_long(Lp, nLp);
    Vector<double> vLx = as_vector(Lx, nLi);

    product_using_only_strict_lower_diagonal_part(&vprod, &vx, &vLi, &vLp, &vLx);

    for (long i = 1; i <= nx; i++) product_out[i - 1] = vprod(i);
}

void gt_product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(
    double *product_out, const double *x, const double *y, long nx,
    const long *Li, long nLi, const long *Lp, long nLp, const double *Lx)
{
    Vector<double> vprod{static_cast<std::size_t>(nx)};
    Vector<double> vx = as_vector(x, nx);
    Vector<double> vy = as_vector(y, nx);
    Vector<long> vLi = as_vector_long(Li, nLi);
    Vector<long> vLp = as_vector_long(Lp, nLp);
    Vector<double> vLx = as_vector(Lx, nLi);

    product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(
        &vprod, &vx, &vy, &vLi, &vLp, &vLx);

    for (long i = 1; i <= nx; i++) product_out[i - 1] = vprod(i);
}

/** The tridiagonal part of (K + diag(y)), used as the solver's preconditioner.
 * `diag_out` is nx long, `udiag_out` nx-1. */
void gt_get_diag_strict_lower_matrix_plus_identity_by_vector(
    double *diag_out, double *udiag_out, const double *y, long nx,
    const long *Li, long nLi, const long *Lp, long nLp, const double *Lx)
{
    Vector<double> vdiag{static_cast<std::size_t>(nx)};
    Vector<double> vudiag{static_cast<std::size_t>(nx - 1)};
    Vector<double> vy = as_vector(y, nx);
    Vector<long> vLi = as_vector_long(Li, nLi);
    Vector<long> vLp = as_vector_long(Lp, nLp);
    Vector<double> vLx = as_vector(Lx, nLi);

    get_diag_strict_lower_matrix_plus_identity_by_vector(&vdiag, &vudiag, &vy,
                                                         &vLi, &vLp, &vLx);

    for (long i = 1; i <= nx; i++) diag_out[i - 1] = vdiag(i);
    for (long i = 1; i <= nx - 1; i++) udiag_out[i - 1] = vudiag(i);
}

/** out = k * (y + K x). */
void gt_product_matrix_using_lower_part_by_vector_plus_vector(
    double k, double *out, const double *y, const double *x, long nx,
    const long *Li, long nLi, const long *Lp, long nLp, const double *Lx)
{
    Vector<double> vout{static_cast<std::size_t>(nx)};
    Vector<double> vy = as_vector(y, nx);
    Vector<double> vx = as_vector(x, nx);
    Vector<long> vLi = as_vector_long(Li, nLi);
    Vector<long> vLp = as_vector_long(Lp, nLp);
    Vector<double> vLx = as_vector(Lx, nLi);

    product_matrix_using_lower_part_by_vector_plus_vector(k, &vout, &vy, &vx,
                                                          &vLi, &vLp, &vLx);

    for (long i = 1; i <= nx; i++) out[i - 1] = vout(i);
}

/** Solve (K + diag(y)) x = b. `x` is IN/OUT: the solver never forms the
 * initial residual, it sets r = b outright and then accumulates into x, so
 * whatever x holds on entry is added to the answer. Returns the iteration
 * count, or -1 if the preconditioner solve hit a zero pivot -- the count is
 * what has to match, not just the solution. */
long gt_BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
    double tol_rel, double tol_min, double tol_max,
    double *x, const double *b, const double *y, long nx,
    const long *Li, long nLi, const long *Lp, long nLp, const double *Lx)
{
    Vector<double> vx = as_vector(x, nx);
    Vector<double> vb = as_vector(b, nx);
    Vector<double> vy = as_vector(y, nx);
    Vector<long> vLi = as_vector_long(Li, nLi);
    Vector<long> vLp = as_vector_long(Lp, nLp);
    Vector<double> vLx = as_vector(Lx, nLi);

    long iter = BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        tol_rel, tol_min, tol_max, &vx, &vb, &vy, &vLi, &vLp, &vLx);

    for (long i = 1; i <= nx; i++) x[i - 1] = vx(i);
    return iter;
}

}  // extern "C"

/* ---------------------------------------------------------- radiation.cc
 * Solar geometry: pure functions of a timestamp/position, no ALLDATA
 * dependency, so wrapped directly like every other law in this file. */

extern "C" {

/** sun-earth distance factor, equation of time [rad], declination [rad]. */
void gt_sun(double JDfrom0, double *E0, double *Et, double *Delta)
{ sun(JDfrom0, E0, Et, Delta); }

double gt_SolarHeight(double JD, double latitude, double Delta, double dh)
{ return SolarHeight(JD, latitude, Delta, dh); }

double gt_SolarAzimuth(double JD, double latitude, double Delta, double dh)
{ return SolarAzimuth(JD, latitude, Delta, dh); }

/** `hor` is `n` rows of (azimuth, elevation), flat row-major, 0-based --
 * the same convention gt_meteo_load's rows use. */
short gt_shadows_point(const double *hor, long n, double alpha, double azimuth,
                       double tol_mount, double tol_flat)
{
    std::vector<double> storage;
    auto rows = as_rows(hor, n, 2, storage);
    return shadows_point(rows.data(), n, alpha, azimuth, tol_mount, tol_flat);
}

}  // extern "C"

/* ------------------------------------------------------------- clouds.cc
 * Cloudiness inferred from a station's own shortwave record. All three take
 * `meteo` as `meteolines` rows of `nmet` columns (gt_meteo_load's own
 * convention) and `horizon` as `horizonlines` rows of (azimuth, elevation).
 * `find_cloudiness`/`find_sunset` read but do not write `meteo`, so no
 * output copy is needed for them.
 */

extern "C" {

double gt_find_cloudiness(long n, const double *meteo, long meteolines,
                          double lat, double lon, double ST, double Z,
                          double sky, double SWrefl_surr, double rotation,
                          double Lozone, double alpha, double beta, double albedo)
{
    std::vector<double> storage;
    auto rows = as_rows(meteo, meteolines, nmet, storage);
    return find_cloudiness(n, rows.data(), meteolines, lat, lon, ST, Z, sky,
                           SWrefl_surr, rotation, Lozone, alpha, beta, albedo);
}

/** Returns n1; writes n0 to `*n0_out`. */
long gt_find_sunset(long nist, const double *meteo, long meteolines,
                    const double *horizon, long horizonlines,
                    double lat, double lon, double ST, double rotation,
                    long *n0_out)
{
    std::vector<double> mstorage;
    auto mrows = as_rows(meteo, meteolines, nmet, mstorage);
    std::vector<double> hstorage;
    auto hrows = as_rows(horizon, horizonlines, 2, hstorage);

    long n0 = 0, n1 = 0;
    find_sunset(nist, &n0, &n1, mrows.data(), meteolines, hrows.data(),
               horizonlines, lat, lon, ST, rotation);
    *n0_out = n0;
    return n1;
}

/** Fills `meteo[n][itauC]` for every row and copies the result to `tauC_out`
 * (`meteolines` long); returns 1 if the column was computed (there was
 * shortwave to infer from), 0 otherwise -- matching `added_cloud`. */
short gt_fill_meteo_data_with_cloudiness(
    double *meteo, long meteolines,
    const double *horizon, long horizonlines,
    double lat, double lon, double ST, double Z, double sky,
    double SWrefl_surr, long ndivday, double rotation,
    double Lozone, double alpha, double beta, double albedo,
    double *tauC_out)
{
    std::vector<double> mstorage;
    auto mrows = as_rows(meteo, meteolines, nmet, mstorage);
    std::vector<double> hstorage;
    auto hrows = as_rows(horizon, horizonlines, 2, hstorage);

    short added = fill_meteo_data_with_cloudiness(
        mrows.data(), meteolines, hrows.data(), horizonlines, lat, lon, ST, Z,
        sky, SWrefl_surr, ndivday, rotation, Lozone, alpha, beta, albedo);

    for (long i = 0; i < meteolines; i++) tauC_out[i] = mrows[i][itauC];
    return added;
}

}  // extern "C"

/* ---------------------------------- libraries/ascii, libraries/geomorphology
 * The ESRI-ASCII raster reader, and the grid operators a point simulation
 * reaches through it when its topography comes from maps.
 */

extern "C" {

/** `read_map(0, ...)`: read a raster and report both the grid and the header
 * GEOtop derives from it. Writes `header_out` as
 * (Dy, Dx, Y0, X0, nrows, ncols, sign of novalue, novalue) -- the first four
 * in GEOtop's own UV->U order -- and the cells row-major into `data_out`.
 * Returns the cell count, or -1 if it would not fit in `maxcells`. */
long gt_read_map(const char *filename, double no_value, double *header_out,
                 double *data_out, long maxcells)
{
    T_INIT UV;
    Matrix<double> dummy{1, 1};
    Matrix<double> *M = read_map(0, const_cast<char *>(filename), &dummy, &UV,
                                 no_value);
    long nr = M->nrh, nc = M->nch;
    if (nr * nc > maxcells) { delete M; return -1; }

    header_out[0] = (*UV.U)(1);
    header_out[1] = (*UV.U)(2);
    header_out[2] = (*UV.U)(3);
    header_out[3] = (*UV.U)(4);
    header_out[4] = (double)nr;
    header_out[5] = (double)nc;
    header_out[6] = (*UV.V)(1);
    header_out[7] = (*UV.V)(2);

    for (long r = 1; r <= nr; r++)
        for (long c = 1; c <= nc; c++)
            data_out[(r - 1) * nc + (c - 1)] = (*M)(r, c);

    delete M;
    return nr * nc;
}

/** `Zin`/`Zout` are row-major `nr` x `nc` buffers, 0-based at this boundary. */
void gt_multipass_topofilter(long ntimes, long nr, long nc, const double *Zin,
                             double *Zout, long novalue, long n)
{
    Matrix<double> in{static_cast<std::size_t>(nr), static_cast<std::size_t>(nc)};
    Matrix<double> out{static_cast<std::size_t>(nr), static_cast<std::size_t>(nc)};
    for (long r = 1; r <= nr; r++)
        for (long c = 1; c <= nc; c++) in(r, c) = Zin[(r - 1) * nc + (c - 1)];

    multipass_topofilter(ntimes, &in, &out, novalue, n);

    for (long r = 1; r <= nr; r++)
        for (long c = 1; c <= nc; c++) Zout[(r - 1) * nc + (c - 1)] = out(r, c);
}

/** `topo` and the four curvature outputs are row-major `nr` x `nc` buffers. */
void gt_curvature(double deltax, double deltay, long nr, long nc,
                  const double *topo, double *c1_out, double *c2_out,
                  double *c3_out, double *c4_out, long undef)
{
    std::size_t R = static_cast<std::size_t>(nr), C = static_cast<std::size_t>(nc);
    Matrix<double> Z{R, C}, c1{R, C}, c2{R, C}, c3{R, C}, c4{R, C};
    for (long r = 1; r <= nr; r++)
        for (long c = 1; c <= nc; c++) Z(r, c) = topo[(r - 1) * nc + (c - 1)];

    curvature(deltax, deltay, &Z, &c1, &c2, &c3, &c4, undef);

    for (long r = 1; r <= nr; r++) {
        for (long c = 1; c <= nc; c++) {
            long k = (r - 1) * nc + (c - 1);
            c1_out[k] = c1(r, c);
            c2_out[k] = c2(r, c);
            c3_out[k] = c3(r, c);
            c4_out[k] = c4(r, c);
        }
    }
}

/** `row`/`col` take the two header entries they actually read: the cell size
 * along that axis and the corner coordinate. */
long gt_row(double N, long nrows, double Dy, double Y0, long novalue)
{
    T_INIT UV;
    UV.U.reset(new Vector<double>{4});
    (*UV.U)(1) = Dy;
    (*UV.U)(3) = Y0;
    return row(N, nrows, &UV, novalue);
}

long gt_col(double E, long ncols, double Dx, double X0, long novalue)
{
    T_INIT UV;
    UV.U.reset(new Vector<double>{4});
    (*UV.U)(2) = Dx;
    (*UV.U)(4) = X0;
    return col(E, ncols, &UV, novalue);
}

/** `existing_file`: 0 none, 2 grass ascii, 3 esri ascii. */
short gt_existing_file(const char *name)
{ return existing_file(const_cast<char *>(name)); }

}  // extern "C"
