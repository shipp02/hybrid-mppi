/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> // memcpy
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "mppi_bicycle_tracker_model/mppi_bicycle_tracker_model.h"


#include "mppi_bicycle_tracker_constraints/mppi_bicycle_tracker_constraints.h"
#include "mppi_bicycle_tracker_cost/mppi_bicycle_tracker_cost.h"



#include "acados_solver_ocp_mppi_bicycle_tracker_e3ac8181.h"

#define NX     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NX
#define NZ     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NZ
#define NU     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NU
#define NP     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NP
#define NP_GLOBAL     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NP_GLOBAL
#define NY0    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NY0
#define NY     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NY
#define NYN    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NYN

#define NBX    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NBX
#define NBX0   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NBX0
#define NBU    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NBU
#define NG     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NG
#define NBXN   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NBXN
#define NGN    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NGN

#define NH     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NH
#define NHN    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NHN
#define NH0    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NH0
#define NPHI   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NPHI
#define NPHIN  OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NPHIN
#define NPHI0  OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NPHI0
#define NR     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NR

#define NS     OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NS
#define NS0    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NS0
#define NSN    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSN

#define NSBX   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSBX
#define NSBU   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSBU
#define NSH0   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSH0
#define NSH    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSH
#define NSHN   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSHN
#define NSG    OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSG
#define NSPHI0 OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSPHI0
#define NSPHI  OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSPHI
#define NSPHIN OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSPHIN
#define NSGN   OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSGN
#define NSBXN  OCP_MPPI_BICYCLE_TRACKER_E3AC8181_NSBXN
// initial value of stagewise parameters
static const double p_init[] = {0,0,10,};





// ** solver data **

ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule * ocp_mppi_bicycle_tracker_e3ac8181_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule));
    ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule *capsule = (ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule *) capsule_mem;

    return capsule;
}


int ocp_mppi_bicycle_tracker_e3ac8181_acados_free_capsule(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int ocp_mppi_bicycle_tracker_e3ac8181_acados_create(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    int N_shooting_intervals = OCP_MPPI_BICYCLE_TRACKER_E3AC8181_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return ocp_mppi_bicycle_tracker_e3ac8181_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int ocp_mppi_bicycle_tracker_e3ac8181_acados_update_time_steps(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "ocp_mppi_bicycle_tracker_e3ac8181_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for ocp_mppi_bicycle_tracker_e3ac8181_acados_create: step 1
 */
void ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = NONLINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = NONLINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = NONLINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_dimensions(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;

    nbxe[0] = 6;

    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for ocp_mppi_bicycle_tracker_e3ac8181_acados_create: step 3
 */
void ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_functions(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        MAP_CASADI_FNC(nl_constr_h_0_fun_jac, mppi_bicycle_tracker_constr_h_0_fun_jac_uxt_zt);
        MAP_CASADI_FNC(nl_constr_h_0_fun, mppi_bicycle_tracker_constr_h_0_fun);
        // constraints.constr_type == "BGH" and dims.nh > 0
        capsule->nl_constr_h_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun_jac[i], mppi_bicycle_tracker_constr_h_fun_jac_uxt_zt);
        }
        capsule->nl_constr_h_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun[i], mppi_bicycle_tracker_constr_h_fun);
        }

        // nonlinear least squares function
        MAP_CASADI_FNC(cost_y_0_fun, mppi_bicycle_tracker_cost_y_0_fun);
        MAP_CASADI_FNC(cost_y_0_fun_jac_ut_xt, mppi_bicycle_tracker_cost_y_0_fun_jac_ut_xt);




        // discrete dynamics
        capsule->discr_dyn_phi_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun[i], mppi_bicycle_tracker_dyn_disc_phi_fun);
        }

        capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], mppi_bicycle_tracker_dyn_disc_phi_fun_jac);
        }






        // nonlinear least squares cost
        capsule->cost_y_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(cost_y_fun[i], mppi_bicycle_tracker_cost_y_fun);
        }

        capsule->cost_y_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(cost_y_fun_jac_ut_xt[i], mppi_bicycle_tracker_cost_y_fun_jac_ut_xt);
        }
    } // N > 0
    MAP_CASADI_FNC(nl_constr_h_e_fun_jac, mppi_bicycle_tracker_constr_h_e_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_e_fun, mppi_bicycle_tracker_constr_h_e_fun);



    // nonlinear least square function
    MAP_CASADI_FNC(cost_y_e_fun, mppi_bicycle_tracker_cost_y_e_fun);
    MAP_CASADI_FNC(cost_y_e_fun_jac_ut_xt, mppi_bicycle_tracker_cost_y_e_fun_jac_ut_xt);

#undef MAP_CASADI_FNC
}


/**
 * Internal function for ocp_mppi_bicycle_tracker_e3ac8181_acados_create: step 5
 */
void ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_default_parameters(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;

    // initialize parameters to initial value

    double* p = malloc(NP*sizeof(double));
    memcpy(p, p_init, NP*sizeof(double));

    for (int i = 0; i <= N; i++) {
        ocp_mppi_bicycle_tracker_e3ac8181_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for ocp_mppi_bicycle_tracker_e3ac8181_acados_create: step 5
 */
void ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_nlp_in_numerical_values(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        ocp_mppi_bicycle_tracker_e3ac8181_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps

        double time_step = 0.05;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.05;
        cost_scaling[1] = 0.05;
        cost_scaling[2] = 0.05;
        cost_scaling[3] = 0.05;
        cost_scaling[4] = 0.05;
        cost_scaling[5] = 0.05;
        cost_scaling[6] = 0.05;
        cost_scaling[7] = 0.05;
        cost_scaling[8] = 0.05;
        cost_scaling[9] = 0.05;
        cost_scaling[10] = 0.05;
        cost_scaling[11] = 0.05;
        cost_scaling[12] = 0.05;
        cost_scaling[13] = 0.05;
        cost_scaling[14] = 0.05;
        cost_scaling[15] = 0.05;
        cost_scaling[16] = 0.05;
        cost_scaling[17] = 0.05;
        cost_scaling[18] = 0.05;
        cost_scaling[19] = 0.05;
        cost_scaling[20] = 0.05;
        cost_scaling[21] = 0.05;
        cost_scaling[22] = 0.05;
        cost_scaling[23] = 0.05;
        cost_scaling[24] = 0.05;
        cost_scaling[25] = 0.05;
        cost_scaling[26] = 0.05;
        cost_scaling[27] = 0.05;
        cost_scaling[28] = 0.05;
        cost_scaling[29] = 0.05;
        cost_scaling[30] = 0.05;
        cost_scaling[31] = 0.05;
        cost_scaling[32] = 0.05;
        cost_scaling[33] = 0.05;
        cost_scaling[34] = 0.05;
        cost_scaling[35] = 0.05;
        cost_scaling[36] = 0.05;
        cost_scaling[37] = 0.05;
        cost_scaling[38] = 0.05;
        cost_scaling[39] = 0.05;
        cost_scaling[40] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 13.5;
    W_0[1+(NY0) * 1] = 13.5;
    W_0[2+(NY0) * 2] = 13;
    W_0[3+(NY0) * 3] = 5.5;
    W_0[4+(NY0) * 4] = 0.01;
    W_0[5+(NY0) * 5] = 1;
    W_0[6+(NY0) * 6] = 0.01;
    W_0[7+(NY0) * 7] = 10;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 13.5;
    W[1+(NY) * 1] = 13.5;
    W[2+(NY) * 2] = 13;
    W[3+(NY) * 3] = 5.5;
    W[4+(NY) * 4] = 0.01;
    W[5+(NY) * 5] = 1;
    W[6+(NY) * 6] = 0.01;
    W[7+(NY) * 7] = 10;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 13.5;
    W_e[1+(NYN) * 1] = 13.5;
    W_e[2+(NYN) * 2] = 13;
    W_e[3+(NYN) * 3] = 5.5;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);



    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    Zl[0] = 10000;
    Zu[0] = 10000;
    zl[0] = 100;
    zu[0] = 100;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);


    // slacks terminal
    double* zluemem = calloc(4*NSN, sizeof(double));
    double* Zl_e = zluemem+NSN*0;
    double* Zu_e = zluemem+NSN*1;
    double* zl_e = zluemem+NSN*2;
    double* zu_e = zluemem+NSN*3;

    // change only the non-zero elements:
    Zl_e[0] = 10000;
    Zu_e[0] = 10000;
    zl_e[0] = 100;
    zu_e[0] = 100;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", Zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", Zu_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", zu_e);
    free(zluemem);

    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(6 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);



    // set up nonlinear constraints for last stage
    double* luh_0 = calloc(2*NH0, sizeof(double));
    double* lh_0 = luh_0;
    double* uh_0 = luh_0 + NH0;
    lh_0[0] = -0.15707963267948966;
    uh_0[0] = 0.15707963267948966;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lh", lh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "uh", uh_0);
    free(luh_0);









    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -3;
    ubu[0] = 3;
    lbu[1] = -0.4189;
    ubu[1] = 0.4189;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);






    /* Path constraints */

    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 3;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    ubx[0] = 6;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);


    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;
    lh[0] = -0.15707963267948966;
    lh[1] = -150;
    uh[0] = 0.15707963267948966;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "uh", uh);
    }
    free(luh);








    // set up soft bounds for nonlinear constraints
    int* idxsh = malloc(NSH * sizeof(int));
    idxsh[0] = 1;
    double* lush = calloc(2*NSH, sizeof(double));
    double* lsh = lush;
    double* ush = lush + NSH;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ush", ush);
    }
    free(idxsh);
    free(lush);



    /* terminal constraints */

    // set up bounds for last stage
    // x
    int* idxbx_e = malloc(NBXN * sizeof(int));
    idxbx_e[0] = 3;
    double* lubx_e = calloc(2*NBXN, sizeof(double));
    double* lbx_e = lubx_e;
    double* ubx_e = lubx_e + NBXN;
    ubx_e[0] = 6;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "idxbx", idxbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lbx", lbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "ubx", ubx_e);
    free(idxbx_e);
    free(lubx_e);




    // set up nonlinear constraints for last stage
    double* luh_e = calloc(2*NHN, sizeof(double));
    double* lh_e = luh_e;
    double* uh_e = luh_e + NHN;
    lh_e[0] = -150;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "uh", uh_e);
    free(luh_e);



    /* terminal soft constraints */








    // set up soft bounds for nonlinear constraints
    int* idxsh_e = malloc(NSHN * sizeof(int));
    idxsh_e[0] = 0;
    double* lush_e = calloc(2*NSHN, sizeof(double));
    double* lsh_e = lush_e;
    double* ush_e = lush_e + NSHN;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "idxsh", idxsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lsh", lsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "ush", ush_e);
    free(idxsh_e);
    free(lush_e);





}

// this function only sets external functions, numerical values are set in ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_nlp_in_numerical_values
void ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_nlp_in(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, const int N)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;


    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);



    }

    /**** Cost ****/
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &capsule->cost_y_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &capsule->cost_y_fun_jac_ut_xt[i-1]);
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);

    /**** Constraints ****/

    // bounds for initial stage

    // set up nonlinear constraints for initial stage
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun_jac", &capsule->nl_constr_h_0_fun_jac);
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun", &capsule->nl_constr_h_0_fun);






    // set up nonlinear constraints for stage 1 to N-1
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);




    }



    /* terminal constraints */

    // set up nonlinear constraints for last stage

    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);



    }


static void ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_opts(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params_forw = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params_forw", &with_solution_sens_wrt_params_forw);

    int with_solution_sens_wrt_params_adj = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params_adj", &with_solution_sens_wrt_params_adj);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 10;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for ocp_mppi_bicycle_tracker_e3ac8181_acados_create: step 7
 */
void ocp_mppi_bicycle_tracker_e3ac8181_acados_set_nlp_out(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for ocp_mppi_bicycle_tracker_e3ac8181_acados_create: step 9
 */
int ocp_mppi_bicycle_tracker_e3ac8181_acados_create_precompute(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int ocp_mppi_bicycle_tracker_e3ac8181_acados_create_with_discretization(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != OCP_MPPI_BICYCLE_TRACKER_E3AC8181_N && !new_time_steps) {
        fprintf(stderr, "ocp_mppi_bicycle_tracker_e3ac8181_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, OCP_MPPI_BICYCLE_TRACKER_E3AC8181_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    ocp_mppi_bicycle_tracker_e3ac8181_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_functions(capsule);
    ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_nlp_in(capsule, N);
    ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_nlp_in_numerical_values(capsule, N, new_time_steps);
    ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = ocp_mppi_bicycle_tracker_e3ac8181_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int ocp_mppi_bicycle_tracker_e3ac8181_acados_update_qp_solver_cond_N(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from ocp_mppi_bicycle_tracker_e3ac8181_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = ocp_mppi_bicycle_tracker_e3ac8181_acados_create_precompute(capsule);
    return status;
}


int ocp_mppi_bicycle_tracker_e3ac8181_acados_reset(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar)
{
    // set initialization to all zeros
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    // sets primal and dual iterates to zero
    ocp_nlp_out_set_values_to_zero(nlp_config, nlp_dims, nlp_out);

    // reset integrator memory
    ocp_nlp_solver_reset_integrator_memory(nlp_solver, nlp_in, nlp_out);
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    if (reset_numerical_values)
    {
        // reset parameters to initial values
        ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_default_parameters(capsule);

        // reset numerical values in nlp_in
        ocp_mppi_bicycle_tracker_e3ac8181_acados_create_setup_nlp_in_numerical_values(capsule, N, NULL);
    }

    if (reset_solver_options)
    {
        // reset solver options to initial values
        ocp_mppi_bicycle_tracker_e3ac8181_acados_create_set_opts(capsule);
    }

    if (reset_x_to_x0_bar)
    {double* buffer = calloc(NX, sizeof(double));
        ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, 0, "lbx", buffer);
        for (int i=0; i<N+1; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        }
        free(buffer);
    }
    return 0;
}




int ocp_mppi_bicycle_tracker_e3ac8181_acados_update_params(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 3;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int ocp_mppi_bicycle_tracker_e3ac8181_acados_update_params_sparse(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int ocp_mppi_bicycle_tracker_e3ac8181_acados_set_p_global_and_precompute_dependencies(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, ocp_mppi_bicycle_tracker_e3ac8181_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int ocp_mppi_bicycle_tracker_e3ac8181_acados_solve(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int ocp_mppi_bicycle_tracker_e3ac8181_acados_setup_qp_matrices_and_factorize(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int ocp_mppi_bicycle_tracker_e3ac8181_acados_free(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);




    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);




    // cost
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_external_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun_jac_ut_xt);

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_0_fun_jac);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_0_fun);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun_jac);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun);



    return 0;
}


void ocp_mppi_bicycle_tracker_e3ac8181_acados_print_stats(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[1616];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int ocp_mppi_bicycle_tracker_e3ac8181_acados_custom_update(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_nlp_in(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_nlp_out(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_sens_out(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_nlp_solver(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_nlp_config(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->nlp_config; }
void *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_nlp_opts(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_nlp_dims(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *ocp_mppi_bicycle_tracker_e3ac8181_acados_get_nlp_plan(ocp_mppi_bicycle_tracker_e3ac8181_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
