#include "GPUSolver.h"

/** The number of FSRs */
__constant__ long num_FSRs;

#ifndef NGROUPS
/** The number of energy groups */
__constant__ int NUM_GROUPS;
#endif
/** the number of azimuthal angles */
__constant__ int num_azim;

/** The number of polar angles */
__constant__ int num_polar;

/** Half the number of polar angles */
__constant__ int num_polar_2;

/** The number of polar angles times energy groups */
__constant__ int polar_times_groups;

/** An array for the sines of the polar angle in the Quadrature set */
__constant__ FP_PRECISION sin_thetas[MAX_POLAR_ANGLES_GPU];

/** An array of the weights from the Quadrature set */
__constant__ FP_PRECISION weights[MAX_POLAR_ANGLES_GPU*MAX_AZIM_ANGLES_GPU];

/** The total number of Tracks */
__constant__ long tot_num_tracks;

/** is for 3D Track**/
__constant__ bool  _solve_3D;

/** The type of stabilization to be employed */
__constant__ stabilizationType stabilization_type;
__constant__ double stabilization_factor;

//从这下面是我自己的参数
//建立3维数组的索引
__constant__ double d_x_max;
__constant__ double d_x_min;
__constant__ double d_y_max;
__constant__ double d_y_min;
__constant__ double d_z_max;
__constant__ double d_z_min;
__constant__ int d_num_seg_matrix_columns;
__constant__ double _max_tau;
__constant__ boundaryType d_MaxZBoundaryType;
__constant__ boundaryType d_MinZBoundaryType;
__constant__ boundaryType d_MaxYBoundaryType;
__constant__ boundaryType d_MinYBoundaryType;
__constant__ boundaryType d_MaxXBoundaryType;
__constant__ boundaryType d_MinXBoundaryType;
__constant__ int num_3D_tracks;

//用于在json文件里显示时间标志位
__global__ void test(track_index* d_track_3D_index){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < num_3D_tracks){
    for(int i=0; i<d_track_3D_index[tid].num_segments; i++){
    }
    tid += blockDim.x * gridDim.x; 
  }
}
//因为没有atomicadd加的
__device__ double myatomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void printmateriald(dev_material* matd)
{
  printf("Printing the material %i\n", matd->_id);
  for (int g=0; g<NUM_GROUPS; ++g)
    printf("    sigmaf(%i) = %f\n", g, matd->_sigma_f[g]);
}

/**
 * @brief A struct used to check if a value on the GPU is equal to INF.
 * @details This is used as a predicate in Thrust routines.
 */
struct isinf_test {
  /**
   * @brief Checks if a double precision value is INF.
   * @param a the value to check
   * @return true if equal to INF, false otherwise
   */
  __host__ __device__ bool operator()(double a) {
    return isinf(a);
  }

  /**
   * @brief Checks if a single precision value is INF.
   * @param a the value to check
   * @return true if equal to INF, false otherwise
   */
  __host__ __device__ bool operator()(float a) {
    return isinf(a);
  }
};


/**
 * @brief A struct used to check if a value on the GPU is equal to NaN.
 * @details This is used as a predicate in Thrust routines.
 */
struct isnan_test {
  /**
   * @brief Checks if a double precision value is NaN.
   * @param a the value to check
   * @return true if equal to NaN, false otherwise
   */
  __host__ __device__ bool operator()(double a) {
    return isnan(a);
  }

  /**
   * @brief Checks if a single precision value is NaN.
   * @param a the value to check
   * @return true if equal to NaN, false otherwise
   */
  __host__ __device__ bool operator()(float a) {
    return isnan(a);
  }
};

/**
 * @brief A functor to multiply all elements in a Thrust vector by a constant.
 * @param constant the constant to multiply the vector
 */
template <typename T>
struct multiplyByConstant {

public:
  /* The constant to multiply by */
  const T constant;

  /**
   * @brief Constructor for the functor.
   * @param constant to multiply each element in a Thrust vector
   */
  multiplyByConstant(T constant) : constant(constant) {}

  /**
   * @brief Multiply an element in a Thrust vector.
   * @param VecElem the element to multiply
   */
  __host__ __device__ void operator()(T& VecElem) const {
    VecElem = VecElem * constant;
  }
};


/**
 * @class This provides a templated interface for a strided iterator over
 *        a Thrust device_vector on a GPU.
 * @details This code is taken from the Thrust examples site on 1/20/2015:
 *           https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
 */
template <typename Iterator>
class strided_range {

public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor : public thrust::unary_function<difference_type,difference_type> {

    difference_type stride;

    stride_functor(difference_type stride) : stride(stride) { }

    __host__ __device__ difference_type operator()(const difference_type& i) const {
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<difference_type> CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator>
    TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>
    PermutationIterator;
  typedef PermutationIterator iterator;

  /**
   * @brief The strided iterator constructor.
   */
  strided_range(Iterator first, Iterator last, difference_type stride)
    : first(first), last(last), stride(stride) { }

  /**
   * @brief Get the first element in the iterator.
   * @return the first element in the iterator
   */
  iterator begin(void) const {
    return PermutationIterator(first,
      TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  /**
   * @brief Get the last element in the iterator.
   * @return the last element in the iterator
   */
  iterator end(void) const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

protected:

  /** The first element in the underlying device_vector as set by the constructor */
  Iterator first;

  /** The last element in the underlying device_vector as set by the constructor */
  Iterator last;

  /** The stride to use when iterating over the underlying device_vector */
  difference_type stride;

};


/**
 * @brief Compute the stabilizing flux
 * @param scalar_flux the scalar flux in each FSR and energy group
 * @param stabilizing_flux the array of stabilizing fluxes
 */
__global__ void computeStabilizingFluxOnDevice(FP_PRECISION* scalar_flux,
                                               FP_PRECISION* stabilizing_flux)
{
  if (stabilization_type == GLOBAL)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double multiplicative_factor = 1.0/stabilization_factor - 1.0;
    while (tid < num_FSRs * NUM_GROUPS)
    {
      stabilizing_flux[tid] = multiplicative_factor * scalar_flux[tid];
      tid += blockDim.x * gridDim.x;
    }
  }
}


/**
 * @brief Stabilize the current flux with a previously computed stabilizing flux
 * @param scalar_flux the scalar flux in each FSR and energy group
 * @param stabilizing_flux the array of stabilizing fluxes
 */
__global__ void stabilizeFluxOnDevice(FP_PRECISION* scalar_flux,
                                      FP_PRECISION* stabilizing_flux)
{
  if (stabilization_type == GLOBAL)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < num_FSRs * NUM_GROUPS)
    {
      scalar_flux[tid] += stabilizing_flux[tid];
      scalar_flux[tid] *= stabilization_factor;
      tid += blockDim.x * gridDim.x;
    }
  }
}


/**
 * @brief Compute the total fission source from all FSRs.
 * @param FSR_volumes an array of FSR volumes
 * @param FSR_materials an array of FSR Material indices
 * @param materials an array of dev_materials on the device
 * @param scalar_flux the scalar flux in each FSR and energy group
 * @param fission_sources array of fission sources in each FSR and energy group
 */
__global__ void computeFissionSourcesOnDevice(FP_PRECISION* FSR_volumes,
                                              int* FSR_materials,
                                              dev_material* materials,
                                              FP_PRECISION* scalar_flux,
                                              FP_PRECISION* fission_sources) {

  /* Use a shared memory buffer for each thread's fission source */
  extern __shared__ FP_PRECISION shared_fission_source[];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  dev_material* curr_material;
  FP_PRECISION* nu_sigma_f;
  FP_PRECISION volume, source;

  /* Initialize fission source to zero */
  shared_fission_source[threadIdx.x] = 0;

  /* Iterate over all FSRs */
  while (tid < num_FSRs) {

    curr_material = &materials[FSR_materials[tid]];
    nu_sigma_f = curr_material->_nu_sigma_f;
    volume = FSR_volumes[tid];

    /* Iterate over energy groups and update fission source for
     * this thread block */
    for (int e=0; e < NUM_GROUPS; e++) {
      source = nu_sigma_f[e] * scalar_flux(tid,e) * volume;
      shared_fission_source[threadIdx.x] += source;
    }

    /* Increment thread id */
    tid += blockDim.x * gridDim.x;
  }

  /* Copy this thread's fission source to global memory */
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  fission_sources[tid] = shared_fission_source[threadIdx.x];
}


/**
 * @brief Computes the total source (fission, scattering, fixed) in each FSR.
 * @details This method computes the total source in each region based on
 *          this iteration's current approximation to the scalar flux.
 * @param FSR_materials an array of FSR Material indices
 * @param materials an array of dev_material pointers
 * @param scalar_flux an array of FSR scalar fluxes
 * @param fixed_sources an array of fixed (user-defined) sources
 * @param reduced_sources an array of FSR sources / total xs
 * @param inverse_k_eff the inverse of keff
 * @param zeroNegatives whether to zero out negative fluxes
 */
__global__ void computeFSRSourcesOnDevice(int* FSR_materials,
                                          dev_material* materials,
                                          FP_PRECISION* scalar_flux,
                                          FP_PRECISION* fixed_sources,
                                          FP_PRECISION* reduced_sources,
                                          FP_PRECISION inverse_k_eff,
                                          bool zeroNegatives) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  FP_PRECISION fission_source;
  FP_PRECISION scatter_source;

  dev_material* curr_material;
  FP_PRECISION* sigma_t;
  FP_PRECISION* sigma_s;
  FP_PRECISION* fiss_mat;

  /* Iterate over all FSRs */
  while (tid < num_FSRs) {

    curr_material = &materials[FSR_materials[tid]];

    sigma_t = curr_material->_sigma_t;
    sigma_s = curr_material->_sigma_s;
    fiss_mat = curr_material->_fiss_matrix;

    /* Compute scatter + fission source for group g */
    for (int g=0; g < NUM_GROUPS; g++) {
      scatter_source = 0;
      fission_source = 0;

      for (int g_prime=0; g_prime < NUM_GROUPS; g_prime++) {
        scatter_source += sigma_s[g*NUM_GROUPS+g_prime] * scalar_flux(tid,g_prime);
        fission_source += fiss_mat[g*NUM_GROUPS+g_prime] * scalar_flux(tid,g_prime);
      }

      fission_source *= inverse_k_eff;

      /* Compute total (scatter+fission+fixed) reduced source */
      reduced_sources(tid,g) = fixed_sources(tid,g);
      reduced_sources(tid,g) += scatter_source + fission_source;
      reduced_sources(tid,g) *= ONE_OVER_FOUR_PI;
      if(!_solve_3D) reduced_sources(tid,g) = __fdividef(reduced_sources(tid,g), sigma_t[g]);
      if (zeroNegatives && reduced_sources(tid,g) < 0.0) reduced_sources(tid,g) = 0.0;
    }

    /* Increment the thread id */
    tid += blockDim.x * gridDim.x;
  }
}


/**
 * @brief Computes the total fission source in each FSR in each energy group
 * @details This method is a helper routine for the openmoc.krylov submodule.
 *          This routine computes the total fission source in each FSR. If the
 *          divide_sigma_t parameter is true then the fission source will be
 *          divided by the total cross-section in each FSR.
 * @param FSR_materials an array of FSR Material indices
 * @param materials an array of dev_material pointers
 * @param divide_sigma_t a boolean indicating whether to divide by the total xs
 * @param scalar_flux an array of FSR scalar fluxes
 * @param reduced_sources an array of FSR fission sources
 */
__global__ void computeFSRFissionSourcesOnDevice(int* FSR_materials,
                                                 dev_material* materials,
						 bool divide_sigma_t,
                                                 FP_PRECISION* scalar_flux,
                                                 FP_PRECISION* reduced_sources) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  FP_PRECISION fission_source;

  dev_material* curr_material;
  FP_PRECISION* sigma_t;
  FP_PRECISION* fiss_mat;

  /* Iterate over all FSRs */
  while (tid < num_FSRs) {

    curr_material = &materials[FSR_materials[tid]];

    sigma_t = curr_material->_sigma_t;
    fiss_mat = curr_material->_fiss_matrix;

    /* Compute fission source for group g */
    for (int g=0; g < NUM_GROUPS; g++) {
      fission_source = 0;

      for (int g_prime=0; g_prime < NUM_GROUPS; g_prime++)
        fission_source += fiss_mat[g*NUM_GROUPS+g_prime] * scalar_flux(tid,g_prime);

      /* Set the reduced fission source for FSR tid in group g */
      reduced_sources(tid,g) = fission_source;
      reduced_sources(tid,g) *= ONE_OVER_FOUR_PI;
      if (divide_sigma_t)
        reduced_sources(tid,g) = __fdividef(reduced_sources(tid,g), sigma_t[g]);
    }

    /* Increment the thread id */
    tid += blockDim.x * gridDim.x;
  }
}


/**
 * @brief Computes the total scattering source in each FSR and energy group.
 * @details This method is a helper routine for the openmoc.krylov submodule.
 *          This routine computes the total scatter source in each FSR. If the
 *          divide_sigma_t parameter is true then the scatter source will be
 *          divided by the total cross-section in each FSR.
 * @param FSR_materials an array of FSR Material indices
 * @param materials an array of dev_material pointers
 * @param divide_sigma_t a boolean indicating whether to divide by the total xs
 * @param scalar_flux an array of FSR scalar fluxes
 * @param reduced_sources an array of FSR scatter sources
 */
__global__ void computeFSRScatterSourcesOnDevice(int* FSR_materials,
                                                 dev_material* materials,
						 bool divide_sigma_t,
                                                 FP_PRECISION* scalar_flux,
                                                 FP_PRECISION* reduced_sources) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  FP_PRECISION scatter_source;

  dev_material* curr_material;
  FP_PRECISION* sigma_s;
  FP_PRECISION* sigma_t;

  /* Iterate over all FSRs */
  while (tid < num_FSRs) {

    curr_material = &materials[FSR_materials[tid]];

    sigma_s = curr_material->_sigma_s;
    sigma_t = curr_material->_sigma_t;

    /* Compute total scattering source for this FSR in group g */
    for (int g=0; g < NUM_GROUPS; g++) {
      scatter_source = 0;

      for (int g_prime=0; g_prime < NUM_GROUPS; g_prime++)
        scatter_source += sigma_s[g*NUM_GROUPS+g_prime] * scalar_flux(tid,g_prime);

      /* Set the reduced scatter source for FSR tid in group g */
      reduced_sources(tid,g) = scatter_source;
      reduced_sources(tid,g) *= ONE_OVER_FOUR_PI;
      if (divide_sigma_t)
        reduced_sources(tid,g) = __fdividef(reduced_sources(tid,g), sigma_t[g]);
    }

    /* Increment the thread id */
    tid += blockDim.x * gridDim.x;
  }
}


/**
 * @brief Set all FSR spectra to match chi of a given material
 * @param chi_material pointer to device material's chi spectrum to use
 * @param scalar_flux the array of FSR scalar fluxes
 */
__global__ void flattenFSRFluxesChiSpectrumOnDevice(dev_material* chi_material,
                                                    FP_PRECISION* scalar_flux) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < num_FSRs) {
    for (int g=0; g < NUM_GROUPS; g++) {
      scalar_flux(tid,g) = chi_material->_chi[g];
    }
    tid += blockDim.x * gridDim.x;
  }
}


/**
 * @brief Computes the contribution to the FSR scalar flux from a Track
 *        segment in a single energy group.
 * @details This method integrates the angular flux for a Track segment across
 *        energy groups and polar angles, and tallies it into the FSR scalar
 *        flux, and updates the Track's angular flux.
 * @param curr_segment a pointer to the Track segment of interest
 * @param azim_index a pointer to the azimuthal angle index for this segment
 * @param energy_group the energy group of interest
 * @param materials the array of dev_material pointers
 * @param track_flux a pointer to the Track's angular flux
 * @param reduced_sources the array of FSR sources / total xs
 * @param scalar_flux the array of FSR scalar fluxes
 */
__device__ void tallyScalarFlux(dev_segment* curr_segment,
                                int azim_index,
                                int energy_group,
                                dev_material* materials,
                                float* track_flux,
                                FP_PRECISION* reduced_sources,
                                FP_PRECISION* scalar_flux,
								FP_PRECISION* fsr_flux
								) {

  long fsr_id = curr_segment->_region_uid;
  FP_PRECISION length = curr_segment->_length;
  dev_material* curr_material = &materials[curr_segment->_material_index];
  FP_PRECISION* sigma_t = curr_material->_sigma_t;

  /* The change in angular flux long this Track segment in this FSR */
  FP_PRECISION delta_psi;
  FP_PRECISION exponential;

  /* Zero the FSR scalar flux contribution from this segment and energy group */ 
  if(_solve_3D)
  {
     FP_PRECISION tau= sigma_t[energy_group] * length;
     /* Compute the exponential */
     exponential= dev_exponential_(tau);
	 /* Compute attenuation and tally the contribution to the scalar flux */
     FP_PRECISION delta_psi = (sigma_t[energy_group] * track_flux[energy_group] -reduced_sources(fsr_id, energy_group));

     delta_psi=delta_psi*exponential*length;	
     track_flux[energy_group] -= delta_psi;
     fsr_flux[0] += delta_psi;
  }
  else
  {
     FP_PRECISION fsr_flux = 0.0;
    /* Loop over polar angles */
    for (int p=0; p < num_polar_2; p++) {
      exponential =
        dev_exponential(sigma_t[energy_group] * length / sin_thetas[p]);
      delta_psi = (track_flux[p] - reduced_sources(fsr_id,energy_group));
      delta_psi *= exponential;
      fsr_flux += delta_psi * weights(azim_index,p);
      track_flux[p] -= delta_psi;
    }
    /* Atomically increment the scalar flux for this FSR */
    myatomicAdd(&scalar_flux(fsr_id,energy_group), fsr_flux);
  }
 
}


/**
 * @brief Updates the boundary flux for a Track given boundary conditions.
 * @details For reflective and periodic boundary conditions, the outgoing
 *          boundary flux for the Track is given to the corresponding reflecting
 *          or periodic Track. For vacuum boundary conditions, the outgoing flux
 *          is tallied as leakage. Note: Only one energy group is transferred
 *          by this routine.
 * @param curr_track a pointer to the Track of interest
 * @param azim_index a pointer to the azimuthal angle index for this segment
 * @param track_flux an array of the outgoing Track flux
 * @param boundary_flux an array of all angular fluxes
 * @param weights an array of Quadrature weights
 * @param energy_angle_index the energy group index
 * @param direction the Track direction (forward - true, reverse - false)
 */
__device__ void transferBoundaryFlux(dev_track* curr_track,
                                     int azim_index,
                                     float* track_flux,
                                     float* boundary_flux,
                                     int energy_angle_index,
                                     bool direction
                                     ) {

  int start = 0;
 boundaryType  transfer_flux;
  int track_out_id;

  /* For the "forward" direction */
  if (direction) {
    transfer_flux = curr_track->_transfer_flux_fwd;
    track_out_id = curr_track->_next_track_fwd;
    start = (!curr_track->_next_fwd_is_fwd) * polar_times_groups;
  }

  /* For the "reverse" direction */
  else {
    transfer_flux = curr_track->_transfer_flux_bwd;
    track_out_id = curr_track->_next_track_bwd;
    start = (!curr_track->_next_bwd_is_fwd )* polar_times_groups;
  }

  if(_solve_3D&&(transfer_flux == REFLECTIVE || transfer_flux == PERIODIC))
  {
    float* track_out_flux = &boundary_flux(track_out_id,start);
    track_out_flux[energy_angle_index] = track_flux[energy_angle_index];
  }
  else if(!_solve_3D)
  {		   
	  float* track_out_flux = &boundary_flux(track_out_id,start);

	  for (int p=0; p < num_polar_2; p++)
		track_out_flux[p] = track_flux[p] * transfer_flux;
  }
}

/**
 * @brief This method performs one transport sweep of one halfspace of all
 *        azimuthal angles, tracks, segments, polar angles and energy groups.
 * @details The method integrates the flux along each track and updates the
 *          boundary fluxes for the corresponding output Track, while updating
 *          the scalar flux in each FSR.
 * @param scalar_flux an array of FSR scalar fluxes
 * @param boundary_flux an array of Track boundary fluxes
 * @param reduced_sources an array of FSR sources / total xs
 * @param materials an array of dev_material pointers
 * @param tracks an array of Tracks
 * @param tid_offset the Track offset for azimuthal angle halfspace
 * @param tid_max the upper bound on the Track IDs for this azimuthal
 *                angle halfspace
 */
__global__ void transportSweepOnDevice(FP_PRECISION* scalar_flux,
                                       float* boundary_flux,
                                       float* start_flux,
                                       FP_PRECISION* reduced_sources,
                                       dev_material* materials,
                                       dev_track* tracks,
                                       long tid_offset,
                                       long tid_max) {

  /* Shared memory buffer for each thread's angular flux */
  extern __shared__ FP_PRECISION temp_flux[];
  float* track_flux;

  int tid = tid_offset + threadIdx.x + blockIdx.x * blockDim.x;
  int track_id = tid / NUM_GROUPS;

//  int track_flux_index = threadIdx.x * num_polar;
  int energy_group = tid % NUM_GROUPS;
  int energy_angle_index = energy_group;// * num_polar_2;

  dev_track* curr_track;
  int azim_index;
  int p;
  int num_segments;
  dev_segment* curr_segment;

  /* Iterate over Track with azimuthal angles in (0, pi/2) */
  while (track_id < tid_max) {

    /* Initialize local registers with important data */
    curr_track = &tracks[track_id];
    azim_index = curr_track->_azim_angle_index;
   	p=curr_track->_polar_index;						
    num_segments = curr_track->_num_segments;
    double  weight=weights(azim_index,p);//for 3D
   
    track_flux = &boundary_flux(track_id,0);
    FP_PRECISION fsr_flux[1]={0.0};
    /* Loop over each Track segment in forward direction */
    for (int i=0; i < num_segments; i++) {
      curr_segment = &curr_track->_segments[i];
	  long fsr_id = curr_segment->_region_uid;
      tallyScalarFlux(curr_segment, azim_index,energy_group, materials,
                      track_flux, reduced_sources, scalar_flux,fsr_flux); 
	if (i < num_segments - 1 && _solve_3D
		  &&  fsr_id != ((&curr_track->_segments[i+1])->_region_uid))	 
      {
         myatomicAdd(&scalar_flux(fsr_id,energy_group), weight*fsr_flux[0]);
         fsr_flux[0] =0.0;
	  }
	}
    /* Transfer boundary angular flux to outgoing Track */
    transferBoundaryFlux(curr_track, azim_index, track_flux, start_flux,
                         energy_angle_index, true);

    
    track_flux = &boundary_flux(track_id,NUM_GROUPS);
    for (int i=num_segments-1; i > -1; i--) {
      curr_segment = &curr_track->_segments[i];
	  long fsr_id = curr_segment->_region_uid;  
      tallyScalarFlux(curr_segment, azim_index,energy_group, materials,
                      track_flux, reduced_sources, scalar_flux,fsr_flux);
	  if ((i==0  || (fsr_id != (&curr_track->_segments[i-1])->_region_uid))
		  &&_solve_3D )
	  {
       myatomicAdd(&scalar_flux(fsr_id,energy_group), weight*fsr_flux[0]);
       fsr_flux[0] =0.0;
	  } 
    }

    /* Transfer boundary angular flux to outgoing Track */
    transferBoundaryFlux(curr_track, azim_index, track_flux, start_flux,
                        energy_angle_index, false);

    /* Update the indices for this thread to the next Track, energy group */
    tid += blockDim.x * gridDim.x;
    track_id = tid / NUM_GROUPS;
    energy_group = tid % NUM_GROUPS;
    energy_angle_index = energy_group;// * (num_polar_2);
  }
}


/**
 * @brief Add the source term contribution in the transport equation to
 *        the FSR scalar flux on the GPU.
 * @param scalar_flux an array of FSR scalar fluxes
 * @param reduced_sources an array of FSR sources / total xs
 * @param FSR_volumes an array of FSR volumes
 * @param FSR_materials an array of FSR material indices
 * @param materials an array of dev_material pointers
 */
__global__ void addSourceToScalarFluxOnDevice(FP_PRECISION* scalar_flux,
                                              FP_PRECISION* reduced_sources,
                                              FP_PRECISION* FSR_volumes,
                                              int* FSR_materials,
                                              dev_material* materials) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  FP_PRECISION volume;

  dev_material* curr_material;
  FP_PRECISION* sigma_t;

  /* Iterate over all FSRs */
  while (tid < num_FSRs) {

    curr_material = &materials[FSR_materials[tid]];
    volume = FSR_volumes[tid];
    sigma_t = curr_material->_sigma_t;
    if (volume < FLT_EPSILON)
      volume = 1e30;
    /* Iterate over all energy groups */
    for (int i=0; i < NUM_GROUPS; i++) {
      scalar_flux(tid,i) /=(// __fdividef(scalar_flux(tid,i),
                                     (sigma_t[i] * volume));
      scalar_flux(tid,i) += FOUR_PI * reduced_sources(tid,i)/sigma_t[i];
    }

    /* Increment thread id */
    tid += blockDim.x * gridDim.x;
  }
}


/**
 * @brief Compute the total volume-intergrated fission source from
 *        all FSRs and energy groups.
 * @param FSR_volumes an array of the FSR volumes
 * @param FSR_materials an array of the FSR Material indices
 * @param materials an array of the dev_material pointers
 * @param scalar_flux an array of FSR scalar fluxes
 * @param fission an array of FSR nu-fission rates
 * @param nu whether total neutron production rate should be calculated
 * @param computing_fission_norm This gets set to true if integrating total
 *            fission source, otherwise this kernel calculates the local
 *            fission source. In short, it switches on a parallel reduction.
 */
__global__ void computeFSRFissionRatesOnDevice(FP_PRECISION* FSR_volumes,
                                               int* FSR_materials,
                                               dev_material* materials,
                                               FP_PRECISION* scalar_flux,
                                               FP_PRECISION* fission,
                                               bool nu = false,
                                               bool computing_fission_norm= false) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  dev_material* curr_material;
  FP_PRECISION* fiss_xs;
  FP_PRECISION volume;

  FP_PRECISION fiss = 0.;

  /* Iterate over all FSRs */
  while (tid < num_FSRs) {

    curr_material = &materials[FSR_materials[tid]];

    if (nu) {
      fiss_xs = curr_material->_nu_sigma_f;
    }
    else {
      fiss_xs = curr_material->_sigma_f;
    }

    volume = FSR_volumes[tid];

    FP_PRECISION curr_fiss = 0.;

    /* Compute fission rates rates for this thread block */
    for (int e=0; e < NUM_GROUPS; e++)
      curr_fiss += fiss_xs[e] * scalar_flux(tid,e);

    fiss += curr_fiss * volume;

    if (!computing_fission_norm)
      fission[tid] = curr_fiss * volume;

    /* Increment thread id */
    tid += blockDim.x * gridDim.x;

  }

  /* Copy this thread's fission to global memory */
  if (computing_fission_norm) {
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    fission[tid] = fiss;
  }
}

//从这往下为我自己的cuda函数
//初始化tci
__device__ void devTrackChainIndexes(dev_TrackChainIndexes* tci){
  tci->_azim = -1;
  tci->_x = -1;
  tci->_polar = -1;
  tci->_lz = -1;
  tci->_link = -1;
}

//初始化tsi
__device__ void devTrackStackIndexes(dev_TrackStackIndexes* tsi){
  tsi->_azim = -1;
  tsi->_xy = -1;
  tsi->_polar = -1;
  tsi->_z = -1;
}

//用与索引3维数组
__device__ int get3Dindex(int a, int* d_num_x, int* d_num_y){
    int index = 0;
    for (int i = 0; i < a; i++)
    {
        index =index+ d_num_x[i]+d_num_y[i];
    }
    return index;
}

//建立chain的索引
__device__ int getchainIndex(dev_TrackChainIndexes* tci,
                             int* d_track_2D_chain_link,
                             int* d_num_x){
    int i = 0;
    for (int j = 0; j < tci->_azim; j++)
    {
        i = d_num_x[j] + i;
    }
    i = i+tci->_x;
    int sum =0;
    for (int k = 0; k < i; k++)
    {
        sum = sum + d_track_2D_chain_link[k];
    }
    return sum;
    
}

//getFirst2DTrackLinkIndex函数的移植
__device__ int getFirst2DTrackLinkIndexonGPU(dev_TrackChainIndexes* tci,
                                             dev_track2D* d_track_2D_chain,
                                             int* d_track_2D_chain_link,
                                             int* d_num_x,
                                             int** d_num_z,
                                             int** d_num_l,
                                             double** d_dz_eff,
                                             double** d_dl_eff,
                                             dev_track3D* track_3D){
    int i = getchainIndex(tci, d_track_2D_chain_link, d_num_x);
    dev_track2D track_2D = d_track_2D_chain[i];
    
    double phi = track_2D._phi;
    double cos_phi  = cos(phi);
    double sin_phi  = sin(phi);

    double x_start  = track_2D._start._xyz[0];
    double y_start  = track_2D._start._xyz[1];
    int lz          = tci->_lz;
    int nl          = d_num_l[tci->_azim][tci->_polar];
    int nz          = d_num_z[tci->_azim][tci->_polar];
    double dz       = d_dz_eff[tci->_azim][tci->_polar];
    double dl       = d_dl_eff[tci->_azim][tci->_polar];

    double width_x = d_x_max - d_x_min;
    double width_y = d_y_max - d_y_min;

    double l_start  = 0.0;

    if (tci->_polar < num_polar / 2 && lz < nl)
    l_start = width_y / sin_phi - (lz + 0.5) * dl;
    else if (tci->_polar >= num_polar / 2 && lz >= nz)
    l_start = dl * (lz - nz + 0.5);

    double x_ext = x_start - d_x_min + l_start * cos_phi;
    double y_ext = y_start - d_y_min + l_start * sin_phi;

    bool nudged = false;

    if (fabs(l_start) > FLT_EPSILON) {
    if (fabs(round(x_ext / width_x) * width_x - x_ext) < TINY_MOVE ||
        fabs(round(y_ext / width_y) * width_y - y_ext) < TINY_MOVE) {
      l_start += 10 * TINY_MOVE;
      x_ext = x_start - d_x_min + l_start * cos_phi;
      y_ext = y_start - d_y_min + l_start * sin_phi;
      nudged = true;
    }
  }

  int link_index = abs(int(floor(x_ext / width_x)));
  
  double x1;
  if (x_ext < 0.0)
    x1 = fmod(x_ext, width_x) + d_x_max;
  else
    x1 = fmod(x_ext, width_x) + d_x_min;

  double y1 = fmod(y_ext, width_y) + d_y_min;
 
  double z1;
  double z2;
  if (tci->_polar < num_polar / 2) {
    z1 = d_z_min + max(0., (lz - nl + 0.5)) * dz;
    z2 = d_z_max + min(0., (lz - nz + 0.5)) * dz;
  }
  else {
    z1 = d_z_max + min(0., (lz - nz + 0.5)) * dz;
    z2 = d_z_min + max(0., (lz - nl + 0.5)) * dz;
  }

  if (nudged) {
    x1 -= 10 * TINY_MOVE * cos_phi;
    y1 -= 10 * TINY_MOVE * sin_phi;
  }

  track_3D->_start._xyz[0] = x1;
  track_3D->_start._xyz[1] = y1;
  track_3D->_start._xyz[2] = z1;
  track_3D->_end._xyz[2] = z2;

  return link_index;
}

//setlinkindex函数的移植
__device__ void setLinkIndexonGPU(dev_TrackStackIndexes* tsi, 
                                  dev_TrackChainIndexes* tci,
                                  dev_track2D* d_track_2D_chain,
                                  int* d_track_2D_chain_link,
                                  int* d_num_x,
                                  int** d_num_z,
                                  int** d_num_l,
                                  double** d_dz_eff,
                                  double** d_dl_eff,
                                  int** d_tracks_2D_xy,
                                  int* d_num_y, dev_track2D* dev_tracks_2d){
    dev_track3D track;
    int first_link = getFirst2DTrackLinkIndexonGPU(tci, d_track_2D_chain, d_track_2D_chain_link, 
                                                   d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, &track);
    dev_track2D* track_2D = &dev_tracks_2d[d_tracks_2D_xy[tsi->_azim][tsi->_xy]];
    tci->_link = track_2D->_link_index - first_link;
 }

//tsi转换为tci的函数
__device__ void convertTsitoTCIonGPU(dev_TrackStackIndexes* tsi, 
                                     dev_TrackChainIndexes* tci, 
                                     int* d_first_lz_of_stack, 
                                     int* d_num_x, 
                                     int* d_num_y,
                                     dev_track2D* d_track_2D_chain,
                                     int* d_track_2D_chain_link,
                                     int** d_num_z,
                                     int** d_num_l,
                                     double** d_dz_eff,
                                     double** d_dl_eff,
                                     int** d_tracks_2D_xy, dev_track2D* dev_tracks_2d,int* d_azim_size){
    tci->_azim = tsi->_azim;
    tci->_x = tsi->_xy % d_num_x[tsi->_azim];
    tci->_polar = tsi->_polar;
    tci->_lz = d_first_lz_of_stack[d_azim_size[tsi->_azim]+tsi->_xy*num_polar+tsi->_polar ] + tsi->_z;
    setLinkIndexonGPU(tsi, tci, d_track_2D_chain, d_track_2D_chain_link, 
                      d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, d_tracks_2D_xy, 
                      d_num_y, dev_tracks_2d);
}

//gettheta函数的移植

__device__ double getThetaonGPU(int azim, int polar, double* d_theta){
  if (azim >= num_azim/2)
    azim = num_azim - azim - 1;
  int index = azim*num_polar + polar;
  return d_theta[index];
}

//set3DTrackDataonGPU的移植
__device__ void set3DTrackDataonGPU(dev_TrackChainIndexes* tci, 
                                    dev_track3D* track,
                                    double* d_theta,
                                    dev_track2D* d_track_2D_chain,
                                    int* d_track_2D_chain_link,
                                    int* d_num_x,
                                    int** d_num_z,
                                    int** d_num_l,
                                    double** d_dz_eff,
                                    double** d_dl_eff,
                                    int* d_num_y){
  double theta = getThetaonGPU(tci->_azim, tci->_polar, d_theta);
  bool end_of_chain = false;

  int link = getFirst2DTrackLinkIndexonGPU(tci, d_track_2D_chain, d_track_2D_chain_link, 
                                           d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, track);

  int first_link = link;  
  double x1;
  double y1;
  double z1;

  double x2 = track->_start._xyz[0];
  double y2 = track->_start._xyz[1];
  double z2 = track->_start._xyz[2];

  while (!end_of_chain) {

    int i = getchainIndex(tci, d_track_2D_chain_link, d_num_x);
    i = i + link;
    dev_track2D track_2D = d_track_2D_chain[i];
    double phi = track_2D._phi;

    x1 = x2;
    y1 = y2;
    z1 = z2;

    double dx;
    double dy;
    double dl_xy;

    if (link == first_link) {
      dx = track_2D._end._xyz[0] - x1;
      dy = track_2D._end._xyz[1] - y1;
      dl_xy = sqrt(dx*dx + dy*dy);
    }

    else
      dl_xy = track_2D._segment._length;

    double dl_z;
    if (tci->_polar < num_polar / 2)
      dl_z = (track->_end._xyz[2] - z1) * tan(theta);
    else
      dl_z = (z1 - track->_end._xyz[2]) / tan(theta - M_PI_2);

    double dl = min(dl_z, dl_xy);

    x2 = x1 + dl * cos(phi);
    y2 = y1 + dl * sin(phi);

    if (tci->_polar < num_polar / 2)
      z2 = z1 + dl / tan(theta);
    else
      z2 = z1 - dl * tan(theta - M_PI_2);

    if (fabs(x2 - x1) < TINY_MOVE ||
        fabs(y2 - y1) < TINY_MOVE ||
        fabs(z2 - z1) < TINY_MOVE)
      break;

    if (dl_z < dl_xy || track_2D._xy_index >= d_num_y[tci->_azim] ||
        tci->_link == link - first_link)
      end_of_chain = true;

    link++;

    if (!end_of_chain) {
      if (tci->_azim < num_azim / 4)
        x2 = d_x_min;
      else
        x2 = d_x_max;
    }
  }
  
  x1 = max(d_x_min, min(d_x_max, x1));
  y1 = max(d_y_min, min(d_y_max, y1));
  z1 = max(d_z_min, min(d_z_max, z1));
  x2 = max(d_x_min, min(d_x_max, x2));
  y2 = max(d_y_min, min(d_y_max, y2));
  z2 = max(d_z_min, min(d_z_max, z2));

  track->_start._xyz[0] = x1;
  track->_start._xyz[1] = y1;
  track->_start._xyz[2] = z1;
  track->_end._xyz[0] =x2;
  track->_end._xyz[1] =y2;
  track->_end._xyz[2] =z2;

  if (tci->_link == -1)
    tci->_link = link - first_link;

}

//getNum3DTrackChainLinks的移植
__device__ int getNum3DTrackChainLinksonGPU(dev_TrackChainIndexes* tci, 
                                            dev_track2D* d_track_2D_chain,
                                            int* d_track_2D_chain_link,
                                            int* d_num_x,
                                            int** d_num_z,
                                            int** d_num_l,
                                            double** d_dz_eff,
                                            double** d_dl_eff,
                                            int* d_num_y,
                                            int* d_first_lz_of_stack, 
                                            int* d_tracks_per_stack,int * d_azim_size){
  dev_track3D track_3D;
  int first_link = getFirst2DTrackLinkIndexonGPU(tci, d_track_2D_chain, d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, &track_3D);
  int link = first_link;
  
  int i = getchainIndex(tci, d_track_2D_chain_link, d_num_x);
  i = i + link;
  dev_track2D track_2D = d_track_2D_chain[i];
  int azim = track_2D._azim_index;
  int xy = track_2D._xy_index;
  int polar = tci->_polar;
  int min_lz, max_lz;
  int index = 0;
  while (true) {
    index = getchainIndex(tci, d_track_2D_chain_link, d_num_x);
    index = index + link;
    track_2D = d_track_2D_chain[index];
    azim = track_2D._azim_index;
    xy = track_2D._xy_index;

    min_lz = d_first_lz_of_stack[d_azim_size[azim]+xy*num_polar+polar ];
    max_lz = d_tracks_per_stack[d_azim_size[azim]+xy*num_polar+polar ] + min_lz - 1;

    if (tci->_polar < num_polar / 2 && tci->_lz > max_lz)
      break;
    else if (tci->_polar >= num_polar / 2 && tci->_lz < min_lz)
      break;

    link++;

    if (track_2D._xy_index >= d_num_y[azim])
      break;
  }

  return link - first_link;
}

//convertTCItoTSIonGPU的移植
__device__ void convertTCItoTSIonGPU(dev_TrackChainIndexes* tci, 
                                     dev_TrackStackIndexes* tsi,
                                     dev_track2D* d_track_2D_chain,
                                     int* d_track_2D_chain_link,
                                     int* d_num_x,
                                     int** d_num_z,
                                     int** d_num_l,
                                     double** d_dz_eff,
                                     double** d_dl_eff,
                                     int* d_first_lz_of_stack,
                                     int* d_num_y,int* d_azim_size){
  dev_track3D track;
  int link = getFirst2DTrackLinkIndexonGPU(tci, d_track_2D_chain, d_track_2D_chain_link, 
                                           d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, &track) + tci->_link;
  int i = getchainIndex(tci, d_track_2D_chain_link, d_num_x);
  i = i + link;
  dev_track2D* track_2D = &d_track_2D_chain[i];
  tsi->_azim = tci->_azim;
  tsi->_xy = track_2D->_xy_index;
  tsi->_polar = tci->_polar;
  tsi->_z = tci->_lz - d_first_lz_of_stack[d_azim_size[tsi->_azim]+tsi->_xy*num_polar+tsi->_polar ];
}

//get3DTrackID的移植
__device__ long get3DTrackID(dev_TrackStackIndexes* tsi, long* d_cum_tracks_per_stack, int* d_num_x, int* d_num_y,int* d_azim_size){
  return d_cum_tracks_per_stack[d_azim_size[tsi->_azim]+tsi->_xy*num_polar+tsi->_polar ] + tsi->_z;
}

//setLinkingTracksonGPU的移植
__device__ void setLinkingTracksonGPU(dev_TrackStackIndexes* tsi, 
                                     dev_TrackChainIndexes* tci, 
                                     int* d_first_lz_of_stack, 
                                     int* d_tracks_per_stack,
                                     int* d_num_x, 
                                     int* d_num_y,
                                     dev_track2D* d_track_2D_chain,
                                     int* d_track_2D_chain_link,
                                     int** d_num_z,
                                     int** d_num_l,
                                     double** d_dz_eff,
                                     double** d_dl_eff,
                                     int** d_tracks_2D_xy,
                                     bool outgoing, dev_track3D* track,
                                     dev_track2D* dev_tracks_2d,
                                     long* d_cum_tracks_per_stack,int * d_azim_size){
  dev_track2D* track_2D = &dev_tracks_2d[d_tracks_2D_xy[tsi->_azim][tsi->_xy]];
  
  dev_TrackChainIndexes tci_next;
  dev_TrackChainIndexes tci_prdc;
  dev_TrackChainIndexes tci_refl;
  dev_TrackStackIndexes tsi_next;
  dev_TrackStackIndexes tsi_prdc;
  dev_TrackStackIndexes tsi_refl;
  devTrackChainIndexes(&tci_next);
  devTrackChainIndexes(&tci_prdc);
  devTrackChainIndexes(&tci_refl);
  devTrackStackIndexes(&tsi_next);
  devTrackStackIndexes(&tsi_prdc);
  devTrackStackIndexes(&tsi_refl);

  /* Set the next TCI to the current TCI */
  tci_next._azim  = tci->_azim;
  tci_next._x = tci->_x;
  tci_next._polar = tci->_polar;
  tci_next._lz = tci->_lz;
  tci_next._link = 0;
  
  /* Set the periodic TCI to the current TCI */
  tci_prdc._azim  = tci->_azim;
  tci_prdc._x = tci->_x;
  tci_prdc._polar = tci->_polar;
  tci_prdc._lz = tci->_lz;
  tci_prdc._link = 0;
  
  /* Set the reflective TCI to the current TCI */
  tci_refl._azim  = tci->_azim;
  tci_refl._x     = tci->_x;
  tci_refl._polar = tci->_polar;
  tci_refl._lz    = tci->_lz;
  tci_refl._link = 0;
  int nz = d_num_z[tci->_azim][tci->_polar];
  int nl = d_num_l[tci->_azim][tci->_polar];
  int lz = tci->_lz;
  int ac = num_azim/2 - tci->_azim - 1;
  int pc = num_polar - tci->_polar - 1;

  int num_links = getNum3DTrackChainLinksonGPU(tci, d_track_2D_chain, d_track_2D_chain_link, 
                                               d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, 
                                               d_num_y, d_first_lz_of_stack, d_tracks_per_stack,d_azim_size);
                                               
  bool next_fwd = outgoing;
  boundaryType bc;

  if (outgoing) {
    bc = track_2D->_bc_fwd;
  }
  else {
    bc = track_2D->_bc_bwd;
  }
  //这里的MPI应该不用加上


  /* Tracks pointing in the positive z direction in the lz plane */
  if (tci->_polar < num_polar / 2) {

    /* SURFACE_Z_MAX */
    if (tci->_link == num_links - 1 && lz >= nz && outgoing) {

      bc = d_MaxZBoundaryType;

      tci_prdc._lz    = lz - nz;
      tci_refl._polar = pc;
      tci_refl._lz    = nl + 2 * nz - lz - 1;

    /* PERIODIC or INTERFACE BC */
      if (d_MaxZBoundaryType == PERIODIC ||
          d_MaxZBoundaryType == INTERFACE)
        tci_next._lz    = lz - nz;

      /* REFLECTIVE OR VACUUM BC */ 
      else {
        tci_next._polar = pc;
        tci_next._lz    = nl + 2 * nz - lz - 1;
      }

      /* Check for a double reflection */
      convertTCItoTSIonGPU(&tci_prdc, &tsi_prdc, d_track_2D_chain, 
                           d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, 
                           d_dz_eff, d_dl_eff, d_first_lz_of_stack, d_num_y,d_azim_size);

      if (tsi_prdc._xy != tsi->_xy) {
        tci_refl._azim = ac;
        
        /* Set the next Track */
        boundaryType bc_xy = track_2D->_bc_fwd;
        if (bc_xy != PERIODIC && bc_xy != INTERFACE) {
          tci_next._azim = ac;
        }
        if (bc_xy == INTERFACE && bc != VACUUM) {
          bc = INTERFACE;
        }
        else if (bc_xy == VACUUM)
          bc = VACUUM;
      }
    }

    /* SURFACE_Z_MIN */
    else if (tci->_link == 0 && lz < nl && !outgoing) {

      bc = d_MinZBoundaryType;

      tci_prdc._lz    = lz + nz;
      tci_prdc._link = getNum3DTrackChainLinksonGPU(&tci_prdc, d_track_2D_chain, 
                                                    d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, 
                                                    d_dz_eff, d_dl_eff, d_num_y, d_first_lz_of_stack, 
                                                    d_tracks_per_stack,d_azim_size) - 1;
      tci_refl._polar = pc;
      tci_refl._lz    = nl - lz - 1;
      tci_refl._link = getNum3DTrackChainLinksonGPU(&tci_refl, d_track_2D_chain, 
                                                    d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, 
                                                    d_dz_eff, d_dl_eff, d_num_y, d_first_lz_of_stack, 
                                                    d_tracks_per_stack,d_azim_size) - 1;
                                                    
      /* PERIODIC or INTERFACE BC */
      if (d_MinZBoundaryType == PERIODIC ||
          d_MinZBoundaryType == INTERFACE) {
        tci_next._lz    = lz + nz;
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next, d_track_2D_chain, 
                                                      d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
      }

      /* REFLECTIVE OR VACUUM BC */
      else {
        tci_next._polar = pc;
        tci_next._lz    = nl - lz - 1;
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next, d_track_2D_chain, 
                                                      d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
      }
      
      /* Check for a double reflection */
      convertTCItoTSIonGPU(&tci_prdc, &tsi_prdc, d_track_2D_chain, d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, d_first_lz_of_stack, d_num_y,d_azim_size);

      if (tsi_prdc._xy != tsi->_xy) {

        tci_refl._azim = ac;

        /* Set the next Track */
        boundaryType bc_xy = track_2D->_bc_bwd;
        if (bc_xy != PERIODIC && bc_xy != INTERFACE) {
          tci_next._azim = ac;
        }
        if (bc_xy == INTERFACE && bc != VACUUM) {
          bc = INTERFACE;
        }
        else if (bc_xy == VACUUM)
          bc = VACUUM;
      }
    } 

    /* SURFACE_Y_MIN */
    else if (tci->_link == 0 && lz >= nl && !outgoing) {

      tci_prdc._lz    = lz - nl;
      tci_prdc._x     = dev_tracks_2d[track_2D->_track_prdc_bwd]._xy_index % d_num_x[tci->_azim];
      tci_prdc._link = getNum3DTrackChainLinksonGPU(&tci_prdc, d_track_2D_chain, d_track_2D_chain_link, d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, d_num_y, d_first_lz_of_stack, d_tracks_per_stack,d_azim_size) - 1;
      tci_refl._azim  = ac;
      tci_refl._x     = dev_tracks_2d[track_2D->_track_refl_bwd]._xy_index % d_num_x[tci->_azim];
      tci_refl._polar = pc;
      tci_refl._lz = lz - nl;
      tci_next._lz = lz - nl;

      if (d_MinYBoundaryType == PERIODIC ||
          d_MinYBoundaryType == INTERFACE) {
        tci_next._x     = dev_tracks_2d[track_2D->_track_prdc_bwd]._xy_index % d_num_x[tci->_azim];
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next, 
                                                      d_track_2D_chain, 
                                                      d_track_2D_chain_link, 
                                                      d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, 
                                                      d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
      }
      else {
        tci_next._azim  = ac;
        tci_next._x     = dev_tracks_2d[track_2D->_track_refl_bwd]._xy_index % d_num_x[tci->_azim];
        tci_next._polar = pc;
        next_fwd = true;
      }
    }
/* SURFACE_Y_MAX */
    else if (tci->_link == num_links - 1 && lz < nz && outgoing) {

      tci_prdc._lz    = nl + lz;
      tci_prdc._x     = dev_tracks_2d[track_2D->_track_prdc_fwd]._xy_index % d_num_x[tci->_azim];
      tci_refl._azim  = ac;
      tci_refl._x     = dev_tracks_2d[track_2D->_track_refl_fwd]._xy_index % d_num_x[tci->_azim];
      tci_refl._polar = pc;
      tci_refl._lz = nl + lz;
      tci_refl._link = getNum3DTrackChainLinksonGPU(&tci_refl,
                                                    d_track_2D_chain, 
                                                    d_track_2D_chain_link, 
                                                    d_num_x, d_num_z, d_num_l, 
                                                    d_dz_eff, d_dl_eff, 
                                                    d_num_y, d_first_lz_of_stack, 
                                                    d_tracks_per_stack,d_azim_size) - 1;
      tci_next._lz = nl + lz;

      if (d_MaxYBoundaryType == PERIODIC ||
          d_MaxYBoundaryType == INTERFACE)
        tci_next._x     = dev_tracks_2d[track_2D->_track_prdc_fwd]._xy_index % d_num_x[tci->_azim];
      else {
        tci_next._azim  = ac;
        tci_next._x     = dev_tracks_2d[track_2D->_track_refl_fwd]._xy_index % d_num_x[tci->_azim];
        tci_next._polar = pc;
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next,
                                                      d_track_2D_chain, 
                                                      d_track_2D_chain_link, 
                                                      d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, 
                                                      d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
        next_fwd = false;
      }
    }

    /* SURFACE_X_MIN or SURFACE_X_MAX */
    else if (outgoing) {

      /* Set the link index */
      tci_prdc._link = tci->_link + 1;
      tci_next._link = tci->_link + 1;
      tci_refl._link = tci->_link + 1;
      tci_refl._azim  = ac;

      /* Set the next track */
      if (track_2D->_bc_fwd != PERIODIC &&
          track_2D->_bc_fwd != INTERFACE)
        tci_next._azim  = ac;
    }

    /* Tracks hitting any of the four x or y surfaces */
    else {

      /* Set the link index */
      tci_prdc._link = tci->_link - 1;
      tci_next._link = tci->_link - 1;
      tci_refl._link = tci->_link - 1;
      tci_refl._azim  = ac;

      /* Set the next track */
      if (track_2D->_bc_bwd != PERIODIC &&
          track_2D->_bc_bwd != INTERFACE)
        tci_next._azim  = ac;
    }
  }
  
  else {

    /* SURFACE_Z_MAX */
    if (tci->_link == 0 && lz >= nz && !outgoing) {

      bc = d_MaxZBoundaryType;

      tci_prdc._lz    = lz - nz;
      tci_prdc._link = getNum3DTrackChainLinksonGPU(&tci_prdc, 
                                               d_track_2D_chain, 
                                               d_track_2D_chain_link, 
                                               d_num_x, d_num_z, d_num_l, 
                                               d_dz_eff, d_dl_eff, 
                                               d_num_y, d_first_lz_of_stack, 
                                               d_tracks_per_stack,d_azim_size) - 1;
      tci_refl._polar = pc;
      tci_refl._lz    = nl + 2 * nz - lz - 1;
      tci_refl._link = getNum3DTrackChainLinksonGPU(&tci_refl,
                                                    d_track_2D_chain, 
                                                    d_track_2D_chain_link, 
                                                    d_num_x, d_num_z, d_num_l, 
                                                    d_dz_eff, d_dl_eff, 
                                                    d_num_y, d_first_lz_of_stack, 
                                                    d_tracks_per_stack,d_azim_size) - 1;

      /* PERIODIC or INTERFACE BC */
      if (d_MaxZBoundaryType == PERIODIC ||
          d_MaxZBoundaryType == INTERFACE) {
        tci_next._lz    = lz - nz;
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next,
                                                      d_track_2D_chain, 
                                                      d_track_2D_chain_link, 
                                                      d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, 
                                                      d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
      }

      /* REFLECTIVE OR VACUUM BC */
      else {
        tci_next._polar = pc;
        tci_next._lz    = nl + 2 * nz - lz - 1;
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next,
                                                      d_track_2D_chain, 
                                                      d_track_2D_chain_link, 
                                                      d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, 
                                                      d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
      }

      /* Check for a double reflection */
      convertTCItoTSIonGPU(&tci_prdc, &tsi_prdc, 
                           d_track_2D_chain, d_track_2D_chain_link, 
                           d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, 
                           d_first_lz_of_stack, d_num_y,d_azim_size);

      if (tsi_prdc._xy != tsi->_xy) {

        tci_refl._azim = ac;

        /* Set the next Track */
        boundaryType bc_xy = track_2D->_bc_bwd;
        if (bc_xy != PERIODIC && bc_xy != INTERFACE) {
          tci_next._azim = ac;
        }
        if (bc_xy == INTERFACE && bc != VACUUM) {
          bc = INTERFACE;
        }
        else if (bc_xy == VACUUM)
          bc = VACUUM;
      }
    }
    
    /* SURFACE_Z_MIN */
    //走到这个分支先1后0
    else if (tci->_link == num_links - 1 && lz < nl && outgoing) {

      bc = d_MinZBoundaryType;

      tci_prdc._lz    = lz + nz;
      tci_refl._polar = pc;
      tci_refl._lz    = nl - lz - 1;

      /* PERIODIC or INTERFACE BC */
      if (d_MinZBoundaryType == PERIODIC ||
          d_MinZBoundaryType == INTERFACE)
        tci_next._lz    = lz + nz;

      /* REFLECTIVE OR VACUUM BC */
      else {
        tci_next._polar = pc;
        tci_next._lz    = nl - lz - 1;
      }

      /* Check for a double reflection */
      convertTCItoTSIonGPU(&tci_prdc, &tsi_prdc,
                      d_track_2D_chain, d_track_2D_chain_link, 
                      d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, 
                      d_first_lz_of_stack, d_num_y,d_azim_size);

      if (tsi_prdc._xy != tsi->_xy) {    
        tci_refl._azim = ac;

        /* Set the next Track */
        boundaryType bc_xy = track_2D->_bc_fwd;

        if (bc_xy != PERIODIC && bc_xy != INTERFACE) {
          tci_next._azim = ac;
        }
        if (bc_xy == INTERFACE && bc != VACUUM) {

          bc = INTERFACE;
        }
        else if (bc_xy == VACUUM)
          bc = VACUUM;
      }
    }
    
    /* SURFACE_Y_MIN */
    else if (tci->_link == 0 && lz < nz && !outgoing) {

      tci_prdc._lz    = lz + nl;
      tci_prdc._x     = dev_tracks_2d[track_2D->_track_prdc_bwd]._xy_index % d_num_x[tci->_azim];
      tci_prdc._link = getNum3DTrackChainLinksonGPU(&tci_prdc,
                                                    d_track_2D_chain, 
                                                    d_track_2D_chain_link, 
                                                    d_num_x, d_num_z, d_num_l, 
                                                    d_dz_eff, d_dl_eff, 
                                                    d_num_y, d_first_lz_of_stack, 
                                                    d_tracks_per_stack,d_azim_size) - 1;
      tci_refl._azim  = ac;
      tci_refl._x     = dev_tracks_2d[track_2D->_track_refl_bwd]._xy_index  % d_num_x[tci->_azim];
      tci_refl._polar = pc;
      tci_next._lz    = lz + nl;
      tci_refl._lz    = lz + nl;

      if (d_MinYBoundaryType == PERIODIC ||
          d_MinYBoundaryType == INTERFACE) {
        tci_next._x     = dev_tracks_2d[track_2D->_track_prdc_bwd]._xy_index % d_num_x[tci->_azim];
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next,
                                                      d_track_2D_chain, 
                                                      d_track_2D_chain_link, 
                                                      d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, 
                                                      d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
      }
      else {
        tci_next._azim  = ac;
        tci_next._x     = dev_tracks_2d[track_2D->_track_refl_bwd]._xy_index % d_num_x[tci->_azim];
        tci_next._polar = pc;
        next_fwd = true;
      }
    }

    /* SURFACE_Y_MAX */
    else if (tci->_link == num_links - 1 && lz >= nl && outgoing) {

      tci_prdc._lz    = lz - nl;
      tci_prdc._x     = dev_tracks_2d[track_2D->_track_prdc_fwd]._xy_index % d_num_x[tci->_azim];
      tci_refl._azim  = ac;
      tci_refl._x     = dev_tracks_2d[track_2D->_track_refl_fwd]._xy_index % d_num_x[tci->_azim];
      tci_refl._polar = pc;
      tci_refl._lz    = lz - nl;
      tci_refl._link = getNum3DTrackChainLinksonGPU(&tci_refl,
                                                    d_track_2D_chain, 
                                                    d_track_2D_chain_link, 
                                                    d_num_x, d_num_z, d_num_l, 
                                                    d_dz_eff, d_dl_eff, 
                                                    d_num_y, d_first_lz_of_stack, 
                                                    d_tracks_per_stack,d_azim_size) - 1;
      tci_next._lz    = lz - nl;

      if (d_MaxYBoundaryType == PERIODIC ||
          d_MaxYBoundaryType == INTERFACE)
        tci_next._x     = dev_tracks_2d[track_2D->_track_prdc_fwd]._xy_index % d_num_x[tci->_azim];
      else {
        tci_next._azim  = ac;
        tci_next._x     = dev_tracks_2d[track_2D->_track_refl_fwd]._xy_index % d_num_x[tci->_azim];
        tci_next._polar = pc;
        tci_next._link = getNum3DTrackChainLinksonGPU(&tci_next,
                                                      d_track_2D_chain, 
                                                      d_track_2D_chain_link, 
                                                      d_num_x, d_num_z, d_num_l, 
                                                      d_dz_eff, d_dl_eff, 
                                                      d_num_y, d_first_lz_of_stack, 
                                                      d_tracks_per_stack,d_azim_size) - 1;
        next_fwd = false;
      }
    }

    /* SURFACE_X_MIN or SURFACE_X_MAX */
    //走到这先1后0
    else if (outgoing) {

      /* Set the link index */
      tci_prdc._link = tci->_link + 1;
      tci_next._link = tci->_link + 1;
      tci_refl._link = tci->_link + 1;
      tci_refl._azim  = ac;

      /* Set the next track */
      if (track_2D->_bc_fwd != PERIODIC &&
          track_2D->_bc_fwd != INTERFACE)
        tci_next._azim  = ac;
    }

    /* Tracks hitting any of the four x or y surfaces */
    else {

      /* Set the link index */
      tci_prdc._link = tci->_link - 1;
      tci_next._link = tci->_link - 1;
      tci_refl._link = tci->_link - 1;
      tci_refl._azim  = ac;

      /* Set the next track */
      if (track_2D->_bc_bwd != PERIODIC &&
          track_2D->_bc_bwd != INTERFACE)
        tci_next._azim  = ac;
    }
  }

  convertTCItoTSIonGPU(&tci_next, &tsi_next,
                       d_track_2D_chain, d_track_2D_chain_link, 
                       d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, 
                       d_first_lz_of_stack, d_num_y,d_azim_size);
  convertTCItoTSIonGPU(&tci_prdc, &tsi_prdc,
                       d_track_2D_chain, d_track_2D_chain_link, 
                       d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, 
                       d_first_lz_of_stack, d_num_y,d_azim_size);
  convertTCItoTSIonGPU(&tci_refl, &tsi_refl,
                       d_track_2D_chain, d_track_2D_chain_link, 
                       d_num_x, d_num_z, d_num_l, d_dz_eff, d_dl_eff, 
                       d_first_lz_of_stack, d_num_y,d_azim_size);

  if (outgoing) {
    track->_track_next_fwd = get3DTrackID(&tsi_next, d_cum_tracks_per_stack, d_num_x, d_num_y,d_azim_size);
    
    track->_track_prdc_fwd = get3DTrackID(&tsi_prdc, d_cum_tracks_per_stack, d_num_x, d_num_y,d_azim_size);
    track->_track_refl_fwd = get3DTrackID(&tsi_refl, d_cum_tracks_per_stack, d_num_x, d_num_y,d_azim_size);
    track->_next_fwd_fwd = next_fwd;
    track->_bc_fwd = bc;
  }

  else {
    track->_track_next_bwd = get3DTrackID(&tsi_next, d_cum_tracks_per_stack, d_num_x, d_num_y,d_azim_size);
    track->_track_prdc_bwd = get3DTrackID(&tsi_prdc, d_cum_tracks_per_stack, d_num_x, d_num_y,d_azim_size);
    track->_track_refl_bwd = get3DTrackID(&tsi_refl, d_cum_tracks_per_stack, d_num_x, d_num_y,d_azim_size);
    track->_next_bwd_fwd = next_fwd;
    track->_bc_bwd = bc;
  }
}

//gettrackotf函数的移植
__device__ void getTrackOTFonGPU(dev_track3D* track, 
                                 dev_TrackStackIndexes* tsi, 
                                 int** d_tracks_2D_xy, 
                                 int* d_num_x, int* d_num_y, 
                                 int* d_first_lz_of_stack,
                                 int* d_tracks_per_stack,
                                 dev_track2D* d_track_2D_chain,
                                 int* d_track_2D_chain_link,
                                 int** d_num_z,
                                 int** d_num_l,
                                 double** d_dz_eff,
                                 double** d_dl_eff,
                                 double* d_theta,
                                 dev_track2D* dev_tracks_2d,
                                 long* d_cum_tracks_per_stack,int * d_azim_size){
    dev_track2D* track_2D = &dev_tracks_2d[d_tracks_2D_xy[tsi->_azim][tsi->_xy]];
    dev_TrackChainIndexes tci;
    devTrackChainIndexes(&tci);
    convertTsitoTCIonGPU(tsi, 
                         &tci, 
                         d_first_lz_of_stack, 
                         d_num_x, 
                         d_num_y, 
                         d_track_2D_chain, 
                         d_track_2D_chain_link, 
                         d_num_z, 
                         d_num_l, 
                         d_dz_eff, 
                         d_dl_eff, 
                         d_tracks_2D_xy, dev_tracks_2d,d_azim_size);

    track->_phi = track_2D->_phi;
    set3DTrackDataonGPU(&tci, track, d_theta, 
                        d_track_2D_chain, 
                        d_track_2D_chain_link,
                        d_num_x, d_num_z, d_num_l, 
                        d_dz_eff, d_dl_eff, d_num_y);

    track->_phi = track_2D->_phi;
    track->_theta = getThetaonGPU(tsi->_azim, tsi->_polar, d_theta);
    track->_azim_index = tsi->_azim;
    track->_xy_index = tsi->_xy;
    track->_polar_index = tsi->_polar;

    setLinkingTracksonGPU(tsi, &tci, d_first_lz_of_stack, d_tracks_per_stack, d_num_x, d_num_y, d_track_2D_chain, d_track_2D_chain_link, d_num_z, d_num_l, d_dz_eff, d_dl_eff, d_tracks_2D_xy, true, track, dev_tracks_2d, d_cum_tracks_per_stack,d_azim_size);
    setLinkingTracksonGPU(tsi, &tci, d_first_lz_of_stack, d_tracks_per_stack, d_num_x, d_num_y, d_track_2D_chain, d_track_2D_chain_link, d_num_z, d_num_l, d_dz_eff, d_dl_eff, d_tracks_2D_xy, false, track, dev_tracks_2d, d_cum_tracks_per_stack,d_azim_size);

    track->_uid = get3DTrackID(tsi, d_cum_tracks_per_stack, d_num_x, d_num_y,d_azim_size);
}

//findMeshIndexonGPU的移植
__device__ int findMeshIndexonGPU(double* values, int size,
                                  double val, int sign){
  int imin = 0;
  int imax = size-1;

  while (imax - imin > 1) {

    int imid = (imin + imax) / 2;

    if (val > values[imid])
      imin = imid;
    else if (val < values[imid])
      imax = imid;
    else {
      if (sign > 0)
        return imid;
      else
        return imid-1;
    }
  }
  return imin;
}

//execute的移植
__device__ void executeonGPU(double length, int _material_index, 
                             long fsr_id,
                             int track_idx, double x_start,
                             double y_start, double z_start,
                             double phi, double theta, 
                             dev_material* materials,
                             dev_track3D* track_3D,
                             dev_segment* _segments){
  /* Determine the number of cuts on the segment */
  double sin_theta = sin(theta);
  double max_sigma_t = materials[_material_index]._max_sigma_t;
  int num_cuts = 1;
  if (length * max_sigma_t * sin_theta > _max_tau)
    num_cuts = length * max_sigma_t * sin_theta / _max_tau + 1;
  double temp_length = _max_tau / (max_sigma_t * sin_theta);

  /* Add segment information */
  for (int i=0; i < num_cuts-1; i++) {
    _segments[track_3D->_num_segments]._length = temp_length;
    _segments[track_3D->_num_segments]._material_index = _material_index;
    _segments[track_3D->_num_segments]._region_uid = fsr_id;
    length -= temp_length;
    x_start += temp_length * sin_theta * cos(phi);
    y_start += temp_length * sin_theta * sin(phi);
    z_start += temp_length * cos(theta);
    track_3D->_num_segments++;
  }
  _segments[track_3D->_num_segments]._length = length;
  _segments[track_3D->_num_segments]._material_index = _material_index;
  _segments[track_3D->_num_segments]._region_uid = fsr_id;
  track_3D->_num_segments++;
}

//traceSegmentsOTF的移植
__device__ void traceSegmentsOTFonGPU(dev_track2D* flattened_track, dev_Point* start,
                                      double theta, 
                                      dev_ExtrudedFSR* d_extruded_FSR_lookup,
                                      dev_material* materials,
                                      dev_track3D* track_3D,
                                      dev_segment* _segments){                                   
  /* Create unit vector */
  double phi = flattened_track->_phi;
  double cos_phi = cos(phi);
  double sin_phi = sin(phi);
  double cos_theta = cos(theta);
  double sin_theta = sin(theta);
  int sign = (cos_theta > 0) - (cos_theta < 0);     
  
  /* Extract starting coordinates */
  double x_start_3D = start->_xyz[0];
  double x_start_2D = flattened_track->_start._xyz[0];
  double x_coord = x_start_3D;
  double y_coord = start->_xyz[1];
  double z_coord = start->_xyz[2];   

  /* Find 2D distance from 2D edge to start of track */
  double start_dist_2D = (x_start_3D - x_start_2D) / cos_phi; 

  /* Find starting 2D segment */
  int seg_start = 0;
  device_segment* segments_2D = flattened_track->segments_2D;
  for (int s=0; s < flattened_track->_num_segments; s++) {
    /* Determine if start point of track is beyond current 2D segment */
    double seg_len_2D = segments_2D[s]._length;
    if (start_dist_2D > seg_len_2D) {
      start_dist_2D -= seg_len_2D;
      seg_start++;
    }
    else {
      break;
    }
  }
 
  int num_fsrs;
  double* axial_mesh;
  bool contains_global_z_mesh;
  contains_global_z_mesh = false;
  int extruded_fsr_id = segments_2D[seg_start]._region_id;
  dev_ExtrudedFSR* extruded_FSR = &d_extruded_FSR_lookup[extruded_fsr_id];
  num_fsrs = extruded_FSR->_num_fsrs;
  axial_mesh = extruded_FSR->_mesh;

  int z_ind = findMeshIndexonGPU(axial_mesh, num_fsrs+1, z_coord, sign);

  bool first_segment = true;
  bool segments_complete = false;
  
  for (int s=seg_start; s < flattened_track->_num_segments; s++) {
    /* Extract extruded FSR */
    int extruded_fsr_id = segments_2D[s]._region_id;
    dev_ExtrudedFSR* extruded_FSR = &d_extruded_FSR_lookup[extruded_fsr_id];

    /* Determine new mesh and z index */
    if (first_segment || contains_global_z_mesh) {
      first_segment = false;
    }
    else {
      /* Determine the axial region */
      num_fsrs = extruded_FSR->_num_fsrs;
      axial_mesh = extruded_FSR->_mesh;
      z_ind = findMeshIndexonGPU(axial_mesh, num_fsrs+1, z_coord, sign);
    }

    /* Extract 2D segment length */
    double remaining_length_2D = segments_2D[s]._length - start_dist_2D;
    start_dist_2D = 0;
  
    /* Transport along the 2D segment until it is completed */
    while (remaining_length_2D > 0) {
      /* Calculate 3D distance to z intersection */
      double z_dist_3D;
      if (sign > 0)
        z_dist_3D = (axial_mesh[z_ind+1] - z_coord) / cos_theta;
      else
        z_dist_3D = (axial_mesh[z_ind] - z_coord) / cos_theta;

      /* Calculate 3D distance to end of segment */
      double seg_dist_3D = remaining_length_2D / sin_theta;

      /* Calcualte shortest distance to intersection */
      double dist_2D;
      double dist_3D;
      int z_move;
      if (z_dist_3D <= seg_dist_3D) {
        dist_2D = z_dist_3D * sin_theta;
        dist_3D = z_dist_3D;
        z_move = sign;
      }
      else {
        dist_2D = remaining_length_2D;
        dist_3D = seg_dist_3D;
        z_move = 0;
      }

      /* Get the 3D FSR */
      long fsr_id = extruded_FSR->_fsr_ids[z_ind];
  
      /* Operate on segment */
      if (dist_3D > TINY_MOVE) {
        executeonGPU(dist_3D, extruded_FSR->_material_index[z_ind], fsr_id, 0,
                     x_coord, y_coord, z_coord, phi, theta, materials,track_3D,
                    _segments);
      }

      /* Move axial height to end of segment */
      x_coord += dist_3D * sin_theta * cos_phi;
      y_coord += dist_3D * sin_theta * sin_phi;
      z_coord += dist_3D * cos_theta;

      /* Shorten remaining 2D segment length and move axial level */
      remaining_length_2D -= dist_2D;
      z_ind += z_move;
    
      /* Check if the track has crossed a Z boundary */
      if (z_ind < 0 or z_ind >= num_fsrs) {

        /* Reset z index */
        if (z_ind < 0)
          z_ind = 0;
        else
          z_ind = num_fsrs - 1;

        /* Mark the 2D segment as complete */
        segments_complete = true;
        break;
      }
    }

    /* Check if the track is completed due to an axial boundary */
    if (segments_complete)
      break;
  }
}

//用于计算通量
__device__ void mytransportSweepOnDevice(FP_PRECISION* scalar_flux,
                                       float* boundary_flux,
                                       float* start_flux,
                                       FP_PRECISION* reduced_sources,
                                       dev_material* materials,
                                       dev_track* track) {
  float* track_flux;

  dev_track* curr_track;
  int azim_index;
  int p;
  int num_segments;
  dev_segment* curr_segment;
  curr_track = track;
  azim_index = curr_track->_azim_angle_index;
  p=curr_track->_polar_index;						
  num_segments = curr_track->_num_segments;
  double  weight=weights(azim_index,p);
  int track_id = curr_track->_uid;

  for (int i = 0; i < NUM_GROUPS; i++){
    //对一根线放到7个能群里算，我将能群的遍历放到for循环算
    int energy_group = i;
    int energy_angle_index = energy_group;
    track_flux = &boundary_flux(track_id,0);
    FP_PRECISION fsr_flux[1]={0.0};
    /* Loop over each Track segment in forward direction */
    for (int i=0; i < num_segments; i++) {
      curr_segment = &curr_track->_segments[i];
	  long fsr_id = curr_segment->_region_uid;
      tallyScalarFlux(curr_segment, azim_index,energy_group, materials,
                      track_flux, reduced_sources, scalar_flux,fsr_flux); 
	if (i < num_segments - 1 && _solve_3D
		  &&  fsr_id != ((&curr_track->_segments[i+1])->_region_uid))	 
      {
         myatomicAdd(&scalar_flux(fsr_id,energy_group), weight*fsr_flux[0]);
         fsr_flux[0] =0.0;
	  }
	}
  /* Transfer boundary angular flux to outgoing Track */
  transferBoundaryFlux(curr_track, azim_index, track_flux, start_flux,
                         energy_angle_index, true);

    
    track_flux = &boundary_flux(track_id,NUM_GROUPS);
    for (int i=num_segments-1; i > -1; i--) {
      curr_segment = &curr_track->_segments[i];
	  long fsr_id = curr_segment->_region_uid;  
      tallyScalarFlux(curr_segment, azim_index,energy_group, materials,
                      track_flux, reduced_sources, scalar_flux,fsr_flux);
	  if ((i==0  || (fsr_id != (&curr_track->_segments[i-1])->_region_uid))
		  &&_solve_3D )
	  {
       myatomicAdd(&scalar_flux(fsr_id,energy_group), weight*fsr_flux[0]);
       fsr_flux[0] =0.0;
	  } 
    }

    /* Transfer boundary angular flux to outgoing Track */
    transferBoundaryFlux(curr_track, azim_index, track_flux, start_flux,
                        energy_angle_index, false);
  }
}

//主核函数
__global__ void OTFsolver(dev_track2D* dev_tracks_2d, 
                          int* d_num_x, int* d_num_y, 
                          int* d_first_lz_of_stack, 
                          int* d_tracks_per_stack,  
                          int** d_tracks_2D_xy,
                          dev_track2D* d_track_2D_chain,
                          int* d_track_2D_chain_link,
                          int** d_num_z, int** d_num_l,
                          double** d_dz_eff, double** d_dl_eff,
                          double* d_theta, long* d_cum_tracks_per_stack,
                          dev_ExtrudedFSR* d_extruded_FSR_lookup,
                          FP_PRECISION* scalar_flux,
                          float* boundary_flux, float* start_flux,
                          FP_PRECISION* reduced_sources,
                          dev_material* materials, track_index* d_track_3D_index,int* d_azim_size) { 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < num_3D_tracks){
    dev_TrackStackIndexes tsi;
    devTrackStackIndexes(&tsi);
    int track_2D_index = d_track_3D_index[tid].track_index;
    dev_track2D flattened_track = dev_tracks_2d[track_2D_index];
    tsi._azim = flattened_track._azim_index;
    tsi._xy = flattened_track._xy_index;
    tsi._polar = d_track_3D_index[tid].polar;
    tsi._z = d_track_3D_index[tid].z;

    //开始计算特征线
    dev_track3D track_3D;
    getTrackOTFonGPU(&track_3D, &tsi, d_tracks_2D_xy, 
                     d_num_x, d_num_y, d_first_lz_of_stack,
                     d_tracks_per_stack, d_track_2D_chain, 
                     d_track_2D_chain_link, d_num_z, d_num_l,  
                     d_dz_eff, d_dl_eff, d_theta, dev_tracks_2d,
                     d_cum_tracks_per_stack,d_azim_size);

    double theta = track_3D._theta;
    dev_Point* start = &track_3D._start;

    dev_segment _segments[5000];
    track_3D._num_segments = 0;

    // 对特征线进行分段
    traceSegmentsOTFonGPU(&flattened_track, start, theta, 
                          d_extruded_FSR_lookup,
                          materials,&track_3D, _segments);
    
    // 迭代计算
    dev_track track;
        
    track._uid = track_3D._uid;
    track._azim_angle_index = track_3D._azim_index;
    track._segments = _segments;
    track._num_segments = track_3D._num_segments;
    track._next_track_fwd = track_3D._track_next_fwd;
    track._next_track_bwd = track_3D._track_next_bwd;
    track._next_fwd_is_fwd = track_3D._next_fwd_fwd;
    track._next_bwd_is_fwd = track_3D._next_bwd_fwd;
    track._transfer_flux_fwd = track_3D._bc_fwd;
    track._transfer_flux_bwd = track_3D._bc_bwd;
    track._polar_index =track_3D._polar_index;
    mytransportSweepOnDevice(scalar_flux, boundary_flux, start_flux, reduced_sources,
                             materials, &track);
    //按照网格继续遍历
    tid += blockDim.x * gridDim.x;                    
  }           
}

// 预先计算段数量
__global__ void calsegmentsnum(dev_track2D* dev_tracks_2d, int* d_num_x, int* d_num_y, 
                              int* d_first_lz_of_stack, int* d_tracks_per_stack,  
                              int** d_tracks_2D_xy, dev_track2D* d_track_2D_chain,
                              int* d_track_2D_chain_link, int** d_num_z, int** d_num_l,
                              double** d_dz_eff, double** d_dl_eff, double* d_theta, 
                              long* d_cum_tracks_per_stack, dev_ExtrudedFSR* d_extruded_FSR_lookup,
                              dev_material* materials, track_index* d_track_3D_index,int * d_azim_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < num_3D_tracks){
    dev_TrackStackIndexes tsi;
    devTrackStackIndexes(&tsi);
    int track_2D_index = d_track_3D_index[tid].track_index;
    dev_track2D flattened_track = dev_tracks_2d[track_2D_index];
    tsi._azim = flattened_track._azim_index;
    tsi._xy = flattened_track._xy_index;
    tsi._polar = d_track_3D_index[tid].polar;
    tsi._z = d_track_3D_index[tid].z;
    //开始计算特征线
    dev_track3D track_3D;
    getTrackOTFonGPU(&track_3D, &tsi, d_tracks_2D_xy, 
                     d_num_x, d_num_y, d_first_lz_of_stack,
                     d_tracks_per_stack, d_track_2D_chain, 
                     d_track_2D_chain_link, d_num_z, d_num_l,  
                     d_dz_eff, d_dl_eff, d_theta, dev_tracks_2d,
                     d_cum_tracks_per_stack,d_azim_size);

    double theta = track_3D._theta;
    dev_Point* start = &track_3D._start;

    dev_segment _segments[1310];
    track_3D._num_segments = 0;

    // 对特征线进行分段
    traceSegmentsOTFonGPU(&flattened_track, start, theta, 
                          d_extruded_FSR_lookup,
                          materials,&track_3D, _segments);
    
    d_track_3D_index[tid].num_segments = track_3D._num_segments;
    //按照网格继续遍历
    tid += blockDim.x * gridDim.x;                    
  }                    
}

/**
 * @brief 预先加载特征线进行计算
 * @details 先加载3D特征线数据存储在gpu中，后续直接使用
 *          特征线数据生成段然后进行计算
 */
__global__ void generate3DtracksonGPU(dev_track2D* dev_tracks_2d, int* d_num_x, int* d_num_y, 
                              int* d_first_lz_of_stack, int* d_tracks_per_stack,  
                              int** d_tracks_2D_xy, dev_track2D* d_track_2D_chain,
                              int* d_track_2D_chain_link, int** d_num_z, int** d_num_l,
                              double** d_dz_eff, double** d_dl_eff, double* d_theta, 
                              long* d_cum_tracks_per_stack, dev_ExtrudedFSR* d_extruded_FSR_lookup,
                              dev_material* materials, track_index* d_track_3D_index, 
                              dev_track* d_track_3D, dev_Point* d_start, int pre_segments_index,int* d_azim_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < num_3D_tracks){
    dev_TrackStackIndexes tsi;
    devTrackStackIndexes(&tsi);
    int track_2D_index = d_track_3D_index[tid].track_index;
    dev_track2D flattened_track = dev_tracks_2d[track_2D_index];
    tsi._azim = flattened_track._azim_index;
    tsi._xy = flattened_track._xy_index;
    tsi._polar = d_track_3D_index[tid].polar;
    tsi._z = d_track_3D_index[tid].z;

    //开始计算特征线
    dev_track3D track_3D;
    getTrackOTFonGPU(&track_3D, &tsi, d_tracks_2D_xy, 
                     d_num_x, d_num_y, d_first_lz_of_stack,
                     d_tracks_per_stack, d_track_2D_chain, 
                     d_track_2D_chain_link, d_num_z, d_num_l,  
                     d_dz_eff, d_dl_eff, d_theta, dev_tracks_2d,
                     d_cum_tracks_per_stack,d_azim_size);
    d_track_3D[tid]._uid = track_3D._uid;
    d_track_3D[tid]._azim_angle_index = track_3D._azim_index;
    d_track_3D[tid]._next_track_fwd = track_3D._track_next_fwd;
    d_track_3D[tid]._next_track_bwd = track_3D._track_next_bwd;
    d_track_3D[tid]._next_fwd_is_fwd = track_3D._next_fwd_fwd;
    d_track_3D[tid]._next_bwd_is_fwd = track_3D._next_bwd_fwd;
    d_track_3D[tid]._transfer_flux_fwd = track_3D._bc_fwd;
    d_track_3D[tid]._transfer_flux_bwd = track_3D._bc_bwd;
    d_track_3D[tid]._polar_index = track_3D._polar_index;
    d_track_3D[tid]._theta = track_3D._theta;
    d_track_3D[tid]._num_segments = d_track_3D_index[tid].num_segments;
    d_start[tid]._xyz[0] = track_3D._start._xyz[0];
    d_start[tid]._xyz[1] = track_3D._start._xyz[1];
    d_start[tid]._xyz[2] = track_3D._start._xyz[2];

    //后面加的特征线
    if(tid < pre_segments_index){
    double theta = track_3D._theta;
    dev_Point* start = &track_3D._start;

    track_3D._num_segments = 0;

    traceSegmentsOTFonGPU(&flattened_track, start, theta, 
                          d_extruded_FSR_lookup,
                          materials,&track_3D, d_track_3D[tid]._segments);
    }
    
    tid += blockDim.x * gridDim.x;                    
  }                    
}

/**
 * @brief 只执行分段和迭代计算的核函数
 * @details 先加载3D特征线数据存储在gpu中，该核函数只执行分段，
 *          实现实时分段实时计算的步骤
 */
// __global__ void onlycalsegments(FP_PRECISION* scalar_flux,float* boundary_flux, float* start_flux, 
//                                 FP_PRECISION* reduced_sources, dev_material* materials, 
//                                 track_index* d_track_3D_index, dev_track* d_track_3D, int i) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     while(tid < num_3D_tracks){
//     // 适配对应的track数据结构
//     dev_track track;
//     track._uid = d_track_3D[tid]._uid;
//     track._azim_angle_index = d_track_3D[tid]._azim_index;
//     track._num_segments = d_track_3D[tid]._num_segments;
//     track._next_track_fwd = d_track_3D[tid]._track_next_fwd;
//     track._next_track_bwd = d_track_3D[tid]._track_next_bwd;
//     track._next_fwd_is_fwd = d_track_3D[tid]._next_fwd_fwd;
//     track._next_bwd_is_fwd = d_track_3D[tid]._next_bwd_fwd;
//     track._transfer_flux_fwd = d_track_3D[tid]._bc_fwd;
//     track._transfer_flux_bwd = d_track_3D[tid]._bc_bwd;
//     track._polar_index = d_track_3D[tid]._polar_index;
//     track._segments = d_track_3D[tid]._segments;

//     // 迭代计算
//     mytransportSweepOnDevice(scalar_flux, boundary_flux, start_flux, reduced_sources,
//                              materials, &track);               
//     tid += blockDim.x * gridDim.x;                    
//   }          
// }

/**
 * @brief 只执行分段+计算的核函数
 * @details 先加载3D特征线对应得段数据存储在gpu中，该核函数只执行分段和迭代计算两个步骤
 */
__global__ void onlygeneratesegments(dev_track2D* dev_tracks_2d, dev_ExtrudedFSR* d_extruded_FSR_lookup,
                                     dev_track* d_track_3D, FP_PRECISION* scalar_flux,
                                     float* boundary_flux, float* start_flux, FP_PRECISION* reduced_sources, dev_material* materials, 
                                     track_index* d_track_3D_index, dev_Point* d_start, int pre_segments_index){
    int tid = pre_segments_index + threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < num_3D_tracks){
    double theta = d_track_3D[tid]._theta;
    dev_Point* start = &d_start[tid];
    int track_2D_index = d_track_3D_index[tid].track_index;
    dev_track2D flattened_track = dev_tracks_2d[track_2D_index];
    // 对特征线进行分段
    dev_segment _segments[3000];
    dev_track3D track_3D;
    track_3D._num_segments = 0;
    traceSegmentsOTFonGPU(&flattened_track, start, theta, 
                          d_extruded_FSR_lookup,
                          materials, &track_3D, _segments);
    d_track_3D[tid]._segments = _segments;
    
    //迭代计算
    mytransportSweepOnDevice(scalar_flux, boundary_flux, start_flux, reduced_sources,
                             materials, &d_track_3D[tid]);
    tid += blockDim.x * gridDim.x;                    
  }                    
}
 

/**
 * @brief Constructor initializes arrays for dev_tracks and dev_materials..
 * @details The constructor initalizes the number of CUDA threads and thread
 *          blocks each to a default of 64.
 * @param track_generator an optional pointer to the TrackjGenerator
 */
GPUSolver::GPUSolver(TrackGenerator* track_generator,configure* configure) :

  Solver(track_generator) {

  /* The default number of thread blocks and threads per thread block */
  if (configure != NULL)
  {
    _B = configure->_B;
    _T = configure->_T;
    _configure=configure;
  }

  _materials = NULL;
  _dev_tracks = NULL;
  dev_tracks_2d=NULL;
  _FSR_materials = NULL;
  _dev_chi_spectrum_material = NULL;
  _segment_formation =  track_generator->getSegmentFormation();
  if (track_generator != NULL)
    setTrackGenerator(track_generator);

  _gpu_solver = true;

  /* Since only global stabilization is implemented, let that be default */
  _stabilization_type = GLOBAL;
#ifdef MPIx
  _track_message_size = 0;
  _MPI_requests = NULL;
  _MPI_sends = NULL;
  _MPI_receives = NULL;
  _neighbor_connections.clear();
  int id,numprocs;
 
  // MPI_Comm_rank(MPI_COMM_WORLD,&id); 
  // MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  // attach_gpu(id%4);
#endif
}


/**
 * @brief Solver destructor frees all memory on the device, including arrays
 *        for the FSR scalar fluxes and sources and Track boundary fluxes.
 */
GPUSolver::~GPUSolver() {

  if (_FSR_volumes != NULL) {
    cudaFree(_FSR_volumes);
    _FSR_volumes = NULL;
  }

  if (_FSR_materials != NULL) {
    cudaFree(_FSR_materials);
    _FSR_materials = NULL;
  }

  if (_materials != NULL) {
    cudaFree(_materials);
    _materials = NULL;
  }

  if (_dev_tracks != NULL) {
    cudaFree(_dev_tracks);
    _dev_tracks = NULL;
  }
  if (dev_tracks_2d != NULL) {
    cudaFree(dev_tracks_2d);
    dev_tracks_2d = NULL;
  }
  
  getLastCudaError();
}


/**
 * @brief Returns the number of thread blocks to execute on the GPU.
 * @return the number of thread blocks
 */
int GPUSolver::getNumThreadBlocks() {
  return _B;
}


/**
 * @brief Returns the number of threads per block to execute on the GPU.
 * @return the number of threads per block
 */
int GPUSolver::getNumThreadsPerBlock() {
  return _T;
}


/**
 * @brief Returns the source for some energy group for a flat source region
 * @details This is a helper routine used by the openmoc.process module.
 * @param fsr_id the ID for the FSR of interest
 * @param group the energy group of interest
 * @return the flat source region source
 */
double GPUSolver::getFSRSource(long fsr_id, int group) {

  if (fsr_id >= _num_FSRs)
    log_printf(ERROR, "Unable to return a source for FSR ID = %d "
               "since the max FSR ID = %d", fsr_id, _num_FSRs-1);

  else if (fsr_id < 0)
    log_printf(ERROR, "Unable to return a source for FSR ID = %d "
               "since FSRs do not have negative IDs", fsr_id);

  else if (group-1 >= _NUM_GROUPS)
    log_printf(ERROR, "Unable to return a source in group %d "
               "since there are only %d groups", group, _NUM_GROUPS);

  else if (group <= 0)
    log_printf(ERROR, "Unable to return a source in group %d "
               "since groups must be greater or equal to 1", group);

  else if (dev_scalar_flux.size() == 0)
    log_printf(ERROR, "Unable to return a source "
               "since it has not yet been computed");

  /* Get host material */
  Material* host_material = _geometry->findFSRMaterial(fsr_id);

  /* Get cross sections and scalar flux */
  FP_PRECISION* sigma_s = host_material->getSigmaS();
  FP_PRECISION* fiss_mat = host_material->getFissionMatrix();

  FP_PRECISION* fsr_scalar_fluxes = new FP_PRECISION[_NUM_GROUPS];
  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);
  cudaMemcpy(fsr_scalar_fluxes, &scalar_flux[fsr_id*_NUM_GROUPS],
             _NUM_GROUPS * sizeof(FP_PRECISION),
             cudaMemcpyDeviceToHost);
  getLastCudaError();

  FP_PRECISION fission_source = 0.0;
  FP_PRECISION scatter_source = 0.0;
  FP_PRECISION total_source;

  /* Compute total scattering and fission sources for this FSR */
  for (int g=0; g < _NUM_GROUPS; g++) {
    scatter_source += sigma_s[(group-1)*(_NUM_GROUPS)+g]
                      * fsr_scalar_fluxes[g];
    fission_source += fiss_mat[(group-1)*(_NUM_GROUPS)+g]
                      * fsr_scalar_fluxes[g];
  }

  fission_source /= _k_eff;

  /* Compute the total source */
  total_source = fission_source + scatter_source;

  /* Add in fixed source (if specified by user) *///TODO  fix fixedsource//TODO
  total_source += _fixed_sources(fsr_id,group-1);

  /* Normalize to solid angle for isotropic approximation */
  total_source *= ONE_OVER_FOUR_PI;

  delete [] fsr_scalar_fluxes;

  return total_source;
}


/**
 * @brief Returns the scalar flux for some FSR and energy group.
 * @param fsr_id the ID for the FSR of interest
 * @param group the energy group of interest
 * @return the FSR scalar flux
 */
double GPUSolver::getFlux(long fsr_id, int group) {

  if (fsr_id >= _num_FSRs)
    log_printf(ERROR, "Unable to return a scalar flux for FSR ID = %d "
               "since the max FSR ID = %d", fsr_id, _num_FSRs-1);

  else if (fsr_id < 0)
    log_printf(ERROR, "Unable to return a scalar flux for FSR ID = %d "
               "since FSRs do not have negative IDs", fsr_id);

  else if (group-1 >= _NUM_GROUPS)
    log_printf(ERROR, "Unable to return a scalar flux in group %d "
               "since there are only %d groups", group, _NUM_GROUPS);

  else if (group <= 0)
    log_printf(ERROR, "Unable to return a scalar flux in group %d "
               "since groups must be greater or equal to 1", group);

  if (dev_scalar_flux.size() == 0)
    log_printf(ERROR, "Unable to return a scalar flux "
               "since it has not yet been computed");

  return _scalar_flux(fsr_id,group-1);
}


/**
 * @brief Fills an array with the scalar fluxes on the GPU.
 * @details This class method is a helper routine called by the OpenMOC
 *          Python "openmoc.krylov" module for Krylov subspace methods.
 *          Although this method appears to require two arguments, in
 *          reality it only requires one due to SWIG and would be called
 *          from within Python as follows:
 *
 * @code
 *          num_fluxes = NUM_GROUPS * num_FSRs
 *          fluxes = solver.getFluxes(num_fluxes)
 * @endcode
 *
 * @param fluxes an array of FSR scalar fluxes in each energy group
 * @param num_fluxes the total number of FSR flux values
 */
void GPUSolver::getFluxes(FP_PRECISION* out_fluxes, int num_fluxes) {

  if (num_fluxes != _NUM_GROUPS * _num_FSRs)
    log_printf(ERROR, "Unable to get FSR scalar fluxes since there are "
               "%d groups and %d FSRs which does not match the requested "
               "%d flux values", _NUM_GROUPS, _num_FSRs, num_fluxes);

  else if (dev_scalar_flux.size() == 0)
    log_printf(ERROR, "Unable to get FSR scalar fluxes since they "
               "have not yet been allocated on the device");

  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);

  /* Copy the fluxes from the GPU to the input array */
  cudaMemcpy(out_fluxes, scalar_flux,
            num_fluxes * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
  getLastCudaError();
}


/**
 * @brief Sets the number of thread blocks (>0) for CUDA kernels.
 * @param num_blocks the number of thread blocks
 */
void GPUSolver::setNumThreadBlocks(int num_blocks) {

  if (num_blocks < 0)
    log_printf(ERROR, "Unable to set the number of CUDA thread blocks "
               "to %d since it is a negative number", num_blocks);

  _B = num_blocks;
}


/**
 * @brief Sets the number of threads per block (>0) for CUDA kernels.
 * @param num_threads the number of threads per block
 */
void GPUSolver::setNumThreadsPerBlock(int num_threads) {

  if (num_threads < 0)
    log_printf(ERROR, "Unable to set the number of CUDA threads per block "
               "to %d since it is a negative number", num_threads);

  _T = num_threads;
}


/**
 * @brief Sets the Geometry for the Solver.
 * @details This is a private setter method for the Solver and is not
 *          intended to be called by the user.
 * @param geometry a pointer to a Geometry object
 */
void GPUSolver::setGeometry(Geometry* geometry) {

  Solver::setGeometry(geometry);

  std::map<int, Material*> host_materials=_geometry->getAllMaterials();
  std::map<int, Material*>::iterator iter;
  int material_index = 0;

  /* Iterate through all Materials and clone them as dev_material structs
   * on the device */
  for (iter=host_materials.begin(); iter != host_materials.end(); ++iter) {
    _material_IDs_to_indices[iter->second->getId()] = material_index;
    material_index++;
  }
}


/**
 * @brief Sets the Solver's TrackGenerator with characteristic Tracks.
 * @details The TrackGenerator must already have generated Tracks and have
 *          used ray tracing to segmentize them across the Geometry. This
 *          should be initated in Python prior to assigning the TrackGenerator
 *          to the Solver:
 *
 * @code
 *          track_generator.generateTracks()
 *          solver.setTrackGenerator(track_generator)
 * @endcode
 *
 * @param track_generator a pointer to a TrackGenerator object
 */
void GPUSolver::setTrackGenerator(TrackGenerator* track_generator) {
  Solver::setTrackGenerator(track_generator);
  initializeTracks();
  copyQuadrature();
}


/**
 * @brief Set the flux array for use in transport sweep source calculations.
 * @detail This is a helper method for the checkpoint restart capabilities,
 *         as well as the IRAMSolver in the openmoc.krylov submodule. This
 *         routine may be used as follows from within Python:
 *
 * @code
 *          num_FSRs = solver.getGeometry.getNumFSRs()
 *          NUM_GROUPS = solver.getGeometry.getNumEnergyGroups()
 *          fluxes = numpy.random.rand(num_FSRs * NUM_GROUPS, dtype=np.float)
 *          solver.setFluxes(fluxes)
 * @endcode
 *
 *          NOTE: This routine stores a pointer to the fluxes for the Solver
 *          to use during transport sweeps and other calculations. Hence, the
 *          flux array pointer is shared between NumPy and the Solver.
 *
 * @param in_fluxes an array with the fluxes to use
 * @param num_fluxes the number of flux values (# groups x # FSRs)
 */
void GPUSolver::setFluxes(FP_PRECISION* in_fluxes, int num_fluxes) {
  if (num_fluxes != _NUM_GROUPS * _num_FSRs)
    log_printf(ERROR, "Unable to set an array with %d flux values for %d "
               " groups and %d FSRs", num_fluxes, _NUM_GROUPS, _num_FSRs);

  /* Allocate array if flux arrays have not yet been initialized */
  if (dev_scalar_flux.size() == 0)
    initializeFluxArrays();

  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);

  /* Copy the input fluxes onto the GPU */
  cudaMemcpy(scalar_flux, in_fluxes,
             num_fluxes * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
  getLastCudaError();
  _user_fluxes = true;
}


/**
 * @brief Creates a polar quadrature object for the GPUSolver on the GPU.
 */
void GPUSolver::copyQuadrature() {

  log_printf(INFO, "Copying quadrature on the GPU...");

  if (_num_polar / 2 > MAX_POLAR_ANGLES_GPU)
    log_printf(ERROR, "Unable to initialize a polar quadrature with %d "
               "angles for the GPUSolver which is limited to %d polar "
               "angles. Update the MAX_POLAR_ANGLES_GPU macro in constants.h "
               "and recompile.", _num_polar/2, MAX_POLAR_ANGLES_GPU);
			   
  /* Copy half the number of polar angles to constant memory on the GPU */
  bool   SOLVE_3D =_SOLVE_3D ;
  cudaMemcpyToSymbol(_solve_3D, &SOLVE_3D, sizeof(bool),0, 
                     cudaMemcpyHostToDevice);
  getLastCudaError();																	  

  /* Copy half the number of polar angles to constant memory on the GPU */
  int polar2 = _num_polar/2;
  cudaMemcpyToSymbol(num_polar_2, &polar2, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  getLastCudaError();

  /* Copy the number of polar angles to constant memory on the GPU */
  cudaMemcpyToSymbol(num_polar, &_num_polar,
                     sizeof(int), 0, cudaMemcpyHostToDevice);
  getLastCudaError();

  /* Copy the number of azimuthal angles to constant memory on the GPU */
  cudaMemcpyToSymbol(num_azim, &_num_azim,
                    sizeof(int), 0, cudaMemcpyHostToDevice); 
  getLastCudaError();

  /* Copy the weights to constant memory on the GPU */
  FP_PRECISION total_weights[_num_azim * _num_polar];
  for (int a=0; a < _num_azim; a++)
    for (int p=0; p < _num_polar; p++){
      total_weights[a*_num_polar + p] = _quad->getWeight(a, p);
	}
  cudaMemcpyToSymbol(weights, total_weights,
      _num_polar * _num_azim * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
  getLastCudaError();

  /* Copy the sines of the polar angles which is needed for the rational
   * approximation to the 1-exp(-x) function
   * Something really confusing: sin(theta) list is *always* in
   * double precision! Need to go through and convert a few npolar/2 of them.
   */
  auto host_sin_thetas = _quad->getSinThetas();
  std::vector<FP_PRECISION> fp_precision_sines(_num_polar/2);
  for (int j=0; j<_num_polar/2; ++j)
    fp_precision_sines[j] = (FP_PRECISION)host_sin_thetas[0][j];
  cudaMemcpyToSymbol(sin_thetas, &fp_precision_sines[0],
                     _num_polar/2 * sizeof(FP_PRECISION), 0,
                     cudaMemcpyHostToDevice);

  getLastCudaError();
}


/**
 * @brief Since rational expression exponential evaluation is easily
          done in a standalone function on GPU, do no exp evaluator setup.
 */
void GPUSolver::initializeExpEvaluators() {}


/**
 * @brief Explicitly disallow construction of CMFD, for now.
 */
void GPUSolver::initializeCmfd() {
  /* Raise an error only if CMFD was attempted to be set.
     Otherwise this fails every time. */
  if (_cmfd != NULL)
    log_printf(ERROR, "CMFD not implemented for GPUSolver yet. Get to work!");
}


/**
 * @brief Initializes the FSR volumes and dev_materials array on the GPU.
 * @details This method assigns each FSR a unique, monotonically increasing
 *          ID, sets the Material for each FSR, and assigns a volume based on
 *          the cumulative length of all of the segments inside the FSR.
 */
void GPUSolver::initializeFSRs() {

  log_printf(NORMAL, "Initializing FSRs on the GPU...");

  /* Delete old FSRs array if it exists */
  if (_FSR_volumes != NULL) {
    cudaFree(_FSR_volumes);
    getLastCudaError();
    _FSR_volumes = NULL;
  }

  if (_FSR_materials != NULL) {
    cudaFree(_FSR_materials);
    getLastCudaError();
    _FSR_materials = NULL;
  }

  Solver::initializeFSRs();

  /* Allocate memory for all FSR volumes and dev_materials on the device */
  try{

    /* Store pointers to arrays of FSR data created on the host by the
     * the parent class Solver::initializeFSRs() routine */
    FP_PRECISION* host_FSR_volumes = _FSR_volumes;
    int* host_FSR_materials = _FSR_materials;
    cudaMalloc(&_FSR_volumes, _num_FSRs * sizeof(FP_PRECISION));
    getLastCudaError();
    cudaMalloc(&_FSR_materials, _num_FSRs * sizeof(int));
    getLastCudaError();

    /* Create a temporary FSR to material indices array */
    int* FSRs_to_material_indices = new int[_num_FSRs];

    /* Populate FSR Material indices array */
    for (long i = 0; i < _num_FSRs; i++)
      FSRs_to_material_indices[i] = _material_IDs_to_indices[_geometry->
        findFSRMaterial(i)->getId()];

    /* Copy the arrays of FSR data to the device */
    cudaMemcpy(_FSR_volumes, host_FSR_volumes,
      _num_FSRs * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
    getLastCudaError();
    cudaMemcpy(_FSR_materials, FSRs_to_material_indices,
      _num_FSRs * sizeof(int), cudaMemcpyHostToDevice);
    getLastCudaError();

    /* Copy the number of FSRs into constant memory on the GPU */
    cudaMemcpyToSymbol(num_FSRs, &_num_FSRs, sizeof(long), 0,
      cudaMemcpyHostToDevice);
    getLastCudaError();

    /* There isn't any other great place to put what comes next */
    cudaMemcpyToSymbol(stabilization_type, &_stabilization_type,
        sizeof(stabilizationType), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(stabilization_factor, &_stabilization_factor,
        sizeof(double), 0, cudaMemcpyHostToDevice);
    getLastCudaError();

    /* Free the array of FSRs data allocated by the Solver parent class */
    free(host_FSR_materials);

    /* Free the temporary array of FSRs to material indices on the host */
    free(FSRs_to_material_indices);
  }
  catch(std::exception &e) {
    log_printf(DEBUG, e.what());
    log_printf(ERROR, "Could not allocate memory for FSRs on GPU");
  }
}


/**
 * @brief Allocates all Materials data on the GPU.`
 * @details This method loops over the materials in the host_materials map.
 *          Since CUDA does not support std::map data types on the device,
 *          the materials map must be converted to an array and a map created
 *          that maps a material ID to an indice in the new materials array. In
 *          initializeTracks, this map is used to convert the Material ID
 *          associated with every segment to an index in the materials array.
 * @param mode the solution type (FORWARD or ADJOINT)
 */
void GPUSolver::initializeMaterials(solverMode mode) {

  Solver::initializeMaterials(mode);

  log_printf(INFO, "Initializing materials on the GPU...");
_num_materials = _track_generator->getGeometry()->getNumMaterials();
  /* Sanity check */
  if (_num_materials <= 0)
    log_printf(ERROR, "Attempt to initialize GPU materials with zero or less materials.");
  if (_NUM_GROUPS <= 0)
    log_printf(ERROR, "Attempt to initialize GPU XS data with zero or less energy groups.");

  /* Copy the number of energy groups to constant memory on the GPU */
#ifndef NGROUPS
  cudaMemcpyToSymbol(NUM_GROUPS, &_NUM_GROUPS, sizeof(int));
  getLastCudaError();
#endif

  /* Copy the number of polar angles times energy groups to constant memory
   * on the GPU */
  cudaMemcpyToSymbol(polar_times_groups, &_fluxes_per_track,
                     sizeof(int), 0, cudaMemcpyHostToDevice);
  getLastCudaError();

  /* Delete old materials array if it exists */
  if (_materials != NULL)
  {
    cudaFree(_materials);
    getLastCudaError();
  }

  /* Allocate memory for all dev_materials on the device */
  try{

    std::map<int, Material*> host_materials=_geometry->getAllMaterials();
    std::map<int, Material*>::iterator iter;
    int material_index = 0;

    /* Iterate through all Materials and clone them as dev_material structs
     * on the device */
    cudaMalloc(&_materials, _num_materials * sizeof(dev_material));
    getLastCudaError();
    
    for (iter=host_materials.begin(); iter != host_materials.end(); ++iter) {
      clone_material(iter->second, &_materials[material_index]);

      material_index++;
    }
  }

  catch(std::exception &e) {
    log_printf(DEBUG, e.what());
    log_printf(ERROR, "Could not allocate memory for Materials on GPU");
  }
}

/**
 * @brief Reset all fixed sources and fixed sources moments to 0.
 * @Add by lsder 从Solver 继承，算是为了不报错吧
 */
void GPUSolver::resetFixedSources() {

  //TODO
}

/**
 * @brief Allocates memory for all Tracks on the GPU
 */
void GPUSolver::initializeTracks() {

  log_printf(INFO, "Initializing tracks on the GPU...");

  /* Delete old Tracks array if it exists */
  if (_dev_tracks != NULL)
    cudaFree(_dev_tracks);
  if (dev_tracks_2d != NULL)
    cudaFree(dev_tracks_2d);
if(_segment_formation==EXPLICIT_3D)
  {
  /* Allocate memory for all Tracks and Track offset indices on the device */
  try {
    /* Iterate through all Tracks and clone them as dev_tracks on the device */
    int index;
    if (is3D()) 
    {
      /* Allocate array of dev_tracks */
        cudaMalloc(&_dev_tracks, _tot_num_tracks * sizeof(dev_track));
        getLastCudaError();
        for (int i=0; i < _tot_num_tracks ; i++) {
              clone_track(_tracks_3D[i], &_dev_tracks[i], _material_IDs_to_indices);
              /* Get indices to next tracks along "forward" and "reverse" directions */
              index = _tracks_3D[i]->getTrackNextFwd();
              cudaMemcpy(&_dev_tracks[i]._next_track_fwd,
                  &index, sizeof(int), cudaMemcpyHostToDevice);
              index = _tracks_3D[i]->getTrackNextBwd();
              cudaMemcpy(&_dev_tracks[i]._next_track_bwd,
                  &index, sizeof(int), cudaMemcpyHostToDevice);
        } 
      }
    else
    {
    for (int i=0; i < _tot_num_tracks; i++) {

    clone_track(_tracks[i], &_dev_tracks[i], _material_IDs_to_indices);

    /* Get indices to next tracks along "forward" and "reverse" directions */
    index = _tracks[i]->getTrackNextFwd();
    cudaMemcpy(&_dev_tracks[i]._next_track_fwd,
          &index, sizeof(int), cudaMemcpyHostToDevice);

    index = _tracks[i]->getTrackNextBwd();
    cudaMemcpy(&_dev_tracks[i]._next_track_bwd,
          &index, sizeof(int), cudaMemcpyHostToDevice);
      }
      }
    
    /* Copy the total number of Tracks into constant memory on GPU */
    cudaMemcpyToSymbol(tot_num_tracks, &_tot_num_tracks,
              sizeof(long), 0, cudaMemcpyHostToDevice);
    }
    catch(std::exception &e) {
      log_printf(DEBUG, e.what());
      log_printf(ERROR, "Could not allocate memory for Tracks on GPU");
    }
  }
  else
  {
  TrackGenerator3D* track_generator_3D =
          dynamic_cast<TrackGenerator3D*>(_track_generator);
    dev_track2D* _h_tracks_2D  = new dev_track2D[_tot_num_track_2d];
    cpytrack(_tracks, _h_tracks_2D, true);
    cudaMalloc(&dev_tracks_2d, sizeof(dev_track2D)*_tot_num_track_2d);
    clone_storage += sizeof(dev_track2D)*_tot_num_track_2d;

    cudaMemcpy(dev_tracks_2d, _h_tracks_2D, sizeof(dev_track2D)*_tot_num_track_2d, 
              cudaMemcpyHostToDevice);
    h_tracks_2D=_h_tracks_2D;
  //delete [] tracks_2D1;
  }
}


/**
 * @brief Allocates memory for Track boundary angular and FSR scalar fluxes.
 * @details Deletes memory for old flux vectors if they were allocated for a
 *          previous simulation.
 */
void GPUSolver::initializeFluxArrays() {

  log_printf(INFO, "Initializing flux vectors on the GPU...");

  /* Allocate memory for all flux arrays on the device */
  try {
    long size = 2 * _tot_num_tracks * _fluxes_per_track;
    long max_size = size;
#ifdef MPIX
    if (_geometry->isDomainDecomposed())
      MPI_Allreduce(&size, &max_size, 1, MPI_LONG, MPI_MAX,
                    _geometry->getMPICart());
#endif
    double max_size_mb = (double) (2 * max_size * sizeof(float))
        / (double) (1e6);
    max_size_mb /= 2;
    log_printf(NORMAL, "Max boundary angular flux storage per domain = %6.2f "
               "MB", max_size_mb);

    //for gpu
    dev_boundary_flux.resize(size);
    dev_start_flux.resize(size);
     //for cpu
     _boundary_flux = new float[size]();
     _start_flux = new float[size]();
    size = _num_FSRs * _NUM_GROUPS;
    max_size = size;
#ifdef MPIX
    if (_geometry->isDomainDecomposed())
      MPI_Allreduce(&size, &max_size, 1, MPI_LONG, MPI_MAX,
                    _geometry->getMPICart());
#endif

    /* Determine the amount of memory allocated */
    int num_flux_arrays = 2;
    if (_stabilize_transport)
      num_flux_arrays++;

    max_size_mb = (double) (num_flux_arrays * max_size * sizeof(FP_PRECISION))
        / (double) (1e6);
    log_printf(NORMAL, "Max scalar flux storage per domain = %6.2f MB",
               max_size_mb);

    /* Allocate scalar fluxes */
    dev_scalar_flux.resize(size);
    dev_old_scalar_flux.resize(size);

    _scalar_flux = new FP_PRECISION[size]();
    _old_scalar_flux = new FP_PRECISION[size]();

    if (_stabilize_transport)
      dev_stabilizing_flux.resize(size);
#ifdef MPIx
    /* Allocate memory for angular flux exchanging buffers */
    if (_geometry->isDomainDecomposed())
      setupMPIBuffers();
#endif
  }
  catch(std::exception &e) {
    log_printf(DEBUG, e.what());
    log_printf(ERROR, "Could not allocate memory for fluxes on GPU");
  }
}


/**
 * @brief Allocates memory for FSR source vectors on the GPU.
 * @details Deletes memory for old source vectors if they were allocated
 *          for a previous simulation.
 */
void GPUSolver::initializeSourceArrays() {

  log_printf(INFO, "Initializing source vectors on the GPU...");
  int size = _num_FSRs * _NUM_GROUPS;

  /* Allocate memory for all source arrays on the device */
  try{
    dev_reduced_sources.resize(size);
    dev_fixed_sources.resize(size);
   
	long max_size = size;
#ifdef MPIX
  if (_geometry->isDomainDecomposed())
    MPI_Allreduce(&size, &max_size, 1, MPI_LONG, MPI_MAX,
                  _geometry->getMPICart());
#endif
  double max_size_mb = (double) (max_size * sizeof(FP_PRECISION))
        / (double) (1e6);
  if (_fixed_sources_on)
    max_size_mb *= 2;
  log_printf(NORMAL, "Max source storage per domain = %6.2f MB",
             max_size_mb);
  }
  catch(std::exception &e) {
    log_printf(DEBUG, e.what());
    log_printf(ERROR, "Could not allocate memory for sources on GPU");
  }

  /* Initialize fixed sources to zero */
  thrust::fill(dev_fixed_sources.begin(), dev_fixed_sources.end(), 0.0);

  /* Fill fixed sources with those assigned by Cell, Material or FSR */
  initializeFixedSources();
}


/**
 * @brief Populates array of fixed sources assigned by FSR.
 */
void GPUSolver::initializeFixedSources() {

  Solver::initializeFixedSources();

  long fsr_id, group;
  std::pair<int, int> fsr_group_key;
  std::map< std::pair<int, int>, FP_PRECISION >::iterator fsr_iter;

  /* Populate fixed source array with any user-defined sources */
  for (fsr_iter = _fix_src_FSR_map.begin();
       fsr_iter != _fix_src_FSR_map.end(); ++fsr_iter) {

    /* Get the FSR with an assigned fixed source */
    fsr_group_key = fsr_iter->first;
    fsr_id = fsr_group_key.first;
    group = fsr_group_key.second;

    if (group <= 0 || group > _NUM_GROUPS)
      log_printf(ERROR,"Unable to use fixed source for group %d in "
                 "a %d energy group problem", group, _NUM_GROUPS);

    if (fsr_id < 0 || fsr_id >= _num_FSRs)
      log_printf(ERROR,"Unable to use fixed source for FSR %d with only "
                 "%d FSRs in the geometry", fsr_id, _num_FSRs);

    _fixed_sources(fsr_id, group-1) = _fix_src_FSR_map[fsr_group_key];//TODO
  }
}


/**
 * @brief Zero each Track's boundary fluxes for each energy group and polar
 *        angle in the "forward" and "reverse" directions.
 */
void GPUSolver::zeroTrackFluxes() {
  thrust::fill(dev_boundary_flux.begin(), dev_boundary_flux.end(), 0.0);
  thrust::fill(dev_start_flux.begin(), dev_start_flux.end(), 0.0);
}


/**
 * @brief Set the scalar flux for each FSR and energy group to some value.
 * @param value the value to assign to each FSR scalar flux
 */
void GPUSolver::flattenFSRFluxes(FP_PRECISION value) {
  thrust::fill(dev_scalar_flux.begin(), dev_scalar_flux.end(), value);
}

/**
 * @brief Set the scalar flux for each FSR to match the fission energy
 *        distribution of the material called chi_spectrum_material.
 */
void GPUSolver::flattenFSRFluxesChiSpectrum() {
    if (_dev_chi_spectrum_material == NULL)
        log_printf(ERROR, "Chi spectrum material not set on GPU. If you set "
                "it on the CPU, but still see this error, there's a problem.");

    FP_PRECISION* scalar_flux = thrust::raw_pointer_cast(&dev_scalar_flux[0]);
    flattenFSRFluxesChiSpectrumOnDevice<<<_B, _T>>>(_dev_chi_spectrum_material,
                                                    scalar_flux);
}


/**
 * @brief Stores the FSR scalar fluxes in the old scalar flux array.
 */
void GPUSolver::storeFSRFluxes() {
  thrust::copy(dev_scalar_flux.begin(), dev_scalar_flux.end(),
               dev_old_scalar_flux.begin());
}

void GPUSolver::computeStabilizingFlux() {
  if (!_stabilize_transport) return;

  if (_stabilization_type == GLOBAL) {
    FP_PRECISION* scalar_flux = thrust::raw_pointer_cast(&dev_scalar_flux[0]);
    FP_PRECISION* stabilizing_flux = thrust::raw_pointer_cast(&dev_stabilizing_flux[0]);
    computeStabilizingFluxOnDevice<<<_B, _T>>>(scalar_flux, stabilizing_flux);
  }
  else
    log_printf(ERROR, "Only global stabilization works on GPUSolver now.");
}

void GPUSolver::stabilizeFlux() {
  if (!_stabilize_transport) return;

  if (_stabilization_type == GLOBAL) {
    FP_PRECISION* scalar_flux = thrust::raw_pointer_cast(&dev_scalar_flux[0]);
    FP_PRECISION* stabilizing_flux = thrust::raw_pointer_cast(&dev_stabilizing_flux[0]);
    stabilizeFluxOnDevice<<<_B, _T>>>(scalar_flux, stabilizing_flux);
  }
}

/**
 * @brief Normalizes all FSR scalar fluxes and Track boundary angular
 *        fluxes to the total fission source (times \f$ \nu \f$).
 */
double GPUSolver::normalizeFluxes() {

  /** Create Thrust vector of fission sources in each FSR */
  thrust::device_vector<FP_PRECISION> fission_sources_vec;
  fission_sources_vec.resize(_B * _T);
  FP_PRECISION* fission_sources =
       thrust::raw_pointer_cast(&fission_sources_vec[0]);

  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);

  int shared_mem = sizeof(FP_PRECISION) * _T;

  computeFissionSourcesOnDevice<<<_B, _T, shared_mem>>>(_FSR_volumes,
                                                        _FSR_materials,
                                                        _materials,
                                                        scalar_flux,
                                                        fission_sources);

  /* Compute the total fission source */
  double tot_fission_source =  thrust::reduce(fission_sources_vec.begin(),fission_sources_vec.end());


/* Get the total number of source regions */
  long total_num_FSRs = _num_FSRs;

#ifdef MPIx
  /* Reduce total fission rates across domains */
  if (_geometry->isDomainDecomposed()) {

    /* Get the communicator */
    MPI_Comm comm = _geometry->getMPICart();

    /* Reduce fission rates */
    double reduced_fission;
    MPI_Allreduce(&tot_fission_source, &reduced_fission, 1, MPI_DOUBLE,
                  MPI_SUM, comm);
    tot_fission_source = reduced_fission;

    /* Get total number of FSRs across all domains */
    MPI_Allreduce(&_num_FSRs, &total_num_FSRs, 1, MPI_LONG, MPI_SUM, comm);
  }
#endif
 
  double norm_factor = total_num_FSRs /tot_fission_source;

  /* Multiply all scalar and angular fluxes by the normalization constant */
  thrust::transform(dev_scalar_flux.begin(), dev_scalar_flux.end(),
                    thrust::constant_iterator<double>(norm_factor),
                    dev_scalar_flux.begin(), thrust::multiplies<FP_PRECISION>());
  // thrust::transform(dev_old_scalar_flux.begin(), dev_old_scalar_flux.end(),
  //                 thrust::constant_iterator<FP_PRECISION>(norm_factor),
  //                 dev_old_scalar_flux.begin(),
  //                thrust::multiplies<FP_PRECISION>());
  thrust::transform(dev_boundary_flux.begin(), dev_boundary_flux.end(),
                    thrust::constant_iterator<double>(norm_factor),
                    dev_boundary_flux.begin(), thrust::multiplies<FP_PRECISION>());
  thrust::transform(dev_start_flux.begin(), dev_start_flux.end(),
                    thrust::constant_iterator<double>(norm_factor),
                    dev_start_flux.begin(), thrust::multiplies<FP_PRECISION>());

  return norm_factor;
}


/**
 * @brief Computes the total source (fission, scattering, fixed) in each FSR.
 * @details This method computes the total source in each FSR based on
 *          this iteration's current approximation to the scalar flux.
 */
void GPUSolver::computeFSRSources(int iteration) {

  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);
  FP_PRECISION* fixed_sources =
       thrust::raw_pointer_cast(&dev_fixed_sources[0]);
  FP_PRECISION* reduced_sources =
       thrust::raw_pointer_cast(&dev_reduced_sources[0]);

  // Zero sources if under 30 iterations, as is custom in CPUSolver
  bool zeroSources;
  if (iteration < 30)
    zeroSources = true;
  else
    zeroSources = false;

  computeFSRSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials,
                                        scalar_flux, fixed_sources,
                                        reduced_sources, 1.0 / _k_eff,
                                        zeroSources);
}


/**
 * @brief Computes the fission source in each FSR.
 * @details This method computes the fission source in each FSR based on
 *          this iteration's current approximation to the scalar flux.
 */
void GPUSolver::computeFSRFissionSources() {

  log_printf(DEBUG, "compute FSR fission sources\n");

  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);
  FP_PRECISION* reduced_sources =
       thrust::raw_pointer_cast(&dev_reduced_sources[0]);

  computeFSRFissionSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, true,
                                               scalar_flux, reduced_sources);
}


/**
 * @brief Computes the scatter source in each FSR.
 * @details This method computes the scatter source in each FSR based on
 *          this iteration's current approximation to the scalar flux.
 */
void GPUSolver::computeFSRScatterSources() {

  log_printf(DEBUG, "compute fsr scatter sources\n");

  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);
  FP_PRECISION* reduced_sources =
       thrust::raw_pointer_cast(&dev_reduced_sources[0]);

  computeFSRScatterSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, true,
                                               scalar_flux, reduced_sources);
}


/**
 * @brief This method performs one transport sweep of all azimuthal angles,
 *        Tracks, Track segments, polar angles and energy groups.
 * @details The method integrates the flux along each Track and updates the
 *          boundary fluxes for the corresponding output Track, while updating
 *          the scalar flux in each flat source region.
 */
void GPUSolver::transportSweep() {
  int shared_mem = _T * _num_polar * sizeof(FP_PRECISION);

  log_printf(DEBUG, "Transport sweep on device with %d blocks and %d threads",
             _B, _T);
  _timer->startTimer();
  /* Get device pointer to the Thrust vectors */
  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);
  float* boundary_flux =
       thrust::raw_pointer_cast(&dev_boundary_flux[0]);
  float* start_flux =
       thrust::raw_pointer_cast(&dev_start_flux[0]);
  FP_PRECISION* reduced_sources =
       thrust::raw_pointer_cast(&dev_reduced_sources[0]);

  log_printf(DEBUG, "Obtained device pointers to thrust vectors.\n");

  /* Initialize flux in each FSR to zero */
  flattenFSRFluxes(0.0);

  /* Copy starting flux to current flux */
  cudaMemcpy(boundary_flux, start_flux, 2 * _tot_num_tracks *
             _fluxes_per_track * sizeof(float),
             cudaMemcpyDeviceToDevice);
  getLastCudaError();

  log_printf(DEBUG, "Copied host to device flux.");

  /* Perform transport sweep on all tracks */
  _timer->startTimer();
  if(_segment_formation==OTF_TRACKS)
  {
  
  /* Perform transport sweep on all tracks */
  if(!_configure->if_pre_track){
  //  cpu测试时间
   _timer->startTimer();
  OTFsolver<<<_B,_T>>>(dev_tracks_2d, d_num_x, d_num_y, d_first_lz_of_stack, 
                         d_tracks_per_stack, d_tracks_2D_xy, d_track_2D_chain, 
                         d_track_2D_chain_link, d_num_z, d_num_l, d_dz_eff, 
                         d_dl_eff, d_theta, d_cum_tracks_per_stack, d_extruded_FSR_lookup,
                         scalar_flux, boundary_flux,
                         start_flux, reduced_sources, _materials,d_track_3D_index,d_azim_size);
  cudaDeviceSynchronize();
  getLastCudaError();

  _timer->stopTimer();
  float Time1 =_timer->getTime();
  
  log_printf(NORMAL, "OTFsolver time: %1.4E sec",Time1);
  // test<<<64,64>>>(d_track_3D_index);
   }
  else{
    log_printf(NORMAL, "ONLY CAL SEGMENTS AND TRANSPORTSWEEP");
    // cudaThreadSynchronize();
    _timer->startTimer();
    transportSweepOnDevice<<<_B, _T, shared_mem>>>(scalar_flux, boundary_flux,
                                                   start_flux, reduced_sources,
                                                   _materials, d_track_3D,
                                                   0, pre_segments_index);
    if(pre_segments_index < _num_3D_tracks){
      onlygeneratesegments<<<_B,_T>>>(dev_tracks_2d, d_extruded_FSR_lookup,
                                      d_track_3D,
                                      scalar_flux, boundary_flux, start_flux, 
                                      reduced_sources, _materials, d_track_3D_index, d_start, pre_segments_index);
    }
    cudaDeviceSynchronize();
    getLastCudaError();
   _timer->stopTimer();
   float Time1 =_timer->getTime();
   log_printf(NORMAL, "OTFsolver time: %1.4E sec",Time1);
   }
  }
  else if(_segment_formation==EXPLICIT_3D)
 {
     _timer->startTimer();
    transportSweepOnDevice<<<_B, _T, shared_mem>>>(scalar_flux, boundary_flux,
                                                  start_flux, reduced_sources,
                                                  _materials, _dev_tracks,
                                                  0, _tot_num_tracks);
    cudaDeviceSynchronize();
    getLastCudaError();
    _timer->stopTimer();
    float Time2 =_timer->getTime();
  log_printf(NORMAL, "transportSweepOnDevice time: %1.4E sec", Time2);
 }
  else if(_segment_formation==EXPLICIT_2D)
 {
  //    _timer->startTimer();
  //   transportSweepOnDevice<<<_B, _T, shared_mem>>>(scalar_flux, boundary_flux,
  //                                                 start_flux, reduced_sources,
  //                                                 _materials, dev_tracks_2d,
  //                                                 0, _tot_num_tracks);
  //   cudaDeviceSynchronize();
  //   getLastCudaError();
  //   _timer->stopTimer();
  //   float Time2 =_timer->getTime();
  // log_printf(NORMAL, "transportSweepOnDevice time: %1.4E sec", Time2);
 }
 
  
  _timer->stopTimer();
  _timer->recordSplit("Transport Sweep kernel");
  log_printf(DEBUG, "Finished sweep on GPU.\n");
  cudaMemcpy(_boundary_flux, boundary_flux, 2 * _tot_num_tracks *
          _fluxes_per_track * sizeof(float),
          cudaMemcpyDeviceToHost);
  getLastCudaError();
  cudaMemcpy(_start_flux, start_flux, 2 * _tot_num_tracks *
          _fluxes_per_track * sizeof(float),
          cudaMemcpyDeviceToHost);

  getLastCudaError();
  cudaMemcpy(_scalar_flux, scalar_flux, _NUM_GROUPS * _num_FSRs * sizeof(FP_PRECISION), 
          cudaMemcpyDeviceToHost);
  getLastCudaError();

#ifdef MPIx
  /* Transfer all interface fluxes after the transport sweep */
  if (_track_generator->getGeometry()->isDomainDecomposed())
    transferAllInterfaceFluxes();
#endif
  cudaMemcpy(boundary_flux,_boundary_flux, 2 * _tot_num_tracks *
              _fluxes_per_track * sizeof(float),
              cudaMemcpyHostToDevice);

  getLastCudaError();
  cudaMemcpy(start_flux, _start_flux, 2 * _tot_num_tracks *
                _fluxes_per_track * sizeof(float),
                cudaMemcpyHostToDevice);
  getLastCudaError();
  _timer->stopTimer();
  _timer->recordSplit("Transport Sweep");
}


/**
 * @brief Add the source term contribution in the transport equation to
 *        the FSR scalar flux.
 */
void GPUSolver::addSourceToScalarFlux() {
  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);
  FP_PRECISION* reduced_sources =
       thrust::raw_pointer_cast(&dev_reduced_sources[0]);

  addSourceToScalarFluxOnDevice<<<_B,_T>>>(scalar_flux, reduced_sources,
                                           _FSR_volumes, _FSR_materials,
                                           _materials);
}


/**
 * @brief Compute \f$ k_{eff} \f$ from successive fission sources.
 * @details This method computes the current approximation to the
 *          multiplication factor on this iteration as follows:
 *          \f$ k_{eff} = \frac{\displaystyle\sum_{i \in I}
 *                        \displaystyle\sum_{g \in G} \nu \Sigma^F_g \Phi V_{i}}
 *                        {\displaystyle\sum_{i \in I}
 *                        \displaystyle\sum_{g \in G} (\Sigma^T_g \Phi V_{i} -
 *                        \Sigma^S_g \Phi V_{i} - L_{i,g})} \f$
 */
void GPUSolver::computeKeff() {


  double fission;

  thrust::device_vector<FP_PRECISION> fission_vec;
  fission_vec.resize(_B * _T);

  FP_PRECISION* fiss_ptr = thrust::raw_pointer_cast(&fission_vec[0]);
  FP_PRECISION* flux = thrust::raw_pointer_cast(&dev_scalar_flux[0]);

  /* Compute the total, fission and scattering reaction rates on device.
   * This kernel stores partial rates in a Thrust vector with as many
   * entries as CUDAthreads executed by the kernel */
  computeFSRFissionRatesOnDevice<<<_B, _T>>>(_FSR_volumes, _FSR_materials,
                                             _materials, flux, fiss_ptr, true, true);

  /* Compute the total fission source */
  fission = thrust::reduce(fission_vec.begin(), fission_vec.end());

  /* Get the total number of source regions */
  long total_num_FSRs = _num_FSRs;
  
#ifdef MPIx
  /* Reduce rates across domians */
  if (_geometry->isDomainDecomposed()) {

    /* Get the communicator */
    MPI_Comm comm = _geometry->getMPICart();

    /* Copy local rates */
    FP_PRECISION local_rates ;

      local_rates = fission;
     /* Reduce computed rates */
    MPI_Allreduce(&local_rates, &fission, 1, MPI_DOUBLE, MPI_SUM, comm);

    /* Get total number of FSRs across all domains */
    MPI_Allreduce(&_num_FSRs, &total_num_FSRs, 1, MPI_LONG, MPI_SUM, comm);
  }
#endif
    _k_eff *= fission / total_num_FSRs;
}


/**
 * @brief Computes the residual between source/flux iterations.
 * @param res_type the type of residuals to compute
 *        (SCALAR_FLUX, FISSION_SOURCE, TOTAL_SOURCE)
 * @return the average residual in each flat source region
 */
double GPUSolver::computeResidual(residualType res_type) {

  int norm;
  double residual;
  isinf_test inf_test;
  isnan_test nan_test;

  /* Allocate Thrust vector for residuals in each FSR */
  thrust::device_vector<double> residuals(_num_FSRs);

  if (res_type == SCALAR_FLUX) {

    norm = _num_FSRs;

    /* Allocate Thrust vector for residuals */
    thrust::device_vector<FP_PRECISION> fp_residuals(_num_FSRs * _NUM_GROUPS);
    thrust::device_vector<FP_PRECISION> FSR_fp_residuals(_num_FSRs);

    /* Compute the relative flux change in each FSR and group */
    thrust::transform(dev_scalar_flux.begin(), dev_scalar_flux.end(),
                      dev_old_scalar_flux.begin(), fp_residuals.begin(),
                      thrust::minus<FP_PRECISION>());
    thrust::transform(fp_residuals.begin(), fp_residuals.end(),
                      dev_old_scalar_flux.begin(), fp_residuals.begin(),
                      thrust::divides<FP_PRECISION>());

    /* Replace INF and NaN values (from divide by zero) with 0. */
    thrust::replace_if(fp_residuals.begin(), fp_residuals.end(), inf_test, 0);
    thrust::replace_if(fp_residuals.begin(), fp_residuals.end(), nan_test, 0);

    /* Square the residuals */
    thrust::transform(fp_residuals.begin(), fp_residuals.end(),
                      fp_residuals.begin(), fp_residuals.begin(),
                      thrust::multiplies<FP_PRECISION>());

    typedef thrust::device_vector<FP_PRECISION>::iterator Iterator;

    /* Reduce flux residuals across energy groups within each FSR */
    for (int e=0; e < _NUM_GROUPS; e++) {
      strided_range<Iterator> strider(fp_residuals.begin() + e,
                                      fp_residuals.end(), _NUM_GROUPS);
      thrust::transform(FSR_fp_residuals.begin(), FSR_fp_residuals.end(),
                        strider.begin(), FSR_fp_residuals.begin(),
                        thrust::plus<FP_PRECISION>());
    }

    /* Copy the FP_PRECISION residual to the double precision residual */
    thrust::copy(FSR_fp_residuals.begin(),
                 FSR_fp_residuals.end(), residuals.begin());

    /* Sum up the residuals */
    residual = thrust::reduce(residuals.begin(), residuals.end());

    /* Normalize the residual */
    residual = sqrt(residual / norm);

    return residual;
  }

  else if (res_type == FISSION_SOURCE) {

    if (_num_fissionable_FSRs == 0)
      // log_printf(ERROR, "The Solver is unable to compute a "
      //            "FISSION_SOURCE residual without fissionable FSRs");
      log_printf(NORMAL, "The Solver is unable to compute a "
      "FISSION_SOURCE residual without fissionable FSRs    %ld",_num_fissionable_FSRs);

    norm = _num_fissionable_FSRs;

    /* Allocate Thrust vectors for fission sources in each FSR, group */
    thrust::device_vector<FP_PRECISION> new_fission_sources_vec(_num_FSRs * _NUM_GROUPS);
    thrust::device_vector<FP_PRECISION> old_fission_sources_vec(_num_FSRs * _NUM_GROUPS);

    /* Allocate Thrust vectors for energy-integrated fission sources in each FSR */
    thrust::device_vector<FP_PRECISION> FSR_old_fiss_src(_num_FSRs);
    thrust::device_vector<FP_PRECISION> FSR_new_fiss_src(_num_FSRs);

    /* Cast Thrust vectors as array pointers */
    FP_PRECISION* old_fission_sources =
         thrust::raw_pointer_cast(&old_fission_sources_vec[0]);
    FP_PRECISION* new_fission_sources =
         thrust::raw_pointer_cast(&new_fission_sources_vec[0]);
    FP_PRECISION* scalar_flux =
         thrust::raw_pointer_cast(&dev_scalar_flux[0]);
    FP_PRECISION* old_scalar_flux =
         thrust::raw_pointer_cast(&dev_old_scalar_flux[0]);

    /* Compute the old and new nu-fission sources in each FSR, group */
    computeFSRFissionSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, false,
                                                 old_scalar_flux, old_fission_sources);
    computeFSRFissionSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, false,
                                                 scalar_flux, new_fission_sources);

    typedef thrust::device_vector<FP_PRECISION>::iterator Iterator;

    /* Reduce nu-fission sources across energy groups within each FSR */
    for (int e=0; e < _NUM_GROUPS; e++) {
      strided_range<Iterator> old_strider(old_fission_sources_vec.begin() + e,
                                          old_fission_sources_vec.end(), _NUM_GROUPS);
      strided_range<Iterator> new_strider(new_fission_sources_vec.begin() + e,
                                          new_fission_sources_vec.end(), _NUM_GROUPS);
      thrust::transform(FSR_old_fiss_src.begin(), FSR_old_fiss_src.end(),
                        old_strider.begin(), FSR_old_fiss_src.begin(),
                        thrust::plus<FP_PRECISION>());
      thrust::transform(FSR_new_fiss_src.begin(), FSR_new_fiss_src.end(),
                        new_strider.begin(), FSR_new_fiss_src.begin(),
                        thrust::plus<FP_PRECISION>());
    }

    /* Compute the relative nu-fission source change in each FSR */
    thrust::transform(FSR_new_fiss_src.begin(), FSR_new_fiss_src.end(),
                      FSR_old_fiss_src.begin(), residuals.begin(),
                      thrust::minus<FP_PRECISION>());
    thrust::transform(residuals.begin(), residuals.end(),
                      FSR_old_fiss_src.begin(), residuals.begin(),
                      thrust::divides<FP_PRECISION>());
  }

  else if (res_type == TOTAL_SOURCE) {

    norm = _num_FSRs;

    /* Allocate Thrust vectors for fission/scatter sources in each FSR, group */
    thrust::device_vector<FP_PRECISION> new_sources_vec(_num_FSRs * _NUM_GROUPS);
    thrust::device_vector<FP_PRECISION> old_sources_vec(_num_FSRs * _NUM_GROUPS);
    thrust::fill(new_sources_vec.begin(), new_sources_vec.end(), 0.0);
    thrust::fill(old_sources_vec.begin(), old_sources_vec.end(), 0.0);

    /* Allocate Thrust vectors for energy-integrated fission/scatter sources in each FSR */
    thrust::device_vector<FP_PRECISION> FSR_old_src(_num_FSRs);
    thrust::device_vector<FP_PRECISION> FSR_new_src(_num_FSRs);
    thrust::fill(FSR_old_src.begin(), FSR_old_src.end(), 0.);
    thrust::fill(FSR_new_src.begin(), FSR_new_src.end(), 0.);

    /* Cast Thrust vectors as array pointers */
    FP_PRECISION* old_sources =
         thrust::raw_pointer_cast(&old_sources_vec[0]);
    FP_PRECISION* new_sources =
         thrust::raw_pointer_cast(&new_sources_vec[0]);
    FP_PRECISION* scalar_flux =
         thrust::raw_pointer_cast(&dev_scalar_flux[0]);
    FP_PRECISION* old_scalar_flux =
         thrust::raw_pointer_cast(&dev_old_scalar_flux[0]);

    /* Compute nu-fission source */

    /* Compute the old and new nu-fission sources in each FSR, group */
    computeFSRFissionSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, false,
                                                 old_scalar_flux, old_sources);
    computeFSRFissionSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, false,
                                                 scalar_flux, new_sources);

    typedef thrust::device_vector<FP_PRECISION>::iterator Iterator;

    /* Reduce nu-fission sources across energy groups within each FSR */
    for (int e=0; e < _NUM_GROUPS; e++) {
      strided_range<Iterator> old_strider(old_sources_vec.begin() + e,
                                          old_sources_vec.end(), _NUM_GROUPS);
      strided_range<Iterator> new_strider(new_sources_vec.begin() + e,
                                          new_sources_vec.end(), _NUM_GROUPS);
      thrust::transform(FSR_old_src.begin(), FSR_old_src.end(),
                        old_strider.begin(), FSR_old_src.begin(),
                        thrust::plus<FP_PRECISION>());
      thrust::transform(FSR_new_src.begin(), FSR_new_src.end(),
                        new_strider.begin(), FSR_new_src.begin(),
                        thrust::plus<FP_PRECISION>());
    }

    /* Multiply fission sources by inverse keff */
    thrust::for_each(FSR_new_src.begin(), FSR_new_src.end(),
                     multiplyByConstant<FP_PRECISION>(1. / _k_eff));
    thrust::for_each(FSR_old_src.begin(), FSR_old_src.end(),
                     multiplyByConstant<FP_PRECISION>(1. / _k_eff));

    /* Compute scatter source */

    /* Reset sources Thrust vectors to zero */
    thrust::fill(new_sources_vec.begin(), new_sources_vec.end(), 0.0);
    thrust::fill(old_sources_vec.begin(), old_sources_vec.end(), 0.0);

    /* Compute the old and new scattering sources in each FSR, group */
    computeFSRScatterSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, false,
                                                 old_scalar_flux, old_sources);
    computeFSRScatterSourcesOnDevice<<<_B, _T>>>(_FSR_materials, _materials, false,
                                                 scalar_flux, new_sources);

    /* Reduce scatter sources across energy groups within each FSR */
    for (int e=0; e < _NUM_GROUPS; e++) {
      strided_range<Iterator> old_strider(old_sources_vec.begin() + e,
                                          old_sources_vec.end(), _NUM_GROUPS);
      strided_range<Iterator> new_strider(new_sources_vec.begin() + e,
                                          new_sources_vec.end(), _NUM_GROUPS);
      thrust::transform(FSR_old_src.begin(), FSR_old_src.end(),
                        old_strider.begin(), FSR_old_src.begin(),
                        thrust::plus<FP_PRECISION>());
      thrust::transform(FSR_new_src.begin(), FSR_new_src.end(),
                        new_strider.begin(), FSR_new_src.begin(),
                        thrust::plus<FP_PRECISION>());
    }

    /* Compute the relative total source change in each FSR */
    thrust::transform(FSR_new_src.begin(), FSR_new_src.end(),
                      FSR_old_src.begin(), residuals.begin(),
                      thrust::minus<FP_PRECISION>());
    thrust::transform(residuals.begin(), residuals.end(),
                      FSR_old_src.begin(), residuals.begin(),
                      thrust::divides<FP_PRECISION>());
  }

  /* Replace INF and NaN values (from divide by zero) with 0. */
  thrust::replace_if(residuals.begin(), residuals.end(), inf_test, 0);
  thrust::replace_if(residuals.begin(), residuals.end(), nan_test, 0);

  /* Square the residuals */
  thrust::transform(residuals.begin(), residuals.end(),
                    residuals.begin(), residuals.begin(),
                    thrust::multiplies<double>());

  /* Sum up the residuals */
  residual = thrust::reduce(residuals.begin(), residuals.end());

#ifdef MPIx
  /* Reduce residuals across domains */
  if (_geometry->isDomainDecomposed()) {

    /* Get the communicator */
    MPI_Comm comm = _geometry->getMPICart();

    /* Reduce residuals */
    double reduced_res;
    MPI_Allreduce(&residual, &reduced_res, 1, MPI_DOUBLE, MPI_SUM, comm);
    residual = reduced_res;

    /* Reduce normalization factors */
    long reduced_norm;
    MPI_Allreduce(&norm, &reduced_norm, 1, MPI_LONG, MPI_SUM, comm);
    norm = reduced_norm;
  }
#endif

  if (res_type == FISSION_SOURCE && norm == 0)
      log_printf(ERROR, "The Solver is unable to compute a "
                 "FISSION_SOURCE residual without fissionable FSRs");

  /* Error check residual componenets */
  if (residual < 0.0) {
    log_printf(WARNING, "MOC residual mean square error %6.4f less than zero",
               residual);
    residual = 0.0;
  }
  if (norm <= 0) {
    log_printf(WARNING, "MOC residual norm %d less than one", norm);
    norm = 1;
  }

  /* Normalize the residual */
  residual = sqrt(residual / norm);

  return residual;
}


/**
 * @brief Computes the volume-averaged, energy-integrated nu-fission rate in
 *        each FSR and stores them in an array indexed by FSR ID.
 * @details This is a helper method for SWIG to allow users to retrieve
 *          FSR nu-fission rates as a NumPy array. An example of how this method
 *          can be called from Python is as follows:
 *
 * @code
 *          num_FSRs = geometry.getNumFSRs()
 *          fission_rates = solver.computeFSRFissionRates(num_FSRs)
 * @endcode
 *
 * @param fission_rates an array to store the nu-fission rates (implicitly
 *                      passed in as a NumPy array from Python)
 * @param num_FSRs the number of FSRs passed in from Python
 */
void GPUSolver::computeFSRFissionRates(double* fission_rates, long num_FSRs, bool nu) {

  log_printf(INFO, "Computing FSR fission rates...");

  /* Allocate memory for the FSR nu-fission rates on the device and host */
  FP_PRECISION* dev_fission_rates;
  cudaMalloc(&dev_fission_rates, _num_FSRs * sizeof(FP_PRECISION));
  getLastCudaError();
  FP_PRECISION* host_fission_rates = new FP_PRECISION[_num_FSRs];

  FP_PRECISION* scalar_flux =
       thrust::raw_pointer_cast(&dev_scalar_flux[0]);

  /* Compute the FSR nu-fission rates on the device */
  computeFSRFissionRatesOnDevice<<<_B, _T>>>(_FSR_volumes, _FSR_materials,
                                             _materials, scalar_flux,
                                             dev_fission_rates, nu, false);

  /* Copy the nu-fission rate array from the device to the host */
  cudaMemcpy(host_fission_rates, dev_fission_rates,
             _num_FSRs * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
  getLastCudaError();

#ifdef MPIx
  if (_geometry->isDomainDecomposed()) {

    /* Allocate buffer for communication */
    long num_total_FSRs = _geometry->getNumTotalFSRs();
    double* temp_fission_rates = new double[num_total_FSRs];
    for (int i=0; i < num_total_FSRs; i++)
      temp_fission_rates[i] = 0;

    int rank = 0;
    MPI_Comm comm = _geometry->getMPICart();
    MPI_Comm_rank(comm, &rank);
    for (long r=0; r < num_total_FSRs; r++) {

      /* Determine the domain and local FSR ID */
      long fsr_id = r;
      int domain = 0;
      _geometry->getLocalFSRId(r, fsr_id, domain);

      /* Set data if in the correct domain */
      if (domain == rank)
        temp_fission_rates[r] =host_fission_rates[fsr_id];
    }

    MPI_Allreduce(temp_fission_rates,host_fission_rates, num_total_FSRs,
                  MPI_DOUBLE, MPI_SUM, comm);
    delete [] temp_fission_rates;
  }
#endif
 /* Populate the double precision NumPy array for the output */
  for (int i=0; i < _num_FSRs; i++)
    fission_rates[i] = host_fission_rates[i];

  /* Deallocate the memory assigned to store the fission rates on the device */
  cudaFree(dev_fission_rates);
  getLastCudaError();
  delete [] host_fission_rates;
}

#ifdef MPIx
/**
 * @brief Buffers used to transfer angular flux information are initialized
 * @details Track connection book-keeping information is also saved for
 *          efficiency during angular flux packing.
 */
void GPUSolver::setupMPIBuffers() {

  /* Determine the size of the buffers */
  _track_message_size = _fluxes_per_track + 3;
  int message_length = TRACKS_PER_BUFFER * _track_message_size;

  /* Initialize MPI requests and status */
  if (_geometry->isDomainDecomposed()) {

    if (_send_buffers.size() > 0)
      deleteMPIBuffers();

    log_printf(NORMAL, "Setting up MPI Buffers for angular flux exchange...");

    /* Fill the hash map of send buffers */
    int idx = 0;
    for (int dx=-1; dx <= 1; dx++) {
      for (int dy=-1; dy <= 1; dy++) {
        for (int dz=-1; dz <= 1; dz++) {
          if (abs(dx) + abs(dy) == 1 ||
              (dx == 0 && dy == 0 && dz != 0)) {
            int domain = _geometry->getNeighborDomain(dx, dy, dz);
            if (domain != -1) {
              _neighbor_connections.insert({domain, idx});
              _neighbor_domains.push_back(domain);
              idx++;

              /* Inititalize vector that shows how filled send_buffers are */
              _send_buffers_index.push_back(0);
            }
          }
        }
      }
    }

    /* Estimate and print size of flux transfer buffers */
    int num_domains = _neighbor_domains.size();
    int size = 2 * message_length * num_domains * sizeof(float);
    int max_size;
    MPI_Allreduce(&size, &max_size, 1, MPI_INT, MPI_MAX,
                  _geometry->getMPICart());
    log_printf(INFO_ONCE, "Max track fluxes transfer buffer storage = %.2f MB",
               max_size / 1e6);

    /* Allocate track fluxes transfer buffers */
    _send_buffers.resize(num_domains);
    _receive_buffers.resize(num_domains);
    for (int i=0; i < num_domains; i++) {
#ifdef ONLYVACUUMBC
      /* Increase capacity because buffers will overflow and need a resize */
      _send_buffers.at(i).reserve(3*message_length);
      _receive_buffers.at(i).reserve(3*message_length);
#endif
      _send_buffers.at(i).resize(message_length);
      _receive_buffers.at(i).resize(message_length);
    }

    /* Setup Track communication information for all neighbor domains */
    _boundary_tracks.resize(num_domains);
    for (int i=0; i < num_domains; i++) {

      /* Initialize Track ID's to -1 */
      int start_idx = _fluxes_per_track + 1;
      for (int idx = start_idx; idx < message_length;
           idx += _track_message_size) {
        long* track_info_location =
             reinterpret_cast<long*>(&_send_buffers.at(i)[idx]);
        track_info_location[0] = -1;
        track_info_location =
             reinterpret_cast<long*>(&_receive_buffers.at(i)[idx]);
        track_info_location[0] = -1;
      }
    }

    /* Allocate vector of send/receive buffer sizes */
    _send_size.resize(num_domains, 0);
    _receive_size.resize(num_domains, 0);

    /* Build array of Track connections */
    _track_connections.resize(2);
    _track_connections.at(0).resize(_tot_num_tracks);
    _track_connections.at(1).resize(_tot_num_tracks);

#ifdef ONLYVACUUMBC
    _domain_connections.resize(2);
    _domain_connections.at(0).resize(_tot_num_tracks);
    _domain_connections.at(1).resize(_tot_num_tracks);
#endif

    /* Determine how many Tracks communicate with each neighbor domain */
    log_printf(NORMAL, "Initializing Track connections accross domains...");
    std::vector<long> num_tracks;
    num_tracks. resize(num_domains, 0);

#pragma omp parallel for
    for (long t=0; t<_tot_num_tracks; t++) {

      Track* track;
      /* Get 3D Track data */
      if (_SOLVE_3D) {
        TrackStackIndexes tsi;
        track = new Track3D();
        TrackGenerator3D* track_generator_3D =
          dynamic_cast<TrackGenerator3D*>(_track_generator);
        track_generator_3D->getTSIByIndex(t, &tsi);
        track_generator_3D->getTrackOTF(dynamic_cast<Track3D*>(track), &tsi);
      }
      /* Get 2D Track data */
      else {
        Track** tracks = _track_generator->get2DTracksArray();
        track = tracks[t];
      }

      /* Save the index of the forward and backward connecting Tracks */
      _track_connections.at(0).at(t) = track->getTrackNextFwd();
      _track_connections.at(1).at(t) = track->getTrackNextBwd();

      /* Determine the indexes of connecting domains */
      int domains[2];
      domains[0] = track->getDomainFwd();
      domains[1] = track->getDomainBwd();
      bool interface[2];
      interface[0] = track->getBCFwd() == INTERFACE;
      interface[1] = track->getBCBwd() == INTERFACE;
      for (int d=0; d < 2; d++) {
        if (domains[d] != -1 && interface[d]) {
          int neighbor = _neighbor_connections.at(domains[d]);
#pragma omp atomic update
          num_tracks[neighbor]++;
        }
      }
      if (_SOLVE_3D)
        delete track;

    }

    /* Resize the buffers for the counted number of Tracks */
    for (int i=0; i < num_domains; i++) {
      _boundary_tracks.at(i).resize(num_tracks[i]);
      num_tracks[i] = 0;
    }

    /* Determine which Tracks communicate with each neighbor domain */
#ifndef ONLYVACUUMBC
#pragma omp parallel for
#endif
    for (long t=0; t<_tot_num_tracks; t++) {

      Track* track;
      /* Get 3D Track data */
      if (_SOLVE_3D) {
        TrackStackIndexes tsi;
        track = new Track3D();
        TrackGenerator3D* track_generator_3D =
          dynamic_cast<TrackGenerator3D*>(_track_generator);
        track_generator_3D->getTSIByIndex(t, &tsi);
        track_generator_3D->getTrackOTF(dynamic_cast<Track3D*>(track), &tsi);
      }
      /* Get 2D Track data */
      else {
        Track** tracks = _track_generator->get2DTracksArray();
        track = tracks[t];
      }

      /* Determine the indexes of connecting domains */
      int domains[2];
      domains[0] = track->getDomainFwd();
      domains[1] = track->getDomainBwd();
      bool interface[2];
      interface[0] = track->getBCFwd() == INTERFACE;
      interface[1] = track->getBCBwd() == INTERFACE;
      for (int d=0; d < 2; d++) {
        if (domains[d] != -1 && interface[d]) {
          int neighbor = _neighbor_connections.at(domains[d]);

          long slot;
#pragma omp atomic capture
          {
            slot = num_tracks[neighbor];
            num_tracks[neighbor]++;
          }
          _boundary_tracks.at(neighbor).at(slot) = 2*t + d;
#ifdef ONLYVACUUMBC
          //NOTE _boundary_tracks needs to be ordered if ONLYVACUUMBC is used
          _domain_connections.at(d).at(t) = domains[d];
#endif
        }
#ifdef ONLYVACUUMBC
        /* Keep a list of the tracks that start at vacuum BC */
        else {
#pragma omp critical
          {
            _tracks_from_vacuum.push_back(2*t + 1 - d);
          }
          //NOTE domains[d] can be set for a track ending on an interface
          // with vacuum and another track
          _domain_connections.at(d).at(t) = -1;
        }
#endif
      }
      if (_SOLVE_3D)
        delete track;
    }

    printLoadBalancingReport();
    log_printf(NORMAL, "Finished setting up MPI buffers...");

    /* Setup MPI communication bookkeeping */
    _MPI_requests = new MPI_Request[2*num_domains];
    _MPI_sends = new bool[num_domains];
    _MPI_receives = new bool[num_domains];
    for (int i=0; i < num_domains; i++) {
      _MPI_sends[i] = false;
      _MPI_receives[i] = false;
    }
  }
}
/**
 * @brief Angular flux transfer information is packed into buffers.
 * @details On each domain, angular flux and track connection information
 *          is packed into buffers. Each buffer pertains to a neighboring
 *          domain. This function proceeds packing buffers until for each
 *          neighboring domain either all the tracks have been packed or the
 *          associated buffer is full. This provided integer array contains
 *          the index of the last track handled for each neighboring domain.
 *          These numbers are updated at the end with the last track handled.
 * @arg packing_indexes index of last track sent for each neighbor domain
 */
 void GPUSolver::packBuffers(std::vector<long> &packing_indexes) {

  /* Fill send buffers for every domain */
  int num_domains = packing_indexes.size();
  for (int i=0; i < num_domains; i++) {

    /* Reset send buffers : start at beginning if the buffer has not been
       prefilled, else start after what has been prefilled */
    int start_idx = _send_buffers_index.at(i) * _track_message_size +
                    _fluxes_per_track + 1;
    int max_idx = _track_message_size * TRACKS_PER_BUFFER;
#pragma omp parallel for
    for (int idx = start_idx; idx < max_idx; idx += _track_message_size) {
      long* track_info_location =
        reinterpret_cast<long*>(&_send_buffers.at(i)[idx]);
      track_info_location[0] = -1;
    }

    /* Fill send buffers with Track information :
       -start from last track packed (packing_indexes)
       -take into account the pre-filling
       (pre-filling only if ONLYVACUUMBC flag is used)
       -only fill to max_buffer_fill */
    int max_buffer_idx = _boundary_tracks.at(i).size() -
          packing_indexes.at(i);

    if (_send_buffers_index.at(i) + max_buffer_idx > TRACKS_PER_BUFFER)
      max_buffer_idx = TRACKS_PER_BUFFER - _send_buffers_index.at(i);

    /* Keep track of buffer size to avoid sending more fluxes than needed */
    _send_size.at(i) = std::max(_send_buffers_index.at(i) + max_buffer_idx,
         _send_buffers_index.at(i));

#ifndef ONLYVACUUMBC
#pragma omp parallel for
#endif
    for (int b=0; b < max_buffer_idx; b++) {

      long boundary_track_idx = packing_indexes.at(i) + b;
      long buffer_index = (_send_buffers_index.at(i)+b) * _track_message_size;

#ifdef ONLYVACUUMBC
      /* Exit loop if all fluxes have been sent */
      if (boundary_track_idx >= _boundary_tracks.at(i).size())
        break;
#endif

      /* Get 3D Track data */
      long boundary_track = _boundary_tracks.at(i).at(boundary_track_idx);
      long t = boundary_track / 2;
      int d = boundary_track - 2*t;
      long connect_track = _track_connections.at(d).at(t);

      /* Fill buffer with angular fluxes */
      for (int pe=0; pe < _fluxes_per_track; pe++)
        _send_buffers.at(i)[buffer_index + pe] = _boundary_flux(t,d,pe);

      /* Assign the connecting Track information */
      long idx = buffer_index + _fluxes_per_track;
      _send_buffers.at(i)[idx] = d;
      long* track_info_location =
        reinterpret_cast<long*>(&_send_buffers.at(i)[idx+1]);
      track_info_location[0] = connect_track;

 #ifdef ONLYVACUUMBC
      /* Invalidate track transfer if it has already been sent by prefilling */
      if (_track_flux_sent.at(d).at(t)) {
        track_info_location[0] = long(-2);

        /* Use freed-up spot in send_buffer and keep track of it */
        b--;
        packing_indexes.at(i)++;
      }
#endif
    }

    /* Record the next Track ID, reset index of fluxes in send_buffers */
    packing_indexes.at(i) += std::max(0, max_buffer_idx);
    _send_buffers_index.at(i) = 0;
  }
}


/**
 * @brief Transfers all angular fluxes at interfaces to their appropriate
 *        domain neighbors
 * @details The angular fluxes stored in the _boundary_flux array that
 *          intersect INTERFACE boundaries are transfered to their appropriate
 *          neighbor's _start_flux array at the periodic indexes.
 */
void GPUSolver::transferAllInterfaceFluxes() {

  /* Initialize MPI requests and status */
  MPI_Comm MPI_cart = _geometry->getMPICart();
  MPI_Status stat;

  /* Wait for all MPI Ranks to be done with sweeping */
  _timer->startTimer();
  MPI_Barrier(MPI_cart);
  _timer->stopTimer();
  _timer->recordSplit("Idle time");

  /* Initialize timer for total transfer cost */
  _timer->startTimer();

  /* Get rank of each process */
  int rank;
  MPI_Comm_rank(MPI_cart, &rank);

  /* Create bookkeeping vectors */
  std::vector<long> packing_indexes;

  /* Resize vectors to the number of domains */
  int num_domains = _neighbor_domains.size();
  packing_indexes.resize(num_domains);

  /* Start communication rounds */
  int round_counter = -1;
  while (true) {

    round_counter++;

    /* Pack buffers with angular flux data */
    _timer->startTimer();
    packBuffers(packing_indexes);
    _timer->stopTimer();
    _timer->recordSplit("Packing time");



    /* Set size of received messages, adjust buffer if needed */
    _timer->startTimer();
    for (int i=0; i < num_domains; i++) {

      /* Size of received message, in number of tracks */
      _receive_size.at(i) = TRACKS_PER_BUFFER;
    }

    /* Send and receive from all neighboring domains */
    bool communication_complete = true;


      for (int i=0; i < num_domains; i++) {

        /* Get the communicating neighbor domain */
        int domain = _neighbor_domains.at(i);


        /* Check if a send/receive needs to be created */
        long* first_track_idx =
          reinterpret_cast<long*>(&_send_buffers.at(i)[_fluxes_per_track+1]);
        long first_track = first_track_idx[0];

        if (first_track != -1) {

          /* Send outgoing flux */
          if (_send_size.at(i) > 0 && !_MPI_sends[i]) {
            MPI_Isend(&_send_buffers.at(i)[0], _track_message_size *
                      _send_size.at(i), MPI_FLOAT, domain, 1, MPI_cart,
                      &_MPI_requests[i*2]);
            _MPI_sends[i] = true;
          }
          else
            if (!_MPI_sends[i])
              _MPI_requests[i*2] = MPI_REQUEST_NULL;

          /* Receive incoming flux */
          if (_receive_size.at(i) > 0 && !_MPI_receives[i]) {

            MPI_Irecv(&_receive_buffers.at(i)[0], _track_message_size *
                      _receive_size.at(i), MPI_FLOAT, domain, 1, MPI_cart,
                      &_MPI_requests[i*2+1]);
            _MPI_receives[i] = true;
          }
          else
            if (!_MPI_receives[i])
              _MPI_requests[i*2+1] = MPI_REQUEST_NULL;

          /* Mark communication as ongoing */
          communication_complete = false;
        }
        else {
          if (!_MPI_sends[i])
            _MPI_requests[i*2] = MPI_REQUEST_NULL;
          if (!_MPI_receives[i])
            _MPI_requests[i*2+1] = MPI_REQUEST_NULL;
        }

    }

    /* Check if communication is done */
    if (communication_complete) {
      _timer->stopTimer();
      _timer->recordSplit("Communication time");
      break;
    }

    /* Block for communication round to complete */
    //FIXME Not necessary, buffers could be unpacked while waiting
    MPI_Waitall(2 * num_domains, _MPI_requests, MPI_STATUSES_IGNORE);
    _timer->stopTimer();
    _timer->recordSplit("Communication time");

    /* Reset status for next communication round and copy fluxes */
    _timer->startTimer();
    for (int i=0; i < num_domains; i++) {

      /* Reset send */
      _MPI_sends[i] = false;

      /* Copy angular fluxes if necessary */
      if (_MPI_receives[i]) {

        /* Get the buffer for the connecting domain */
        for (int t=0; t < _receive_size.at(i); t++) {

          /* Get the Track ID */
          float* curr_track_buffer = &_receive_buffers.at(i)[
                                     t*_track_message_size];
          long* track_idx =
            reinterpret_cast<long*>(&curr_track_buffer[_fluxes_per_track+1]);
          long track_id = track_idx[0];

          /* Break out of loop once buffer is finished */
          if (track_id == -1)
            break;

          /* Check if the angular fluxes are active */
          /* -2 : already transfered through pre-filling
           * -1 : padding of buffer */
          if (track_id > -1) {
            int dir = curr_track_buffer[_fluxes_per_track];
            for (int pe=0; pe < _fluxes_per_track; pe++)
              _start_flux(track_id, dir, pe) = curr_track_buffer[pe];
          }
        }
      }

      /* Reset receive */
      _MPI_receives[i] = false;
    }

    _timer->stopTimer();
    _timer->recordSplit("Unpacking time");
  }

  /* Join MPI at the end of communication */
  MPI_Barrier(MPI_cart);
  _timer->stopTimer();
  _timer->recordSplit("Total transfer time");
}

/**
 * @brief The arrays used to store angular flux information are deleted along
 *        with book-keeping information for track connections.
 */
 void GPUSolver::deleteMPIBuffers() {
  for (int i=0; i < _send_buffers.size(); i++) {
    _send_buffers.at(i).clear();
  }
  _send_buffers.clear();

  for (int i=0; i < _receive_buffers.size(); i++) {
    _receive_buffers.at(i).clear();
  }
  _receive_buffers.clear();
  _neighbor_domains.clear();

  for (int i=0; i < _boundary_tracks.size(); i++)
    _boundary_tracks.at(i).clear();
  _boundary_tracks.clear();

  delete [] _MPI_requests;
  delete [] _MPI_sends;
  delete [] _MPI_receives;
}


void GPUSolver::boundaryFluxChecker(){
  log_printf(ERROR, "boundaryFluxChecker() is not yet implemented on GPU.");
  }
/**
 * @brief Prints out tracking information for cycles, traversing domain
 *        interfaces.
 * @details This function prints Track starting and ending points for a cycle
 *          that traverses the entire Geometry.
 * @param track_start The starting Track ID from which the cycle is followed
 * @param domain_start The domain for the starting Track
 * @param length The number of Tracks to follow across the cycle
 */
void GPUSolver::printCycle(long track_start, int domain_start, int length) {

  /* Initialize buffer for MPI communication */
  int message_size = sizeof(sendInfo);

  /* Initialize MPI requests and status */
  MPI_Comm MPI_cart = _geometry->getMPICart();
  int num_ranks;
  MPI_Comm_size(MPI_cart, &num_ranks);
  MPI_Status stat;
  MPI_Request request[num_ranks];

  int rank;
  MPI_Comm_rank(MPI_cart, &rank);

  /* Loop over all tracks and exchange fluxes */
  long curr_track = track_start;
  int curr_rank = domain_start;
  bool fwd = true;
  for (int t=0; t < length; t++) {

    /* Check if this rank is sending the Track */
    if (rank == curr_rank) {

      /* Get 3D Track data */
      TrackStackIndexes tsi;
      Track3D track;
      TrackGenerator3D* track_generator_3D =
        dynamic_cast<TrackGenerator3D*>(_track_generator);
      track_generator_3D->getTSIByIndex(curr_track, &tsi);
      track_generator_3D->getTrackOTF(&track, &tsi);

      /* Get connecting tracks */
      long connect;
      bool connect_fwd;
      Point* start;
      Point* end;
      int next_domain;
      if (fwd) {
        connect = track.getTrackPrdcFwd();
        connect_fwd = track.getNextFwdFwd();
        start = track.getStart();
        end = track.getEnd();
        next_domain = track.getDomainFwd();
      }
      else {
        connect = track.getTrackPrdcBwd();
        connect_fwd = track.getNextBwdFwd();
        start = track.getEnd();
        end = track.getStart();
        next_domain = track.getDomainBwd();
      }

      /* Write information */
      log_printf(NODAL, "Rank %d: Track (%f, %f, %f) -> (%f, %f, %f)", rank,
                 start->getX(), start->getY(), start->getZ(), end->getX(),
                 end->getY(), end->getZ());

      /* Check domain for reflected boundaries */
      if (next_domain == -1) {
        next_domain = curr_rank;
        if (fwd)
          connect = track.getTrackNextFwd();
        else
          connect = track.getTrackNextBwd();
      }

      /* Pack the information */
      sendInfo si;
      si.track_id = connect;
      si.domain = next_domain;
      si.fwd = connect_fwd;

      /* Send the information */
      for (int i=0; i < num_ranks; i++)
        if (i != rank)
          MPI_Isend(&si, message_size, MPI_BYTE, i, 0, MPI_cart, &request[i]);

      /* Copy information */
      curr_rank = next_domain;
      fwd = connect_fwd;
      curr_track = connect;

      /* Wait for sends to complete */
      bool complete = false;
      while (!complete) {
        complete = true;
        for (int i=0; i < num_ranks; i++) {
          if (i != rank) {
            int flag;
            MPI_Test(&request[i], &flag, &stat);
            if (flag == 0)
              complete = false;
          }
        }
      }
    }

    /* Receiving info */
    else {

      /* Create object to receive sent information */
      sendInfo si;

      /* Issue the receive from the current node */
      MPI_Irecv(&si, message_size, MPI_BYTE, curr_rank, 0, MPI_cart,
                &request[0]);

      /* Wait for receive to complete */
      bool complete = false;
      while (!complete) {
        complete = true;
        int flag;
        MPI_Test(&request[0], &flag, &stat);
        if (flag == 0)
          complete = false;
      }

      /* Copy information */
      curr_rank = si.domain;
      fwd = si.fwd;
      curr_track = si.track_id;
    }

    MPI_Barrier(MPI_cart);
  }

  /* Join MPI at the end of communication */
  MPI_Barrier(MPI_cart);
}

/**
 * @brief A function that prints the repartition of integrations and tracks
 *        among domains and interfaces.
 */
void GPUSolver::printLoadBalancingReport() {

  /* Give a measure of the load imbalance for the sweep step (segments) */
  int num_ranks = 1;
  long num_segments = _track_generator->getNumSegments();
  long min_segments = num_segments, max_segments = num_segments,
       total_segments = num_segments;
  if (_geometry->isDomainDecomposed()) {
    MPI_Comm_size(_geometry->getMPICart(), &num_ranks);
    MPI_Reduce(&num_segments, &min_segments, 1, MPI_LONG, MPI_MIN, 0,
               _geometry->getMPICart());
    MPI_Reduce(&num_segments, &max_segments, 1, MPI_LONG, MPI_MAX, 0,
               _geometry->getMPICart());
    MPI_Reduce(&num_segments, &total_segments, 1, MPI_LONG, MPI_SUM, 0,
               _geometry->getMPICart());
  }
  FP_PRECISION mean_segments = float(total_segments) / num_ranks;
  log_printf(INFO_ONCE, "Min / max / mean number of segments in domains: "
             "%.1e / %.1e / %.1e", float(min_segments), float(max_segments),
             mean_segments);

  /* Give a measure of load imbalance for the communication phase */
  FP_PRECISION tracks_x = 0, tracks_y = 0, tracks_z = 0;
  int domain = _geometry->getNeighborDomain(0, 0, 1);
  if (domain != -1)
    tracks_z = _boundary_tracks.at(_neighbor_connections.at(domain)).size();

  domain = _geometry->getNeighborDomain(0, 1, 0);
  if (domain != -1)
    tracks_y = _boundary_tracks.at(_neighbor_connections.at(domain)).size();

  domain = _geometry->getNeighborDomain(1, 0, 0);
  if (domain != -1)
    tracks_x = _boundary_tracks.at(_neighbor_connections.at(domain)).size();

  long sum_border_tracks_200 = std::max(FP_PRECISION(1),
                                        tracks_x + tracks_y + tracks_z) / 100.;
  log_printf(INFO_ONCE, "Percentage of tracks exchanged in X/Y/Z direction: "
             "%.2f / %.2f / %.2f %", tracks_x / sum_border_tracks_200, tracks_y
             / sum_border_tracks_200, tracks_z / sum_border_tracks_200);
}

#endif
/**
 * @brief Initializes most components of Solver. Mostly needed from the
 *        Python side.
 */
void GPUSolver::initializeSolver(solverMode solver_mode) {

  Solver::initializeSolver(solver_mode);
  if(_segment_formation!=EXPLICIT_3D)
     clonedatatoGPU();
}
//我自己加的函数，用来还原OTF
void GPUSolver::clonedatatoGPU()
{
  log_printf(NORMAL, "Clone data to GPU...");
  _timer->startTimer();
 // initializeSolver(_solver_mode);
  TrackGenerator3D* track_generator_3D =
        dynamic_cast<TrackGenerator3D*>(_track_generator);
  //拷贝num_x与num_y到GPU中
  h_num_x = track_generator_3D->getaddrNumX();
  h_num_y = track_generator_3D->getaddrNumY();
  cudaMalloc(&d_num_x,sizeof(int)*_num_azim/2);
  cudaMalloc(&d_num_y,sizeof(int)*_num_azim/2);
  clone_storage += sizeof(int)*_num_azim;
  cudaMemcpy(d_num_x,h_num_x,sizeof(int)*_num_azim/2,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_num_y,h_num_y,sizeof(int)*_num_azim/2,
             cudaMemcpyHostToDevice);
  clone_storage += 2*sizeof(int)*_num_azim/2;
 
  //拷贝_tracks_per_stack  _first_lz_of_stack   _cum_tracks_per_stack
  int*** _tracks_per_stack = track_generator_3D->getTracksPerStack();
  int*** _first_lz_of_stack = track_generator_3D->getFirstLzofStack();
  long*** _cum_tracks_per_stack = track_generator_3D->getcumtracksperstack();
  //get vector length
  int sotrage_tracks_per_stack=0;

  int* h_azim_size=new int[_num_azim/2];
  for (int a=0; a < _num_azim/2; a++) {
    h_azim_size[a]=sotrage_tracks_per_stack;
    // std::cout<<h_azim_size[a]<<"  ";
    for (int i=0; i < h_num_x[a] + h_num_y[a]; i++) {
      for (int p=0; p < _num_polar; p++) {
        sotrage_tracks_per_stack++;
      }
    }
  }
  cudaMalloc(&d_azim_size, sizeof(int)*_num_azim/2);
  clone_storage += sizeof(int)*_num_azim/2;
  cudaMemcpy(d_azim_size, h_azim_size, sizeof(int)*_num_polar, cudaMemcpyHostToDevice);
  _num_2D_tracks = track_generator_3D->getNum2DTracks();
  
  //malloc memory
  cudaMalloc(&d_cum_tracks_per_stack, sizeof(long)*sotrage_tracks_per_stack);
  cudaMalloc(&d_tracks_per_stack, sizeof(int)*sotrage_tracks_per_stack);
  cudaMalloc(&d_first_lz_of_stack, sizeof(int)*sotrage_tracks_per_stack);
  clone_storage += sizeof(int)*3*sotrage_tracks_per_stack;
  //copy to device
  int shift_ =0;
  for (int a=0; a < _num_azim/2; a++) {
    for (int i=0; i < h_num_x[a] + h_num_y[a]; i++) {
      cudaMemcpy(d_cum_tracks_per_stack+shift_, _cum_tracks_per_stack[a][i], sizeof(long)*_num_polar, cudaMemcpyHostToDevice);        
      cudaMemcpy(d_tracks_per_stack+shift_, _tracks_per_stack[a][i], sizeof(int)*_num_polar, cudaMemcpyHostToDevice);
      cudaMemcpy(d_first_lz_of_stack+shift_, _first_lz_of_stack[a][i], sizeof(int)*_num_polar, cudaMemcpyHostToDevice);
      shift_+=_num_polar;
    }
  }

  

  // 拷贝3D特征线数量到GPU
  _num_3D_tracks = track_generator_3D->getNum3DTracks();
  cudaMemcpyToSymbol(num_3D_tracks, &_num_3D_tracks,
                     sizeof(int), 0, cudaMemcpyHostToDevice);
  // clone_storage += sizeof(int);

  // 拷贝按uid排序的2D特征线
  Geometry* geometry = track_generator_3D->getGeometry();


  //将3D特征性的计算特征传入gpu，以方便一根根计算，提升计算效率
  h_track_3D_index = new track_index[_num_3D_tracks];
  int _azim;
  int _xy;
  int track_3D_index_num=0;
  for (int i = 0; i < _tot_num_track_2d; i++)
  {
    _azim = h_tracks_2D[i]._azim_index;
    _xy = h_tracks_2D[i]._xy_index;
    for (int p = 0; p < _num_polar; p++)
    {
      for(int z = 0; z < _tracks_per_stack[_azim][_xy][p]; z++)
      {
        h_track_3D_index[track_3D_index_num].track_index = i;
        h_track_3D_index[track_3D_index_num].polar = p;
        h_track_3D_index[track_3D_index_num].z = z;
        track_3D_index_num++;
      }
    }   
  }
  cudaMalloc(&d_track_3D_index, sizeof(track_index)*_num_3D_tracks);
  cudaMemcpy(d_track_3D_index, h_track_3D_index, sizeof(track_index)*_num_3D_tracks,cudaMemcpyHostToDevice);
  clone_storage +=  sizeof(track_index)*_num_3D_tracks;
  //delete [] h_tracks_2D;//释放类变量空间

  //拷贝按坐标排序的2D特征线
  Track** _tracks_2D = track_generator_3D->get2DTracks();
  cudaMalloc(&d_tracks_2D_xy, sizeof(int*)*_num_azim/2);
  clone_storage += sizeof(int*)*_num_azim/2;
  int** h_tracks_2D_xy;
  h_tracks_2D_xy = new int*[_num_azim/2];
  for (int j = 0; j < _num_azim/2; j++)
  {
    cudaMalloc(&h_tracks_2D_xy[j], sizeof(int)*(h_num_x[j] + h_num_y[j]));
    clone_storage += sizeof(int)*(h_num_x[j] + h_num_y[j]);
    int* temp;
    temp = new int[h_num_x[j] + h_num_y[j]];
    for (int a = 0; a < h_num_x[j] + h_num_y[j]; a++)
    {
        temp[a] = _tracks_2D[j][a].getUid();
    }
    cudaMemcpy(h_tracks_2D_xy[j], temp, sizeof(int)*(h_num_x[j] + h_num_y[j]),cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_tracks_2D_xy, h_tracks_2D_xy, sizeof(int*)*_num_azim/2,cudaMemcpyHostToDevice);
 // delete [] h_tracks_2D_xy;
 
  clone_storage += sizeof(int);

  getLastCudaError();

  Track** tracks_2D1 = track_generator_3D->getTrack2DChainon1D();
  //拷贝按linkindex排序的2D特征线
  dev_track2D* h_tracks_2D_ = new dev_track2D[_tot_num_track_2d];
   
  cpytrack(tracks_2D1, h_tracks_2D_, false);
  cudaMalloc(&d_track_2D_chain, sizeof(dev_track2D)*_tot_num_track_2d);
  clone_storage += sizeof(dev_track2D)*_tot_num_track_2d;
  cudaMemcpy(d_track_2D_chain, h_tracks_2D_, sizeof(dev_track2D)*_tot_num_track_2d, 
            cudaMemcpyHostToDevice);//不拷贝segment
 // delete [] tracks_2D1;
  //delete [] h_tracks_2D_;
    
  

  int* h_track_chain_link = track_generator_3D->getTrack2DChain();
  int sum = 0;
  for (int i = 0; i < _num_azim/2; i++)
  {
      sum = sum+h_num_x[i];
  }
  
  cudaMalloc(&d_track_2D_chain_link, sizeof(int)*sum);
  clone_storage += sizeof(int)*sum;
  cudaMemcpy(d_track_2D_chain_link, h_track_chain_link, sizeof(int)*sum, 
             cudaMemcpyHostToDevice);
  
  //delete [] h_track_chain_link;
  //delete [] h_tracks_2D;
  //拷贝每个轴的轨道数以及每个轴的轴间距
  int** _num_z = track_generator_3D->getnumzarray();
  int** _num_l = track_generator_3D->getnumlarray();
  double** _dz_eff = track_generator_3D->getdzeff();
  double** _dl_eff = track_generator_3D->getdleff();
  cudaMalloc(&d_num_z, sizeof(int*)*_num_azim/2);
  cudaMalloc(&d_num_l, sizeof(int*)*_num_azim/2);
  cudaMalloc(&d_dz_eff, sizeof(double*)*_num_azim/2);
  cudaMalloc(&d_dl_eff, sizeof(double*)*_num_azim/2);
  clone_storage += 2*sizeof(int*)*_num_azim/2;
  clone_storage += 2*sizeof(double*)*_num_azim/2;
  
  int** h_num_z;
  int** h_num_l;
  double** h_dz_eff; 
  double** h_dl_eff;
  h_dz_eff = new double*[_num_azim/2];
  h_dl_eff = new double*[_num_azim/2];
  h_num_z  = new int*[_num_azim/2];
  h_num_l  = new int*[_num_azim/2];
  for(int i = 0; i < _num_azim/2; i++){
    cudaMalloc(&h_num_z[i], sizeof(int)*_num_polar);
    cudaMalloc(&h_num_l[i], sizeof(int)*_num_polar);
    cudaMalloc(&h_dz_eff[i], sizeof(double)*_num_polar);
    cudaMalloc(&h_dl_eff[i], sizeof(double)*_num_polar);
    clone_storage += 2*sizeof(int)*_num_polar;
    clone_storage += 2*sizeof(double)*_num_polar;
    cudaMemcpy(h_num_z[i], _num_z[i], sizeof(int)*_num_polar, cudaMemcpyHostToDevice);
    cudaMemcpy(h_num_l[i], _num_l[i], sizeof(int)*_num_polar, cudaMemcpyHostToDevice);
    cudaMemcpy(h_dz_eff[i], _dz_eff[i], sizeof(double)*_num_polar, cudaMemcpyHostToDevice);
    cudaMemcpy(h_dl_eff[i], _dl_eff[i], sizeof(double)*_num_polar, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_num_z, h_num_z, sizeof(int*)*_num_azim/2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_num_l, h_num_l, sizeof(int*)*_num_azim/2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dz_eff, h_dz_eff, sizeof(double*)*_num_azim/2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dl_eff, h_dl_eff, sizeof(double*)*_num_azim/2, cudaMemcpyHostToDevice);

  //拷贝几何边界的值
  double _x_max = track_generator_3D->getxmax();
  double _x_min = track_generator_3D->getxmin();
  double _y_max = track_generator_3D->getymax();
  double _y_min = track_generator_3D->getymin();
  double _z_max = track_generator_3D->getzmax();
  double _z_min = track_generator_3D->getzmin();
  cudaMemcpyToSymbol(d_x_max, &_x_max,
                     sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_x_min, &_x_min,
                     sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_y_max, &_y_max,
                     sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_y_min, &_y_min,
                     sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_z_max, &_z_max,
                     sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_z_min, &_z_min,
                     sizeof(double), 0, cudaMemcpyHostToDevice);     
  // clone_storage += 6*sizeof(double);  
  // clone_storage += sizeof(int);           
                     
  //拷贝正交的一组极角
  track_generator_3D->getQuadrature()->getthetas();
  h_theta = track_generator_3D->getQuadrature()->getarraysize();
  cudaMalloc(&d_theta, sizeof(double)*_num_polar*_num_azim/2);
  clone_storage += sizeof(double)*_num_polar*_num_azim/2;
  cudaMemcpy(d_theta, h_theta, sizeof(double)*_num_polar*_num_azim/2,cudaMemcpyHostToDevice);
  getLastCudaError();
  //拷贝边界类型
  boundaryType x_max = track_generator_3D->getGeometry()->getMaxXBoundaryType();
  boundaryType x_min = track_generator_3D->getGeometry()->getMinXBoundaryType();
  boundaryType y_max = track_generator_3D->getGeometry()->getMaxYBoundaryType();
  boundaryType y_min = track_generator_3D->getGeometry()->getMinYBoundaryType();
  boundaryType z_max = track_generator_3D->getGeometry()->getMaxZBoundaryType();
  boundaryType z_min = track_generator_3D->getGeometry()->getMinZBoundaryType();
  cudaMemcpyToSymbol(d_MaxXBoundaryType, &x_max,
                     sizeof(boundaryType), 0, cudaMemcpyHostToDevice);
  getLastCudaError();
  cudaMemcpyToSymbol(d_MinXBoundaryType, &x_min,  
                     sizeof(boundaryType), 0, cudaMemcpyHostToDevice);
  getLastCudaError();
  cudaMemcpyToSymbol(d_MaxYBoundaryType, &y_max,
                     sizeof(boundaryType), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_MinYBoundaryType, &y_min,
                     sizeof(boundaryType), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_MaxZBoundaryType, &z_max,
                     sizeof(boundaryType), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_MinZBoundaryType, &z_min,
                     sizeof(boundaryType), 0, cudaMemcpyHostToDevice);  
  getLastCudaError();
  //拷贝geometry中需要使用的数据到gpu中

  int _extruded_FSR_lookup_long = geometry->getextrudedFSRlookupsize();
  dev_ExtrudedFSR* h_extruded_FSR_lookup = new dev_ExtrudedFSR[_extruded_FSR_lookup_long];
  //用于传输区域几何特征
    int domians_index[3];
    geometry->getDomainIndexes(domians_index);
    _domain_index_x = domians_index[0];
    _domain_index_y = domians_index[1];
    _domain_index_z = domians_index[2];
    // geometry->getDomainStructure(_num_domains);
      

  for (int i = 0; i < _extruded_FSR_lookup_long; i++){
    ExtrudedFSR* temp = geometry->getExtrudedFSR(i);
    h_extruded_FSR_lookup[i]._fsr_id = temp->_fsr_id;
    h_extruded_FSR_lookup[i]._num_fsrs = temp->_num_fsrs;
    double* d_mesh;
    double* h_mesh;
    long* d_fsr_ids;
    long* h_fsr_ids;
    int* h_material_index; 
    int* d_material_index; 

    //获得数组长度
    int num_regions = temp->_num_fsrs;

    //开辟空间
    h_fsr_ids = new long[num_regions];
    h_mesh = new double[num_regions+1];
    h_material_index = new int[num_regions];

    //赋值
    for (int s=0; s < num_regions; s++) {
      h_fsr_ids[s] = temp->_fsr_ids[s];
      h_mesh[s] = temp->_mesh[s];
      
      if(temp->_materials[s]->getId()==1)
        h_material_index[s]=0;
        if(temp->_materials[s]->getId()==2)
        h_material_index[s]=1;
        if(temp->_materials[s]->getId()==3)
        h_material_index[s]=2;
        if(temp->_materials[s]->getId()==4)
        h_material_index[s]=3;
        if(temp->_materials[s]->getId()==5)
        h_material_index[s]=4;
        if(temp->_materials[s]->getId()==6)
        h_material_index[s]=5;
        if(temp->_materials[s]->getId()==1000000)
        h_material_index[s]=6;
    }
    h_mesh[num_regions] = temp->_mesh[num_regions];

    //传参
    cudaMalloc(&d_fsr_ids, sizeof(long)*num_regions);
    cudaMalloc(&d_mesh, sizeof(double)*(num_regions+1));
    cudaMalloc(&d_material_index, sizeof(int)*num_regions);
    clone_storage += sizeof(long)*num_regions;
    clone_storage += sizeof(int)*num_regions;
    clone_storage += sizeof(double)*(num_regions+1);
    cudaMemcpy(d_fsr_ids, h_fsr_ids, sizeof(long)*num_regions,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh, h_mesh, sizeof(double)*(num_regions+1),cudaMemcpyHostToDevice);
    cudaMemcpy(d_material_index, h_material_index, sizeof(int)*num_regions,cudaMemcpyHostToDevice);
    h_extruded_FSR_lookup[i]._fsr_ids = d_fsr_ids;
    h_extruded_FSR_lookup[i]._mesh = d_mesh;
    h_extruded_FSR_lookup[i]._material_index = d_material_index;
    delete [] h_mesh;
    delete [] h_fsr_ids;
    delete [] h_material_index;
  }
  getLastCudaError();
  //将赋值好的ExtrudedFSR传入gpu
  cudaMalloc(&d_extruded_FSR_lookup, sizeof(dev_ExtrudedFSR)*_extruded_FSR_lookup_long);
  clone_storage += sizeof(dev_ExtrudedFSR)*_extruded_FSR_lookup_long;
  cudaMemcpy(d_extruded_FSR_lookup, h_extruded_FSR_lookup, sizeof(dev_ExtrudedFSR)*_extruded_FSR_lookup_long,cudaMemcpyHostToDevice);
  delete [] h_extruded_FSR_lookup;

  //拷贝材料属性到GPU中
  

  //拷贝_max_tau
  double h_max_tau = track_generator_3D->retrieveMaxOpticalLength();
  cudaMemcpyToSymbol(_max_tau, &h_max_tau,
                     sizeof(double), 0, cudaMemcpyHostToDevice);

                     getLastCudaError();

  //为了负载均衡预先通过gpu加载段数量，然后对段数量大小进行排序，之后依次放入gpu计算
  log_printf(NORMAL, "Cal segments numbers...");
  calsegmentsnum<<<512, 512>>>(dev_tracks_2d, d_num_x, d_num_y, d_first_lz_of_stack, 
                         d_tracks_per_stack, d_tracks_2D_xy, d_track_2D_chain, 
                         d_track_2D_chain_link, d_num_z, d_num_l, d_dz_eff, 
                         d_dl_eff, d_theta, d_cum_tracks_per_stack, d_extruded_FSR_lookup,
                        _materials,d_track_3D_index,d_azim_size); 
  cudaDeviceSynchronize();
    
  getLastCudaError();
  cudaMemcpy(h_track_3D_index, d_track_3D_index, sizeof(track_index)*_num_3D_tracks,
                          cudaMemcpyDeviceToHost);
  log_printf(NORMAL, "OTF clone 3D track need data storage per domain: %.2f MB", clone_storage/float(1e6));
  if(_configure->if_sort){
    sort(0, _num_3D_tracks-1, h_track_3D_index);
    cudaMemcpy(d_track_3D_index, h_track_3D_index, sizeof(track_index)*_num_3D_tracks,
             cudaMemcpyHostToDevice);
  }
  //进行判断是否使用预先加载3D特征线的方式
  if(_configure->if_pre_track){
    _timer->startTimer();
    cudaMalloc(&d_track_3D, _num_3D_tracks * sizeof(dev_track));
    clone_storage += _num_3D_tracks * sizeof(dev_track);
    // float storage = clone_storage/float(1e6);
    cudaMalloc(&d_start, _num_3D_tracks * sizeof(dev_Point));
    for(;pre_segments_index < _num_3D_tracks; pre_segments_index++){
      if(clone_storage/float(1e6) > 6144.00)
      {
        break;
      }
      clone_storage += h_track_3D_index[pre_segments_index].num_segments*sizeof(dev_segment);
    }
    log_printf(NORMAL, "OTF clone 3D track need data storage per domain: %.2f MB", clone_storage/float(1e6));
    h_track_3D = new dev_track[_num_3D_tracks];
    for(int i=0; i<pre_segments_index; i++){
      // dev_segment* d_segments;
      // cudaMalloc(&d_segments, sizeof(dev_segment)*h_track_3D_index[i].num_segments);
      cudaMalloc(&h_track_3D[i]._segments, sizeof(dev_segment)*h_track_3D_index[i].num_segments);
      // h_track_3D[i]._segments = d_segments;
    }
    cudaMemcpy(d_track_3D, h_track_3D, sizeof(dev_track)*_num_3D_tracks, cudaMemcpyHostToDevice);
    getLastCudaError();

    generate3DtracksonGPU<<<256, 256>>>(dev_tracks_2d, d_num_x, d_num_y, d_first_lz_of_stack, 
                         d_tracks_per_stack, d_tracks_2D_xy, d_track_2D_chain, 
                         d_track_2D_chain_link, d_num_z, d_num_l, d_dz_eff, 
                         d_dl_eff, d_theta, d_cum_tracks_per_stack, d_extruded_FSR_lookup,
                         _materials,d_track_3D_index,d_track_3D, d_start, pre_segments_index,d_azim_size);
    cudaDeviceSynchronize();
    getLastCudaError();
   _timer->stopTimer();
   _timer->recordSplit("pre track Time");
  }
  else{
    for(int i = 0; i < _num_3D_tracks; i++){
    clone_storage += h_track_3D_index[i].num_segments*sizeof(dev_segment);
  }
   log_printf(NORMAL, "OTF clone 3D track need data storage per domain: %.2f MB", clone_storage/float(1e6));
  }
  _timer->stopTimer();
  _timer->recordSplit("Track Generation Time");
}

void GPUSolver::cpytrack(Track** tracks_2D2, dev_track2D* h_tracks_2D, bool signal){
    for (int i = 0; i < _tot_num_track_2d; i++)
    {
        Track* flattern_track = tracks_2D2[i];
        Point* start =  flattern_track->getStart();
        Point* end = flattern_track->getEnd();
        h_tracks_2D[i]._uid =  flattern_track->getUid();
        h_tracks_2D[i]._start._xyz[0] = start->getX();
        h_tracks_2D[i]._start._xyz[1] = start->getY();
        h_tracks_2D[i]._start._xyz[2] = start->getZ();
        h_tracks_2D[i]._end._xyz[0] = end->getX();
        h_tracks_2D[i]._end._xyz[1] = end->getY();
        h_tracks_2D[i]._end._xyz[2] = end->getZ();
        h_tracks_2D[i]._phi = flattern_track->getPhi();
        h_tracks_2D[i]._num_segments = flattern_track->getNumSegments();
        h_tracks_2D[i]._bc_fwd = flattern_track->getBCFwd();
        h_tracks_2D[i]._bc_bwd = flattern_track->getBCBwd();
        h_tracks_2D[i]._azim_index = flattern_track->getAzimIndex();
        h_tracks_2D[i]._xy_index = flattern_track->getXYIndex();
        h_tracks_2D[i]._link_index = flattern_track->getLinkIndex();
        h_tracks_2D[i]._track_next_fwd = flattern_track->getTrackNextFwd();
        h_tracks_2D[i]._track_next_bwd = flattern_track->getTrackNextBwd();
        h_tracks_2D[i]._track_prdc_fwd = flattern_track->getTrackPrdcFwd();
        h_tracks_2D[i]._track_prdc_bwd = flattern_track->getTrackPrdcBwd();
        h_tracks_2D[i]._track_refl_fwd = flattern_track->getTrackReflFwd();
        h_tracks_2D[i]._track_refl_bwd = flattern_track->getTrackReflBwd();
        h_tracks_2D[i]._next_fwd_fwd = flattern_track->getNextFwdFwd();
        h_tracks_2D[i]._next_bwd_fwd = flattern_track->getNextBwdFwd();
        h_tracks_2D[i]._surface_in = flattern_track->getSurfaceIn();
        h_tracks_2D[i]._surface_out = flattern_track->getSurfaceOut();
        h_tracks_2D[i]._domain_fwd = flattern_track->getDomainFwd();
        h_tracks_2D[i]._domain_bwd = flattern_track->getDomainBwd();
        h_tracks_2D[i]._segment._length = flattern_track->getLength(); 
    }
    //将按uid排序的2D特征线的段拷贝到gpu中
    device_segment* dev_segments;
    device_segment* host_segments;
    if(signal){
      for (int i = 0; i < _tot_num_track_2d; i++)
      {
        Track* temp_track = tracks_2D2[i];
        host_segments = new device_segment[temp_track->getNumSegments()];
        for (int j = 0; j < temp_track->getNumSegments(); j++)
        {
          segment* curr = temp_track->getSegment(j);
          host_segments[j]._length = curr->_length;
          host_segments[j]._region_id = curr->_region_id;
          host_segments[j]._track_idx = curr->_track_idx;
          host_segments[j]._cmfd_surface_fwd = curr->_cmfd_surface_fwd;
          host_segments[j]._cmfd_surface_bwd = curr->_cmfd_surface_bwd;
          host_segments[j]._starting_position[0] = curr->_starting_position[0];
          host_segments[j]._starting_position[1] = curr->_starting_position[1];
          host_segments[j]._starting_position[2] = curr->_starting_position[2];
        }
        cudaMalloc(&dev_segments, sizeof(device_segment)*temp_track->getNumSegments());
        cudaMemcpy(dev_segments, host_segments, sizeof(device_segment)*temp_track->getNumSegments(), cudaMemcpyHostToDevice);
        h_tracks_2D[i].segments_2D = dev_segments;
        delete host_segments;
      }
    }
    
}

void GPUSolver::traversedouble2Dto1D(double** array, double* d_array){
    int j = 0;
    for (int i = 0; i < _num_azim/2; i++)
    {
        for (int k = 0; k < _num_polar; k++)
        {
            d_array[j] = array[i][k];
            j++;
        }
        
    }
    
}

void GPUSolver::traverseint2Dto1D(int** array, int* d_array){
    int j = 0;
    for (int i = 0; i < _num_azim/2; i++)
    {
        for (int k = 0; k < _num_polar; k++)
        {
            d_array[j] = array[i][k];
            j++;
        }
        
    }
}

 
//用于rocprof测试执行时间的函数
void GPUSolver::testtime(){
  test<<<64,64>>>(d_track_3D_index);
  cudaDeviceSynchronize();
}

//用于输出区域几何特征
void GPUSolver::printgeometry(){
  _domains_index = _domain_index_z*_num_domains[1]*_num_domains[0]+_domain_index_y*_num_domains[0]+_domain_index_x;
  printf("%d %ld ", _domains_index, sum_segments);
  if(_domain_index_z>0)
    printf("%d ", (_domain_index_z-1)*_num_domains[1]*_num_domains[0]+_domain_index_y*_num_domains[0]+_domain_index_x+1);
  if((_domain_index_z+1)<_num_domains[2])
    printf("%d ", (_domain_index_z+1)*_num_domains[1]*_num_domains[0]+_domain_index_y*_num_domains[0]+_domain_index_x+1);
  if(_domain_index_y>0)
    printf("%d ", _domain_index_z*_num_domains[1]*_num_domains[0]+(_domain_index_y-1)*_num_domains[0]+_domain_index_x+1);
  if((_domain_index_y+1)<_num_domains[1])
    printf("%d ", _domain_index_z*_num_domains[1]*_num_domains[0]+(_domain_index_y+1)*_num_domains[0]+_domain_index_x+1);
  if(_domain_index_x>0)
    printf("%d ", _domain_index_z*_num_domains[1]*_num_domains[0]+_domain_index_y*_num_domains[0]+(_domain_index_x-1)+1);
  if((_domain_index_x+1)<_num_domains[0])
    printf("%d ", _domain_index_z*_num_domains[1]*_num_domains[0]+_domain_index_y*_num_domains[0]+(_domain_index_x+1)+1);
  printf("\n");
}

//计算特征线数量
void GPUSolver::sumsegments(){
  sum_segments=0;
  for(int i=0; i < _num_3D_tracks; i++){
    sum_segments +=  h_track_3D_index[i].num_segments;
  }
  int rank = 0;
  if (_geometry->isDomainDecomposed()) 
  {
    MPI_Comm_rank(_geometry->getMPICart(), &rank);
    if(rank==0)
    {
      std::cout<<"segment count : "<<_domain_index_x<<","<<_domain_index_y<<","<<_domain_index_z<<" :  "<<sum_segments<<std::endl;
      std::cout<<"total segment count: "<<sum_segments<<"memory "<<sum_segments*sizeof(dev_segment)/(1e6)<<"MB"<<std::endl;
      std::cout<<"total track3d "<<_num_3D_tracks<<"memory "<<_num_3D_tracks*sizeof(dev_track3D)/(1e6)<<"MB"<<std::endl;
      std::cout<<"total track2d "<<_tot_num_track_2d <<"memory "<<_tot_num_track_2d*sizeof(dev_track2D)/(1e6)<<"MB"<<std::endl;
    }
  }
}

void GPUSolver::sort(int left, int right, track_index* h_track_3D_index)
{
	if (left >= right) 
		return ;

	int i = left;
	int j = right;
  track_index key = h_track_3D_index[left];

	while (i < j) 
	{
		while (i < j && key.num_segments >= h_track_3D_index[j].num_segments)
 		{  
			j--;
		} 
    h_track_3D_index[i] = h_track_3D_index[j];

		while (i < j && key.num_segments <= h_track_3D_index[j].num_segments) 
		{  
			i++;
		}
    h_track_3D_index[j] = h_track_3D_index[i];
	}

  h_track_3D_index[i] = key;

	sort(left, i - 1, h_track_3D_index);
	sort(i + 1, right, h_track_3D_index);
}

