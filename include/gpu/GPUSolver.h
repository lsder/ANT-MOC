/**
 * @file GPUSolver.h
 * @brief The GPUSolver class and CUDA physics kernels.
 * @date August 5, 2012
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */

#ifndef GPUSOLVER_H_
#define GPUSOLVER_H_

#ifdef __cplusplus
#ifdef SWIG
#include "Python.h"
#endif
#include "Solver.h"
#endif

#define PySys_WriteStdout printf

#include <thrust/copy.h>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include "clone.h"
#include "dev_exponential.h"
#include "GPUQuery.h"
#include <unordered_map>
#include "OTF.h"
#include "DeviceTrack.h"

/** If number of groups is known at compile time */
#ifdef NGROUPS
#define NUM_GROUPS (NGROUPS)
#define _NUM_GROUPS (NGROUPS)
#else
#define _NUM_GROUPS (_num_groups)
#endif

/** Indexing macro for the scalar flux in each FSR and energy group */
#define scalar_flux(tid,e) (scalar_flux[(tid)*NUM_GROUPS + (e)])

/** Indexing macro for the old scalar flux in each FSR and energy group */
#define old_scalar_flux(tid,e) (old_scalar_flux[(tid)*NUM_GROUPS + (e)])

/** Indexing macro for the total source divided by the total cross-section,
 *  \f$ \frac{Q}{\Sigma_t} \f$, in each FSR and energy group */
#define reduced_sources(tid,e) (reduced_sources[(tid)*NUM_GROUPS + (e)])

/** Indexing scheme for fixed sources for each FSR and energy group */
#define fixed_sources(r,e) (fixed_sources[(r)*NUM_GROUPS + (e)])

/** Indexing macro for the azimuthal and polar weights */
#define weights(i,p) (weights[(i)*num_polar + (p)])

#define d_index(a,i,p) (d_azim_size[a]+i*d_num_polar+p);

/** Indexing macro for the angular fluxes for each polar angle and energy
 *  group for a given Track */
#define boundary_flux(t,pe2) (boundary_flux[2*(t)*polar_times_groups+(pe2)])

/** Indexing macro for the starting angular fluxes for each polar angle and
 *  energy group for a given Track. These are copied to the boundary fluxes
 *  array at the beginning of each transport sweep */
#define start_flux(t,pe2) (start_flux[2*(t)*polar_times_groups+(pe2)])

/**
 * @class GPUSolver GPUSolver.h "openmoc/src/dev/gpu/GPUSolver.h"
 * @brief This a subclass of the Solver class for NVIDIA Graphics
 *        Processing Units (GPUs).
 * @details The source code for this class includes C++ coupled with
 *          compute intensive CUDA kernels for execution on the GPU.
 */
class GPUSolver : public Solver {

private:
#ifdef MPIx
  /* Message size when communicating track angular fluxes at interfaces */
  int _track_message_size;

  /* Buffer to send track angular fluxes and associated information */
  std::vector<std::vector<float> > _send_buffers;

  /* Index into send_buffers for pre-filling (ONLYVACUUMBC mode) */
  std::vector<int> _send_buffers_index;

  /* Buffer to receive track angular fluxes and associated information */
  std::vector<std::vector<float> > _receive_buffers;

  /* Vector of vectors containing boundary track ids and direction */
  std::vector<std::vector<long> > _boundary_tracks;

  /* Vector to know how big of a send buffer to send to another domain */
  std::vector<int> _send_size;

  /* Vector to save the size of the receive buffers */
  std::vector<int> _receive_size;

  /* Vector of vectors containing the connecting track id and direction */
  std::vector<std::vector<long> > _track_connections;

  /* Vector of vectors containing the connecting domains */
  std::vector<std::vector<int> > _domain_connections;

  /* Rank of domains neighboring local domain */
  std::vector<int> _neighbor_domains;

  /* Index of neighboring domains in _neighbor_domains */
  std::unordered_map<int, int> _neighbor_connections;

  /* Array to check whether MPI communications are finished */
  MPI_Request* _MPI_requests;

  /* Arrays of booleans to know whether a send/receive call was made */
  bool* _MPI_sends;
  bool* _MPI_receives;
#endif

  /** The number of thread blocks */
  int _B;

  /** The number of threads per thread block */
  int _T;

  /** The FSR Material pointers index by FSR ID */
  int* _FSR_materials;

  /** A pointer to an array of the Materials on the device */
  dev_material* _materials;

  /** Pointer to chi spectrum material on the device */
  dev_material* _dev_chi_spectrum_material;

  /** A pointer to the array of Tracks on the device */
  dev_track* _dev_tracks;

  //dev_track*  dev_tracks_3d;//TODO
  dev_track2D* dev_tracks_2d;//用于存储按UID排列的Track指针的一维数组

  /** Thrust vector of angular fluxes for each track */
  thrust::device_vector<float> dev_boundary_flux;

  /** Thrust vector of starting angular fluxes for each track */
  thrust::device_vector<float> dev_start_flux;

  /** Thrust vector of FSR scalar fluxes */
  thrust::device_vector<FP_PRECISION> dev_scalar_flux;

  /** Thrust vector of old FSR scalar fluxes */
  thrust::device_vector<FP_PRECISION> dev_old_scalar_flux;

  /** Thrust vector of stabilizing flux */
  thrust::device_vector<FP_PRECISION> dev_stabilizing_flux;

  /** Thrust vector of fixed sources in each FSR */
  thrust::device_vector<FP_PRECISION> dev_fixed_sources;

  /** Thrust vector of source / sigma_t in each FSR */
  thrust::device_vector<FP_PRECISION> dev_reduced_sources;

  /** Map of Material IDs to indices in _materials array */
  std::map<int, int> _material_IDs_to_indices;

  void copyQuadrature();

  //将OTF放在这里
  int _num_2D_tracks;//用于建立一维线程块
  
  //通过列表索引
  configure* _configure;
 // dev_track2D* dev_tracks_2d;//用于存储按UID排列的Track指针的一
  dev_track2D* d_track_2D_chain;//用于转换tsi与tci的一种2D特征线排序
  int* d_track_2D_chain_link;//用于2D的链转换
  int** d_tracks_2D_xy;//用于存储按坐标排序的track数组,通过坐标索引
  dev_track2D* h_tracks_2D;
  int* d_tracks_per_stack;//3维数组转换为1维数组，并建立对应的索引关系

  int* d_num_x;//用于建立三维数组的对应关系
  int* d_num_y;//用于建立三维数组的对应关系
  int* d_first_lz_of_stack;//用于转换TSI与TCI的数据
  long* d_cum_tracks_per_stack;

  int* d_azim_size;

  int* h_num_x;
  int* h_num_y;
  int** d_num_z;//每个轴的轨道数
  int** d_num_l;
  double** d_dz_eff;//每根轴的轴间距
  double** d_dl_eff;
  double* h_theta;//用于传输正交的极角
  double* d_theta;
  dev_ExtrudedFSR* d_extruded_FSR_lookup;//用于计算tracesegment
  dev_Point* d_FSRs_to_centroids;
  int _num_materials;

  device_segment* d_temp_segments;
  device_segment* h_temp_segments;
  int h_num_seg_matrix_columns;

  track_index* d_track_3D_index;
  track_index* h_track_3D_index;

  long clone_storage = 0;

  //存储各区域段数
  long sum_segments=0;
  int _num_3D_tracks;

  //各区域几何特征
  int _domain_index_x;//区域坐标
  int _domain_index_y;
  int _domain_index_z;
  int _domains_index;
  int _num_domains[3];//总划分区域

  /*3D特征线不包含段的数据*/
  dev_track* h_track_3D;//这个用来导出数据
  dev_track* d_track_3D;
  dev_Point* d_start;

  /*判断是否先生成特征线*/
  bool pre_track;
  dev_track* d_track_3D_last;
  int pre_segments_index = 0;
public:

  GPUSolver(TrackGenerator* track_generator=NULL,configure* configure=NULL);
  virtual ~GPUSolver();

  int getNumThreadBlocks();

  /**
   * @brief Returns the number of threads per block to execute on the GPU.
   * @return the number of threads per block
   */
  int getNumThreadsPerBlock();
  double getFSRSource(long fsr_id, int group) override;
  double getFlux(long fsr_id, int group) override;
  void getFluxes(FP_PRECISION* out_fluxes, int num_fluxes) override;

  void setNumThreadBlocks(int num_blocks);
  void setNumThreadsPerBlock(int num_threads);
  void setGeometry(Geometry* geometry) override;
  void setTrackGenerator(TrackGenerator* track_generator) override;
  void setFluxes(FP_PRECISION* in_fluxes, int num_fluxes) override;

  void initializeExpEvaluators() override;
  void initializeMaterials(solverMode mode) override;
  void initializeFSRs() override;
  void initializeTracks();
  void initializeFluxArrays() override;
  void initializeSourceArrays() override;
  void initializeFixedSources() override;
  void initializeCmfd() override;
  void initializeSolver(solverMode solver_mode) override;
  void zeroTrackFluxes() override;
  void flattenFSRFluxes(FP_PRECISION value) override;
  void flattenFSRFluxesChiSpectrum() override;
  void storeFSRFluxes() override;
  void computeStabilizingFlux() override;
  void stabilizeFlux() override;
  void computeFSRSources(int iteration) override;
  void computeFSRFissionSources() override;
  void computeFSRScatterSources() override;
  void transportSweep() override;
  void addSourceToScalarFlux() override;
  void computeKeff() override;
  double normalizeFluxes() override;
  double computeResidual(residualType res_type) override;
  void resetFixedSources() override;

  void computeFSRFissionRates(double* fission_rates, long num_FSRs, bool nu = false) override;
  #ifdef MPIx
  void printCycle(long track_start, int domain_start, int length) override;
  void printLoadBalancingReport() override;
  void boundaryFluxChecker() override;
  void setupMPIBuffers();
  void deleteMPIBuffers();
  void packBuffers(std::vector<long> &packing_indexes);
  void transferAllInterfaceFluxes();
#endif 
  //将OTFonGPU的初始化放在这
  void clonedatatoGPU();//克隆的合集，所有克隆操作在这里完成
  void cpytrack(Track** tracks_2D, dev_track2D* h_tracks_2D, bool signal);
  void cpytrack2DchaintoGPU(TrackGenerator3D* track_generator_3D);
  void traverseint2Dto1D(int** array, int* d_array);
  void traversedouble2Dto1D(double** array, double* d_array);
  //用于rocprof测试执行时间的函数
  void testtime();
  //计算各个区域段数
  void sumsegments();
  void printgeometry();
  void sort(int left, int right, track_index* h_track_3D_index);
};


#endif /* GPUSOLVER_H_ */
