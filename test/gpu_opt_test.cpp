#include "CPUSolver.h"
#include "GPUSolver.h"
#include "log.h"
#include <array>
#include <iostream>
#include "cmdline.h"
int main(int argc, char* argv[]) {
#ifdef MPIx
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  log_set_ranks(MPI_COMM_WORLD);
#endif

  /* Define simulation parameters */
#ifdef OPENMP
  int num_threads = omp_get_num_threads();
#else
  int num_threads = 1;
#endif

  double azim_spacing = 0.2;
  int num_azim = 8;
  double polar_spacing = 1.;
  int num_polar = 4;
  double tolerance = 1e-5;
  int max_iters = 10;
  int axial_refines = 1;

  /* Set logging information */
  set_log_level("NORMAL");
  log_printf(TITLE, "Simulating the OECD's C5G7 Benchmark Problem...");
 
  /* Create the geometry */
  log_printf(NORMAL, "Creating geometry...");
  Geometry geometry;

  //geometry.dumpToFile("test2.geo");
 // geometry.PrintToFile("test2.geo");
  // geometry.printString();
   geometry.loadFromFile("20220322.geo"); 
  // // geometry.printString();
   geometry.initializeFlatSourceRegions();

  /* Generate tracks */
  log_printf(NORMAL, "Initializing the track generator...");
  Quadrature* quad = new GLPolarQuad();
  quad->setNumAzimAngles(num_azim);
  quad->setNumPolarAngles(num_polar);
  TrackGenerator3D track_generator(&geometry, num_azim, num_polar, azim_spacing,
                                   polar_spacing);
  track_generator.setNumThreads(num_threads);
  track_generator.setQuadrature(quad);
  track_generator.setSegmentFormation(EXPLICIT_3D);
  std::vector<double> seg_heights {-32.13, 0.0, 32.13};
  track_generator.setSegmentationZones(seg_heights);
  track_generator.generateTracks();

  /* Run simulation */
  DCUSolver solver(&track_generator);
  solver.initializeSolver(FORWARD);
  solver.setConvergenceThreshold(tolerance);
  solver.computeEigenvalue(max_iters);
  solver.printTimerReport();

  CPUSolver csolver(&track_generator);
  csolver.setNumThreads(32);
  csolver.setConvergenceThreshold(tolerance);
  csolver.computeEigenvalue(max_iters);
  csolver.printTimerReport();

  log_printf(TITLE, "Finished");
  return 0;

}
