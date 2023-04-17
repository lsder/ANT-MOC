#include "CPUSolver.h"
#include "GPUSolver.h"
#include "GPUQuery.h"
#include "log.h"
//#include "OTFonGPU.h"
#include <array>
#include <iostream>
#include "cmdline.h"
#include "yaml-cpp/yaml.h"
int main(int argc, char* argv[]) {

CommandLineArgs args(argc,argv);
std::vector<std::string>  config_path ;
if(args.CheckCmdLineFlag("config"))
{
    args.GetCmdLineArguments("config", config_path);//用于获取多个数据
    // log_printf(NORMAL,"load config %s",config_path[0].c_str());
}
YAML::Node config = YAML::LoadFile(config_path[0]);

#ifdef MPIx
  int id,numprocs;
  int provided;
  std::vector<int>  domain ;
  int domain_x,domain_y, domain_z;
  if(args.CheckCmdLineFlag("domain"))
  {
      args.GetCmdLineArguments("domain", domain);//用于获取多个数据
      log_printf(NORMAL,"load config %d %d %d",domain[0],domain[1],domain[2]);
        domain_x=domain[0];
        domain_y=domain[1];
        domain_z=domain[2];
  }
  else{
      domain_x=config["domain_decomposition"]["x"].as<int>();
      domain_y=config["domain_decomposition"]["y"].as<int>();
      domain_z=config["domain_decomposition"]["z"].as<int>();
  }
  int num_domains[3]={domain_x,domain_y,domain_z};
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  log_set_ranks(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD,&id); 
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  log_printf(NORMAL,"Number  processes =%d" ,num_domains[0]*num_domains[1]*num_domains[2]);
  
  if(config["dump_segments"].as<bool>())
    log_printf(NORMAL,"dump_segments");
#endif


  /* Define simulation parameters */
#ifdef OPENMP
  int num_threads = omp_get_max_threads();
#else
  int num_threads = 1;
#endif

  double azim_spacing = config["ray_info"]["azim_spacing"].as<double>();
  int num_azim = config["ray_info"]["num_azim"].as<int>();
  double polar_spacing = config["ray_info"]["polar_spacing"].as<double>();
  int num_polar = config["ray_info"]["num_polar"].as<int>();
  double tolerance = 1e-5;
  int max_iters = config["max_iters"].as<int>();
  int axial_refines = 1;

  /* Domain decomposition / modular ray tracing */
  int nx = 1;
  int ny = 1;
  int nz = 1;
#ifdef MPIx
  num_threads = std::max(1, num_threads / (nx*ny*nz));
#endif

  /* Set logging information */
  set_log_level("NORMAL");
  log_printf(TITLE, "Simulating the OECD's C5G7 Benchmark Problem...");
 
  /* Create the geometry */
  log_printf(NORMAL, "Creating geometry...");
  Geometry geometry;
  //geometry.setRootUniverse(root_universe);
 geometry.loadFromFile("c5g7.geo");
#ifdef MPIx
   geometry.setDomainDecomposition( num_domains[0], num_domains[1], num_domains[2], MPI_COMM_WORLD);
#else
  geometry.setNumDomainModules(nx, ny, nz);
#endif
  geometry.initializeFlatSourceRegions();
//geometry.dumpToFile("c5g7.geo");
  /* Generate tracks */
  log_printf(NORMAL, "Initializing the track generator...");
  Quadrature* quad = new GLPolarQuad();
  quad->setNumPolarAngles(num_polar);
  quad->setNumAzimAngles(num_azim);
  #ifdef MPIx
  attach_gpu(id%4); 
  #endif
  std::vector<double> seg_heights {-32.13, 0.0, 32.13};
  TrackGenerator3D track_generator(&geometry, num_azim, num_polar, azim_spacing,
                                   polar_spacing);
  track_generator.setNumThreads(num_threads);
  track_generator.setQuadrature(quad);
  track_generator.setSegmentFormation(OTF_TRACKS);
  track_generator.setSegmentationZones(seg_heights);
  track_generator.generateTracks();

   /* Run simulation */
  CPUSolver solver(&track_generator);
  solver.setNumThreads(num_threads);
  solver.setConvergenceThreshold(tolerance);
  solver.computeEigenvalue(max_iters);
  solver.printTimerReport();

  configure* _configure = new configure;
  _configure->_T = config["_T"].as<int>();
  _configure->_B = config["_B"].as<int>();
  _configure->if_pre_track = config["if_pre_track"].as<bool>();
  _configure->dump_domain = config["dump_domain"].as<bool>();
  _configure->if_sort = config["if_sort"].as<bool>();
  _configure->onlycal_domain=config["onlycal_domain"].as<bool>();
  GPUSolver Gsolver(&track_generator,_configure);
  Gsolver.initializeSolver(FORWARD);
  if(!config["onlycal_domain"].as<bool>()){
    Gsolver.setConvergenceThreshold(tolerance);
    Gsolver.computeEigenvalue(max_iters);
    Gsolver.printTimerReport(); 
  }
  // log_printf(TITLE, "Finished");


}
