#ifndef OTF_H_
#define OTF_H_

#include "TrackGenerator3D.h"
#include "boundary_type.h"
#include "Quadrature.h"
#include "DeviceTrack.h"
#endif

struct device_segment {
  double _length;
  //Material* _material;后续改,如果不用就不传
  int _region_id;
  int _track_idx;
  int _cmfd_surface_fwd;
  int _cmfd_surface_bwd;
  double _starting_position[3];
  int _material_index;
};

struct dev_track2D
{
  int _uid;
  dev_Point _start;
  dev_Point _end;
  double _phi;
  int _num_segments;
  boundaryType _bc_fwd;
  boundaryType _bc_bwd;
  int _azim_index;
  int _xy_index;
  int _link_index;
  long _track_next_fwd;
  long _track_next_bwd;
  long _track_prdc_fwd;
  long _track_prdc_bwd;
  long _track_refl_fwd;
  long _track_refl_bwd;
  bool _next_fwd_fwd;
  bool _next_bwd_fwd;
  int _surface_in;
  int _surface_out;
  int _domain_fwd;
  int _domain_bwd;
  device_segment _segment;
  device_segment* segments_2D;
};

struct dev_track3D
{
  int _uid;
  dev_Point _start;
  dev_Point _end;
  double _phi;
  int _num_segments;
  boundaryType _bc_fwd;
  boundaryType _bc_bwd;
  int _azim_index;
  int _xy_index;
  int _link_index;
  long _track_next_fwd;
  long _track_next_bwd;
  long _track_prdc_fwd;
  long _track_prdc_bwd;
  long _track_refl_fwd;
  long _track_refl_bwd;
  bool _next_fwd_fwd;
  bool _next_bwd_fwd;
  bool _direction_in_cycle;
  int _surface_in;
  int _surface_out;
  int _domain_fwd;
  int _domain_bwd;
  //下面是3D的特征线数据
  double _theta;
  int _polar_index;
  int _z_index;
  int _lz_index;
  int _cycle_index;
  int _cycle_track_index;
  int _train_index;
  bool _cycle_fwd;
  bool if_load_segments;
  bool if_while;
  dev_segment* _segments;
};


struct dev_TrackStackIndexes {

  /** The azimuthal index (in 0 to _num_azim / 2) */
  int _azim;

  /** The xy index (in 0 to _num_x[_azim] + _num_y[_azim]) */
  int _xy;

  /** The polar index (in 0 to _num_polar) */
  int _polar;

  /** The z index in the z-stack (in 0 to _tracks_per_stack[_azim][_xy][_polar]) */
  int _z;


};

struct dev_TrackChainIndexes {

  /** The azimuthal index (in 0 to _num_azim / 4) */
  int _azim;

  /** The x index (in 0 to _num_x[_azim]) */
  int _x;

  /** The polar index (in 0 to _num_polar) */
  int _polar;

  /** The lz index (in 0 to _num_l[_azim][_polar] + _num_z[_azim][_polar]) */
  int _lz;

  /** The link index of the chain */
  int _link;


};

struct dev_ExtrudedFSR {
  double* _mesh;
  int _fsr_id;
  long* _fsr_ids;
  int _num_fsrs;
  int* _material_index;
};
 
struct configure
{
  int _B;
  int _T;
  bool if_pre_track;
  bool dump_domain;
  bool if_sort;
  bool onlycal_domain;
  configure() {
    _B = 64;
    _T = 64;
    if_pre_track = true;
    dump_domain = false;
    if_sort = true;
    onlycal_domain=false;
  }
};

struct track_index
{
  int track_index;
  int polar;
  int z;
  int num_segments;
};

/** GPU中的Quadrature */
// struct device_Quadrature{
//   // QuadratureType _quad_type;

//   /** The number of azimuthal angles in (0, 2*PI) */
//   // size_t _num_azim;

//   /** The number of polar angles in (0, PI) */
//   // size_t _num_polar;

//   /** An array of the sines of quadrature polar angles */
//   // std::vector<DoubleVec> _sin_thetas;

//   /** An array of the inverse sines of quadrature polar angles */
//   // std::vector<FloatVec> _inv_sin_thetas;

//   /** An array of the quadrature polar angles */
//   // std::vector<DoubleVec> _thetas;

//   /** An array of the quadrature azimuthal angles */
//   // DoubleVec _phis;

//   /** The actual track azimuthal spacing (cm) by azimuthal angle */
//   // DoubleVec _azim_spacings;

//   /** An array of the quadrature azimuthal weights */
//   // DoubleVec _azim_weights;

//   /** The actual track polar spacing (cm) by (azim, polar) */
//   // std::vector<DoubleVec> _polar_spacings;

//   /** An array of the quadrature polar weights */
//   // std::vector<DoubleVec> _polar_weights;

//   /** An array of the total weights for each azimuthal/polar angle pair */
//   // std::vector<FloatVec> _total_weights;
// };


