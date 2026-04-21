#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <yaml-cpp/yaml.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using std::map;
using std::pair;
using std::string;
using std::vector;

namespace {

constexpr double kEps = 1e-12;

Matrix3d skew(const Vector3d & v) {
  Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
      -v.y(), v.x(), 0.0;
  return m;
}

template<typename T>
Eigen::Matrix<T, 3, 3> rpyToRTemplate(const Eigen::Matrix<T, 3, 1> & rpy) {
  const T roll = rpy.x();
  const T pitch = rpy.y();
  const T yaw = rpy.z();

  Eigen::Matrix<T, 3, 3> Rx, Ry, Rz;
  Rx << T(1), T(0), T(0),
        T(0), ceres::cos(roll), -ceres::sin(roll),
        T(0), ceres::sin(roll), ceres::cos(roll);

  Ry << ceres::cos(pitch), T(0), ceres::sin(pitch),
        T(0), T(1), T(0),
        -ceres::sin(pitch), T(0), ceres::cos(pitch);

  Rz << ceres::cos(yaw), -ceres::sin(yaw), T(0),
        ceres::sin(yaw), ceres::cos(yaw), T(0),
        T(0), T(0), T(1);

  return Rz * Ry * Rx;
}

Matrix3d rpyToR(const Vector3d & rpy) {
  return rpyToRTemplate<double>(rpy);
}

template<typename T>
Eigen::Matrix<T, 3, 3> rodriguesTemplate(const Eigen::Matrix<T, 3, 1> & w) {
  const T theta = ceres::sqrt(w.squaredNorm() + T(1e-18));
  Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
  if constexpr (std::is_same_v<T, double>) {
    if (std::abs(theta) < 1e-12) {
      return I;
    }
  }
  const Eigen::Matrix<T, 3, 1> k = w / theta;
  Eigen::Matrix<T, 3, 3> K;
  K << T(0), -k.z(), k.y(),
       k.z(), T(0), -k.x(),
      -k.y(), k.x(), T(0);
  return I + ceres::sin(theta) * K + (T(1) - ceres::cos(theta)) * K * K;
}

Vector3d normalizeVec(const Vector3d & v) {
  const double n = v.norm();
  if (n < kEps) {
    return v;
  }
  return v / n;
}

pair<Vector3d, double> transformPlaneToBody(
  const Matrix3d & R, const Vector3d & t,
  const Vector3d & n_lidar, double d_lidar)
{
  const Vector3d n_body = normalizeVec(R * n_lidar);
  const double d_body = d_lidar - n_body.dot(t);
  return {n_body, d_body};
}

pair<Vector3d, double> transformPlaneBodyToLidar(
  const Matrix3d & R, const Vector3d & t,
  const Vector3d & n_body, double d_body)
{
  const Vector3d n_lidar = normalizeVec(R.transpose() * n_body);
  const double d_lidar = d_body + n_body.dot(t);
  return {n_lidar, d_lidar};
}

Vector3d alignNormal(const Vector3d & n_meas, const Vector3d & n_prior) {
  return (n_meas.dot(n_prior) < 0.0) ? -n_meas : n_meas;
}

double rad2deg(double r) {
  return r * 180.0 / M_PI;
}

double deg2rad(double d) {
  return d * M_PI / 180.0;
}

pair<Vector3d, double> orientPlaneTowardLidar(
  const Vector3d & n_body_in, double d_body_in,
  const Vector3d & lidar_pos_body)
{
  Vector3d n = normalizeVec(n_body_in);
  double d = d_body_in;
  const double side = n.dot(lidar_pos_body) + d;
  if (side < 0.0) {
    n = -n;
    d = -d;
  }
  return {n, d};
}

string sanitizeIpToFrame(const string & ip) {
  string out = "lidar_";
  for (char c : ip) {
    out.push_back(c == '.' ? '_' : c);
  }
  return out;
}

template<typename PointT>
void finalizeCloud(typename pcl::PointCloud<PointT>::Ptr cloud) {
  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = false;
}

pcl::PointXYZRGB makePointRGB(double x, double y, double z, uint8_t r, uint8_t g, uint8_t b) {
  pcl::PointXYZRGB p;
  p.x = static_cast<float>(x);
  p.y = static_cast<float>(y);
  p.z = static_cast<float>(z);
  p.r = r;
  p.g = g;
  p.b = b;
  return p;
}

double angleBetweenNormalsDeg(const Vector3d & a, const Vector3d & b) {
  const double c = std::clamp(normalizeVec(a).dot(normalizeVec(b)), -1.0, 1.0);
  return rad2deg(std::acos(c));
}

struct RoiBox {
  double xmin{-std::numeric_limits<double>::infinity()};
  double xmax{ std::numeric_limits<double>::infinity()};
  double ymin{-std::numeric_limits<double>::infinity()};
  double ymax{ std::numeric_limits<double>::infinity()};
  double zmin{-std::numeric_limits<double>::infinity()};
  double zmax{ std::numeric_limits<double>::infinity()};

  bool contains(double x, double y, double z) const {
    return x >= xmin && x <= xmax &&
           y >= ymin && y <= ymax &&
           z >= zmin && z <= zmax;
  }

  bool contains(const Vector3d & p) const {
    return contains(p.x(), p.y(), p.z());
  }

  bool contains(const pcl::PointXYZ & pt) const {
    return contains(static_cast<double>(pt.x), static_cast<double>(pt.y), static_cast<double>(pt.z));
  }
};

struct OrientedRoiBox {
  Vector3d center_lidar{0.0, 0.0, 0.0};
  std::array<Vector3d, 3> axes_lidar{
    Vector3d::UnitX(),
    Vector3d::UnitY(),
    Vector3d::UnitZ()
  };
  Vector3d half_extents{0.0, 0.0, 0.0};

  bool contains(const Vector3d & p_lidar) const {
    const Vector3d q = p_lidar - center_lidar;
    for (int i = 0; i < 3; ++i) {
      const double local = q.dot(axes_lidar[i]);
      if (std::abs(local) > half_extents[i]) {
        return false;
      }
    }
    return true;
  }

  bool contains(const pcl::PointXYZ & pt) const {
    return contains(Vector3d(pt.x, pt.y, pt.z));
  }
};

struct AxisAlignedRoiBox {
  Vector3d min_corner{
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};
  Vector3d max_corner{
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()};

  bool isValid() const {
    return (min_corner.array() <= max_corner.array()).all();
  }

  bool contains(const Vector3d & p) const {
    return (p.array() >= min_corner.array()).all() &&
           (p.array() <= max_corner.array()).all();
  }

  bool contains(const pcl::PointXYZ & pt) const {
    return contains(Vector3d(pt.x, pt.y, pt.z));
  }
};

OrientedRoiBox transformRoiBoxBodyToLidar(
  const RoiBox & roi_body,
  const Matrix3d & R_lidar_to_body,
  const Vector3d & t_lidar_to_body)
{
  OrientedRoiBox roi_lidar;

  const Vector3d center_body(
    0.5 * (roi_body.xmin + roi_body.xmax),
    0.5 * (roi_body.ymin + roi_body.ymax),
    0.5 * (roi_body.zmin + roi_body.zmax));
  roi_lidar.center_lidar = R_lidar_to_body.transpose() * (center_body - t_lidar_to_body);

  roi_lidar.axes_lidar[0] = normalizeVec(R_lidar_to_body.transpose() * Vector3d::UnitX());
  roi_lidar.axes_lidar[1] = normalizeVec(R_lidar_to_body.transpose() * Vector3d::UnitY());
  roi_lidar.axes_lidar[2] = normalizeVec(R_lidar_to_body.transpose() * Vector3d::UnitZ());

  roi_lidar.half_extents = Vector3d(
    0.5 * (roi_body.xmax - roi_body.xmin),
    0.5 * (roi_body.ymax - roi_body.ymin),
    0.5 * (roi_body.zmax - roi_body.zmin));
  return roi_lidar;
}

std::array<Vector3d, 8> orientedRoiCornersLidar(const OrientedRoiBox & roi_lidar)
{
  std::array<Vector3d, 8> corners;
  int idx = 0;
  for (int sx : {-1, 1}) {
    for (int sy : {-1, 1}) {
      for (int sz : {-1, 1}) {
        corners[idx++] = roi_lidar.center_lidar
          + static_cast<double>(sx) * roi_lidar.half_extents.x() * roi_lidar.axes_lidar[0]
          + static_cast<double>(sy) * roi_lidar.half_extents.y() * roi_lidar.axes_lidar[1]
          + static_cast<double>(sz) * roi_lidar.half_extents.z() * roi_lidar.axes_lidar[2];
      }
    }
  }
  return corners;
}

AxisAlignedRoiBox orientedRoiToAxisAlignedAabb(const OrientedRoiBox & roi_lidar)
{
  AxisAlignedRoiBox aabb;
  const auto corners = orientedRoiCornersLidar(roi_lidar);
  for (const auto & c : corners) {
    aabb.min_corner = aabb.min_corner.cwiseMin(c);
    aabb.max_corner = aabb.max_corner.cwiseMax(c);
  }
  return aabb;
}

AxisAlignedRoiBox unionAxisAlignedAabb(const vector<AxisAlignedRoiBox> & boxes)
{
  AxisAlignedRoiBox out;
  bool has_any = false;
  for (const auto & box : boxes) {
    if (!box.isValid()) {
      continue;
    }
    if (!has_any) {
      out = box;
      has_any = true;
      continue;
    }
    out.min_corner = out.min_corner.cwiseMin(box.min_corner);
    out.max_corner = out.max_corner.cwiseMax(box.max_corner);
  }
  return out;
}


std::array<Vector3d, 8> bodyRoiCorners(const RoiBox & roi_body)
{
  return {
    Vector3d(roi_body.xmin, roi_body.ymin, roi_body.zmin),
    Vector3d(roi_body.xmax, roi_body.ymin, roi_body.zmin),
    Vector3d(roi_body.xmax, roi_body.ymax, roi_body.zmin),
    Vector3d(roi_body.xmin, roi_body.ymax, roi_body.zmin),
    Vector3d(roi_body.xmin, roi_body.ymin, roi_body.zmax),
    Vector3d(roi_body.xmax, roi_body.ymin, roi_body.zmax),
    Vector3d(roi_body.xmax, roi_body.ymax, roi_body.zmax),
    Vector3d(roi_body.xmin, roi_body.ymax, roi_body.zmax)
  };
}

struct PlaneMeasurement {
  Vector3d n_lidar;
  double d_lidar{0.0};
  Vector3d N0_body;
  double D0_body{0.0};  // plane prior in form N^T p = D
  std::size_t plane_index{0};
  std::size_t union_aabb_point_count{0};
  std::size_t body_roi_hit_count{0};
  std::size_t inlier_count{0};
  bool prior_flipped{false};
  Vector3d prior_input_normal;
  double prior_input_d_body_std{0.0};
  Vector3d prior_oriented_normal;
  double prior_oriented_d_body_std{0.0};
  std::size_t roi_point_count{0};
  bool bbox_used{false};
  std::optional<RoiBox> roi_box;
  pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud;
};

struct PlaneDiagnostic {
  std::size_t plane_index{0};
  double dtheta_rad_norm{0.0};
  double dtheta_deg_norm{0.0};
  Vector3d dtheta_rad{0.0, 0.0, 0.0};
  double dD_m{0.0};
  double normal_angle_error_deg{0.0};
  double distance_error_m{0.0};
  double mean_abs_body_plane_dist_m{0.0};
  double max_abs_body_plane_dist_m{0.0};
};

struct CandidateSolution {
  Vector3d rpy_opt;
  Vector3d t_opt;
  Matrix3d R_opt;
  std::size_t union_aabb_point_count{0};
  double initial_cost{0.0};
  double final_cost{0.0};
  bool numeric_success{false};
  bool strict_success{false};
  string summary;
  string strict_failure_reason;
  vector<std::array<double, 3>> dtheta_opt;
  vector<double> dD_opt;
  vector<PlaneMeasurement> used_measurements;
  vector<PlaneDiagnostic> plane_diagnostics;
  int seed_index{0};
};

struct DynamicRefineResult {
  bool attempted{false};
  bool success{false};
  string message;
};

struct PlaneResidual {
  PlaneResidual(
    const Vector3d & n_lidar_meas,
    double d_lidar_meas,
    const Vector3d & N0_body_prior,
    double D0_body_prior,
    double wn,
    double wd,
    double wp,
    double wD)
  : nL_(n_lidar_meas), dL_(d_lidar_meas), N0_(N0_body_prior), D0_(D0_body_prior),
    wn_(wn), wd_(wd), wp_(wp), wD_(wD) {}

  template<typename T>
  bool operator()(const T * const rpy, const T * const t,
                  const T * const dtheta, const T * const dD,
                  T * residuals) const {
    const Eigen::Matrix<T, 3, 1> rpy_v(rpy[0], rpy[1], rpy[2]);
    const Eigen::Matrix<T, 3, 1> t_v(t[0], t[1], t[2]);
    const Eigen::Matrix<T, 3, 1> dtheta_v(dtheta[0], dtheta[1], dtheta[2]);

    const Eigen::Matrix<T, 3, 3> R = rpyToRTemplate<T>(rpy_v);

    const Eigen::Matrix<T, 3, 1> nL = nL_.cast<T>();
    const Eigen::Matrix<T, 3, 1> N0 = N0_.cast<T>();
    const T dL = T(dL_);
    const T D0 = T(D0_);

    const Eigen::Matrix<T, 3, 3> R_delta = rodriguesTemplate<T>(dtheta_v);
    Eigen::Matrix<T, 3, 1> Ni = R_delta * N0;
    const T Ni_norm = ceres::sqrt(Ni.squaredNorm() + T(1e-12));
    Ni /= Ni_norm;
    const T Di = D0 + dD[0];

    Eigen::Matrix<T, 3, 1> nB_hat = R * nL;
    const T nB_hat_norm = ceres::sqrt(nB_hat.squaredNorm() + T(1e-12));
    nB_hat /= nB_hat_norm;
    const T dB_hat = dL - nB_hat.dot(t_v);

    // 第二轮修改：法向残差从分量差改为叉乘残差，更直接约束方向一致性
    const Eigen::Matrix<T, 3, 1> cross_err = nB_hat.cross(Ni);
    residuals[0] = T(wn_) * cross_err[0];
    residuals[1] = T(wn_) * cross_err[1];
    residuals[2] = T(wn_) * cross_err[2];

    residuals[3] = T(wd_) * (dB_hat + Di);
    residuals[4] = T(wp_) * dtheta[0];
    residuals[5] = T(wp_) * dtheta[1];
    residuals[6] = T(wp_) * dtheta[2];
    residuals[7] = T(wD_) * dD[0];
    return true;
  }

  static ceres::CostFunction * Create(
    const Vector3d & n_lidar_meas,
    double d_lidar_meas,
    const Vector3d & N0_body_prior,
    double D0_body_prior,
    double wn,
    double wd,
    double wp,
    double wD)
  {
    return new ceres::AutoDiffCostFunction<PlaneResidual, 8, 3, 3, 3, 1>(
      new PlaneResidual(
        n_lidar_meas, d_lidar_meas, N0_body_prior, D0_body_prior, wn, wd, wp, wD));
  }

  Vector3d nL_;
  double dL_;
  Vector3d N0_;
  double D0_;
  double wn_;
  double wd_;
  double wp_;
  double wD_;
};

struct LidarConfig {
  string ip;
  string topic;
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double roll{0.0};
  double pitch{0.0};
  double yaw{0.0};
  vector<Vector4d> planes;  // Ax+By+Cz+D=0 in body frame.
  vector<std::optional<RoiBox>> plane_rois;
};

struct LidarState {
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub;
  pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated{new pcl::PointCloud<pcl::PointXYZ>()};
  rclcpp::Time accumulation_start{0, 0, RCL_ROS_TIME};
  bool started{false};
  bool calibrated{false};
};

}  // namespace

class MultiMid360Calibrator : public rclcpp::Node {
public:
  MultiMid360Calibrator() : Node("multi_mid360_calibrator") {
    declare_parameter<string>("config_path", "lidar_config.yaml");
    declare_parameter<string>("target_lidar_ip", "");
    declare_parameter<double>("accumulation_time_sec", 2.0);
    declare_parameter<double>("roi_distance_threshold", 0.08);
    declare_parameter<double>("ransac_distance_threshold", 0.02);
    declare_parameter<int>("min_plane_points", 300);
    declare_parameter<double>("voxel_leaf_size", 0.01);

    // 第一轮：外参加边界，使用初始值 ± margin
    declare_parameter<double>("tx_margin", 0.10);
    declare_parameter<double>("ty_margin", 0.10);
    declare_parameter<double>("tz_margin", 0.05);
    declare_parameter<double>("roll_margin_deg", 10.0);
    declare_parameter<double>("pitch_margin_deg", 5.0);
    declare_parameter<double>("yaw_margin_deg", 5.0);

    declare_parameter<double>("wn", 10.0);
    declare_parameter<double>("wd", 20.0);
    declare_parameter<double>("wp", 2.0);
    declare_parameter<double>("wD", 5.0);

    // 第一轮：严格成功判定阈值
    declare_parameter<double>("strict_max_final_cost", 0.30);
    declare_parameter<double>("strict_max_normal_angle_error_deg", 8.0);
    declare_parameter<double>("strict_max_distance_error_m", 0.03);
    declare_parameter<double>("strict_max_mean_reprojection_error_m", 0.03);
    declare_parameter<double>("strict_max_dtheta_deg", 5.0);
    declare_parameter<double>("strict_max_dD_m", 0.03);

    // 第二轮：平面偏差变量边界
    declare_parameter<double>("plane_angle_dev_bound_deg", 5.0);
    declare_parameter<double>("plane_offset_dev_bound_m", 0.03);

    // 第一轮：多初值求解
    declare_parameter<bool>("enable_multi_seed", true);
    declare_parameter<double>("seed_pos_delta_xy", 0.03);
    declare_parameter<double>("seed_pos_delta_z", 0.02);
    declare_parameter<double>("seed_roll_delta_deg", 3.0);
    declare_parameter<double>("seed_pitch_delta_deg", 2.0);
    declare_parameter<double>("seed_yaw_delta_deg", 2.0);

    // 第三轮：动态 8 字接口（骨架）
    declare_parameter<bool>("enable_dynamic_figure8_refine", false);
    declare_parameter<string>("figure8_data_path", "");

    declare_parameter<bool>("publish_tf", true);
    declare_parameter<string>("output_result_path", "/tmp/lidar_body_calib_result.yaml");

    declare_parameter<bool>("save_raw_cloud", true);
    declare_parameter<bool>("save_body_cloud", true);
    declare_parameter<bool>("save_visualization_cloud", true);
    declare_parameter<string>("output_cloud_dir", "/tmp/mid360_calib_outputs");
    declare_parameter<double>("plane_vis_size", 0.8);
    declare_parameter<double>("plane_vis_resolution", 0.05);
    declare_parameter<double>("axis_vis_length", 0.4);
    declare_parameter<double>("axis_vis_step", 0.01);

    config_path_ = get_parameter("config_path").as_string();
    target_lidar_ip_ = get_parameter("target_lidar_ip").as_string();
    accumulation_time_sec_ = get_parameter("accumulation_time_sec").as_double();
    roi_distance_threshold_ = get_parameter("roi_distance_threshold").as_double();
    ransac_distance_threshold_ = get_parameter("ransac_distance_threshold").as_double();
    min_plane_points_ = get_parameter("min_plane_points").as_int();
    voxel_leaf_size_ = get_parameter("voxel_leaf_size").as_double();

    tx_margin_ = get_parameter("tx_margin").as_double();
    ty_margin_ = get_parameter("ty_margin").as_double();
    tz_margin_ = get_parameter("tz_margin").as_double();
    roll_margin_ = deg2rad(get_parameter("roll_margin_deg").as_double());
    pitch_margin_ = deg2rad(get_parameter("pitch_margin_deg").as_double());
    yaw_margin_ = deg2rad(get_parameter("yaw_margin_deg").as_double());

    wn_ = get_parameter("wn").as_double();
    wd_ = get_parameter("wd").as_double();
    wp_ = get_parameter("wp").as_double();
    wD_ = get_parameter("wD").as_double();

    strict_max_final_cost_ = get_parameter("strict_max_final_cost").as_double();
    strict_max_normal_angle_error_deg_ = get_parameter("strict_max_normal_angle_error_deg").as_double();
    strict_max_distance_error_m_ = get_parameter("strict_max_distance_error_m").as_double();
    strict_max_mean_reprojection_error_m_ = get_parameter("strict_max_mean_reprojection_error_m").as_double();
    strict_max_dtheta_deg_ = get_parameter("strict_max_dtheta_deg").as_double();
    strict_max_dD_m_ = get_parameter("strict_max_dD_m").as_double();

    plane_angle_dev_bound_ = deg2rad(get_parameter("plane_angle_dev_bound_deg").as_double());
    plane_offset_dev_bound_ = get_parameter("plane_offset_dev_bound_m").as_double();

    enable_multi_seed_ = get_parameter("enable_multi_seed").as_bool();
    seed_pos_delta_xy_ = get_parameter("seed_pos_delta_xy").as_double();
    seed_pos_delta_z_ = get_parameter("seed_pos_delta_z").as_double();
    seed_roll_delta_ = deg2rad(get_parameter("seed_roll_delta_deg").as_double());
    seed_pitch_delta_ = deg2rad(get_parameter("seed_pitch_delta_deg").as_double());
    seed_yaw_delta_ = deg2rad(get_parameter("seed_yaw_delta_deg").as_double());

    enable_dynamic_figure8_refine_ = get_parameter("enable_dynamic_figure8_refine").as_bool();
    figure8_data_path_ = get_parameter("figure8_data_path").as_string();

    publish_tf_ = get_parameter("publish_tf").as_bool();
    output_result_path_ = get_parameter("output_result_path").as_string();
    save_raw_cloud_ = get_parameter("save_raw_cloud").as_bool();
    save_body_cloud_ = get_parameter("save_body_cloud").as_bool();
    save_visualization_cloud_ = get_parameter("save_visualization_cloud").as_bool();
    output_cloud_dir_ = get_parameter("output_cloud_dir").as_string();
    plane_vis_size_ = get_parameter("plane_vis_size").as_double();
    plane_vis_resolution_ = get_parameter("plane_vis_resolution").as_double();
    axis_vis_length_ = get_parameter("axis_vis_length").as_double();
    axis_vis_step_ = get_parameter("axis_vis_step").as_double();

    if (!loadConfig(config_path_)) {
      throw std::runtime_error("Failed to load config: " + config_path_);
    }

    std::filesystem::create_directories(output_cloud_dir_);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("calib_plane_markers", 1);

    createSubscribers();

    RCLCPP_INFO(get_logger(), "Node started. Loaded %zu lidar configs.", lidar_configs_.size());
  }

private:
  bool loadConfig(const string & path) {
    YAML::Node root = YAML::LoadFile(path);
    if (!root || root.IsNull()) {
      RCLCPP_ERROR(get_logger(), "Config file empty: %s", path.c_str());
      return false;
    }

    for (auto it = root.begin(); it != root.end(); ++it) {
      LidarConfig cfg;
      cfg.ip = it->first.as<string>();
      const YAML::Node node = it->second;

      if (!node["x"] || !node["y"] || !node["z"] ||
          !node["roll"] || !node["pitch"] || !node["yaw"] || !node["planes"]) {
        RCLCPP_ERROR(get_logger(), "Lidar %s config incomplete.", cfg.ip.c_str());
        return false;
      }

      cfg.topic = node["topic"] ? node["topic"].as<string>() : ("/" + sanitizeIpToFrame(cfg.ip) + "/points");
      cfg.x = node["x"].as<double>();
      cfg.y = node["y"].as<double>();
      cfg.z = node["z"].as<double>();
      cfg.roll = deg2rad(node["roll"].as<double>());
      cfg.pitch = deg2rad(node["pitch"].as<double>());
      cfg.yaw = deg2rad(node["yaw"].as<double>());

      for (const auto & plane_node : node["planes"]) {
        auto coeff = plane_node.as<vector<double>>();
        if (coeff.size() != 4) {
          RCLCPP_ERROR(get_logger(), "Plane coeff size must be 4 for lidar %s", cfg.ip.c_str());
          return false;
        }
        Vector4d p;
        p << coeff[0], coeff[1], coeff[2], coeff[3];
        cfg.planes.push_back(p);
      }

      cfg.plane_rois.resize(cfg.planes.size(), std::nullopt);
      if (node["plane_rois"]) {
        const YAML::Node rois = node["plane_rois"];
        if (!rois.IsSequence()) {
          RCLCPP_ERROR(get_logger(), "plane_rois must be a sequence for lidar %s", cfg.ip.c_str());
          return false;
        }
        const std::size_t n = std::min<std::size_t>(rois.size(), cfg.planes.size());
        for (std::size_t i = 0; i < n; ++i) {
          if (rois[i].IsNull()) {
            continue;
          }
          RoiBox box;
          box.xmin = rois[i]["xmin"] ? rois[i]["xmin"].as<double>() : box.xmin;
          box.xmax = rois[i]["xmax"] ? rois[i]["xmax"].as<double>() : box.xmax;
          box.ymin = rois[i]["ymin"] ? rois[i]["ymin"].as<double>() : box.ymin;
          box.ymax = rois[i]["ymax"] ? rois[i]["ymax"].as<double>() : box.ymax;
          box.zmin = rois[i]["zmin"] ? rois[i]["zmin"].as<double>() : box.zmin;
          box.zmax = rois[i]["zmax"] ? rois[i]["zmax"].as<double>() : box.zmax;
          cfg.plane_rois[i] = box;
        }
      }

      if (cfg.planes.size() < 3) {
        RCLCPP_WARN(get_logger(), "Lidar %s has only %zu priors. Calibration is best with >=3 planes.",
          cfg.ip.c_str(), cfg.planes.size());
      }

      lidar_configs_[cfg.ip] = cfg;
      lidar_states_[cfg.ip] = LidarState{};
      RCLCPP_INFO(get_logger(),
        "Loaded lidar %s from topic %s with %zu planes. init_t=[%.3f, %.3f, %.3f], init_rpy_deg=[%.3f, %.3f, %.3f]",
        cfg.ip.c_str(), cfg.topic.c_str(), cfg.planes.size(),
        cfg.x, cfg.y, cfg.z,
        rad2deg(cfg.roll), rad2deg(cfg.pitch), rad2deg(cfg.yaw));
    }

    if (!target_lidar_ip_.empty() && lidar_configs_.find(target_lidar_ip_) == lidar_configs_.end()) {
      RCLCPP_ERROR(get_logger(), "target_lidar_ip %s not found in config.", target_lidar_ip_.c_str());
      return false;
    }

    return true;
  }

  void createSubscribers() {
    for (auto & kv : lidar_configs_) {
      const string ip = kv.first;
      if (!target_lidar_ip_.empty() && ip != target_lidar_ip_) {
        continue;
      }

      const auto & cfg = kv.second;
      auto callback = [this, ip](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        this->pointCloudCallback(ip, msg);
      };
      lidar_states_[ip].sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        cfg.topic, rclcpp::SensorDataQoS(), callback);
      RCLCPP_INFO(get_logger(), "Subscribed %s -> %s", ip.c_str(), cfg.topic.c_str());
    }
  }

  void pointCloudCallback(const string & ip, const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto & state = lidar_states_.at(ip);
    if (state.calibrated) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr frame(new pcl::PointCloud<pcl::PointXYZ>());
    try {
      pcl::fromROSMsg(*msg, *frame);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "fromROSMsg failed for %s: %s", ip.c_str(), e.what());
      return;
    }

    if (!state.started) {
      state.accumulation_start = rclcpp::Time(msg->header.stamp);
      state.started = true;
      state.accumulated->clear();
    }

    *state.accumulated += *frame;
    const rclcpp::Time current(msg->header.stamp);
    const double elapsed = (current - state.accumulation_start).seconds();

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "Accumulating lidar %s: %.2fs / %.2fs, points=%zu",
      ip.c_str(), elapsed, accumulation_time_sec_, state.accumulated->size());

    if (elapsed < accumulation_time_sec_) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>(*state.accumulated));
    finalizeCloud<pcl::PointXYZ>(raw_cloud);

    if (save_raw_cloud_) {
      saveRawCloud(ip, raw_cloud);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    voxelDownsample(raw_cloud, filtered, voxel_leaf_size_);
    RCLCPP_INFO(get_logger(), "Lidar %s accumulated %.2fs, raw=%zu, voxel=%zu",
      ip.c_str(), elapsed, raw_cloud->size(), filtered->size());

    const auto result = calibrateSingleLidar(lidar_configs_.at(ip), filtered, raw_cloud, msg->header.frame_id);
    if (result.has_value()) {
      writeResultYaml(ip, *result);

      if (publish_tf_) {
        publishTransform(ip, *result, msg->header.frame_id);
      }
      publishMarkers(ip, *result, msg->header.frame_id);

      if (save_body_cloud_ || save_visualization_cloud_) {
        saveBodyAndVisualizationClouds(ip, *result, raw_cloud);
      }

      if (enable_dynamic_figure8_refine_) {
        const auto dyn = runDynamicFigure8Refine(*result);
        RCLCPP_WARN(get_logger(), "Dynamic figure-8 refine status: attempted=%s success=%s msg=%s",
          dyn.attempted ? "true" : "false",
          dyn.success ? "true" : "false",
          dyn.message.c_str());
      }

      state.calibrated = true;
      RCLCPP_INFO(get_logger(), "Calibration complete for %s", ip.c_str());
    } else {
      saveExtractionDebugOutputs(ip, lidar_configs_.at(ip), raw_cloud, filtered);
      RCLCPP_WARN(get_logger(), "Calibration failed for %s. Restarting accumulation window.", ip.c_str());
      state.started = false;
      state.accumulated->clear();
    }
  }

  vector<PlaneMeasurement> extractAndFitPlanes(
    const LidarConfig & cfg,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) const
  {
    vector<PlaneMeasurement> out;
    const Matrix3d R0 = rpyToR(Vector3d(cfg.roll, cfg.pitch, cfg.yaw));
    const Vector3d t0(cfg.x, cfg.y, cfg.z);

    vector<std::optional<OrientedRoiBox>> roi_boxes_lidar(cfg.planes.size(), std::nullopt);
    vector<AxisAlignedRoiBox> roi_aabbs_lidar;
    for (std::size_t i = 0; i < cfg.planes.size(); ++i) {
      const auto roi_box_body = (i < cfg.plane_rois.size()) ? cfg.plane_rois[i] : std::nullopt;
      if (!roi_box_body.has_value()) {
        continue;
      }
      roi_boxes_lidar[i] = transformRoiBoxBodyToLidar(*roi_box_body, R0, t0);
      roi_aabbs_lidar.push_back(orientedRoiToAxisAlignedAabb(*roi_boxes_lidar[i]));
    }

    const auto union_lidar_aabb = unionAxisAlignedAabb(roi_aabbs_lidar);
    pcl::PointCloud<pcl::PointXYZ>::Ptr union_subset_lidar(new pcl::PointCloud<pcl::PointXYZ>());
    vector<Vector3d> union_subset_body;
    union_subset_body.reserve(cloud->size());

    for (const auto & pt : cloud->points) {
      if (!pcl::isFinite(pt)) {
        continue;
      }
      const Vector3d p_lidar(pt.x, pt.y, pt.z);
      if (!roi_aabbs_lidar.empty() && !union_lidar_aabb.contains(p_lidar)) {
        continue;
      }
      union_subset_lidar->points.push_back(pt);
      union_subset_body.push_back(R0 * p_lidar + t0);
    }
    finalizeCloud<pcl::PointXYZ>(union_subset_lidar);

    RCLCPP_INFO(get_logger(),
      "Lidar %s union ROI subset: source=%zu, inside_union_aabb=%zu, rois_with_bbox=%zu",
      cfg.ip.c_str(), cloud->size(), union_subset_lidar->size(), roi_aabbs_lidar.size());

    for (std::size_t i = 0; i < cfg.planes.size(); ++i) {
      const auto & p = cfg.planes[i];
      Vector3d N0 = Vector3d(p[0], p[1], p[2]);
      const double norm = N0.norm();
      if (norm < 1e-9) {
        RCLCPP_WARN(get_logger(), "Plane %zu of %s has invalid normal.", i, cfg.ip.c_str());
        continue;
      }
      N0 /= norm;
      const Vector3d prior_input_normal = N0;
      const double prior_input_d_body_std = p[3] / norm;
      double d_body_std = prior_input_d_body_std;

      const auto oriented_prior = orientPlaneTowardLidar(N0, d_body_std, t0);
      const bool prior_flipped = (oriented_prior.first.dot(prior_input_normal) < 0.0);
      N0 = oriented_prior.first;
      d_body_std = oriented_prior.second;
      const double D0 = -d_body_std;

      const auto pred_lidar_plane = transformPlaneBodyToLidar(R0, t0, N0, d_body_std);
      const auto roi_box_body = (i < cfg.plane_rois.size()) ? cfg.plane_rois[i] : std::nullopt;
      std::size_t body_roi_hit_count = 0;
      const auto roi_cloud = extractPlanePointsFromUnionSubset(
        union_subset_lidar,
        union_subset_body,
        pred_lidar_plane.first,
        pred_lidar_plane.second,
        roi_distance_threshold_,
        roi_box_body,
        &body_roi_hit_count);
      RCLCPP_INFO(get_logger(),
        "Plane %zu for %s: union_aabb_points=%zu, body_roi_hits=%zu, plane_roi_points=%zu%s",
        i, cfg.ip.c_str(), union_subset_lidar->size(), body_roi_hit_count, roi_cloud->size(),
        roi_box_body.has_value() ? " with union-lidar-AABB + body-ROI" : " without body ROI");
      if (static_cast<int>(roi_cloud->size()) < min_plane_points_) {
        RCLCPP_WARN(get_logger(), "Plane %zu ROI too small for %s: %zu < %d", i, cfg.ip.c_str(),
          roi_cloud->size(), min_plane_points_);
        continue;
      }

      Vector3d n_fit;
      double d_fit = 0.0;
      std::size_t inlier_count = 0;
      if (!fitPlaneRansac(roi_cloud, n_fit, d_fit, inlier_count)) {
        continue;
      }

      n_fit = alignNormal(n_fit, pred_lidar_plane.first);
      if (n_fit.dot(pred_lidar_plane.first) < 0.0) {
        n_fit = -n_fit;
        d_fit = -d_fit;
      }

      PlaneMeasurement m;
      m.n_lidar = n_fit;
      m.d_lidar = d_fit;
      m.N0_body = N0;
      m.D0_body = D0;
      m.plane_index = i;
      m.union_aabb_point_count = union_subset_lidar->size();
      m.body_roi_hit_count = body_roi_hit_count;
      m.inlier_count = inlier_count;
      m.prior_flipped = prior_flipped;
      m.prior_input_normal = prior_input_normal;
      m.prior_input_d_body_std = prior_input_d_body_std;
      m.prior_oriented_normal = N0;
      m.prior_oriented_d_body_std = d_body_std;
      m.roi_point_count = roi_cloud->size();
      m.bbox_used = roi_box_body.has_value();
      m.roi_box = roi_box_body;
      m.roi_cloud = roi_cloud;
      m.roi_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>(*roi_cloud));
      out.push_back(m);

      RCLCPP_INFO(get_logger(),
        "Plane %zu used: nL=[%.5f %.5f %.5f], dL=%.5f, prior_flipped=%s, D0=%.5f, union_aabb_points=%zu, body_roi_hits=%zu, ransac_inliers=%zu",
        i, n_fit.x(), n_fit.y(), n_fit.z(), d_fit, prior_flipped ? "true" : "false", D0,
        m.union_aabb_point_count, m.body_roi_hit_count, m.inlier_count);
    }
    return out;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanePoints(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const Vector3d & n_lidar,
    double d_lidar,
    double threshold,
    const std::optional<OrientedRoiBox> & roi_box_lidar) const
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto & pt : cloud->points) {
      if (!pcl::isFinite(pt)) {
        continue;
      }

      const Vector3d p_lidar(pt.x, pt.y, pt.z);
      if (roi_box_lidar.has_value() && !roi_box_lidar->contains(p_lidar)) {
        continue;
      }

      const double dist = std::abs(n_lidar.dot(p_lidar) + d_lidar);
      if (dist < threshold) {
        out->points.push_back(pt);
      }
    }
    finalizeCloud<pcl::PointXYZ>(out);
    return out;
  }


  pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanePointsFromUnionSubset(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & union_subset_lidar,
    const vector<Vector3d> & union_subset_body,
    const Vector3d & n_lidar,
    double d_lidar,
    double threshold,
    const std::optional<RoiBox> & roi_box_body,
    std::size_t * body_roi_hit_count) const
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>());
    const std::size_t n = std::min<std::size_t>(union_subset_lidar->points.size(), union_subset_body.size());
    out->points.reserve(n);
    std::size_t body_hits = 0;
    for (std::size_t idx = 0; idx < n; ++idx) {
      const auto & pt = union_subset_lidar->points[idx];
      if (!pcl::isFinite(pt)) {
        continue;
      }
      const bool inside_body_roi = !roi_box_body.has_value() || roi_box_body->contains(union_subset_body[idx]);
      if (!inside_body_roi) {
        continue;
      }
      ++body_hits;
      const Vector3d p_lidar(pt.x, pt.y, pt.z);
      const double dist = std::abs(n_lidar.dot(p_lidar) + d_lidar);
      if (dist < threshold) {
        out->points.push_back(pt);
      }
    }
    if (body_roi_hit_count) {
      *body_roi_hit_count = body_hits;
    }
    finalizeCloud<pcl::PointXYZ>(out);
    return out;
  }

  bool fitPlaneRansac(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    Vector3d & n,
    double & d,
    std::size_t & inlier_count) const
  {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ransac_distance_threshold_);
    seg.setMaxIterations(10000);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coeff);

    if (inliers->indices.empty() || coeff->values.size() < 4) {
      RCLCPP_WARN(get_logger(), "RANSAC failed or no inliers.");
      return false;
    }

    n = normalizeVec(Vector3d(coeff->values[0], coeff->values[1], coeff->values[2]));
    d = coeff->values[3] / Vector3d(coeff->values[0], coeff->values[1], coeff->values[2]).norm();
    inlier_count = inliers->indices.size();
    return true;
  }

  void voxelDownsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & in,
    pcl::PointCloud<pcl::PointXYZ>::Ptr & out,
    double leaf) const
  {
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(in);
    voxel.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
    voxel.filter(*out);
  }

  void applyPoseBounds(
    ceres::Problem & problem,
    const LidarConfig & cfg,
    double * rpy,
    double * t) const
  {
    problem.SetParameterLowerBound(t, 0, cfg.x - tx_margin_);
    problem.SetParameterUpperBound(t, 0, cfg.x + tx_margin_);
    problem.SetParameterLowerBound(t, 1, cfg.y - ty_margin_);
    problem.SetParameterUpperBound(t, 1, cfg.y + ty_margin_);
    problem.SetParameterLowerBound(t, 2, cfg.z - tz_margin_);
    problem.SetParameterUpperBound(t, 2, cfg.z + tz_margin_);

    problem.SetParameterLowerBound(rpy, 0, cfg.roll - roll_margin_);
    problem.SetParameterUpperBound(rpy, 0, cfg.roll + roll_margin_);
    problem.SetParameterLowerBound(rpy, 1, cfg.pitch - pitch_margin_);
    problem.SetParameterUpperBound(rpy, 1, cfg.pitch + pitch_margin_);
    problem.SetParameterLowerBound(rpy, 2, cfg.yaw - yaw_margin_);
    problem.SetParameterUpperBound(rpy, 2, cfg.yaw + yaw_margin_);
  }

  void applyPlaneDeviationBounds(
    ceres::Problem & problem,
    vector<std::array<double, 3>> & dtheta_blocks,
    vector<double> & dD_blocks) const
  {
    for (std::size_t i = 0; i < dtheta_blocks.size(); ++i) {
      for (int k = 0; k < 3; ++k) {
        problem.SetParameterLowerBound(dtheta_blocks[i].data(), k, -plane_angle_dev_bound_);
        problem.SetParameterUpperBound(dtheta_blocks[i].data(), k,  plane_angle_dev_bound_);
      }
      problem.SetParameterLowerBound(&dD_blocks[i], 0, -plane_offset_dev_bound_);
      problem.SetParameterUpperBound(&dD_blocks[i], 0,  plane_offset_dev_bound_);
    }
  }

  vector<std::pair<Vector3d, Vector3d>> buildSeedList(const LidarConfig & cfg) const {
    vector<std::pair<Vector3d, Vector3d>> seeds;
    const Vector3d rpy0(cfg.roll, cfg.pitch, cfg.yaw);
    const Vector3d t0(cfg.x, cfg.y, cfg.z);
    seeds.emplace_back(rpy0, t0);
    if (!enable_multi_seed_) {
      return seeds;
    }

    seeds.emplace_back(rpy0 + Vector3d( seed_roll_delta_, 0.0, 0.0), t0);
    seeds.emplace_back(rpy0 + Vector3d(-seed_roll_delta_, 0.0, 0.0), t0);
    seeds.emplace_back(rpy0 + Vector3d(0.0,  seed_pitch_delta_, 0.0), t0);
    seeds.emplace_back(rpy0 + Vector3d(0.0, -seed_pitch_delta_, 0.0), t0);
    seeds.emplace_back(rpy0 + Vector3d(0.0, 0.0,  seed_yaw_delta_), t0);
    seeds.emplace_back(rpy0 + Vector3d(0.0, 0.0, -seed_yaw_delta_), t0);

    seeds.emplace_back(rpy0, t0 + Vector3d( seed_pos_delta_xy_, 0.0, 0.0));
    seeds.emplace_back(rpy0, t0 + Vector3d(-seed_pos_delta_xy_, 0.0, 0.0));
    seeds.emplace_back(rpy0, t0 + Vector3d(0.0,  seed_pos_delta_xy_, 0.0));
    seeds.emplace_back(rpy0, t0 + Vector3d(0.0, -seed_pos_delta_xy_, 0.0));
    seeds.emplace_back(rpy0, t0 + Vector3d(0.0, 0.0,  seed_pos_delta_z_));
    seeds.emplace_back(rpy0, t0 + Vector3d(0.0, 0.0, -seed_pos_delta_z_));

    return seeds;
  }

  CandidateSolution solveOnce(
    const LidarConfig & cfg,
    const vector<PlaneMeasurement> & measurements,
    const Vector3d & rpy_seed,
    const Vector3d & t_seed,
    int seed_index,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud) const
  {
    double rpy[3] = {rpy_seed.x(), rpy_seed.y(), rpy_seed.z()};
    double t[3] = {t_seed.x(), t_seed.y(), t_seed.z()};
    vector<std::array<double, 3>> dtheta_blocks(measurements.size(), {0.0, 0.0, 0.0});
    vector<double> dD_blocks(measurements.size(), 0.0);

    ceres::Problem problem;
    for (std::size_t i = 0; i < measurements.size(); ++i) {
      const auto & m = measurements[i];
      ceres::CostFunction * cost = PlaneResidual::Create(
        m.n_lidar, m.d_lidar, m.N0_body, m.D0_body,
        wn_, wd_, wp_, wD_);
      problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
        rpy, t, dtheta_blocks[i].data(), &dD_blocks[i]);
    }

    applyPoseBounds(problem, cfg, rpy, t);
    applyPlaneDeviationBounds(problem, dtheta_blocks, dD_blocks);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.function_tolerance = 1e-10;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    CandidateSolution cand;
    cand.seed_index = seed_index;
    cand.rpy_opt = Vector3d(rpy[0], rpy[1], rpy[2]);
    cand.t_opt = Vector3d(t[0], t[1], t[2]);
    cand.R_opt = rpyToR(cand.rpy_opt);
    cand.initial_cost = summary.initial_cost;
    cand.final_cost = summary.final_cost;
    cand.numeric_success = (
      summary.termination_type == ceres::CONVERGENCE ||
      summary.termination_type == ceres::USER_SUCCESS);
    cand.summary = summary.BriefReport();
    cand.used_measurements = measurements;
    cand.union_aabb_point_count = measurements.empty() ? 0 : measurements.front().union_aabb_point_count;
    cand.dtheta_opt = dtheta_blocks;
    cand.dD_opt = dD_blocks;
    cand.plane_diagnostics = evaluatePlaneDiagnostics(cand, raw_cloud);
    return cand;
  }

  vector<PlaneDiagnostic> evaluatePlaneDiagnostics(
    const CandidateSolution & cand,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud) const
  {
    vector<PlaneDiagnostic> out;
    const auto body_cloud = transformCloudToBody(raw_cloud, cand);

    for (std::size_t i = 0; i < cand.used_measurements.size(); ++i) {
      const auto & m = cand.used_measurements[i];
      const auto body_plane = transformPlaneToBody(cand.R_opt, cand.t_opt, m.n_lidar, m.d_lidar);
      PlaneDiagnostic diag;
      diag.plane_index = m.plane_index;
      diag.dtheta_rad = Vector3d(cand.dtheta_opt[i][0], cand.dtheta_opt[i][1], cand.dtheta_opt[i][2]);
      diag.dtheta_rad_norm = diag.dtheta_rad.norm();
      diag.dtheta_deg_norm = rad2deg(diag.dtheta_rad_norm);
      diag.dD_m = cand.dD_opt[i];
      diag.normal_angle_error_deg = angleBetweenNormalsDeg(body_plane.first, m.N0_body);
      diag.distance_error_m = std::abs(body_plane.second + m.D0_body);

      // 第三轮：自动回投检查。把转换到车体系后的点云，对理想平面做统计误差
      double sum_abs = 0.0;
      double max_abs = 0.0;
      std::size_t used = 0;
      for (const auto & pt : body_cloud->points) {
        if (!pcl::isFinite(pt)) {
          continue;
        }
        const Vector3d p(pt.x, pt.y, pt.z);
        const double dist = std::abs(m.N0_body.dot(p) + (-m.D0_body));
        if (dist < roi_distance_threshold_) {
          sum_abs += dist;
          max_abs = std::max(max_abs, dist);
          ++used;
        }
      }
      diag.mean_abs_body_plane_dist_m = (used > 0) ? (sum_abs / static_cast<double>(used)) : std::numeric_limits<double>::infinity();
      diag.max_abs_body_plane_dist_m = (used > 0) ? max_abs : std::numeric_limits<double>::infinity();
      out.push_back(diag);
    }
    return out;
  }

  bool validateCalibrationResult(const LidarConfig & cfg, CandidateSolution & cand) const {
    if (!cand.numeric_success) {
      cand.strict_failure_reason = "numeric_not_converged";
      cand.strict_success = false;
      return false;
    }
    if (cand.final_cost > strict_max_final_cost_) {
      cand.strict_failure_reason = "final_cost_too_high";
      cand.strict_success = false;
      return false;
    }

    if (cand.t_opt.x() < cfg.x - tx_margin_ || cand.t_opt.x() > cfg.x + tx_margin_ ||
        cand.t_opt.y() < cfg.y - ty_margin_ || cand.t_opt.y() > cfg.y + ty_margin_ ||
        cand.t_opt.z() < cfg.z - tz_margin_ || cand.t_opt.z() > cfg.z + tz_margin_) {
      cand.strict_failure_reason = "translation_out_of_bounds";
      cand.strict_success = false;
      return false;
    }

    if (cand.rpy_opt.x() < cfg.roll - roll_margin_ || cand.rpy_opt.x() > cfg.roll + roll_margin_ ||
        cand.rpy_opt.y() < cfg.pitch - pitch_margin_ || cand.rpy_opt.y() > cfg.pitch + pitch_margin_ ||
        cand.rpy_opt.z() < cfg.yaw - yaw_margin_ || cand.rpy_opt.z() > cfg.yaw + yaw_margin_) {
      cand.strict_failure_reason = "rotation_out_of_bounds";
      cand.strict_success = false;
      return false;
    }

    for (const auto & diag : cand.plane_diagnostics) {
      if (diag.normal_angle_error_deg > strict_max_normal_angle_error_deg_) {
        cand.strict_failure_reason = "plane_normal_error_too_large";
        cand.strict_success = false;
        return false;
      }
      if (diag.distance_error_m > strict_max_distance_error_m_) {
        cand.strict_failure_reason = "plane_distance_error_too_large";
        cand.strict_success = false;
        return false;
      }
      if (diag.mean_abs_body_plane_dist_m > strict_max_mean_reprojection_error_m_) {
        cand.strict_failure_reason = "reprojection_error_too_large";
        cand.strict_success = false;
        return false;
      }
      if (diag.dtheta_deg_norm > strict_max_dtheta_deg_) {
        cand.strict_failure_reason = "plane_angle_deviation_too_large";
        cand.strict_success = false;
        return false;
      }
      if (std::abs(diag.dD_m) > strict_max_dD_m_) {
        cand.strict_failure_reason = "plane_offset_deviation_too_large";
        cand.strict_success = false;
        return false;
      }
    }

    cand.strict_success = true;
    cand.strict_failure_reason.clear();
    return true;
  }

  std::optional<CandidateSolution> pickBestCandidate(const vector<CandidateSolution> & candidates) const {
    if (candidates.empty()) {
      return std::nullopt;
    }
    const CandidateSolution * best = nullptr;
    for (const auto & c : candidates) {
      if (!c.strict_success) {
        continue;
      }
      if (!best || c.final_cost < best->final_cost) {
        best = &c;
      }
    }
    if (best) {
      return *best;
    }

    // 都未通过严格验收时，不返回“看起来能用”的坏解
    return std::nullopt;
  }

  std::optional<CandidateSolution> calibrateSingleLidar(
    const LidarConfig & cfg,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud,
    const string & input_frame)
  {
    (void)input_frame;
    const auto measurements = extractAndFitPlanes(cfg, cloud);
    if (measurements.size() < 3) {
      RCLCPP_ERROR(get_logger(), "Need at least 3 valid planes, got %zu for %s",
        measurements.size(), cfg.ip.c_str());
      return std::nullopt;
    }

    const auto seeds = buildSeedList(cfg);
    vector<CandidateSolution> candidates;
    candidates.reserve(seeds.size());

    for (std::size_t i = 0; i < seeds.size(); ++i) {
      auto cand = solveOnce(cfg, measurements, seeds[i].first, seeds[i].second, static_cast<int>(i), raw_cloud);
      validateCalibrationResult(cfg, cand);
      RCLCPP_INFO(get_logger(),
        "Seed %zu result for %s: strict_success=%s, reason=%s, final_cost=%.6f, rpy_deg=[%.3f %.3f %.3f], t=[%.3f %.3f %.3f]",
        i, cfg.ip.c_str(),
        cand.strict_success ? "true" : "false",
        cand.strict_failure_reason.empty() ? "ok" : cand.strict_failure_reason.c_str(),
        cand.final_cost,
        rad2deg(cand.rpy_opt.x()), rad2deg(cand.rpy_opt.y()), rad2deg(cand.rpy_opt.z()),
        cand.t_opt.x(), cand.t_opt.y(), cand.t_opt.z());
      candidates.push_back(std::move(cand));
    }

    const auto best = pickBestCandidate(candidates);
    if (!best.has_value()) {
      RCLCPP_ERROR(get_logger(), "No candidate passed strict validation for %s", cfg.ip.c_str());
      return std::nullopt;
    }

    RCLCPP_INFO(get_logger(), "=== Best calibration result for %s (seed=%d) ===", cfg.ip.c_str(), best->seed_index);
    RCLCPP_INFO(get_logger(), "Summary: %s", best->summary.c_str());
    RCLCPP_INFO(get_logger(), "Cost: %.8f -> %.8f", best->initial_cost, best->final_cost);
    RCLCPP_INFO(get_logger(), "RPY(deg): [%.6f, %.6f, %.6f]",
      rad2deg(best->rpy_opt.x()), rad2deg(best->rpy_opt.y()), rad2deg(best->rpy_opt.z()));
    RCLCPP_INFO(get_logger(), "t(m): [%.6f, %.6f, %.6f]",
      best->t_opt.x(), best->t_opt.y(), best->t_opt.z());

    return best;
  }

  DynamicRefineResult runDynamicFigure8Refine(const CandidateSolution &) const {
    DynamicRefineResult out;
    out.attempted = enable_dynamic_figure8_refine_;
    if (!enable_dynamic_figure8_refine_) {
      out.success = false;
      out.message = "disabled";
      return out;
    }
    out.success = false;
    if (figure8_data_path_.empty()) {
      out.message = "enabled_but_no_figure8_data_path_provided";
      return out;
    }
    out.message = "interface_only_not_implemented_yet";
    return out;
  }

  string makeBaseOutputPrefix(const string & ip) const {
    const std::filesystem::path dir(output_cloud_dir_);
    std::filesystem::create_directories(dir);
    return (dir / sanitizeIpToFrame(ip)).string();
  }

  void saveRawCloud(const string & ip, const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud) const {
    const string path = makeBaseOutputPrefix(ip) + "_raw_lidar.pcd";
    if (pcl::io::savePCDFileBinary(path, *raw_cloud) == 0) {
      RCLCPP_INFO(get_logger(), "Saved raw lidar cloud: %s", path.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to save raw lidar cloud: %s", path.c_str());
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloudToBody(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_lidar,
    const CandidateSolution & result) const
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>());
    out->reserve(cloud_lidar->size());
    for (const auto & pt : cloud_lidar->points) {
      if (!pcl::isFinite(pt)) {
        continue;
      }
      const Vector3d p_l(pt.x, pt.y, pt.z);
      const Vector3d p_b = result.R_opt * p_l + result.t_opt;
      pcl::PointXYZ q;
      q.x = static_cast<float>(p_b.x());
      q.y = static_cast<float>(p_b.y());
      q.z = static_cast<float>(p_b.z());
      out->points.push_back(q);
    }
    finalizeCloud<pcl::PointXYZ>(out);
    return out;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorizeCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    uint8_t r, uint8_t g, uint8_t b) const
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>());
    out->reserve(cloud->size());
    for (const auto & pt : cloud->points) {
      if (!pcl::isFinite(pt)) {
        continue;
      }
      out->points.push_back(makePointRGB(pt.x, pt.y, pt.z, r, g, b));
    }
    finalizeCloud<pcl::PointXYZRGB>(out);
    return out;
  }

  void appendPlanePatch(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
    const Vector3d & n, double d,
    double plane_size, double resolution,
    uint8_t r, uint8_t g, uint8_t b) const
  {
    Vector3d normal = normalizeVec(n);
    Vector3d center = -d * normal;

    Vector3d ref = (std::abs(normal.z()) < 0.9) ? Vector3d::UnitZ() : Vector3d::UnitX();
    Vector3d u = normalizeVec(normal.cross(ref));
    Vector3d v = normalizeVec(normal.cross(u));

    const int steps = std::max(1, static_cast<int>(std::ceil(plane_size / resolution)));
    const double half = plane_size * 0.5;
    for (int i = -steps; i <= steps; ++i) {
      for (int j = -steps; j <= steps; ++j) {
        const double du = static_cast<double>(i) * resolution;
        const double dv = static_cast<double>(j) * resolution;
        if (std::abs(du) > half || std::abs(dv) > half) {
          continue;
        }
        const Vector3d p = center + du * u + dv * v;
        cloud->points.push_back(makePointRGB(p.x(), p.y(), p.z(), r, g, b));
      }
    }
  }

  void appendAxisLine(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
    const Vector3d & origin,
    const Vector3d & direction,
    double length,
    double step,
    uint8_t r, uint8_t g, uint8_t b) const
  {
    const Vector3d dir = normalizeVec(direction);
    const int n = std::max(2, static_cast<int>(std::ceil(length / step)));
    for (int i = 0; i <= n; ++i) {
      const double s = std::min(length, i * step);
      const Vector3d p = origin + s * dir;
      cloud->points.push_back(makePointRGB(p.x(), p.y(), p.z(), r, g, b));
    }
  }

  void appendFrameAxes(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
    const Vector3d & origin,
    const Matrix3d & R,
    double length,
    double step) const
  {
    appendAxisLine(cloud, origin, R.col(0), length, step, 255, 0, 0);
    appendAxisLine(cloud, origin, R.col(1), length, step, 0, 255, 0);
    appendAxisLine(cloud, origin, R.col(2), length, step, 0, 0, 255);
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr buildVisualizationCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_body,
    const CandidateSolution & result) const
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis = colorizeCloud(cloud_body, 180, 180, 180);

    const std::array<std::array<uint8_t, 3>, 6> colors = {{
      {{255, 128, 0}},
      {{255, 0, 255}},
      {{0, 255, 255}},
      {{255, 255, 0}},
      {{0, 128, 255}},
      {{128, 255, 0}}
    }};

    for (std::size_t i = 0; i < result.used_measurements.size(); ++i) {
      const auto & m = result.used_measurements[i];
      const auto plane_body = transformPlaneToBody(result.R_opt, result.t_opt, m.n_lidar, m.d_lidar);
      const auto & c = colors[i % colors.size()];
      appendPlanePatch(vis, plane_body.first, plane_body.second,
        plane_vis_size_, plane_vis_resolution_, c[0], c[1], c[2]);
    }

    appendFrameAxes(vis, Vector3d::Zero(), Matrix3d::Identity(), axis_vis_length_, axis_vis_step_);
    appendFrameAxes(vis, result.t_opt, result.R_opt, axis_vis_length_, axis_vis_step_);

    finalizeCloud<pcl::PointXYZRGB>(vis);
    return vis;
  }

  void saveBodyAndVisualizationClouds(
    const string & ip,
    const CandidateSolution & result,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud) const
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr body_cloud = transformCloudToBody(raw_cloud, result);
    const string base = makeBaseOutputPrefix(ip);

    if (save_body_cloud_) {
      const string body_path = base + "_raw_in_body.pcd";
      if (pcl::io::savePCDFileBinary(body_path, *body_cloud) == 0) {
        RCLCPP_INFO(get_logger(), "Saved body-frame raw cloud: %s", body_path.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "Failed to save body-frame raw cloud: %s", body_path.c_str());
      }
    }

    if (save_visualization_cloud_) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud = buildVisualizationCloud(body_cloud, result);
      const string vis_path = base + "_body_with_planes_axes.pcd";
      if (pcl::io::savePCDFileBinary(vis_path, *vis_cloud) == 0) {
        RCLCPP_INFO(get_logger(), "Saved visualization cloud: %s", vis_path.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "Failed to save visualization cloud: %s", vis_path.c_str());
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr buildPlanePatchCloud(
    const CandidateSolution & result) const
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>());
    const std::array<std::array<uint8_t, 3>, 6> colors = {{
      {{255, 128, 0}},
      {{255, 0, 255}},
      {{0, 255, 255}},
      {{255, 255, 0}},
      {{0, 128, 255}},
      {{128, 255, 0}}
    }};

    for (std::size_t i = 0; i < result.used_measurements.size(); ++i) {
      const auto & m = result.used_measurements[i];
      const auto plane_body = transformPlaneToBody(result.R_opt, result.t_opt, m.n_lidar, m.d_lidar);
      const auto & c = colors[i % colors.size()];
      appendPlanePatch(out, plane_body.first, plane_body.second,
        plane_vis_size_, plane_vis_resolution_, c[0], c[1], c[2]);
    }
    finalizeCloud<pcl::PointXYZRGB>(out);
    return out;
  }


  pcl::PointCloud<pcl::PointXYZRGB>::Ptr buildPlanePatchCloudFromMeasurementsLidar(
    const vector<PlaneMeasurement> & measurements) const
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>());
    const std::array<std::array<uint8_t, 3>, 6> colors = {{
      {{255, 128, 0}},
      {{255, 0, 255}},
      {{0, 255, 255}},
      {{255, 255, 0}},
      {{0, 128, 255}},
      {{128, 255, 0}}
    }};

    for (std::size_t i = 0; i < measurements.size(); ++i) {
      const auto & m = measurements[i];
      const auto & c = colors[i % colors.size()];
      appendPlanePatch(out, m.n_lidar, m.d_lidar,
        plane_vis_size_, plane_vis_resolution_, c[0], c[1], c[2]);
    }
    finalizeCloud<pcl::PointXYZRGB>(out);
    return out;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr buildRoiDebugCloud(
    const vector<PlaneMeasurement> & measurements) const
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>());
    const std::array<std::array<uint8_t, 3>, 6> colors = {{
      {{255, 128, 0}},
      {{255, 0, 255}},
      {{0, 255, 255}},
      {{255, 255, 0}},
      {{0, 128, 255}},
      {{128, 255, 0}}
    }};

    for (std::size_t i = 0; i < measurements.size(); ++i) {
      if (!measurements[i].roi_cloud) {
        continue;
      }
      const auto & c = colors[i % colors.size()];
      for (const auto & pt : measurements[i].roi_cloud->points) {
        if (!pcl::isFinite(pt)) {
          continue;
        }
        out->points.push_back(makePointRGB(pt.x, pt.y, pt.z, c[0], c[1], c[2]));
      }
    }
    finalizeCloud<pcl::PointXYZRGB>(out);
    return out;
  }


  void appendRoiWireframeLidar(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
    const RoiBox & roi_body,
    const Matrix3d & R_lidar_to_body,
    const Vector3d & t_lidar_to_body,
    uint8_t r, uint8_t g, uint8_t b) const
  {
    const auto corners_body = bodyRoiCorners(roi_body);
    std::array<Vector3d, 8> corners_lidar;
    for (std::size_t i = 0; i < corners_body.size(); ++i) {
      corners_lidar[i] = R_lidar_to_body.transpose() * (corners_body[i] - t_lidar_to_body);
    }

    const std::array<std::array<int, 2>, 12> edges = {{
      {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
      {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
      {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}
    }};

    const double wire_step = std::max(0.5 * axis_vis_step_, 0.005);
    for (const auto & edge : edges) {
      const Vector3d & p0 = corners_lidar[edge[0]];
      const Vector3d & p1 = corners_lidar[edge[1]];
      const Vector3d dir = p1 - p0;
      const double len = dir.norm();
      if (len < 1e-9) {
        cloud->points.push_back(makePointRGB(p0.x(), p0.y(), p0.z(), r, g, b));
        continue;
      }
      const Vector3d unit = dir / len;
      const int n = std::max(2, static_cast<int>(std::ceil(len / wire_step)));
      for (int i = 0; i <= n; ++i) {
        const double s = std::min(len, i * wire_step);
        const Vector3d p = p0 + s * unit;
        cloud->points.push_back(makePointRGB(p.x(), p.y(), p.z(), r, g, b));
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr buildExtractionDebugCloud(
    const LidarConfig & cfg,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & filtered_cloud,
    const vector<PlaneMeasurement> & measurements) const
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis(new pcl::PointCloud<pcl::PointXYZRGB>());
    vis->reserve(raw_cloud->size() + filtered_cloud->size());

    for (const auto & pt : raw_cloud->points) {
      if (!pcl::isFinite(pt)) continue;
      vis->points.push_back(makePointRGB(pt.x, pt.y, pt.z, 100, 180, 255));
    }
    for (const auto & pt : filtered_cloud->points) {
      if (!pcl::isFinite(pt)) continue;
      vis->points.push_back(makePointRGB(pt.x, pt.y, pt.z, 160, 160, 160));
    }

    const std::array<std::array<uint8_t, 3>, 6> colors = {{
      {{255, 64, 64}},
      {{64, 255, 64}},
      {{64, 128, 255}},
      {{255, 180, 0}},
      {{255, 64, 255}},
      {{0, 220, 220}}
    }};

    const std::array<std::array<uint8_t, 3>, 3> roi_wire_colors = {{
      {{255, 255, 0}},
      {{255, 0, 255}},
      {{0, 0, 0}}
    }};
    const Matrix3d R0 = rpyToR(Vector3d(cfg.roll, cfg.pitch, cfg.yaw));
    const Vector3d t0(cfg.x, cfg.y, cfg.z);
    for (std::size_t i = 0; i < std::min<std::size_t>(3, cfg.plane_rois.size()); ++i) {
      if (!cfg.plane_rois[i].has_value()) {
        continue;
      }
      const auto & c = roi_wire_colors[i % roi_wire_colors.size()];
      appendRoiWireframeLidar(vis, *cfg.plane_rois[i], R0, t0, c[0], c[1], c[2]);
    }

    for (std::size_t i = 0; i < measurements.size(); ++i) {
      const auto & m = measurements[i];
      const auto & c = colors[i % colors.size()];

      if (m.roi_cloud) {
        for (const auto & pt : m.roi_cloud->points) {
          if (!pcl::isFinite(pt)) continue;
          vis->points.push_back(makePointRGB(pt.x, pt.y, pt.z, c[0], c[1], c[2]));
        }
      }

      appendPlanePatch(vis, m.n_lidar, m.d_lidar,
        plane_vis_size_, plane_vis_resolution_, c[0], c[1], c[2]);

      Vector3d center = -m.d_lidar * normalizeVec(m.n_lidar);
      appendAxisLine(vis, center, normalizeVec(m.n_lidar), axis_vis_length_ * 0.6,
        axis_vis_step_, 255, 255, 255);
    }

    appendFrameAxes(vis, Vector3d::Zero(), Matrix3d::Identity(), axis_vis_length_, axis_vis_step_);
    finalizeCloud<pcl::PointXYZRGB>(vis);
    return vis;
  }

  void saveExtractionDebugOutputs(
    const string & ip,
    const LidarConfig & cfg,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & filtered_cloud) const
  {
    const auto measurements = extractAndFitPlanes(cfg, filtered_cloud);
    const string base = makeBaseOutputPrefix(ip);

    const string voxel_path = base + "_voxel_lidar.pcd";
    if (pcl::io::savePCDFileBinary(voxel_path, *filtered_cloud) == 0) {
      RCLCPP_INFO(get_logger(), "Saved voxel lidar cloud: %s", voxel_path.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to save voxel lidar cloud: %s", voxel_path.c_str());
    }

    for (std::size_t i = 0; i < measurements.size(); ++i) {
      if (!measurements[i].roi_cloud) {
        continue;
      }
      const string roi_path = base + "_plane_" + std::to_string(i) + "_roi_lidar.pcd";
      if (pcl::io::savePCDFileBinary(roi_path, *measurements[i].roi_cloud) == 0) {
        RCLCPP_INFO(get_logger(), "Saved plane %zu ROI cloud: %s", i, roi_path.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "Failed to save plane %zu ROI cloud: %s", i, roi_path.c_str());
      }
    }

    const auto debug_cloud = buildExtractionDebugCloud(cfg, raw_cloud, filtered_cloud, measurements);
    const string debug_path = base + "_debug_planes_lidar.pcd";
    if (pcl::io::savePCDFileBinary(debug_path, *debug_cloud) == 0) {
      RCLCPP_INFO(get_logger(),
        "Saved extraction debug cloud: %s (light blue=raw, grey=voxel, yellow/purple/black=YAML body ROI wireframes in lidar frame, red/green/blue=ROI+plane fit results)",
        debug_path.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to save extraction debug cloud: %s", debug_path.c_str());
    }
  }


  void writeResultYaml(const string & ip, const CandidateSolution & result) const {
    YAML::Node root;
    root["ip"] = ip;
    root["numeric_success"] = result.numeric_success;
    root["strict_success"] = result.strict_success;
    root["strict_failure_reason"] = result.strict_failure_reason;
    root["summary"] = result.summary;
    root["initial_cost"] = result.initial_cost;
    root["final_cost"] = result.final_cost;
    root["selected_seed_index"] = result.seed_index;
    root["union_aabb_point_count"] = static_cast<int>(result.union_aabb_point_count);

    YAML::Node rpy;
    rpy.push_back(result.rpy_opt.x());
    rpy.push_back(result.rpy_opt.y());
    rpy.push_back(result.rpy_opt.z());
    root["rpy_rad"] = rpy;

    YAML::Node rpy_deg;
    rpy_deg.push_back(rad2deg(result.rpy_opt.x()));
    rpy_deg.push_back(rad2deg(result.rpy_opt.y()));
    rpy_deg.push_back(rad2deg(result.rpy_opt.z()));
    root["rpy_deg"] = rpy_deg;

    YAML::Node t;
    t.push_back(result.t_opt.x());
    t.push_back(result.t_opt.y());
    t.push_back(result.t_opt.z());
    root["t_m"] = t;

    YAML::Node R;
    for (int i = 0; i < 3; ++i) {
      YAML::Node row;
      for (int j = 0; j < 3; ++j) {
        row.push_back(result.R_opt(i, j));
      }
      R.push_back(row);
    }
    root["R"] = R;

    YAML::Node bounds;
    bounds["tx_min"] = lidar_configs_.at(ip).x - tx_margin_;
    bounds["tx_max"] = lidar_configs_.at(ip).x + tx_margin_;
    bounds["ty_min"] = lidar_configs_.at(ip).y - ty_margin_;
    bounds["ty_max"] = lidar_configs_.at(ip).y + ty_margin_;
    bounds["tz_min"] = lidar_configs_.at(ip).z - tz_margin_;
    bounds["tz_max"] = lidar_configs_.at(ip).z + tz_margin_;
    bounds["roll_min_deg"] = rad2deg(lidar_configs_.at(ip).roll - roll_margin_);
    bounds["roll_max_deg"] = rad2deg(lidar_configs_.at(ip).roll + roll_margin_);
    bounds["pitch_min_deg"] = rad2deg(lidar_configs_.at(ip).pitch - pitch_margin_);
    bounds["pitch_max_deg"] = rad2deg(lidar_configs_.at(ip).pitch + pitch_margin_);
    bounds["yaw_min_deg"] = rad2deg(lidar_configs_.at(ip).yaw - yaw_margin_);
    bounds["yaw_max_deg"] = rad2deg(lidar_configs_.at(ip).yaw + yaw_margin_);
    root["pose_bounds"] = bounds;

    YAML::Node validation;
    validation["strict_max_final_cost"] = strict_max_final_cost_;
    validation["strict_max_normal_angle_error_deg"] = strict_max_normal_angle_error_deg_;
    validation["strict_max_distance_error_m"] = strict_max_distance_error_m_;
    validation["strict_max_mean_reprojection_error_m"] = strict_max_mean_reprojection_error_m_;
    validation["strict_max_dtheta_deg"] = strict_max_dtheta_deg_;
    validation["strict_max_dD_m"] = strict_max_dD_m_;
    root["validation_thresholds"] = validation;

    YAML::Node plane_dev_bounds;
    plane_dev_bounds["angle_bound_deg"] = rad2deg(plane_angle_dev_bound_);
    plane_dev_bounds["offset_bound_m"] = plane_offset_dev_bound_;
    root["plane_deviation_bounds"] = plane_dev_bounds;

    YAML::Node planes;
    for (std::size_t i = 0; i < result.used_measurements.size(); ++i) {
      const auto & m = result.used_measurements[i];
      YAML::Node item;
      item["plane_index"] = static_cast<int>(m.plane_index);
      item["union_aabb_point_count"] = static_cast<int>(m.union_aabb_point_count);
      item["body_roi_hit_count"] = static_cast<int>(m.body_roi_hit_count);
      item["inlier_count"] = static_cast<int>(m.inlier_count);
      item["roi_point_count"] = static_cast<int>(m.roi_point_count);
      item["prior_flipped"] = m.prior_flipped;
      item["bbox_used"] = m.bbox_used;

      YAML::Node prior_input_n;
      prior_input_n.push_back(m.prior_input_normal.x());
      prior_input_n.push_back(m.prior_input_normal.y());
      prior_input_n.push_back(m.prior_input_normal.z());
      item["prior_input_normal"] = prior_input_n;
      item["prior_input_d_body_std"] = m.prior_input_d_body_std;

      YAML::Node prior_oriented_n;
      prior_oriented_n.push_back(m.prior_oriented_normal.x());
      prior_oriented_n.push_back(m.prior_oriented_normal.y());
      prior_oriented_n.push_back(m.prior_oriented_normal.z());
      item["prior_oriented_normal"] = prior_oriented_n;
      item["prior_oriented_d_body_std"] = m.prior_oriented_d_body_std;

      if (m.roi_box.has_value()) {
        YAML::Node roi;
        roi["xmin"] = m.roi_box->xmin;
        roi["xmax"] = m.roi_box->xmax;
        roi["ymin"] = m.roi_box->ymin;
        roi["ymax"] = m.roi_box->ymax;
        roi["zmin"] = m.roi_box->zmin;
        roi["zmax"] = m.roi_box->zmax;
        item["roi_box_body"] = roi;
      }

      YAML::Node nL;
      nL.push_back(m.n_lidar.x());
      nL.push_back(m.n_lidar.y());
      nL.push_back(m.n_lidar.z());
      item["n_lidar"] = nL;
      item["d_lidar"] = m.d_lidar;
      YAML::Node N0;
      N0.push_back(m.N0_body.x());
      N0.push_back(m.N0_body.y());
      N0.push_back(m.N0_body.z());
      item["N0_body"] = N0;
      item["D0_body"] = m.D0_body;

      const auto plane_body = transformPlaneToBody(result.R_opt, result.t_opt, m.n_lidar, m.d_lidar);
      YAML::Node nB;
      nB.push_back(plane_body.first.x());
      nB.push_back(plane_body.first.y());
      nB.push_back(plane_body.first.z());
      item["n_body_est"] = nB;
      item["d_body_est"] = plane_body.second;

      YAML::Node dtheta;
      dtheta.push_back(result.dtheta_opt[i][0]);
      dtheta.push_back(result.dtheta_opt[i][1]);
      dtheta.push_back(result.dtheta_opt[i][2]);
      item["dtheta_rad"] = dtheta;
      item["dtheta_deg_norm"] = result.plane_diagnostics[i].dtheta_deg_norm;
      item["dD_m"] = result.dD_opt[i];
      item["normal_angle_error_deg"] = result.plane_diagnostics[i].normal_angle_error_deg;
      item["distance_error_m"] = result.plane_diagnostics[i].distance_error_m;
      item["mean_abs_body_plane_dist_m"] = result.plane_diagnostics[i].mean_abs_body_plane_dist_m;
      item["max_abs_body_plane_dist_m"] = result.plane_diagnostics[i].max_abs_body_plane_dist_m;

      planes.push_back(item);
    }
    root["used_planes"] = planes;

    YAML::Node dynamic_node;
    dynamic_node["enable_dynamic_figure8_refine"] = enable_dynamic_figure8_refine_;
    dynamic_node["figure8_data_path"] = figure8_data_path_;
    dynamic_node["status"] = enable_dynamic_figure8_refine_ ? "interface_only" : "disabled";
    root["dynamic_figure8_refine"] = dynamic_node;

    YAML::Node outputs;
    const string base = makeBaseOutputPrefix(ip);
    if (save_raw_cloud_) {
      outputs["raw_lidar_pcd"] = base + "_raw_lidar.pcd";
    }
    if (save_body_cloud_) {
      outputs["raw_body_pcd"] = base + "_raw_in_body.pcd";
    }
    if (save_visualization_cloud_) {
      outputs["visualization_body_pcd"] = base + "_body_with_planes_axes.pcd";
    }
    outputs["voxel_lidar_pcd"] = base + "_voxel_lidar.pcd";
    outputs["debug_lidar_pcd"] = base + "_debug_planes_lidar.pcd";
    root["output_files"] = outputs;

    string path = output_result_path_;
    const auto pos = path.rfind(".yaml");
    if (pos != string::npos) {
      path.insert(pos, "_" + sanitizeIpToFrame(ip));
    } else {
      path += "_" + sanitizeIpToFrame(ip) + ".yaml";
    }
    std::ofstream ofs(path);
    ofs << root;
    ofs.close();
    RCLCPP_INFO(get_logger(), "Wrote result YAML: %s", path.c_str());
  }

  void publishTransform(const string & ip, const CandidateSolution & result, const string & input_frame) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = now();
    tf.header.frame_id = "body";
    tf.child_frame_id = input_frame.empty() ? sanitizeIpToFrame(ip) : input_frame;

    const Matrix3d R_bl = result.R_opt.transpose();
    const Vector3d t_bl = -R_bl * result.t_opt;
    tf.transform.translation.x = t_bl.x();
    tf.transform.translation.y = t_bl.y();
    tf.transform.translation.z = t_bl.z();

    tf2::Quaternion q;
    Eigen::Vector3d rpy_bl = R_bl.eulerAngles(0, 1, 2);
    q.setRPY(rpy_bl.x(), rpy_bl.y(), rpy_bl.z());
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf);
  }

  void publishMarkers(const string & ip, const CandidateSolution & result, const string & input_frame) {
    visualization_msgs::msg::MarkerArray array;
    int id = 0;
    for (const auto & m : result.used_measurements) {
      visualization_msgs::msg::Marker mk;
      mk.header.stamp = now();
      mk.header.frame_id = input_frame.empty() ? sanitizeIpToFrame(ip) : input_frame;
      mk.ns = "planes";
      mk.id = id++;
      mk.type = visualization_msgs::msg::Marker::ARROW;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.scale.x = 0.4;
      mk.scale.y = 0.06;
      mk.scale.z = 0.08;
      mk.color.a = 1.0;
      mk.color.g = 1.0;
      geometry_msgs::msg::Point p0, p1;
      p0.x = 0.0; p0.y = 0.0; p0.z = 0.0;
      p1.x = m.n_lidar.x() * 0.5;
      p1.y = m.n_lidar.y() * 0.5;
      p1.z = m.n_lidar.z() * 0.5;
      mk.points.push_back(p0);
      mk.points.push_back(p1);
      array.markers.push_back(mk);
    }
    marker_pub_->publish(array);
  }

  string config_path_;
  string target_lidar_ip_;
  string output_result_path_;
  string output_cloud_dir_;
  string figure8_data_path_;

  double accumulation_time_sec_{2.0};
  double roi_distance_threshold_{0.08};
  double ransac_distance_threshold_{0.02};
  int min_plane_points_{300};
  double voxel_leaf_size_{0.01};

  double wn_{10.0};
  double wd_{20.0};
  double wp_{2.0};
  double wD_{5.0};

  double plane_vis_size_{0.8};
  double plane_vis_resolution_{0.05};
  double axis_vis_length_{0.4};
  double axis_vis_step_{0.01};

  double tx_margin_{0.10};
  double ty_margin_{0.10};
  double tz_margin_{0.05};
  double roll_margin_{deg2rad(10.0)};
  double pitch_margin_{deg2rad(5.0)};
  double yaw_margin_{deg2rad(5.0)};

  double strict_max_final_cost_{0.30};
  double strict_max_normal_angle_error_deg_{8.0};
  double strict_max_distance_error_m_{0.03};
  double strict_max_mean_reprojection_error_m_{0.03};
  double strict_max_dtheta_deg_{5.0};
  double strict_max_dD_m_{0.03};

  double plane_angle_dev_bound_{deg2rad(5.0)};
  double plane_offset_dev_bound_{0.03};

  bool enable_multi_seed_{true};
  double seed_pos_delta_xy_{0.03};
  double seed_pos_delta_z_{0.02};
  double seed_roll_delta_{deg2rad(3.0)};
  double seed_pitch_delta_{deg2rad(2.0)};
  double seed_yaw_delta_{deg2rad(2.0)};

  bool enable_dynamic_figure8_refine_{false};
  bool publish_tf_{true};
  bool save_raw_cloud_{true};
  bool save_body_cloud_{true};
  bool save_visualization_cloud_{true};

  map<string, LidarConfig> lidar_configs_;
  map<string, LidarState> lidar_states_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<MultiMid360Calibrator>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    fprintf(stderr, "Fatal: %s\n", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
