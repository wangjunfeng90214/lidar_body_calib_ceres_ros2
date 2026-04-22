#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <pcl/common/point_tests.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;

namespace {

constexpr double kEps = 1e-12;
constexpr double kSmallAngleThreshold = 1e-10;
constexpr double kMaxReasonablePointCoordAbs = 1e6;

enum class ExitCode : int {
  kStrictAccept = 0,
  kRuntimeError = 1,
  kUsableButNotStrict = 2,
  kSolveFailed = 3,
};

enum class LogLevel : int {
  kInfo = 0,
  kWarn = 1,
  kError = 2,
};

struct RoiBox {
  double xmin{-std::numeric_limits<double>::infinity()};
  double xmax{ std::numeric_limits<double>::infinity()};
  double ymin{-std::numeric_limits<double>::infinity()};
  double ymax{ std::numeric_limits<double>::infinity()};
  double zmin{-std::numeric_limits<double>::infinity()};
  double zmax{ std::numeric_limits<double>::infinity()};

  bool contains(const Vector3d & p) const {
    return p.x() >= xmin && p.x() <= xmax &&
           p.y() >= ymin && p.y() <= ymax &&
           p.z() >= zmin && p.z() <= zmax;
  }
};

struct OrientedRoiBox {
  Vector3d center_lidar{0.0, 0.0, 0.0};
  std::array<Vector3d, 3> axes_lidar{Vector3d::UnitX(), Vector3d::UnitY(), Vector3d::UnitZ()};
  Vector3d half_extents{0.0, 0.0, 0.0};
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
};

struct PlaneMeasurement {
  std::size_t plane_index{0};
  Vector3d n_lidar{0.0, 0.0, 1.0};
  double d_lidar{0.0};
  Vector3d n0_body{0.0, 0.0, 1.0};
  double d0_body{0.0};
  std::size_t roi_point_count{0};
  std::size_t inlier_count{0};
  bool prior_flipped{false};
  bool meas_flipped{false};
  double init_normal_angle_err_deg{0.0};
  double init_plane_distance_err_m{0.0};
};

struct PerPlaneDiagnostics {
  std::size_t plane_index{0};
  Vector3d n_body_hat{0.0, 0.0, 1.0};
  double d_body_hat{0.0};
  double normal_angle_err_deg{0.0};
  double plane_distance_err_m{0.0};
};

struct SolverResult {
  bool success{false};
  bool strict_accept{false};
  std::string summary;
  double initial_cost{0.0};
  double final_cost{0.0};
  Vector3d rpy_rad{0.0, 0.0, 0.0};
  Vector3d t_m{0.0, 0.0, 0.0};
  Matrix3d R{Matrix3d::Identity()};
  int seed_index{-1};
  std::vector<PerPlaneDiagnostics> per_plane;
};

struct BoundsConfig {
  double dtheta_deg{5.0};
  double dD_m{0.05};
  Vector3d t_xyz_m{0.10, 0.10, 0.10};
  Vector3d rpy_deg{5.0, 5.0, 5.0};
};

struct GateConfig {
  double max_normal_angle_error_deg{20.0};
  double max_plane_distance_error_m{0.08};
};

struct AcceptanceConfig {
  double strict_max_final_cost{5.0};
  double strict_max_angle_error_deg{3.0};
  double strict_max_distance_error_m{0.02};
  double strict_max_translation_delta_m{0.10};
  double strict_max_rotation_delta_deg{10.0};
};

struct SeedConfig {
  double scale_fraction{0.4};
  Vector3d max_rpy_deg{2.0, 2.0, 2.0};
  Vector3d max_t_m{0.02, 0.02, 0.02};
  bool enable_diagonal_seeds{true};
};

struct Config {
  std::string lidar_ip;
  std::string pcd_path;
  std::string yaml_path;
  std::string angle_unit_cli;
  double ransac_distance_threshold{0.02};
  int ransac_iterations{1000};
  int min_plane_points{80};
  double wn{10.0};
  double wd{20.0};
  double wp{2.0};
  double wD{5.0};
  int multi_seed_mode{1};
  LogLevel log_level{LogLevel::kInfo};
  BoundsConfig bounds;
  GateConfig gates;
  AcceptanceConfig acceptance;
  SeedConfig seeds;
};

struct YamlInput {
  std::string angle_unit{"deg"};
  Vector3d t_init{0.0, 0.0, 0.0};
  Vector3d rpy_init_rad{0.0, 0.0, 0.0};
  Matrix3d R_init{Matrix3d::Identity()};
  std::array<Vector4d, 3> planes_body;
  std::array<RoiBox, 3> rois_body;
};

struct AlignedPlaneResult {
  Vector3d n{0.0, 0.0, 1.0};
  double d{0.0};
  bool flipped{false};
};

struct RansacFitResult {
  bool success{false};
  Vector3d n{0.0, 0.0, 1.0};
  double d{0.0};
  pcl::PointIndices::Ptr inliers{new pcl::PointIndices()};
};

struct SeedOffset {
  Vector3d rpy_offset_rad{0.0, 0.0, 0.0};
  Vector3d t_offset_m{0.0, 0.0, 0.0};
};

[[noreturn]] void fail(const std::string & msg) {
  throw std::runtime_error(msg);
}

const char * logLevelTag(LogLevel level) {
  switch (level) {
    case LogLevel::kInfo: return "info";
    case LogLevel::kWarn: return "warn";
    case LogLevel::kError: return "error";
  }
  return "info";
}

bool shouldLog(LogLevel cur, LogLevel msg_level) {
  return static_cast<int>(msg_level) >= static_cast<int>(cur);
}

void logMessage(const Config & cfg, LogLevel level, const std::string & msg) {
  if (!shouldLog(cfg.log_level, level)) return;
  std::ostream & os = (level == LogLevel::kError) ? std::cerr : std::cout;
  os << "[" << logLevelTag(level) << "] " << msg << "\n";
}

double deg2rad(double deg) {
  return deg * M_PI / 180.0;
}

double rad2deg(double rad) {
  return rad * 180.0 / M_PI;
}

bool isFiniteVec(const Vector3d & v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool isReasonablePoint(const Vector3d & v) {
  return isFiniteVec(v) &&
         std::abs(v.x()) < kMaxReasonablePointCoordAbs &&
         std::abs(v.y()) < kMaxReasonablePointCoordAbs &&
         std::abs(v.z()) < kMaxReasonablePointCoordAbs;
}

Vector3d normalizeVecOrFail(const Vector3d & v, const std::string & name) {
  const double n = v.norm();
  if (!std::isfinite(n) || n < kEps) {
    fail(name + " norm is too small or invalid");
  }
  return v / n;
}

double clampDot(double x) {
  return std::max(-1.0, std::min(1.0, x));
}

double angleBetweenNormalsDeg(const Vector3d & a, const Vector3d & b) {
  const Vector3d an = normalizeVecOrFail(a, "normal a");
  const Vector3d bn = normalizeVecOrFail(b, "normal b");
  const double c = clampDot(an.dot(bn));
  return rad2deg(std::acos(c));
}

template<typename T>
Eigen::Matrix<T, 3, 3> rpyToRTemplate(const Eigen::Matrix<T, 3, 1> & rpy) {
  const T roll = rpy.x();
  const T pitch = rpy.y();
  const T yaw = rpy.z();

  Eigen::Matrix<T, 3, 3> Rx, Ry, Rz;
  Rx << T(1), T(0), T(0),
        T(0), ceres::cos(roll), -ceres::sin(roll),
        T(0), ceres::sin(roll),  ceres::cos(roll);
  Ry << ceres::cos(pitch), T(0), ceres::sin(pitch),
        T(0), T(1), T(0),
        -ceres::sin(pitch), T(0), ceres::cos(pitch);
  Rz << ceres::cos(yaw), -ceres::sin(yaw), T(0),
        ceres::sin(yaw),  ceres::cos(yaw), T(0),
        T(0), T(0), T(1);
  return Rz * Ry * Rx;
}

Matrix3d rpyToR(const Vector3d & rpy) {
  return rpyToRTemplate<double>(rpy);
}

template<typename T>
Eigen::Matrix<T, 3, 3> rodriguesTemplate(const Eigen::Matrix<T, 3, 1> & w) {
  const T theta2 = w.squaredNorm();
  if (theta2 < T(kSmallAngleThreshold)) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }
  T R_arr[9];
  const T angle_axis[3] = {w.x(), w.y(), w.z()};
  ceres::AngleAxisToRotationMatrix(angle_axis, R_arr);
  Eigen::Matrix<T, 3, 3> R;
  R << R_arr[0], R_arr[1], R_arr[2],
       R_arr[3], R_arr[4], R_arr[5],
       R_arr[6], R_arr[7], R_arr[8];
  return R;
}

std::array<Vector3d, 8> orientedRoiCornersLidar(const OrientedRoiBox & roi) {
  std::array<Vector3d, 8> corners;
  int idx = 0;
  for (int sx : {-1, 1}) {
    for (int sy : {-1, 1}) {
      for (int sz : {-1, 1}) {
        corners[idx++] = roi.center_lidar
          + static_cast<double>(sx) * roi.half_extents.x() * roi.axes_lidar[0]
          + static_cast<double>(sy) * roi.half_extents.y() * roi.axes_lidar[1]
          + static_cast<double>(sz) * roi.half_extents.z() * roi.axes_lidar[2];
      }
    }
  }
  return corners;
}

OrientedRoiBox transformRoiBoxBodyToLidar(
  const RoiBox & roi_body,
  const Matrix3d & R_lidar_to_body,
  const Vector3d & t_lidar_to_body)
{
  OrientedRoiBox roi;
  const Vector3d center_body(
    0.5 * (roi_body.xmin + roi_body.xmax),
    0.5 * (roi_body.ymin + roi_body.ymax),
    0.5 * (roi_body.zmin + roi_body.zmax));

  roi.center_lidar = R_lidar_to_body.transpose() * (center_body - t_lidar_to_body);
  roi.axes_lidar[0] = normalizeVecOrFail(R_lidar_to_body.transpose() * Vector3d::UnitX(), "roi axis x");
  roi.axes_lidar[1] = normalizeVecOrFail(R_lidar_to_body.transpose() * Vector3d::UnitY(), "roi axis y");
  roi.axes_lidar[2] = normalizeVecOrFail(R_lidar_to_body.transpose() * Vector3d::UnitZ(), "roi axis z");
  roi.half_extents = Vector3d(
    0.5 * (roi_body.xmax - roi_body.xmin),
    0.5 * (roi_body.ymax - roi_body.ymin),
    0.5 * (roi_body.zmax - roi_body.zmin));
  return roi;
}

AxisAlignedRoiBox orientedRoiToAabb(const OrientedRoiBox & roi) {
  AxisAlignedRoiBox aabb;
  const auto corners = orientedRoiCornersLidar(roi);
  for (const auto & c : corners) {
    aabb.min_corner = aabb.min_corner.cwiseMin(c);
    aabb.max_corner = aabb.max_corner.cwiseMax(c);
  }
  return aabb;
}

AxisAlignedRoiBox unionAabb(const std::vector<AxisAlignedRoiBox> & boxes) {
  AxisAlignedRoiBox out;
  bool has_any = false;
  for (const auto & box : boxes) {
    if (!box.isValid()) continue;
    if (!has_any) {
      out = box;
      has_any = true;
    } else {
      out.min_corner = out.min_corner.cwiseMin(box.min_corner);
      out.max_corner = out.max_corner.cwiseMax(box.max_corner);
    }
  }
  return out;
}

std::pair<Vector3d, double> transformPlaneToBody(
  const Matrix3d & R,
  const Vector3d & t,
  const Vector3d & n_lidar,
  double d_lidar)
{
  const Vector3d n_body = normalizeVecOrFail(R * n_lidar, "transformPlaneToBody normal");
  const double d_body = d_lidar - n_body.dot(t);
  return {n_body, d_body};
}

std::pair<Vector3d, double> transformPlaneBodyToLidar(
  const Matrix3d & R,
  const Vector3d & t,
  const Vector3d & n_body,
  double d_body)
{
  const Vector3d n_lidar = normalizeVecOrFail(R.transpose() * n_body, "transformPlaneBodyToLidar normal");
  const double d_lidar = d_body + n_body.dot(t);
  return {n_lidar, d_lidar};
}

std::pair<Vector3d, double> orientPlaneTowardLidar(
  const Vector3d & n_body_in,
  double d_body_in,
  const Vector3d & lidar_pos_body)
{
  Vector3d n = normalizeVecOrFail(n_body_in, "orientPlaneTowardLidar input normal");
  double d = d_body_in;
  const Vector3d p0 = -d * n;
  const Vector3d toward_lidar = lidar_pos_body - p0;
  if (n.dot(toward_lidar) < 0.0) {
    n = -n;
    d = -d;
  }
  return {n, d};
}

AlignedPlaneResult alignPlaneToPrior(
  const Vector3d & n_meas_in,
  double d_meas_in,
  const Vector3d & n_prior_in)
{
  AlignedPlaneResult out;
  out.n = normalizeVecOrFail(n_meas_in, "alignPlaneToPrior measured normal");
  out.d = d_meas_in;
  if (out.n.dot(normalizeVecOrFail(n_prior_in, "alignPlaneToPrior prior normal")) < 0.0) {
    out.n = -out.n;
    out.d = -out.d;
    out.flipped = true;
  }
  return out;
}

std::string planeEquationStr(const Vector3d & n, double d) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6)
      << n.x() << " * x + "
      << n.y() << " * y + "
      << n.z() << " * z + "
      << d << " = 0";
  return oss.str();
}

RansacFitResult fitPlaneRansac(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
  double distance_threshold,
  int max_iterations)
{
  RansacFitResult out;
  if (!cloud || cloud->size() < 3) return out;

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(distance_threshold);
  seg.setMaxIterations(max_iterations);
  seg.setInputCloud(cloud);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::ModelCoefficients coeff;
  seg.segment(*inliers, coeff);

  if (inliers->indices.size() < 3 || coeff.values.size() != 4) return out;

  Vector3d n(coeff.values[0], coeff.values[1], coeff.values[2]);
  const double norm = n.norm();
  if (norm < kEps || !std::isfinite(norm)) return out;

  out.success = true;
  out.n = n / norm;
  out.d = static_cast<double>(coeff.values[3]) / norm;
  out.inliers = inliers;
  return out;
}

bool fitPlaneLSE(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
  const pcl::PointIndices::Ptr & inliers,
  Vector3d & n_fit,
  double & d_fit)
{
  if (!cloud || !inliers || inliers->indices.size() < 3) return false;

  Eigen::MatrixXd X(inliers->indices.size(), 3);
  for (std::size_t i = 0; i < inliers->indices.size(); ++i) {
    const auto & pt = cloud->points[inliers->indices[i]];
    X(static_cast<Eigen::Index>(i), 0) = static_cast<double>(pt.x);
    X(static_cast<Eigen::Index>(i), 1) = static_cast<double>(pt.y);
    X(static_cast<Eigen::Index>(i), 2) = static_cast<double>(pt.z);
  }

  const Vector3d centroid = X.colwise().mean();
  Eigen::MatrixXd Xc = X.rowwise() - centroid.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xc, Eigen::ComputeThinV);
  Vector3d n = svd.matrixV().col(2);
  const double norm = n.norm();
  if (norm < kEps || !std::isfinite(norm)) return false;

  n /= norm;
  n_fit = n;
  d_fit = -n.dot(centroid);
  return std::isfinite(d_fit) && isFiniteVec(n_fit);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr loadPcd(const std::string & path) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPCDFile(path, *cloud) != 0) {
    fail("failed to load pcd: " + path);
  }
  if (cloud->empty()) {
    fail("pcd is empty: " + path);
  }
  return cloud;
}

RoiBox parseRoiBox(const YAML::Node & node) {
  if (!node || !node.IsMap()) {
    fail("plane_rois entry must be a map with xmin/xmax/ymin/ymax/zmin/zmax");
  }
  RoiBox roi;
  roi.xmin = node["xmin"].as<double>();
  roi.xmax = node["xmax"].as<double>();
  roi.ymin = node["ymin"].as<double>();
  roi.ymax = node["ymax"].as<double>();
  roi.zmin = node["zmin"].as<double>();
  roi.zmax = node["zmax"].as<double>();
  if (!(std::isfinite(roi.xmin) && std::isfinite(roi.xmax) &&
        std::isfinite(roi.ymin) && std::isfinite(roi.ymax) &&
        std::isfinite(roi.zmin) && std::isfinite(roi.zmax))) {
    fail("ROI bounds must be finite");
  }
  if (roi.xmin > roi.xmax || roi.ymin > roi.ymax || roi.zmin > roi.zmax) {
    fail("invalid ROI bounds: require xmin<=xmax, ymin<=ymax, zmin<=zmax");
  }
  return roi;
}

double readAngleWithUnit(const YAML::Node & parent, const char * key, const std::string & angle_unit) {
  if (!parent[key]) fail(std::string("missing angle field: ") + key);
  const double v = parent[key].as<double>();
  if (angle_unit == "deg") return deg2rad(v);
  if (angle_unit == "rad") return v;
  fail("angle_unit must be 'deg' or 'rad'");
}

void validateNonNegative(const std::string & name, double value) {
  if (!std::isfinite(value) || value < 0.0) {
    fail(name + " must be finite and >= 0");
  }
}

void validateConfig(const Config & cfg) {
  validateNonNegative("ransac_distance_threshold", cfg.ransac_distance_threshold);
  if (cfg.ransac_iterations <= 0) fail("ransac_iterations must be > 0");
  if (cfg.min_plane_points < 3) fail("min_plane_points must be >= 3");
  validateNonNegative("wn", cfg.wn);
  validateNonNegative("wd", cfg.wd);
  validateNonNegative("wp", cfg.wp);
  validateNonNegative("wD", cfg.wD);
  validateNonNegative("bounds.dtheta_deg", cfg.bounds.dtheta_deg);
  validateNonNegative("bounds.dD_m", cfg.bounds.dD_m);
  for (int i = 0; i < 3; ++i) {
    validateNonNegative("bounds.t_xyz_m", cfg.bounds.t_xyz_m[i]);
    validateNonNegative("bounds.rpy_deg", cfg.bounds.rpy_deg[i]);
    validateNonNegative("seed.scale_fraction", cfg.seeds.scale_fraction);
    validateNonNegative("seed.max_rpy_deg", cfg.seeds.max_rpy_deg[i]);
    validateNonNegative("seed.max_t_m", cfg.seeds.max_t_m[i]);
  }
  validateNonNegative("gates.max_normal_angle_error_deg", cfg.gates.max_normal_angle_error_deg);
  validateNonNegative("gates.max_plane_distance_error_m", cfg.gates.max_plane_distance_error_m);
  validateNonNegative("acceptance.strict_max_final_cost", cfg.acceptance.strict_max_final_cost);
  validateNonNegative("acceptance.strict_max_angle_error_deg", cfg.acceptance.strict_max_angle_error_deg);
  validateNonNegative("acceptance.strict_max_distance_error_m", cfg.acceptance.strict_max_distance_error_m);
  validateNonNegative("acceptance.strict_max_translation_delta_m", cfg.acceptance.strict_max_translation_delta_m);
  validateNonNegative("acceptance.strict_max_rotation_delta_deg", cfg.acceptance.strict_max_rotation_delta_deg);
  if (cfg.multi_seed_mode < 0) fail("multi_seed_mode must be >= 0");
}

void loadYamlOptionalConfig(const YAML::Node & node, Config & cfg) {
  if (node["bounds"]) {
    const auto b = node["bounds"];
    if (b["dtheta_deg"]) cfg.bounds.dtheta_deg = b["dtheta_deg"].as<double>();
    if (b["dD_m"]) cfg.bounds.dD_m = b["dD_m"].as<double>();
    if (b["t_xyz_m"]) {
      const auto a = b["t_xyz_m"].as<std::vector<double>>();
      if (a.size() != 3) fail("bounds.t_xyz_m must have 3 elements");
      cfg.bounds.t_xyz_m = Vector3d(a[0], a[1], a[2]);
    }
    if (b["rpy_deg"]) {
      const auto a = b["rpy_deg"].as<std::vector<double>>();
      if (a.size() != 3) fail("bounds.rpy_deg must have 3 elements");
      cfg.bounds.rpy_deg = Vector3d(a[0], a[1], a[2]);
    }
  }
  if (node["gates"]) {
    const auto g = node["gates"];
    if (g["max_normal_angle_error_deg"]) cfg.gates.max_normal_angle_error_deg = g["max_normal_angle_error_deg"].as<double>();
    if (g["max_plane_distance_error_m"]) cfg.gates.max_plane_distance_error_m = g["max_plane_distance_error_m"].as<double>();
  }
  if (node["acceptance"]) {
    const auto a = node["acceptance"];
    if (a["strict_max_final_cost"]) cfg.acceptance.strict_max_final_cost = a["strict_max_final_cost"].as<double>();
    if (a["strict_max_angle_error_deg"]) cfg.acceptance.strict_max_angle_error_deg = a["strict_max_angle_error_deg"].as<double>();
    if (a["strict_max_distance_error_m"]) cfg.acceptance.strict_max_distance_error_m = a["strict_max_distance_error_m"].as<double>();
    if (a["strict_max_translation_delta_m"]) cfg.acceptance.strict_max_translation_delta_m = a["strict_max_translation_delta_m"].as<double>();
    if (a["strict_max_rotation_delta_deg"]) cfg.acceptance.strict_max_rotation_delta_deg = a["strict_max_rotation_delta_deg"].as<double>();
  }
  if (node["multi_seed_mode"]) cfg.multi_seed_mode = node["multi_seed_mode"].as<int>();
  if (node["seed_config"]) {
    const auto s = node["seed_config"];
    if (s["scale_fraction"]) cfg.seeds.scale_fraction = s["scale_fraction"].as<double>();
    if (s["max_rpy_deg"]) {
      const auto a = s["max_rpy_deg"].as<std::vector<double>>();
      if (a.size() != 3) fail("seed_config.max_rpy_deg must have 3 elements");
      cfg.seeds.max_rpy_deg = Vector3d(a[0], a[1], a[2]);
    }
    if (s["max_t_m"]) {
      const auto a = s["max_t_m"].as<std::vector<double>>();
      if (a.size() != 3) fail("seed_config.max_t_m must have 3 elements");
      cfg.seeds.max_t_m = Vector3d(a[0], a[1], a[2]);
    }
    if (s["enable_diagonal_seeds"]) cfg.seeds.enable_diagonal_seeds = s["enable_diagonal_seeds"].as<bool>();
  }
}

YamlInput loadYamlInput(const std::string & yaml_path, const std::string & lidar_ip, Config & cfg) {
  const YAML::Node root = YAML::LoadFile(yaml_path);
  if (!root[lidar_ip]) {
    fail("lidar ip not found in yaml: " + lidar_ip);
  }
  const YAML::Node node = root[lidar_ip];

  YamlInput out;
  out.angle_unit = cfg.angle_unit_cli.empty()
      ? (node["angle_unit"] ? node["angle_unit"].as<std::string>() : std::string("deg"))
      : cfg.angle_unit_cli;
  loadYamlOptionalConfig(node, cfg);
  validateConfig(cfg);

  out.t_init = Vector3d(node["x"].as<double>(), node["y"].as<double>(), node["z"].as<double>());
  out.rpy_init_rad = Vector3d(
    readAngleWithUnit(node, "roll", out.angle_unit),
    readAngleWithUnit(node, "pitch", out.angle_unit),
    readAngleWithUnit(node, "yaw", out.angle_unit));
  out.R_init = rpyToR(out.rpy_init_rad);

  const YAML::Node planes = node["planes"];
  const YAML::Node rois = node["plane_rois"];
  if (!planes || !planes.IsSequence() || planes.size() < 3) {
    fail("yaml must contain at least 3 planes");
  }
  if (!rois || !rois.IsSequence() || rois.size() < 3) {
    fail("yaml must contain at least 3 plane_rois");
  }

  for (std::size_t i = 0; i < 3; ++i) {
    const auto coeff = planes[i].as<std::vector<double>>();
    if (coeff.size() != 4) fail("plane entry must be [a,b,c,d]");
    Vector4d p(coeff[0], coeff[1], coeff[2], coeff[3]);
    const double norm = p.head<3>().norm();
    if (norm < kEps || !std::isfinite(norm)) fail("plane normal norm too small");
    out.planes_body[i] = Vector4d(p[0] / norm, p[1] / norm, p[2] / norm, p[3] / norm);
    out.rois_body[i] = parseRoiBox(rois[i]);
  }
  return out;
}

std::vector<PlaneMeasurement> extractPlaneMeasurements(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud,
  const YamlInput & yin,
  const Config & cfg)
{
  std::vector<OrientedRoiBox> rois_lidar;
  std::vector<AxisAlignedRoiBox> rois_aabb;
  rois_lidar.reserve(3);
  rois_aabb.reserve(3);

  for (std::size_t i = 0; i < 3; ++i) {
    const auto roi_lidar = transformRoiBoxBodyToLidar(yin.rois_body[i], yin.R_init, yin.t_init);
    rois_lidar.push_back(roi_lidar);
    rois_aabb.push_back(orientedRoiToAabb(roi_lidar));
  }

  const AxisAlignedRoiBox union_box = unionAabb(rois_aabb);
  if (!union_box.isValid()) fail("invalid union AABB from 3 ROIs");

  pcl::PointCloud<pcl::PointXYZ>::Ptr union_cloud_lidar(new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<Vector3d> union_points_body;
  union_points_body.reserve(raw_cloud->size());

  for (const auto & pt : raw_cloud->points) {
    if (!pcl::isFinite(pt)) continue;
    const Vector3d p_lidar(pt.x, pt.y, pt.z);
    if (!isReasonablePoint(p_lidar)) continue;
    if (!union_box.contains(p_lidar)) continue;
    const Vector3d p_body = yin.R_init * p_lidar + yin.t_init;
    if (!isReasonablePoint(p_body)) continue;
    union_cloud_lidar->points.push_back(pt);
    union_points_body.push_back(p_body);
  }

  union_cloud_lidar->width = static_cast<uint32_t>(union_cloud_lidar->points.size());
  union_cloud_lidar->height = 1;
  union_cloud_lidar->is_dense = false;

  if (union_cloud_lidar->empty()) fail("no points inside union AABB");

  {
    std::ostringstream oss;
    oss << "union lidar AABB min=" << union_box.min_corner.transpose()
        << ", max=" << union_box.max_corner.transpose();
    logMessage(cfg, LogLevel::kInfo, oss.str());
  }
  {
    std::ostringstream oss;
    oss << "raw points=" << raw_cloud->size() << " union_aabb_points=" << union_cloud_lidar->size();
    logMessage(cfg, LogLevel::kInfo, oss.str());
  }

  std::vector<PlaneMeasurement> measurements;
  measurements.reserve(3);

  for (std::size_t i = 0; i < 3; ++i) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (std::size_t k = 0; k < union_cloud_lidar->points.size(); ++k) {
      if (yin.rois_body[i].contains(union_points_body[k])) {
        roi_cloud->points.push_back(union_cloud_lidar->points[k]);
      }
    }
    roi_cloud->width = static_cast<uint32_t>(roi_cloud->points.size());
    roi_cloud->height = 1;
    roi_cloud->is_dense = false;

    {
      std::ostringstream oss;
      oss << "roi " << i << ": body hit points=" << roi_cloud->size();
      logMessage(cfg, LogLevel::kInfo, oss.str());
    }
    if (static_cast<int>(roi_cloud->size()) < cfg.min_plane_points) {
      fail("roi " + std::to_string(i) + " has too few points for plane fitting");
    }

    const auto ransac_fit = fitPlaneRansac(roi_cloud, cfg.ransac_distance_threshold, cfg.ransac_iterations);
    if (!ransac_fit.success) fail("RANSAC failed for roi " + std::to_string(i));

    Vector3d n_fit = ransac_fit.n;
    double d_fit = ransac_fit.d;
    if (!fitPlaneLSE(roi_cloud, ransac_fit.inliers, n_fit, d_fit)) {
      fail("LSE refine failed for roi " + std::to_string(i));
    }

    const Vector4d prior_plane = yin.planes_body[i];
    const Vector3d prior_input_n = normalizeVecOrFail(prior_plane.head<3>(), "yaml prior plane normal");
    const double prior_input_d = prior_plane[3];
    const auto oriented_prior = orientPlaneTowardLidar(prior_input_n, prior_input_d, yin.t_init);
    const bool prior_flipped = oriented_prior.first.dot(prior_input_n) < 0.0;

    const auto pred_lidar_plane = transformPlaneBodyToLidar(
      yin.R_init, yin.t_init, oriented_prior.first, oriented_prior.second);
    const auto aligned_meas = alignPlaneToPrior(n_fit, d_fit, pred_lidar_plane.first);

    const double init_normal_angle_err_deg = angleBetweenNormalsDeg(aligned_meas.n, pred_lidar_plane.first);
    const double init_plane_distance_err_m = std::abs(aligned_meas.d - pred_lidar_plane.second);
    if (init_normal_angle_err_deg > cfg.gates.max_normal_angle_error_deg) {
      fail("roi " + std::to_string(i) + " normal angle error too large: " + std::to_string(init_normal_angle_err_deg) + " deg");
    }
    if (init_plane_distance_err_m > cfg.gates.max_plane_distance_error_m) {
      fail("roi " + std::to_string(i) + " plane distance error too large: " + std::to_string(init_plane_distance_err_m) + " m");
    }

    PlaneMeasurement m;
    m.plane_index = i;
    m.n_lidar = aligned_meas.n;
    m.d_lidar = aligned_meas.d;
    m.n0_body = oriented_prior.first;
    m.d0_body = oriented_prior.second;
    m.roi_point_count = roi_cloud->size();
    m.inlier_count = ransac_fit.inliers->indices.size();
    m.prior_flipped = prior_flipped;
    m.meas_flipped = aligned_meas.flipped;
    m.init_normal_angle_err_deg = init_normal_angle_err_deg;
    m.init_plane_distance_err_m = init_plane_distance_err_m;
    measurements.push_back(m);

    std::ostringstream oss;
    oss << "plane " << i
        << ": nL=" << m.n_lidar.transpose()
        << " dL=" << std::fixed << std::setprecision(6) << m.d_lidar
        << " inliers=" << m.inlier_count
        << " prior_flipped=" << (m.prior_flipped ? "true" : "false")
        << " meas_flipped=" << (m.meas_flipped ? "true" : "false")
        << " d0=" << m.d0_body
        << " init_angle_err_deg=" << m.init_normal_angle_err_deg
        << " init_dist_err_m=" << m.init_plane_distance_err_m;
    logMessage(cfg, LogLevel::kInfo, oss.str());
  }

  return measurements;
}

struct PlaneResidual {
  PlaneResidual(
    const Vector3d & n_lidar_meas,
    double d_lidar_meas,
    const Vector3d & n0_body_prior,
    double d0_body_prior,
    double wn,
    double wd,
    double wp,
    double wD)
  : nL_(n_lidar_meas), dL_(d_lidar_meas), n0_(n0_body_prior), d0_(d0_body_prior),
    wn_(wn), wd_(wd), wp_(wp), wD_(wD) {}

  template<typename T>
  bool operator()(const T * const rpy, const T * const t,
                  const T * const dtheta, const T * const dD,
                  T * residuals) const {
    const Eigen::Matrix<T, 3, 1> rpy_v(rpy[0], rpy[1], rpy[2]);
    const Eigen::Matrix<T, 3, 1> t_v(t[0], t[1], t[2]);
    const Eigen::Matrix<T, 3, 1> dtheta_v(dtheta[0], dtheta[1], dtheta[2]);

    const Eigen::Matrix<T, 3, 3> R = rpyToRTemplate<T>(rpy_v);
    const Eigen::Matrix<T, 3, 3> R_delta = rodriguesTemplate<T>(dtheta_v);

    Eigen::Matrix<T, 3, 1> ni = R_delta * n0_.cast<T>();
    ni /= ceres::sqrt(ni.squaredNorm() + T(1e-12));
    const T di = T(d0_) + dD[0];

    Eigen::Matrix<T, 3, 1> n_body_hat = R * nL_.cast<T>();
    n_body_hat /= ceres::sqrt(n_body_hat.squaredNorm() + T(1e-12));
    const T d_body_hat = T(dL_) - n_body_hat.dot(t_v);

    const Eigen::Matrix<T, 3, 1> cross_err = n_body_hat.cross(ni);
    residuals[0] = T(wn_) * cross_err[0];
    residuals[1] = T(wn_) * cross_err[1];
    residuals[2] = T(wn_) * cross_err[2];
    residuals[3] = T(wd_) * (d_body_hat - di);
    residuals[4] = T(wp_) * dtheta[0];
    residuals[5] = T(wp_) * dtheta[1];
    residuals[6] = T(wp_) * dtheta[2];
    residuals[7] = T(wD_) * dD[0];
    return true;
  }

  static ceres::CostFunction * Create(
    const Vector3d & n_lidar_meas,
    double d_lidar_meas,
    const Vector3d & n0_body_prior,
    double d0_body_prior,
    double wn,
    double wd,
    double wp,
    double wD)
  {
    return new ceres::AutoDiffCostFunction<PlaneResidual, 8, 3, 3, 3, 1>(
      new PlaneResidual(n_lidar_meas, d_lidar_meas, n0_body_prior, d0_body_prior, wn, wd, wp, wD));
  }

  Vector3d nL_;
  double dL_;
  Vector3d n0_;
  double d0_;
  double wn_;
  double wd_;
  double wp_;
  double wD_;
};

std::vector<PerPlaneDiagnostics> evaluatePerPlaneDiagnostics(
  const std::vector<PlaneMeasurement> & measurements,
  const SolverResult & result)
{
  std::vector<PerPlaneDiagnostics> out;
  out.reserve(measurements.size());
  for (const auto & m : measurements) {
    const auto body_plane = transformPlaneToBody(result.R, result.t_m, m.n_lidar, m.d_lidar);
    PerPlaneDiagnostics d;
    d.plane_index = m.plane_index;
    d.n_body_hat = body_plane.first;
    d.d_body_hat = body_plane.second;
    d.normal_angle_err_deg = angleBetweenNormalsDeg(body_plane.first, m.n0_body);
    d.plane_distance_err_m = std::abs(body_plane.second - m.d0_body);
    out.push_back(d);
  }
  return out;
}

bool passesStrictAcceptance(
  const SolverResult & result,
  const std::vector<PlaneMeasurement> & measurements,
  const YamlInput & yin,
  const Config & cfg)
{
  if (!result.success) return false;
  if (result.final_cost > cfg.acceptance.strict_max_final_cost) return false;
  if ((result.t_m - yin.t_init).norm() > cfg.acceptance.strict_max_translation_delta_m) return false;
  if (rad2deg((result.rpy_rad - yin.rpy_init_rad).norm()) > cfg.acceptance.strict_max_rotation_delta_deg) return false;
  for (const auto & d : result.per_plane) {
    if (d.normal_angle_err_deg > cfg.acceptance.strict_max_angle_error_deg) return false;
    if (d.plane_distance_err_m > cfg.acceptance.strict_max_distance_error_m) return false;
  }
  return result.per_plane.size() == measurements.size();
}

std::vector<SeedOffset> buildSeedOffsets(const Config & cfg) {
  std::vector<SeedOffset> seeds;
  seeds.push_back({});
  if (cfg.multi_seed_mode == 0) return seeds;

  Vector3d r_deg = cfg.bounds.rpy_deg * cfg.seeds.scale_fraction;
  Vector3d t_m = cfg.bounds.t_xyz_m * cfg.seeds.scale_fraction;
  for (int i = 0; i < 3; ++i) {
    r_deg[i] = std::min(r_deg[i], cfg.seeds.max_rpy_deg[i]);
    t_m[i] = std::min(t_m[i], cfg.seeds.max_t_m[i]);
  }
  const Vector3d r_rad(deg2rad(r_deg.x()), deg2rad(r_deg.y()), deg2rad(r_deg.z()));

  auto push_axis_pair = [&](int axis) {
    if (r_rad[axis] > 0.0) {
      SeedOffset sp, sn;
      sp.rpy_offset_rad[axis] =  r_rad[axis];
      sn.rpy_offset_rad[axis] = -r_rad[axis];
      seeds.push_back(sp);
      seeds.push_back(sn);
    }
    if (t_m[axis] > 0.0) {
      SeedOffset sp, sn;
      sp.t_offset_m[axis] =  t_m[axis];
      sn.t_offset_m[axis] = -t_m[axis];
      seeds.push_back(sp);
      seeds.push_back(sn);
    }
  };

  for (int axis = 0; axis < 3; ++axis) push_axis_pair(axis);

  if (cfg.seeds.enable_diagonal_seeds) {
    for (int sx : {-1, 1}) {
      for (int sy : {-1, 1}) {
        SeedOffset s;
        s.rpy_offset_rad = Vector3d(sx * r_rad.x(), sy * r_rad.y(), 0.0);
        if (s.rpy_offset_rad.norm() > 0.0) seeds.push_back(s);
      }
    }
    for (int sx : {-1, 1}) {
      for (int sy : {-1, 1}) {
        SeedOffset s;
        s.t_offset_m = Vector3d(sx * t_m.x(), sy * t_m.y(), 0.0);
        if (s.t_offset_m.norm() > 0.0) seeds.push_back(s);
      }
    }
  }
  return seeds;
}

SolverResult solveSingleSeed(
  const std::vector<PlaneMeasurement> & measurements,
  const YamlInput & yin,
  const Config & cfg,
  const SeedOffset & seed,
  int seed_index)
{
  double rpy[3] = {
    yin.rpy_init_rad.x() + seed.rpy_offset_rad.x(),
    yin.rpy_init_rad.y() + seed.rpy_offset_rad.y(),
    yin.rpy_init_rad.z() + seed.rpy_offset_rad.z()};
  double t[3] = {
    yin.t_init.x() + seed.t_offset_m.x(),
    yin.t_init.y() + seed.t_offset_m.y(),
    yin.t_init.z() + seed.t_offset_m.z()};
  std::vector<std::array<double, 3>> dtheta_blocks(3, {0.0, 0.0, 0.0});
  std::vector<double> dD_blocks(3, 0.0);

  ceres::Problem problem;
  for (std::size_t i = 0; i < measurements.size(); ++i) {
    const auto & m = measurements[i];
    ceres::CostFunction * cost = PlaneResidual::Create(
      m.n_lidar, m.d_lidar, m.n0_body, m.d0_body,
      cfg.wn, cfg.wd, cfg.wp, cfg.wD);
    problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), rpy, t, dtheta_blocks[i].data(), &dD_blocks[i]);

    const double dtheta_lim = deg2rad(cfg.bounds.dtheta_deg);
    for (int j = 0; j < 3; ++j) {
      problem.SetParameterLowerBound(dtheta_blocks[i].data(), j, -dtheta_lim);
      problem.SetParameterUpperBound(dtheta_blocks[i].data(), j,  dtheta_lim);
    }
    problem.SetParameterLowerBound(&dD_blocks[i], 0, -cfg.bounds.dD_m);
    problem.SetParameterUpperBound(&dD_blocks[i], 0,  cfg.bounds.dD_m);
  }

  for (int j = 0; j < 3; ++j) {
    problem.SetParameterLowerBound(t, j, yin.t_init[j] - cfg.bounds.t_xyz_m[j]);
    problem.SetParameterUpperBound(t, j, yin.t_init[j] + cfg.bounds.t_xyz_m[j]);
    const double lim = deg2rad(cfg.bounds.rpy_deg[j]);
    const double lower = yin.rpy_init_rad[j] - lim;
    const double upper = yin.rpy_init_rad[j] + lim;
    const double seeded = std::min(std::max(rpy[j], lower), upper);
    rpy[j] = seeded;
    problem.SetParameterLowerBound(rpy, j, lower);
    problem.SetParameterUpperBound(rpy, j, upper);
  }

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

  SolverResult out;
  out.success = summary.IsSolutionUsable();
  out.summary = summary.BriefReport();
  out.initial_cost = summary.initial_cost;
  out.final_cost = summary.final_cost;
  out.rpy_rad = Vector3d(rpy[0], rpy[1], rpy[2]);
  out.t_m = Vector3d(t[0], t[1], t[2]);
  out.R = rpyToR(out.rpy_rad);
  out.seed_index = seed_index;
  out.per_plane = evaluatePerPlaneDiagnostics(measurements, out);
  out.strict_accept = passesStrictAcceptance(out, measurements, yin, cfg);
  return out;
}

SolverResult solveCeres(
  const std::vector<PlaneMeasurement> & measurements,
  const YamlInput & yin,
  const Config & cfg)
{
  if (measurements.size() != 3) fail("solveCeres requires exactly 3 plane measurements");

  const auto seeds = buildSeedOffsets(cfg);
  SolverResult best;
  bool has_best = false;

  for (std::size_t i = 0; i < seeds.size(); ++i) {
    const SolverResult cur = solveSingleSeed(measurements, yin, cfg, seeds[i], static_cast<int>(i));
    std::ostringstream oss;
    oss << "seed " << i
        << " final_cost=" << cur.final_cost
        << " strict_accept=" << (cur.strict_accept ? "true" : "false")
        << " t=" << cur.t_m.transpose()
        << " rpy_deg=" << rad2deg(cur.rpy_rad.x()) << ","
        << rad2deg(cur.rpy_rad.y()) << ","
        << rad2deg(cur.rpy_rad.z());
    logMessage(cfg, LogLevel::kInfo, oss.str());

    if (!has_best) {
      best = cur;
      has_best = true;
      continue;
    }

    const auto better = [&](const SolverResult & a, const SolverResult & b) {
      if (a.strict_accept != b.strict_accept) return a.strict_accept;
      if (a.success != b.success) return a.success;
      return a.final_cost < b.final_cost;
    };
    if (better(cur, best)) best = cur;
  }

  return best;
}

void printExitCodeGuide() {
  std::cout << "\n------------------------------------------------------------------------\n";
  std::cout << "exit code meaning\n";
  std::cout << "------------------------------------------------------------------------\n";
  std::cout << "0: success and strict_accept=true\n";
  std::cout << "1: runtime/config/input error\n";
  std::cout << "2: solver returned usable result, but strict acceptance failed\n";
  std::cout << "3: solver failed or result unusable\n";
}

void printResult(const std::vector<PlaneMeasurement> & measurements, const SolverResult & result, const YamlInput & yin) {
  std::cout << "\n------------------------------------------------------------------------\n";
  std::cout << "Ceres result\n";
  std::cout << "------------------------------------------------------------------------\n";
  std::cout << "success: " << (result.success ? "true" : "false") << "\n";
  std::cout << "strict_accept: " << (result.strict_accept ? "true" : "false") << "\n";
  std::cout << "seed_index: " << result.seed_index << "\n";
  std::cout << "summary: " << result.summary << "\n";
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "initial_cost: " << result.initial_cost << "\n";
  std::cout << "final_cost  : " << result.final_cost << "\n";

  std::cout << "\n------------------------------------------------------------------------\n";
  std::cout << "解算后的外参下，雷达在小车坐标系下的位置和姿态\n";
  std::cout << "------------------------------------------------------------------------\n";
  std::cout << std::setprecision(6);
  std::cout << "position in body (m): x=" << result.t_m.x()
            << ", y=" << result.t_m.y()
            << ", z=" << result.t_m.z() << "\n";
  std::cout << "attitude in body (deg): roll=" << rad2deg(result.rpy_rad.x())
            << ", pitch=" << rad2deg(result.rpy_rad.y())
            << ", yaw=" << rad2deg(result.rpy_rad.z()) << "\n";
  std::cout << std::setprecision(10);
  std::cout << "attitude in body (rad): roll=" << result.rpy_rad.x()
            << ", pitch=" << result.rpy_rad.y()
            << ", yaw=" << result.rpy_rad.z() << "\n";
  std::cout << std::setprecision(6);
  std::cout << "delta t from init (m): " << (result.t_m - yin.t_init).transpose() << "\n";
  std::cout << "delta rpy from init (deg): "
            << rad2deg(result.rpy_rad.x() - yin.rpy_init_rad.x()) << ", "
            << rad2deg(result.rpy_rad.y() - yin.rpy_init_rad.y()) << ", "
            << rad2deg(result.rpy_rad.z() - yin.rpy_init_rad.z()) << "\n";

  std::cout << "\n------------------------------------------------------------------------\n";
  std::cout << "optimized rotation matrix R (lidar -> body)\n";
  std::cout << "------------------------------------------------------------------------\n";
  std::cout << std::setprecision(10) << result.R << "\n";

  std::cout << "\n------------------------------------------------------------------------\n";
  std::cout << "measured planes in lidar frame used by Ceres\n";
  std::cout << "------------------------------------------------------------------------\n";
  std::cout << std::setprecision(6);
  for (const auto & m : measurements) {
    std::cout << "plane " << m.plane_index << ": "
              << planeEquationStr(m.n_lidar, m.d_lidar)
              << " | roi_points=" << m.roi_point_count
              << " | inliers=" << m.inlier_count
              << " | init_angle_err_deg=" << m.init_normal_angle_err_deg
              << " | init_dist_err_m=" << m.init_plane_distance_err_m
              << "\n";
  }

  std::cout << "\n------------------------------------------------------------------------\n";
  std::cout << "fitted planes transformed to body frame after optimization\n";
  std::cout << "------------------------------------------------------------------------\n";
  for (const auto & d : result.per_plane) {
    std::cout << "fitted plane " << d.plane_index << ": "
              << planeEquationStr(d.n_body_hat, d.d_body_hat)
              << " | final_angle_err_deg=" << d.normal_angle_err_deg
              << " | final_dist_err_m=" << d.plane_distance_err_m
              << "\n";
  }

  printExitCodeGuide();
}

LogLevel parseLogLevel(const std::string & s) {
  if (s == "info") return LogLevel::kInfo;
  if (s == "warn") return LogLevel::kWarn;
  if (s == "error") return LogLevel::kError;
  fail("--log-level must be info|warn|error");
}

Config parseArgs(int argc, char ** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const std::string & name) -> std::string {
      if (i + 1 >= argc) fail("missing value for " + name);
      return argv[++i];
    };

    if (arg == "--pcd-file") {
      cfg.pcd_path = need(arg);
    } else if (arg == "--yaml-path") {
      cfg.yaml_path = need(arg);
    } else if (arg == "--lidar-ip") {
      cfg.lidar_ip = need(arg);
    } else if (arg == "--angle-unit") {
      cfg.angle_unit_cli = need(arg);
    } else if (arg == "--ransac-distance-threshold") {
      cfg.ransac_distance_threshold = std::stod(need(arg));
    } else if (arg == "--ransac-iterations") {
      cfg.ransac_iterations = std::stoi(need(arg));
    } else if (arg == "--min-plane-points") {
      cfg.min_plane_points = std::stoi(need(arg));
    } else if (arg == "--wn") {
      cfg.wn = std::stod(need(arg));
    } else if (arg == "--wd") {
      cfg.wd = std::stod(need(arg));
    } else if (arg == "--wp") {
      cfg.wp = std::stod(need(arg));
    } else if (arg == "--wD") {
      cfg.wD = std::stod(need(arg));
    } else if (arg == "--log-level") {
      cfg.log_level = parseLogLevel(need(arg));
    } else if (arg == "--help" || arg == "-h") {
      std::cout
        << "Usage:\n"
        << "  mid360_body_plane_calibrator_3planes_v3 \\\n"
        << "    --pcd-file input.pcd \\\n"
        << "    --yaml-path lidar_config.yaml \\\n"
        << "    --lidar-ip 192.168.1.135 [--angle-unit deg|rad]\n\n"
        << "Optional:\n"
        << "  --ransac-distance-threshold 0.02\n"
        << "  --ransac-iterations 1000\n"
        << "  --min-plane-points 80\n"
        << "  --wn 10 --wd 20 --wp 2 --wD 5\n"
        << "  --log-level info|warn|error\n"
        << "\nPlane convention:\n"
        << "  This version uses only n^T x + d = 0 everywhere.\n"
        << "\nExit codes:\n"
        << "  0 strict accept\n"
        << "  1 runtime/config/input error\n"
        << "  2 usable result but strict acceptance failed\n"
        << "  3 solve failed or unusable result\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + arg);
    }
  }

  if (cfg.pcd_path.empty()) fail("--pcd-file is required");
  if (cfg.yaml_path.empty()) fail("--yaml-path is required");
  if (cfg.lidar_ip.empty()) fail("--lidar-ip is required");
  if (!cfg.angle_unit_cli.empty() && cfg.angle_unit_cli != "deg" && cfg.angle_unit_cli != "rad") {
    fail("--angle-unit must be deg or rad");
  }
  validateConfig(cfg);
  return cfg;
}



RoiBox expandRoiBox(const RoiBox & roi, double pad) {
  RoiBox out = roi;
  out.xmin -= pad; out.xmax += pad;
  out.ymin -= pad; out.ymax += pad;
  out.zmin -= pad; out.zmax += pad;
  return out;
}

std::vector<PlaneMeasurement> extractPlaneMeasurementsWithPadding(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & raw_cloud,
  const YamlInput & yin,
  const Config & cfg,
  double roi_padding)
{
  YamlInput padded = yin;
  for (std::size_t i = 0; i < 3; ++i) {
    padded.rois_body[i] = expandRoiBox(yin.rois_body[i], roi_padding);
  }
  return extractPlaneMeasurements(raw_cloud, padded, cfg);
}

std::string loadTopicFromYaml(const std::string & yaml_path, const std::string & lidar_ip) {
  const YAML::Node root = YAML::LoadFile(yaml_path);
  if (!root[lidar_ip]) {
    fail("lidar ip not found in yaml when reading topic: " + lidar_ip);
  }
  const YAML::Node node = root[lidar_ip];
  if (node["topic"] && node["topic"].IsScalar()) {
    return node["topic"].as<std::string>();
  }
  return "/livox/lidar";
}

void ensureParentDir(const std::string & file_path) {
  if (file_path.empty()) return;
  const auto parent = std::filesystem::path(file_path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

void ensureDir(const std::string & dir_path) {
  if (dir_path.empty()) return;
  std::filesystem::create_directories(dir_path);
}

void saveSolverResultYaml(
  const std::string & output_result_path,
  const std::string & lidar_ip,
  const SolverResult & result,
  const YamlInput & yin,
  const std::vector<PlaneMeasurement> & measurements)
{
  if (output_result_path.empty()) {
    return;
  }

  ensureParentDir(output_result_path);

  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << lidar_ip << YAML::Value;
  out << YAML::BeginMap;
  out << YAML::Key << "angle_unit" << YAML::Value << "deg";
  out << YAML::Key << "x" << YAML::Value << result.t_m.x();
  out << YAML::Key << "y" << YAML::Value << result.t_m.y();
  out << YAML::Key << "z" << YAML::Value << result.t_m.z();
  out << YAML::Key << "roll" << YAML::Value << rad2deg(result.rpy_rad.x());
  out << YAML::Key << "pitch" << YAML::Value << rad2deg(result.rpy_rad.y());
  out << YAML::Key << "yaw" << YAML::Value << rad2deg(result.rpy_rad.z());
  out << YAML::Key << "success" << YAML::Value << result.success;
  out << YAML::Key << "strict_accept" << YAML::Value << result.strict_accept;
  out << YAML::Key << "initial_cost" << YAML::Value << result.initial_cost;
  out << YAML::Key << "final_cost" << YAML::Value << result.final_cost;
  out << YAML::Key << "seed_index" << YAML::Value << result.seed_index;

  out << YAML::Key << "delta_t_m" << YAML::Value << YAML::Flow << YAML::BeginSeq
      << (result.t_m.x() - yin.t_init.x())
      << (result.t_m.y() - yin.t_init.y())
      << (result.t_m.z() - yin.t_init.z())
      << YAML::EndSeq;
  out << YAML::Key << "delta_rpy_deg" << YAML::Value << YAML::Flow << YAML::BeginSeq
      << rad2deg(result.rpy_rad.x() - yin.rpy_init_rad.x())
      << rad2deg(result.rpy_rad.y() - yin.rpy_init_rad.y())
      << rad2deg(result.rpy_rad.z() - yin.rpy_init_rad.z())
      << YAML::EndSeq;

  out << YAML::Key << "per_plane" << YAML::Value << YAML::BeginSeq;
  for (std::size_t i = 0; i < result.per_plane.size(); ++i) {
    const auto & d = result.per_plane[i];
    const auto & m = measurements[i];
    out << YAML::BeginMap;
    out << YAML::Key << "plane_index" << YAML::Value << static_cast<int>(d.plane_index);
    out << YAML::Key << "fitted_plane_body" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << d.n_body_hat.x() << d.n_body_hat.y() << d.n_body_hat.z() << d.d_body_hat << YAML::EndSeq;
    out << YAML::Key << "final_normal_angle_error_deg" << YAML::Value << d.normal_angle_err_deg;
    out << YAML::Key << "final_plane_distance_error_m" << YAML::Value << d.plane_distance_err_m;
    out << YAML::Key << "init_normal_angle_error_deg" << YAML::Value << m.init_normal_angle_err_deg;
    out << YAML::Key << "init_plane_distance_error_m" << YAML::Value << m.init_plane_distance_err_m;
    out << YAML::Key << "roi_point_count" << YAML::Value << static_cast<int>(m.roi_point_count);
    out << YAML::Key << "inlier_count" << YAML::Value << static_cast<int>(m.inlier_count);
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  out << YAML::EndMap;
  out << YAML::EndMap;

  std::ofstream ofs(output_result_path);
  if (!ofs) {
    fail("failed to open output_result_path for write: " + output_result_path);
  }
  ofs << out.c_str();
  ofs.close();
}
}  // namespace


class MultiMid360CalibratorNode : public rclcpp::Node {
public:
  MultiMid360CalibratorNode()
  : rclcpp::Node("multi_mid360_calibrator")
  {
    declare_parameter<std::string>("config_path", "");
    declare_parameter<std::string>("target_lidar_ip", "");
    declare_parameter<std::string>("input_topic", "");
    declare_parameter<double>("accumulation_time_sec", 3.0);
    declare_parameter<double>("roi_distance_threshold", 0.02);
    declare_parameter<double>("ransac_distance_threshold", 0.01);
    declare_parameter<std::string>("output_result_path", "");
    declare_parameter<std::string>("output_cloud_dir", "");
    declare_parameter<int>("min_plane_points", 80);
    declare_parameter<int>("ransac_iterations", 1000);
    declare_parameter<double>("wn", 10.0);
    declare_parameter<double>("wd", 20.0);
    declare_parameter<double>("wp", 2.0);
    declare_parameter<double>("wD", 5.0);
    declare_parameter<int>("multi_seed_mode", 1);
    declare_parameter<std::string>("log_level", "info");
    declare_parameter<std::string>("angle_unit", "");

    get_parameter("config_path", config_path_);
    get_parameter("target_lidar_ip", target_lidar_ip_);
    get_parameter("input_topic", input_topic_);
    get_parameter("accumulation_time_sec", accumulation_time_sec_);
    get_parameter("roi_distance_threshold", roi_distance_threshold_);
    get_parameter("ransac_distance_threshold", ransac_distance_threshold_);
    get_parameter("output_result_path", output_result_path_);
    get_parameter("output_cloud_dir", output_cloud_dir_);

    cfg_.yaml_path = config_path_;
    cfg_.lidar_ip = target_lidar_ip_;
    cfg_.ransac_distance_threshold = ransac_distance_threshold_;
    cfg_.min_plane_points = get_parameter("min_plane_points").as_int();
    cfg_.ransac_iterations = get_parameter("ransac_iterations").as_int();
    cfg_.wn = get_parameter("wn").as_double();
    cfg_.wd = get_parameter("wd").as_double();
    cfg_.wp = get_parameter("wp").as_double();
    cfg_.wD = get_parameter("wD").as_double();
    cfg_.multi_seed_mode = get_parameter("multi_seed_mode").as_int();
    cfg_.log_level = parseLogLevel(get_parameter("log_level").as_string());
    cfg_.angle_unit_cli = get_parameter("angle_unit").as_string();

    if (config_path_.empty()) {
      throw std::runtime_error("parameter config_path is empty");
    }
    if (target_lidar_ip_.empty()) {
      throw std::runtime_error("parameter target_lidar_ip is empty");
    }
    if (accumulation_time_sec_ <= 0.0) {
      throw std::runtime_error("accumulation_time_sec must be > 0");
    }

    if (input_topic_.empty()) {
      input_topic_ = loadTopicFromYaml(config_path_, target_lidar_ip_);
    }

    validateConfig(cfg_);
    ensureDir(output_cloud_dir_);
    raw_accum_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    raw_accum_cloud_->reserve(300000);

    RCLCPP_INFO(get_logger(), "config_path: %s", config_path_.c_str());
    RCLCPP_INFO(get_logger(), "target_lidar_ip: %s", target_lidar_ip_.c_str());
    RCLCPP_INFO(get_logger(), "input_topic: %s", input_topic_.c_str());
    RCLCPP_INFO(get_logger(), "accumulation_time_sec: %.3f", accumulation_time_sec_);
    RCLCPP_INFO(get_logger(), "roi_distance_threshold: %.4f", roi_distance_threshold_);
    RCLCPP_INFO(get_logger(), "ransac_distance_threshold: %.4f", ransac_distance_threshold_);

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&MultiMid360CalibratorNode::onCloud, this, std::placeholders::_1));
  }

  int exitCode() const { return exit_code_; }

private:
  void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (finished_) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    try {
      pcl::fromROSMsg(*msg, cloud);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "failed to convert PointCloud2 to PCL: %s", e.what());
      finished_ = true;
      rclcpp::shutdown();
      return;
    }

    if (!started_) {
      start_time_ = msg->header.stamp;
      started_ = true;
      RCLCPP_INFO(get_logger(), "first cloud received, start accumulation");
    }

    std::size_t appended = 0;
    for (const auto & pt : cloud.points) {
      if (!pcl::isFinite(pt)) continue;
      const Vector3d p(pt.x, pt.y, pt.z);
      if (!isReasonablePoint(p)) continue;
      raw_accum_cloud_->points.push_back(pt);
      ++appended;
    }
    raw_accum_cloud_->width = static_cast<std::uint32_t>(raw_accum_cloud_->points.size());
    raw_accum_cloud_->height = 1;
    raw_accum_cloud_->is_dense = false;

    const double elapsed = elapsedSec(msg->header.stamp, start_time_);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
      "accumulating... latest_appended=%zu total_points=%zu elapsed=%.3f / %.3f sec",
      appended, raw_accum_cloud_->points.size(), elapsed, accumulation_time_sec_);

    if (elapsed >= accumulation_time_sec_) {
      finished_ = true;
      finalizeSolve();
      rclcpp::shutdown();
    }
  }

  double elapsedSec(const builtin_interfaces::msg::Time & now,
                    const builtin_interfaces::msg::Time & then) const {
    const double s_now = static_cast<double>(now.sec) + 1e-9 * static_cast<double>(now.nanosec);
    const double s_then = static_cast<double>(then.sec) + 1e-9 * static_cast<double>(then.nanosec);
    return s_now - s_then;
  }

  void finalizeSolve() {
    try {
      if (!raw_accum_cloud_ || raw_accum_cloud_->empty()) {
        throw std::runtime_error("no accumulated point cloud available for calibration");
      }

      if (!output_cloud_dir_.empty()) {
        const std::string raw_pcd = (std::filesystem::path(output_cloud_dir_) /
          (std::string("lidar_") + sanitizeIp(target_lidar_ip_) + "_accumulated_raw.pcd")).string();
        pcl::io::savePCDFileBinary(raw_pcd, *raw_accum_cloud_);
        RCLCPP_INFO(get_logger(), "saved accumulated raw cloud to: %s", raw_pcd.c_str());
      }

      YamlInput yin = loadYamlInput(cfg_.yaml_path, cfg_.lidar_ip, cfg_);
      auto measurements = extractPlaneMeasurementsWithPadding(raw_accum_cloud_, yin, cfg_, roi_distance_threshold_);
      auto result = solveCeres(measurements, yin, cfg_);
      printResult(measurements, result, yin);
      saveSolverResultYaml(output_result_path_, target_lidar_ip_, result, yin, measurements);

      if (!output_result_path_.empty()) {
        RCLCPP_INFO(get_logger(), "saved calibration yaml to: %s", output_result_path_.c_str());
      }

      if (!result.success) {
        exit_code_ = static_cast<int>(ExitCode::kSolveFailed);
        RCLCPP_ERROR(get_logger(), "solver finished but result is unusable");
      } else if (!result.strict_accept) {
        exit_code_ = static_cast<int>(ExitCode::kUsableButNotStrict);
        RCLCPP_WARN(get_logger(), "solver returned usable result, but strict acceptance failed");
      } else {
        exit_code_ = static_cast<int>(ExitCode::kStrictAccept);
        RCLCPP_INFO(get_logger(), "solver succeeded and strict acceptance passed");
      }
    } catch (const std::exception & e) {
      exit_code_ = static_cast<int>(ExitCode::kRuntimeError);
      RCLCPP_ERROR(get_logger(), "calibration failed: %s", e.what());
    }
  }

  std::string sanitizeIp(const std::string & ip) const {
    std::string out = ip;
    std::replace(out.begin(), out.end(), '.', '_');
    std::replace(out.begin(), out.end(), ':', '_');
    return out;
  }

private:
  Config cfg_;
  std::string config_path_;
  std::string target_lidar_ip_;
  std::string input_topic_;
  double accumulation_time_sec_{3.0};
  double roi_distance_threshold_{0.02};
  double ransac_distance_threshold_{0.01};
  std::string output_result_path_;
  std::string output_cloud_dir_;

  bool started_{false};
  bool finished_{false};
  builtin_interfaces::msg::Time start_time_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_accum_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  int exit_code_{static_cast<int>(ExitCode::kRuntimeError)};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  int exit_code = static_cast<int>(ExitCode::kRuntimeError);
  try {
    auto node = std::make_shared<MultiMid360CalibratorNode>();
    rclcpp::spin(node);
    exit_code = node->exitCode();
  } catch (const std::exception & e) {
    std::cerr << "[error] multi_mid360_calibrator fatal: " << e.what() << "\n";
    exit_code = static_cast<int>(ExitCode::kRuntimeError);
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return exit_code;
}
