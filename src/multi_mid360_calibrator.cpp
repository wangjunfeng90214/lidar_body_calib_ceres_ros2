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
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>
#include <map>
#include <set>
#include <numeric>

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

enum class MeasurementChannel : int {
  kRaw = 0,
  kPadding = 1,
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

struct ChannelSolveResult {
  MeasurementChannel channel{MeasurementChannel::kRaw};
  bool measurement_ok{false};
  bool solve_ok{false};
  bool strict_accept{false};
  std::string failure_reason;
  std::vector<PlaneMeasurement> measurements;
  SolverResult solver_result;
  double max_final_normal_angle_error_deg{1e9};
  double max_final_plane_distance_error_m{1e9};
  double mean_final_normal_angle_error_deg{1e9};
  double mean_final_plane_distance_error_m{1e9};
  double channel_score{1e18};
};

struct DualChannelSolveResult {
  ChannelSolveResult raw;
  ChannelSolveResult padding;
  bool has_selected{false};
  MeasurementChannel selected_channel{MeasurementChannel::kRaw};
  ChannelSolveResult selected;
  bool channels_consistent{false};
  double delta_t_norm_m{1e9};
  double delta_angle_deg{1e9};
};

struct CalibrationRunRecord {
  int run_index{0};
  std::string timestamp_iso8601;
  std::string lidar_ip;
  bool run_success{false};
  bool strict_accept{false};
  DualChannelSolveResult dual_result;
  Vector3d final_t_m{0.0, 0.0, 0.0};
  Vector3d final_rpy_rad{0.0, 0.0, 0.0};
  Matrix3d final_R{Matrix3d::Identity()};
  double final_cost{1e18};
  int seed_index{-1};
  std::string selected_channel{"raw"};
  std::string reject_reason;
};

struct BatchCalibrationSummary {
  int requested_runs{0};
  int completed_runs{0};
  int accepted_runs{0};
  int rejected_runs{0};
  std::vector<CalibrationRunRecord> all_runs;
  std::vector<int> accepted_run_indices;
  std::vector<int> rejected_run_indices;
  Vector3d robust_t_m{0.0, 0.0, 0.0};
  Eigen::Quaterniond robust_q{1.0, 0.0, 0.0, 0.0};
  Vector3d robust_rpy_rad{0.0, 0.0, 0.0};
  Vector3d mean_t_m{0.0, 0.0, 0.0};
  Vector3d std_t_m{0.0, 0.0, 0.0};
  Vector3d mean_rpy_deg{0.0, 0.0, 0.0};
  Vector3d std_rpy_deg{0.0, 0.0, 0.0};
  double max_pairwise_t_diff_m{0.0};
  double max_pairwise_angle_diff_deg{0.0};
  int representative_run_index{-1};
  std::string aggregation_method{"robust_median_then_select_nearest"};
  bool batch_stable{false};
  std::string batch_comment;
};

struct Figure8VerificationConfig {
  bool enable{false};
  std::string data_path;

  // 验收门限
  double max_rotation_error_deg{0.5};
  double max_translation_error_m{0.02};

  // 时间对齐搜索
  double max_time_offset_sec{0.08};
  double time_offset_step_sec{0.01};
  double association_tolerance_sec{0.02};

  // 运动对筛选
  double min_relative_translation_m{0.05};
  double min_relative_rotation_deg{3.0};
  int min_motion_pairs{15};

  // 是否做小范围微调
  bool enable_micro_refine{true};
  bool replace_final_result_on_success{false};
  int max_num_iterations{80};
};

struct Figure8VerificationResult {
  bool enabled{false};
  bool success{false};
  bool pass{false};

  int matched_pose_pairs{0};
  int motion_pairs{0};

  double estimated_time_offset_sec{0.0};
  double rotation_consistency_deg{1e9};
  double translation_consistency_m{1e9};

  bool refined_result_available{false};
  Vector3d refined_rpy_rad{0.0, 0.0, 0.0};
  Vector3d refined_t_m{0.0, 0.0, 0.0};

  std::string summary;
  std::string reject_reason;
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

struct TimedPose {
  double t_sec{0.0};
  Vector3d p{0.0, 0.0, 0.0};
  Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};
};

struct RelativeMotionPair {
  int idx0{-1};
  int idx1{-1};
  Matrix3d R_a{Matrix3d::Identity()};
  Vector3d t_a{0.0, 0.0, 0.0};
  Matrix3d R_b{Matrix3d::Identity()};
  Vector3d t_b{0.0, 0.0, 0.0};
};

struct InterpolatedPose {
  bool ok{false};
  Vector3d p{0.0, 0.0, 0.0};
  Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};
};

struct Figure8Dataset {
  std::vector<TimedPose> body_traj;
  std::vector<TimedPose> lidar_traj;
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

const char * measurementChannelTag(MeasurementChannel c) {
  switch (c) {
    case MeasurementChannel::kRaw: return "raw";
    case MeasurementChannel::kPadding: return "padding";
  }
  return "raw";
}

std::string nowIso8601Local() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &tt);
#else
  localtime_r(&tt, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
  return oss.str();
}

Eigen::Quaterniond rotationMatrixToQuaternion(const Matrix3d & R) {
  Eigen::Quaterniond q(R);
  if (q.w() < 0.0) {
    q.coeffs() *= -1.0;
  }
  return q.normalized();
}

Vector3d medianTranslation(const std::vector<Vector3d> & ts) {
  if (ts.empty()) return Vector3d::Zero();
  Vector3d out;
  for (int k = 0; k < 3; ++k) {
    std::vector<double> vals;
    vals.reserve(ts.size());
    for (const auto & t : ts) vals.push_back(t[k]);
    std::sort(vals.begin(), vals.end());
    const std::size_t n = vals.size();
    out[k] = (n % 2 == 1) ? vals[n / 2] : 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
  }
  return out;
}

Eigen::Quaterniond averageQuaternionChordal(const std::vector<Eigen::Quaterniond> & quats) {
  if (quats.empty()) return Eigen::Quaterniond::Identity();
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  Eigen::Vector4d ref(quats.front().w(), quats.front().x(), quats.front().y(), quats.front().z());
  for (auto q : quats) {
    q.normalize();
    Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
    if (ref.dot(v) < 0.0) v = -v;
    A += v * v.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(A);
  Eigen::Vector4d v = es.eigenvectors().col(3);
  Eigen::Quaterniond q(v[0], v[1], v[2], v[3]);
  if (q.w() < 0.0) q.coeffs() *= -1.0;
  return q.normalized();
}

double quaternionAngularDistanceDeg(const Eigen::Quaterniond & a, const Eigen::Quaterniond & b) {
  const double d = std::abs(a.normalized().dot(b.normalized()));
  const double c = std::max(-1.0, std::min(1.0, d));
  return rad2deg(2.0 * std::acos(c));
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

std::vector<std::string> splitCsvLine(const std::string & line) {
  std::vector<std::string> out;
  std::string cur;
  std::stringstream ss(line);
  while (std::getline(ss, cur, ',')) {
    out.push_back(cur);
  }
  return out;
}

double parseDoubleStrict(const std::string & s, const std::string & field_name) {
  try {
    size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size()) {
      fail("invalid numeric field '" + field_name + "': " + s);
    }
    if (!std::isfinite(v)) {
      fail("non-finite numeric field '" + field_name + "'");
    }
    return v;
  } catch (const std::exception &) {
    fail("failed to parse numeric field '" + field_name + "': " + s);
  }
}

std::vector<TimedPose> loadTrajectoryCsv(const std::string & path) {
  std::ifstream ifs(path);
  if (!ifs) {
    fail("failed to open trajectory csv: " + path);
  }

  std::vector<TimedPose> traj;
  std::string line;
  bool header_checked = false;

  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    if (!header_checked) {
      header_checked = true;
      continue;
    }

    const auto cols = splitCsvLine(line);
    if (cols.size() < 8) {
      fail("trajectory csv requires 8 columns: timestamp_sec,tx,ty,tz,qx,qy,qz,qw");
    }

    TimedPose s;
    s.t_sec = parseDoubleStrict(cols[0], "timestamp_sec");
    s.p = Vector3d(
      parseDoubleStrict(cols[1], "tx"),
      parseDoubleStrict(cols[2], "ty"),
      parseDoubleStrict(cols[3], "tz"));

    const double qx = parseDoubleStrict(cols[4], "qx");
    const double qy = parseDoubleStrict(cols[5], "qy");
    const double qz = parseDoubleStrict(cols[6], "qz");
    const double qw = parseDoubleStrict(cols[7], "qw");
    Eigen::Quaterniond q(qw, qx, qy, qz);
    if (q.norm() < kEps) {
      fail("invalid quaternion norm in: " + path);
    }
    s.q = q.normalized();
    if (s.q.w() < 0.0) s.q.coeffs() *= -1.0;

    traj.push_back(s);
  }

  if (traj.size() < 2) {
    fail("trajectory csv has too few samples: " + path);
  }

  std::sort(traj.begin(), traj.end(), [](const TimedPose & a, const TimedPose & b) {
    return a.t_sec < b.t_sec;
  });

  return traj;
}

InterpolatedPose interpolatePose(const std::vector<TimedPose> & traj, double t_sec) {
  InterpolatedPose out;
  if (traj.size() < 2) return out;
  if (t_sec < traj.front().t_sec || t_sec > traj.back().t_sec) return out;

  auto it = std::lower_bound(
    traj.begin(), traj.end(), t_sec,
    [](const TimedPose & a, double t) { return a.t_sec < t; });

  if (it == traj.begin()) {
    out.ok = true;
    out.p = it->p;
    out.q = it->q;
    return out;
  }
  if (it == traj.end()) {
    out.ok = true;
    out.p = traj.back().p;
    out.q = traj.back().q;
    return out;
  }

  const auto & s1 = *it;
  const auto & s0 = *(it - 1);
  const double dt = s1.t_sec - s0.t_sec;
  if (dt < 1e-9) return out;

  const double alpha = (t_sec - s0.t_sec) / dt;
  out.ok = true;
  out.p = (1.0 - alpha) * s0.p + alpha * s1.p;
  out.q = s0.q.slerp(alpha, s1.q).normalized();
  if (out.q.w() < 0.0) out.q.coeffs() *= -1.0;
  return out;
}

Figure8Dataset loadFigure8Dataset(const Figure8VerificationConfig & cfg) {
  Figure8Dataset ds;
  const std::filesystem::path root(cfg.data_path);
  ds.body_traj = loadTrajectoryCsv((root / "body_trajectory.csv").string());
  ds.lidar_traj = loadTrajectoryCsv((root / "lidar_trajectory.csv").string());
  return ds;
}

Eigen::Isometry3d makePose(const Vector3d & p, const Eigen::Quaterniond & q) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = q.normalized().toRotationMatrix();
  T.translation() = p;
  return T;
}

double rotationAngleDeg(const Matrix3d & R) {
  Eigen::AngleAxisd aa(R);
  return rad2deg(std::abs(aa.angle()));
}

std::vector<RelativeMotionPair> buildRelativeMotionPairs(
  const std::vector<TimedPose> & body_aligned,
  const std::vector<TimedPose> & lidar_aligned,
  const Figure8VerificationConfig & cfg)
{
  std::vector<RelativeMotionPair> pairs;
  const int n = static_cast<int>(std::min(body_aligned.size(), lidar_aligned.size()));
  if (n < 2) return pairs;

  for (int i = 0; i < n - 1; ++i) {
    const Eigen::Isometry3d A0 = makePose(body_aligned[i].p, body_aligned[i].q);
    const Eigen::Isometry3d A1 = makePose(body_aligned[i + 1].p, body_aligned[i + 1].q);
    const Eigen::Isometry3d B0 = makePose(lidar_aligned[i].p, lidar_aligned[i].q);
    const Eigen::Isometry3d B1 = makePose(lidar_aligned[i + 1].p, lidar_aligned[i + 1].q);

    const Eigen::Isometry3d dA = A0.inverse() * A1;
    const Eigen::Isometry3d dB = B0.inverse() * B1;

    const double trans_a = dA.translation().norm();
    const double rot_a_deg = rotationAngleDeg(dA.linear());

    if (trans_a < cfg.min_relative_translation_m &&
        rot_a_deg < cfg.min_relative_rotation_deg) {
      continue;
        }

    RelativeMotionPair mp;
    mp.idx0 = i;
    mp.idx1 = i + 1;
    mp.R_a = dA.linear();
    mp.t_a = dA.translation();
    mp.R_b = dB.linear();
    mp.t_b = dB.translation();
    pairs.push_back(mp);
  }
  return pairs;
}

bool buildAlignedTrajectoryPairs(
  const Figure8Dataset & ds,
  double time_offset_sec,
  double assoc_tol_sec,
  std::vector<TimedPose> & body_aligned,
  std::vector<TimedPose> & lidar_aligned)
{
  body_aligned.clear();
  lidar_aligned.clear();

  for (const auto & body_s : ds.body_traj) {
    const double t_lidar = body_s.t_sec + time_offset_sec;
    const auto interp = interpolatePose(ds.lidar_traj, t_lidar);
    if (!interp.ok) continue;

    auto it = std::lower_bound(
      ds.lidar_traj.begin(), ds.lidar_traj.end(), t_lidar,
      [](const TimedPose & a, double t) { return a.t_sec < t; });

    double nearest_dt = 1e9;
    if (it != ds.lidar_traj.end()) {
      nearest_dt = std::min(nearest_dt, std::abs(it->t_sec - t_lidar));
    }
    if (it != ds.lidar_traj.begin()) {
      nearest_dt = std::min(nearest_dt, std::abs((it - 1)->t_sec - t_lidar));
    }
    if (nearest_dt > assoc_tol_sec) continue;

    body_aligned.push_back(body_s);
    TimedPose ls;
    ls.t_sec = t_lidar;
    ls.p = interp.p;
    ls.q = interp.q;
    lidar_aligned.push_back(ls);
  }

  return body_aligned.size() >= 2 && lidar_aligned.size() == body_aligned.size();
}

void evaluateFigure8Consistency(
  const std::vector<RelativeMotionPair> & pairs,
  const Matrix3d & R_lidar_to_body,
  const Vector3d & t_lidar_to_body,
  double & rotation_consistency_deg,
  double & translation_consistency_m)
{
  rotation_consistency_deg = 1e9;
  translation_consistency_m = 1e9;
  if (pairs.empty()) return;

  std::vector<double> rot_errs_deg;
  std::vector<double> trans_errs_m;
  rot_errs_deg.reserve(pairs.size());
  trans_errs_m.reserve(pairs.size());

  const Matrix3d R = R_lidar_to_body;
  const Vector3d t = t_lidar_to_body;

  for (const auto & mp : pairs) {
    const Matrix3d left_R = mp.R_a * R;
    const Matrix3d right_R = R * mp.R_b;
    const Matrix3d R_err = left_R.transpose() * right_R;
    rot_errs_deg.push_back(rotationAngleDeg(R_err));

    const Vector3d left_t = mp.R_a * t + mp.t_a;
    const Vector3d right_t = R * mp.t_b + t;
    trans_errs_m.push_back((left_t - right_t).norm());
  }

  auto mean_of = [](const std::vector<double> & v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  };

  rotation_consistency_deg = mean_of(rot_errs_deg);
  translation_consistency_m = mean_of(trans_errs_m);
}

struct Figure8MotionResidual {
  struct PackedMotion {
    Matrix3d R_a{Matrix3d::Identity()};
    Vector3d t_a{0.0, 0.0, 0.0};
    Matrix3d R_b{Matrix3d::Identity()};
    Vector3d t_b{0.0, 0.0, 0.0};
    Matrix3d R_x0{Matrix3d::Identity()};
    Vector3d t_x0{0.0, 0.0, 0.0};
  };

  Figure8MotionResidual(const RelativeMotionPair & mp, double w_rot, double w_trans)
  : w_rot_(w_rot), w_trans_(w_trans) {
    mp_.R_a = mp.R_a;
    mp_.t_a = mp.t_a;
    mp_.R_b = mp.R_b;
    mp_.t_b = mp.t_b;
  }

  template<typename T>
  bool operator()(const T * const d_rpy, const T * const d_t, T * residuals) const {
    const Eigen::Matrix<T, 3, 1> drpy(d_rpy[0], d_rpy[1], d_rpy[2]);
    const Eigen::Matrix<T, 3, 1> dt(d_t[0], d_t[1], d_t[2]);
    const Eigen::Matrix<T, 3, 3> dR = rpyToRTemplate<T>(drpy);

    const Eigen::Matrix<T, 3, 3> R0 = mp_.R_x0.cast<T>();
    const Eigen::Matrix<T, 3, 1> t0 = mp_.t_x0.cast<T>();

    const Eigen::Matrix<T, 3, 3> R = dR * R0;
    const Eigen::Matrix<T, 3, 1> t = t0 + dt;

    const Eigen::Matrix<T, 3, 3> left_R  = mp_.R_a.cast<T>() * R;
    const Eigen::Matrix<T, 3, 3> right_R = R * mp_.R_b.cast<T>();
    const Eigen::Matrix<T, 3, 3> R_err = left_R.transpose() * right_R;

    const Eigen::Matrix<T, 3, 1> rot_vec(
      R_err(2,1) - R_err(1,2),
      R_err(0,2) - R_err(2,0),
      R_err(1,0) - R_err(0,1));

    const Eigen::Matrix<T, 3, 1> left_t  = mp_.R_a.cast<T>() * t + mp_.t_a.cast<T>();
    const Eigen::Matrix<T, 3, 1> right_t = R * mp_.t_b.cast<T>() + t;
    const Eigen::Matrix<T, 3, 1> trans_err = left_t - right_t;

    residuals[0] = T(w_rot_)   * rot_vec[0];
    residuals[1] = T(w_rot_)   * rot_vec[1];
    residuals[2] = T(w_rot_)   * rot_vec[2];
    residuals[3] = T(w_trans_) * trans_err[0];
    residuals[4] = T(w_trans_) * trans_err[1];
    residuals[5] = T(w_trans_) * trans_err[2];
    return true;
  }

  static ceres::CostFunction * Create(
    const RelativeMotionPair & mp,
    const Matrix3d & R_x0,
    const Vector3d & t_x0,
    double w_rot,
    double w_trans)
  {
    auto * obj = new Figure8MotionResidual(mp, w_rot, w_trans);
    obj->mp_.R_x0 = R_x0;
    obj->mp_.t_x0 = t_x0;
    return new ceres::AutoDiffCostFunction<Figure8MotionResidual, 6, 3, 3>(obj);
  }

  PackedMotion mp_;
  double w_rot_{1.0};
  double w_trans_{1.0};
};

bool refineExtrinsicByFigure8(
  const std::vector<RelativeMotionPair> & pairs,
  const Figure8VerificationConfig & cfg,
  Vector3d & rpy_rad,
  Vector3d & t_m)
{
  if (pairs.size() < static_cast<size_t>(cfg.min_motion_pairs)) return false;

  double d_rpy[3] = {0.0, 0.0, 0.0};
  double d_t[3] = {0.0, 0.0, 0.0};

  ceres::Problem problem;
  const Matrix3d R0 = rpyToR(rpy_rad);
  for (const auto & mp : pairs) {
    auto * cost = Figure8MotionResidual::Create(mp, R0, t_m, 1.0, 5.0);
    problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), d_rpy, d_t);
  }

  const double max_rot = deg2rad(1.0);
  const double max_trans = 0.02;
  for (int i = 0; i < 3; ++i) {
    problem.SetParameterLowerBound(d_rpy, i, -max_rot);
    problem.SetParameterUpperBound(d_rpy, i,  max_rot);
    problem.SetParameterLowerBound(d_t, i, -max_trans);
    problem.SetParameterUpperBound(d_t, i,  max_trans);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = cfg.max_num_iterations;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.function_tolerance = 1e-10;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-10;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (!summary.IsSolutionUsable()) return false;

  rpy_rad += Vector3d(d_rpy[0], d_rpy[1], d_rpy[2]);
  t_m += Vector3d(d_t[0], d_t[1], d_t[2]);
  return true;
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


std::vector<std::string> splitCommaSeparated(const std::string & s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), item.end());
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

std::vector<std::string> loadAllLidarIpsFromYaml(const std::string & yaml_path) {
  const YAML::Node root = YAML::LoadFile(yaml_path);
  if (!root || !root.IsMap()) {
    fail("yaml root must be a map of lidar_ip -> config");
  }
  std::vector<std::string> ips;
  ips.reserve(root.size());
  for (auto it = root.begin(); it != root.end(); ++it) {
    if (!it->first.IsScalar()) continue;
    const std::string key = it->first.as<std::string>();
    if (key.empty()) continue;
    ips.push_back(key);
  }
  if (ips.empty()) {
    fail("no lidar entries found in yaml");
  }
  std::sort(ips.begin(), ips.end());
  return ips;
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

void computeChannelDiagnostics(ChannelSolveResult & r) {
  if (r.solver_result.per_plane.empty()) return;
  double max_ang = 0.0, max_dist = 0.0;
  double sum_ang = 0.0, sum_dist = 0.0;
  for (const auto & p : r.solver_result.per_plane) {
    max_ang = std::max(max_ang, p.normal_angle_err_deg);
    max_dist = std::max(max_dist, std::abs(p.plane_distance_err_m));
    sum_ang += p.normal_angle_err_deg;
    sum_dist += std::abs(p.plane_distance_err_m);
  }
  const double n = static_cast<double>(r.solver_result.per_plane.size());
  r.max_final_normal_angle_error_deg = max_ang;
  r.max_final_plane_distance_error_m = max_dist;
  r.mean_final_normal_angle_error_deg = sum_ang / n;
  r.mean_final_plane_distance_error_m = sum_dist / n;
}

double computeChannelScore(const ChannelSolveResult & r) {
  if (!r.measurement_ok) return 1e18;
  if (!r.solve_ok) return 1e17;
  if (!r.strict_accept) return 1e16 + r.solver_result.final_cost;
  double score = 0.0;
  score += 1000.0 * r.max_final_normal_angle_error_deg;
  score += 5000.0 * r.max_final_plane_distance_error_m;
  score += r.solver_result.final_cost;
  std::size_t total_inliers = 0;
  for (const auto & m : r.measurements) total_inliers += m.inlier_count;
  score -= 0.001 * static_cast<double>(total_inliers);
  return score;
}

ChannelSolveResult runSingleChannel(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
  const YamlInput & yin,
  const Config & cfg,
  MeasurementChannel channel,
  double roi_padding_m)
{
  ChannelSolveResult out;
  out.channel = channel;
  try {
    if (channel == MeasurementChannel::kRaw) {
      out.measurements = extractPlaneMeasurements(cloud, yin, cfg);
    } else {
      out.measurements = extractPlaneMeasurementsWithPadding(cloud, yin, cfg, roi_padding_m);
    }
    out.measurement_ok = (out.measurements.size() == 3);
    if (!out.measurement_ok) {
      out.failure_reason = "measurement count != 3";
      return out;
    }
    out.solver_result = solveCeres(out.measurements, yin, cfg);
    out.solve_ok = out.solver_result.success;
    out.strict_accept = out.solver_result.strict_accept;
    computeChannelDiagnostics(out);
    out.channel_score = computeChannelScore(out);
    if (!out.solve_ok) {
      out.failure_reason = "solver unusable";
    } else if (!out.strict_accept) {
      out.failure_reason = "strict acceptance failed";
    }
  } catch (const std::exception & e) {
    out.failure_reason = e.what();
    out.channel_score = 1e18;
  }
  return out;
}

DualChannelSolveResult runDualChannelSolve(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
  const YamlInput & yin,
  const Config & cfg,
  bool enable_padding_channel,
  double roi_padding_m)
{
  DualChannelSolveResult out;
  out.raw = runSingleChannel(cloud, yin, cfg, MeasurementChannel::kRaw, roi_padding_m);
  if (enable_padding_channel) {
    out.padding = runSingleChannel(cloud, yin, cfg, MeasurementChannel::kPadding, roi_padding_m);
  } else {
    out.padding.channel = MeasurementChannel::kPadding;
    out.padding.failure_reason = "padding channel disabled";
    out.padding.channel_score = 1e18;
  }

  if (out.raw.solve_ok && enable_padding_channel && out.padding.solve_ok) {
    const auto dt = out.raw.solver_result.t_m - out.padding.solver_result.t_m;
    out.delta_t_norm_m = dt.norm();
    const Eigen::Quaterniond qr = rotationMatrixToQuaternion(out.raw.solver_result.R);
    const Eigen::Quaterniond qp = rotationMatrixToQuaternion(out.padding.solver_result.R);
    out.delta_angle_deg = quaternionAngularDistanceDeg(qr, qp);
    out.channels_consistent = (out.delta_t_norm_m < 0.005 && out.delta_angle_deg < 0.3);
  }

  const bool raw_better = out.raw.channel_score <= out.padding.channel_score;
  out.selected = raw_better ? out.raw : out.padding;
  out.selected_channel = raw_better ? MeasurementChannel::kRaw : MeasurementChannel::kPadding;
  out.has_selected = out.selected.measurement_ok;
  return out;
}

bool isAcceptedRun(const CalibrationRunRecord & r) {
  return r.run_success && r.strict_accept;
}

CalibrationRunRecord processOneRun(
  int run_index,
  const std::string & lidar_ip,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
  const YamlInput & yin,
  const Config & cfg,
  bool enable_padding_channel,
  double roi_padding_m)
{
  CalibrationRunRecord rec;
  rec.run_index = run_index;
  rec.timestamp_iso8601 = nowIso8601Local();
  rec.lidar_ip = lidar_ip;
  rec.dual_result = runDualChannelSolve(cloud, yin, cfg, enable_padding_channel, roi_padding_m);
  rec.run_success = rec.dual_result.has_selected;
  if (!rec.run_success) {
    rec.reject_reason = "dual channel solve failed";
    return rec;
  }
  const auto & sel = rec.dual_result.selected;
  rec.strict_accept = sel.strict_accept;
  rec.final_t_m = sel.solver_result.t_m;
  rec.final_rpy_rad = sel.solver_result.rpy_rad;
  rec.final_R = sel.solver_result.R;
  rec.final_cost = sel.solver_result.final_cost;
  rec.seed_index = sel.solver_result.seed_index;
  rec.selected_channel = measurementChannelTag(rec.dual_result.selected_channel);
  if (!rec.strict_accept) {
    rec.reject_reason = sel.failure_reason.empty() ? "strict acceptance failed" : sel.failure_reason;
  }
  return rec;
}

BatchCalibrationSummary aggregateBatchResults(const std::vector<CalibrationRunRecord> & runs) {
  BatchCalibrationSummary summary;
  summary.requested_runs = static_cast<int>(runs.size());
  summary.completed_runs = static_cast<int>(runs.size());
  summary.all_runs = runs;

  std::vector<Vector3d> accepted_t;
  std::vector<Eigen::Quaterniond> accepted_q;
  std::vector<Vector3d> accepted_rpy_deg;
  std::vector<const CalibrationRunRecord *> accepted_records;

  for (const auto & r : runs) {
    if (isAcceptedRun(r)) {
      summary.accepted_run_indices.push_back(r.run_index);
      accepted_t.push_back(r.final_t_m);
      accepted_q.push_back(rotationMatrixToQuaternion(r.final_R));
      accepted_rpy_deg.push_back(Vector3d(rad2deg(r.final_rpy_rad.x()), rad2deg(r.final_rpy_rad.y()), rad2deg(r.final_rpy_rad.z())));
      accepted_records.push_back(&r);
    } else {
      summary.rejected_run_indices.push_back(r.run_index);
    }
  }
  summary.accepted_runs = static_cast<int>(summary.accepted_run_indices.size());
  summary.rejected_runs = static_cast<int>(summary.rejected_run_indices.size());

  if (accepted_records.empty()) {
    summary.batch_comment = "no accepted runs";
    return summary;
  }

  summary.robust_t_m = medianTranslation(accepted_t);
  summary.robust_q = averageQuaternionChordal(accepted_q);
  summary.robust_rpy_rad = summary.robust_q.toRotationMatrix().eulerAngles(2, 1, 0).reverse();

  for (const auto & t : accepted_t) summary.mean_t_m += t;
  summary.mean_t_m /= static_cast<double>(accepted_t.size());
  for (const auto & rpy_deg : accepted_rpy_deg) summary.mean_rpy_deg += rpy_deg;
  summary.mean_rpy_deg /= static_cast<double>(accepted_rpy_deg.size());
  for (const auto & t : accepted_t) {
    const Vector3d d = t - summary.mean_t_m;
    summary.std_t_m += d.cwiseProduct(d);
  }
  for (const auto & rpy_deg : accepted_rpy_deg) {
    const Vector3d d = rpy_deg - summary.mean_rpy_deg;
    summary.std_rpy_deg += d.cwiseProduct(d);
  }
  summary.std_t_m = (summary.std_t_m / static_cast<double>(accepted_t.size())).cwiseSqrt();
  summary.std_rpy_deg = (summary.std_rpy_deg / static_cast<double>(accepted_rpy_deg.size())).cwiseSqrt();

  double best_dist = 1e18;
  for (const auto * r : accepted_records) {
    const double t_dist = (r->final_t_m - summary.robust_t_m).norm();
    const double q_dist = quaternionAngularDistanceDeg(rotationMatrixToQuaternion(r->final_R), summary.robust_q);
    const double dist = t_dist + 0.01 * q_dist;
    if (dist < best_dist) {
      best_dist = dist;
      summary.representative_run_index = r->run_index;
    }
  }

  for (std::size_t i = 0; i < accepted_records.size(); ++i) {
    for (std::size_t j = i + 1; j < accepted_records.size(); ++j) {
      summary.max_pairwise_t_diff_m = std::max(summary.max_pairwise_t_diff_m,
        (accepted_records[i]->final_t_m - accepted_records[j]->final_t_m).norm());
      summary.max_pairwise_angle_diff_deg = std::max(summary.max_pairwise_angle_diff_deg,
        quaternionAngularDistanceDeg(rotationMatrixToQuaternion(accepted_records[i]->final_R),
                                     rotationMatrixToQuaternion(accepted_records[j]->final_R)));
    }
  }

  summary.batch_stable =
    (summary.accepted_runs >= 3) &&
    (summary.std_t_m.x() < 0.003) &&
    (summary.std_t_m.y() < 0.003) &&
    (summary.std_t_m.z() < 0.005) &&
    (summary.std_rpy_deg.x() < 0.2) &&
    (summary.std_rpy_deg.y() < 0.2) &&
    (summary.std_rpy_deg.z() < 0.2);

  summary.batch_comment = summary.batch_stable ? "stable" : "not stable enough";
  return summary;
}

Figure8VerificationResult verifyFigure8(
  const Figure8VerificationConfig & cfg,
  const BatchCalibrationSummary & batch)
{
  Figure8VerificationResult out;
  out.enabled = cfg.enable;
  if (!cfg.enable) {
    out.success = true;
    out.pass = true;
    out.summary = "figure8 verification disabled";
    return out;
  }

  if (cfg.data_path.empty()) {
    out.success = false;
    out.pass = false;
    out.reject_reason = "figure8_data_path is empty";
    out.summary = out.reject_reason;
    return out;
  }

  if (batch.representative_run_index < 0 ||
      batch.representative_run_index >= static_cast<int>(batch.all_runs.size())) {
    out.success = false;
    out.pass = false;
    out.reject_reason = "batch has no representative run";
    out.summary = out.reject_reason;
    return out;
  }

  try {
    const auto ds = loadFigure8Dataset(cfg);
    const auto & rep = batch.all_runs[batch.representative_run_index];

    double best_score = 1e18;
    double best_time_offset = 0.0;
    std::vector<TimedPose> best_body_aligned;
    std::vector<TimedPose> best_lidar_aligned;
    std::vector<RelativeMotionPair> best_pairs;
    double best_rot_deg = 1e9;
    double best_trans_m = 1e9;

    for (double dt = -cfg.max_time_offset_sec;
         dt <= cfg.max_time_offset_sec + 1e-12;
         dt += cfg.time_offset_step_sec) {

      std::vector<TimedPose> body_aligned;
      std::vector<TimedPose> lidar_aligned;
      if (!buildAlignedTrajectoryPairs(ds, dt, cfg.association_tolerance_sec,
                                       body_aligned, lidar_aligned)) {
        continue;
      }

      auto pairs = buildRelativeMotionPairs(body_aligned, lidar_aligned, cfg);
      if (pairs.size() < static_cast<size_t>(cfg.min_motion_pairs)) {
        continue;
      }

      double rot_deg = 1e9;
      double trans_m = 1e9;
      evaluateFigure8Consistency(pairs, rep.final_R, rep.final_t_m, rot_deg, trans_m);

      const double score = rot_deg + 100.0 * trans_m;
      if (score < best_score) {
        best_score = score;
        best_time_offset = dt;
        best_body_aligned = body_aligned;
        best_lidar_aligned = lidar_aligned;
        best_pairs = pairs;
        best_rot_deg = rot_deg;
        best_trans_m = trans_m;
      }
    }

    if (best_pairs.size() < static_cast<size_t>(cfg.min_motion_pairs)) {
      out.success = false;
      out.pass = false;
      out.reject_reason = "not enough valid motion pairs for figure8 verification";
      out.summary = out.reject_reason;
      return out;
    }

    out.success = true;
    out.matched_pose_pairs = static_cast<int>(best_body_aligned.size());
    out.motion_pairs = static_cast<int>(best_pairs.size());
    out.estimated_time_offset_sec = best_time_offset;
    out.rotation_consistency_deg = best_rot_deg;
    out.translation_consistency_m = best_trans_m;

    Vector3d refined_rpy = rep.final_rpy_rad;
    Vector3d refined_t = rep.final_t_m;
    if (cfg.enable_micro_refine) {
      if (refineExtrinsicByFigure8(best_pairs, cfg, refined_rpy, refined_t)) {
        out.refined_result_available = true;
        out.refined_rpy_rad = refined_rpy;
        out.refined_t_m = refined_t;

        double rot2 = 1e9;
        double trans2 = 1e9;
        evaluateFigure8Consistency(best_pairs, rpyToR(refined_rpy), refined_t, rot2, trans2);

        out.rotation_consistency_deg = rot2;
        out.translation_consistency_m = trans2;
      }
    }

    out.pass =
      out.rotation_consistency_deg <= cfg.max_rotation_error_deg &&
      out.translation_consistency_m <= cfg.max_translation_error_m;

    std::ostringstream oss;
    oss << "figure8 verification "
        << (out.pass ? "pass" : "fail")
        << ", matched_pose_pairs=" << out.matched_pose_pairs
        << ", motion_pairs=" << out.motion_pairs
        << ", time_offset_sec=" << std::fixed << std::setprecision(4) << out.estimated_time_offset_sec
        << ", rot_deg=" << out.rotation_consistency_deg
        << ", trans_m=" << out.translation_consistency_m;
    out.summary = oss.str();

    if (!out.pass) {
      out.reject_reason = out.summary;
    }
    return out;
  } catch (const std::exception & e) {
    out.success = false;
    out.pass = false;
    out.reject_reason = e.what();
    out.summary = std::string("figure8 verification failed: ") + e.what();
    return out;
  }
}

void saveBatchCalibrationYaml(
  const std::string & output_result_path,
  const std::string & lidar_ip,
  const BatchCalibrationSummary & summary,
  const Figure8VerificationResult & fig8)
{
  if (output_result_path.empty()) return;
  ensureParentDir(output_result_path);
  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << lidar_ip << YAML::Value;
  out << YAML::BeginMap;
  out << YAML::Key << "format_version" << YAML::Value << 2;
  out << YAML::Key << "calibration_mode" << YAML::Value << "static_three_plane_batch";
  out << YAML::Key << "angle_unit" << YAML::Value << "deg";

  out << YAML::Key << "final_result" << YAML::Value << YAML::BeginMap;
  if (summary.representative_run_index >= 0) {
    const auto & rep = summary.all_runs.at(static_cast<std::size_t>(summary.representative_run_index));
    out << YAML::Key << "x" << YAML::Value << rep.final_t_m.x();
    out << YAML::Key << "y" << YAML::Value << rep.final_t_m.y();
    out << YAML::Key << "z" << YAML::Value << rep.final_t_m.z();
    out << YAML::Key << "roll" << YAML::Value << rad2deg(rep.final_rpy_rad.x());
    out << YAML::Key << "pitch" << YAML::Value << rad2deg(rep.final_rpy_rad.y());
    out << YAML::Key << "yaw" << YAML::Value << rad2deg(rep.final_rpy_rad.z());
    out << YAML::Key << "selected_channel" << YAML::Value << rep.selected_channel;
    out << YAML::Key << "success" << YAML::Value << rep.run_success;
    out << YAML::Key << "strict_accept" << YAML::Value << rep.strict_accept;
  }
  out << YAML::Key << "representative_run_index" << YAML::Value << summary.representative_run_index;
  out << YAML::EndMap;

  out << YAML::Key << "aggregate_statistics" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "requested_runs" << YAML::Value << summary.requested_runs;
  out << YAML::Key << "completed_runs" << YAML::Value << summary.completed_runs;
  out << YAML::Key << "accepted_runs" << YAML::Value << summary.accepted_runs;
  out << YAML::Key << "rejected_runs" << YAML::Value << summary.rejected_runs;
  out << YAML::Key << "aggregation_method" << YAML::Value << summary.aggregation_method;
  out << YAML::Key << "batch_stable" << YAML::Value << summary.batch_stable;
  out << YAML::Key << "mean_t_m" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.mean_t_m.x() << summary.mean_t_m.y() << summary.mean_t_m.z() << YAML::EndSeq;
  out << YAML::Key << "std_t_m" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.std_t_m.x() << summary.std_t_m.y() << summary.std_t_m.z() << YAML::EndSeq;
  out << YAML::Key << "mean_rpy_deg" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.mean_rpy_deg.x() << summary.mean_rpy_deg.y() << summary.mean_rpy_deg.z() << YAML::EndSeq;
  out << YAML::Key << "std_rpy_deg" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.std_rpy_deg.x() << summary.std_rpy_deg.y() << summary.std_rpy_deg.z() << YAML::EndSeq;
  out << YAML::Key << "max_pairwise_t_diff_m" << YAML::Value << summary.max_pairwise_t_diff_m;
  out << YAML::Key << "max_pairwise_angle_diff_deg" << YAML::Value << summary.max_pairwise_angle_diff_deg;
  out << YAML::Key << "batch_comment" << YAML::Value << summary.batch_comment;
  out << YAML::EndMap;

  out << YAML::Key << "runs" << YAML::Value << YAML::BeginSeq;
  for (const auto & run : summary.all_runs) {
    out << YAML::BeginMap;
    out << YAML::Key << "run_index" << YAML::Value << run.run_index;
    out << YAML::Key << "timestamp" << YAML::Value << run.timestamp_iso8601;
    out << YAML::Key << "run_success" << YAML::Value << run.run_success;
    out << YAML::Key << "strict_accept" << YAML::Value << run.strict_accept;
    out << YAML::Key << "selected_channel" << YAML::Value << run.selected_channel;
    out << YAML::Key << "final_cost" << YAML::Value << run.final_cost;
    out << YAML::Key << "seed_index" << YAML::Value << run.seed_index;
    out << YAML::Key << "x" << YAML::Value << run.final_t_m.x();
    out << YAML::Key << "y" << YAML::Value << run.final_t_m.y();
    out << YAML::Key << "z" << YAML::Value << run.final_t_m.z();
    out << YAML::Key << "roll" << YAML::Value << rad2deg(run.final_rpy_rad.x());
    out << YAML::Key << "pitch" << YAML::Value << rad2deg(run.final_rpy_rad.y());
    out << YAML::Key << "yaw" << YAML::Value << rad2deg(run.final_rpy_rad.z());
    out << YAML::Key << "reject_reason" << YAML::Value << run.reject_reason;
    out << YAML::Key << "channel_consistency" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "delta_t_norm_m" << YAML::Value << run.dual_result.delta_t_norm_m;
    out << YAML::Key << "delta_angle_deg" << YAML::Value << run.dual_result.delta_angle_deg;
    out << YAML::Key << "channels_consistent" << YAML::Value << run.dual_result.channels_consistent;
    out << YAML::EndMap;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  out << YAML::Key << "figure8_verification" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "enabled" << YAML::Value << fig8.enabled;
  out << YAML::Key << "success" << YAML::Value << fig8.success;
  out << YAML::Key << "pass" << YAML::Value << fig8.pass;
  out << YAML::Key << "matched_pose_pairs" << YAML::Value << fig8.matched_pose_pairs;
  out << YAML::Key << "motion_pairs" << YAML::Value << fig8.motion_pairs;
  out << YAML::Key << "estimated_time_offset_sec" << YAML::Value << fig8.estimated_time_offset_sec;
  out << YAML::Key << "rotation_consistency_deg" << YAML::Value << fig8.rotation_consistency_deg;
  out << YAML::Key << "translation_consistency_m" << YAML::Value << fig8.translation_consistency_m;
  out << YAML::Key << "refined_result_available" << YAML::Value << fig8.refined_result_available;

  if (fig8.refined_result_available) {
    out << YAML::Key << "refined_result" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "x" << YAML::Value << fig8.refined_t_m.x();
    out << YAML::Key << "y" << YAML::Value << fig8.refined_t_m.y();
    out << YAML::Key << "z" << YAML::Value << fig8.refined_t_m.z();
    out << YAML::Key << "roll_rad" << YAML::Value << fig8.refined_rpy_rad.x();
    out << YAML::Key << "pitch_rad" << YAML::Value << fig8.refined_rpy_rad.y();
    out << YAML::Key << "yaw_rad" << YAML::Value << fig8.refined_rpy_rad.z();
    out << YAML::EndMap;
  }

  out << YAML::Key << "summary" << YAML::Value << fig8.summary;
  out << YAML::Key << "reject_reason" << YAML::Value << fig8.reject_reason;
  out << YAML::EndMap;

  out << YAML::EndMap;
  out << YAML::EndMap;
  std::ofstream ofs(output_result_path);
  if (!ofs) fail("failed to open output_result_path for write: " + output_result_path);
  ofs << out.c_str();
}


void saveMultiBatchCalibrationYaml(
  const std::string & output_result_path,
  const std::vector<std::string> & lidar_ips,
  const std::map<std::string, BatchCalibrationSummary> & summaries,
  const std::map<std::string, Figure8VerificationResult> & fig8_results)
{
  if (output_result_path.empty()) return;
  ensureParentDir(output_result_path);
  YAML::Emitter out;
  out << YAML::BeginMap;
  for (const auto & lidar_ip : lidar_ips) {
    auto sit = summaries.find(lidar_ip);
    if (sit == summaries.end()) continue;
    const auto & summary = sit->second;
    const auto fit = fig8_results.find(lidar_ip);
    const Figure8VerificationResult empty_fig8{};
    const auto & fig8 = (fit != fig8_results.end()) ? fit->second : empty_fig8;

    out << YAML::Key << lidar_ip << YAML::Value;
    out << YAML::BeginMap;
    out << YAML::Key << "format_version" << YAML::Value << 2;
    out << YAML::Key << "calibration_mode" << YAML::Value << "static_three_plane_batch";
    out << YAML::Key << "angle_unit" << YAML::Value << "deg";

    out << YAML::Key << "final_result" << YAML::Value << YAML::BeginMap;
    if (summary.representative_run_index >= 0) {
      const auto & rep = summary.all_runs.at(static_cast<std::size_t>(summary.representative_run_index));
      out << YAML::Key << "x" << YAML::Value << rep.final_t_m.x();
      out << YAML::Key << "y" << YAML::Value << rep.final_t_m.y();
      out << YAML::Key << "z" << YAML::Value << rep.final_t_m.z();
      out << YAML::Key << "roll" << YAML::Value << rad2deg(rep.final_rpy_rad.x());
      out << YAML::Key << "pitch" << YAML::Value << rad2deg(rep.final_rpy_rad.y());
      out << YAML::Key << "yaw" << YAML::Value << rad2deg(rep.final_rpy_rad.z());
      out << YAML::Key << "selected_channel" << YAML::Value << rep.selected_channel;
      out << YAML::Key << "success" << YAML::Value << rep.run_success;
      out << YAML::Key << "strict_accept" << YAML::Value << rep.strict_accept;
    }
    out << YAML::Key << "representative_run_index" << YAML::Value << summary.representative_run_index;
    out << YAML::EndMap;

    out << YAML::Key << "aggregate_statistics" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "requested_runs" << YAML::Value << summary.requested_runs;
    out << YAML::Key << "completed_runs" << YAML::Value << summary.completed_runs;
    out << YAML::Key << "accepted_runs" << YAML::Value << summary.accepted_runs;
    out << YAML::Key << "rejected_runs" << YAML::Value << summary.rejected_runs;
    out << YAML::Key << "aggregation_method" << YAML::Value << summary.aggregation_method;
    out << YAML::Key << "batch_stable" << YAML::Value << summary.batch_stable;
    out << YAML::Key << "mean_t_m" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.mean_t_m.x() << summary.mean_t_m.y() << summary.mean_t_m.z() << YAML::EndSeq;
    out << YAML::Key << "std_t_m" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.std_t_m.x() << summary.std_t_m.y() << summary.std_t_m.z() << YAML::EndSeq;
    out << YAML::Key << "mean_rpy_deg" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.mean_rpy_deg.x() << summary.mean_rpy_deg.y() << summary.mean_rpy_deg.z() << YAML::EndSeq;
    out << YAML::Key << "std_rpy_deg" << YAML::Value << YAML::Flow << YAML::BeginSeq << summary.std_rpy_deg.x() << summary.std_rpy_deg.y() << summary.std_rpy_deg.z() << YAML::EndSeq;
    out << YAML::Key << "max_pairwise_t_diff_m" << YAML::Value << summary.max_pairwise_t_diff_m;
    out << YAML::Key << "max_pairwise_angle_diff_deg" << YAML::Value << summary.max_pairwise_angle_diff_deg;
    out << YAML::Key << "batch_comment" << YAML::Value << summary.batch_comment;
    out << YAML::EndMap;

    out << YAML::Key << "runs" << YAML::Value << YAML::BeginSeq;
    for (const auto & run : summary.all_runs) {
      out << YAML::BeginMap;
      out << YAML::Key << "run_index" << YAML::Value << run.run_index;
      out << YAML::Key << "timestamp" << YAML::Value << run.timestamp_iso8601;
      out << YAML::Key << "run_success" << YAML::Value << run.run_success;
      out << YAML::Key << "strict_accept" << YAML::Value << run.strict_accept;
      out << YAML::Key << "selected_channel" << YAML::Value << run.selected_channel;
      out << YAML::Key << "final_cost" << YAML::Value << run.final_cost;
      out << YAML::Key << "seed_index" << YAML::Value << run.seed_index;
      out << YAML::Key << "x" << YAML::Value << run.final_t_m.x();
      out << YAML::Key << "y" << YAML::Value << run.final_t_m.y();
      out << YAML::Key << "z" << YAML::Value << run.final_t_m.z();
      out << YAML::Key << "roll" << YAML::Value << rad2deg(run.final_rpy_rad.x());
      out << YAML::Key << "pitch" << YAML::Value << rad2deg(run.final_rpy_rad.y());
      out << YAML::Key << "yaw" << YAML::Value << rad2deg(run.final_rpy_rad.z());
      out << YAML::Key << "reject_reason" << YAML::Value << run.reject_reason;
      out << YAML::Key << "channel_consistency" << YAML::Value << YAML::BeginMap;
      out << YAML::Key << "delta_t_norm_m" << YAML::Value << run.dual_result.delta_t_norm_m;
      out << YAML::Key << "delta_angle_deg" << YAML::Value << run.dual_result.delta_angle_deg;
      out << YAML::Key << "channels_consistent" << YAML::Value << run.dual_result.channels_consistent;
      out << YAML::EndMap;
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    out << YAML::Key << "figure8_verification" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "enabled" << YAML::Value << fig8.enabled;
    out << YAML::Key << "success" << YAML::Value << fig8.success;
    out << YAML::Key << "pass" << YAML::Value << fig8.pass;
    out << YAML::Key << "matched_pose_pairs" << YAML::Value << fig8.matched_pose_pairs;
    out << YAML::Key << "motion_pairs" << YAML::Value << fig8.motion_pairs;
    out << YAML::Key << "estimated_time_offset_sec" << YAML::Value << fig8.estimated_time_offset_sec;
    out << YAML::Key << "rotation_consistency_deg" << YAML::Value << fig8.rotation_consistency_deg;
    out << YAML::Key << "translation_consistency_m" << YAML::Value << fig8.translation_consistency_m;
    out << YAML::Key << "refined_result_available" << YAML::Value << fig8.refined_result_available;
    if (fig8.refined_result_available) {
      out << YAML::Key << "refined_result" << YAML::Value << YAML::BeginMap;
      out << YAML::Key << "x" << YAML::Value << fig8.refined_t_m.x();
      out << YAML::Key << "y" << YAML::Value << fig8.refined_t_m.y();
      out << YAML::Key << "z" << YAML::Value << fig8.refined_t_m.z();
      out << YAML::Key << "roll_rad" << YAML::Value << fig8.refined_rpy_rad.x();
      out << YAML::Key << "pitch_rad" << YAML::Value << fig8.refined_rpy_rad.y();
      out << YAML::Key << "yaw_rad" << YAML::Value << fig8.refined_rpy_rad.z();
      out << YAML::EndMap;
    }
    out << YAML::Key << "summary" << YAML::Value << fig8.summary;
    out << YAML::Key << "reject_reason" << YAML::Value << fig8.reject_reason;
    out << YAML::EndMap;

    out << YAML::EndMap;
  }
  out << YAML::EndMap;
  std::ofstream ofs(output_result_path);
  if (!ofs) fail("failed to open output_result_path for write: " + output_result_path);
  ofs << out.c_str();
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
    declare_parameter<std::string>("target_lidar_ips", "");
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
    declare_parameter<int>("num_repeats", 5);
    declare_parameter<bool>("enable_padding_channel", true);
    declare_parameter<double>("padding_roi_margin_m", 0.02);
    declare_parameter<bool>("enable_figure8_verification", false);
    declare_parameter<std::string>("figure8_data_path", "");
    declare_parameter<bool>("figure8_enable_micro_refine", true);
    declare_parameter<bool>("figure8_replace_final_result_on_success", false);
    declare_parameter<double>("figure8_max_rotation_error_deg", 0.5);
    declare_parameter<double>("figure8_max_translation_error_m", 0.02);
    declare_parameter<double>("figure8_max_time_offset_sec", 0.08);
    declare_parameter<double>("figure8_time_offset_step_sec", 0.01);
    declare_parameter<double>("figure8_association_tolerance_sec", 0.02);
    declare_parameter<double>("figure8_min_relative_translation_m", 0.05);
    declare_parameter<double>("figure8_min_relative_rotation_deg", 3.0);
    declare_parameter<int>("figure8_min_motion_pairs", 15);
    declare_parameter<int>("figure8_max_num_iterations", 80);

    get_parameter("config_path", config_path_);
    get_parameter("target_lidar_ip", target_lidar_ip_);
    get_parameter("target_lidar_ips", target_lidar_ips_csv_);
    get_parameter("input_topic", input_topic_);
    get_parameter("accumulation_time_sec", accumulation_time_sec_);
    get_parameter("roi_distance_threshold", roi_distance_threshold_);
    get_parameter("ransac_distance_threshold", ransac_distance_threshold_);
    get_parameter("output_result_path", output_result_path_);
    get_parameter("output_cloud_dir", output_cloud_dir_);
    get_parameter("num_repeats", num_repeats_);
    get_parameter("enable_padding_channel", enable_padding_channel_);
    get_parameter("padding_roi_margin_m", padding_roi_margin_);
    get_parameter("enable_figure8_verification", figure8_cfg_template_.enable);
    get_parameter("figure8_data_path", figure8_cfg_template_.data_path);
    get_parameter("figure8_enable_micro_refine", figure8_cfg_template_.enable_micro_refine);
    get_parameter("figure8_replace_final_result_on_success", figure8_cfg_template_.replace_final_result_on_success);
    get_parameter("figure8_max_rotation_error_deg", figure8_cfg_template_.max_rotation_error_deg);
    get_parameter("figure8_max_translation_error_m", figure8_cfg_template_.max_translation_error_m);
    get_parameter("figure8_max_time_offset_sec", figure8_cfg_template_.max_time_offset_sec);
    get_parameter("figure8_time_offset_step_sec", figure8_cfg_template_.time_offset_step_sec);
    get_parameter("figure8_association_tolerance_sec", figure8_cfg_template_.association_tolerance_sec);
    get_parameter("figure8_min_relative_translation_m", figure8_cfg_template_.min_relative_translation_m);
    get_parameter("figure8_min_relative_rotation_deg", figure8_cfg_template_.min_relative_rotation_deg);
    figure8_cfg_template_.min_motion_pairs = get_parameter("figure8_min_motion_pairs").as_int();
    figure8_cfg_template_.max_num_iterations = get_parameter("figure8_max_num_iterations").as_int();

    base_cfg_.yaml_path = config_path_;
    base_cfg_.ransac_distance_threshold = ransac_distance_threshold_;
    base_cfg_.min_plane_points = get_parameter("min_plane_points").as_int();
    base_cfg_.ransac_iterations = get_parameter("ransac_iterations").as_int();
    base_cfg_.wn = get_parameter("wn").as_double();
    base_cfg_.wd = get_parameter("wd").as_double();
    base_cfg_.wp = get_parameter("wp").as_double();
    base_cfg_.wD = get_parameter("wD").as_double();
    base_cfg_.multi_seed_mode = get_parameter("multi_seed_mode").as_int();
    base_cfg_.log_level = parseLogLevel(get_parameter("log_level").as_string());
    base_cfg_.angle_unit_cli = get_parameter("angle_unit").as_string();

    if (config_path_.empty()) throw std::runtime_error("parameter config_path is empty");
    if (accumulation_time_sec_ <= 0.0) throw std::runtime_error("accumulation_time_sec must be > 0");
    if (num_repeats_ <= 0) throw std::runtime_error("num_repeats must be > 0");

    target_lidar_ips_ = resolveTargetLidarIps();
    if (target_lidar_ips_.empty()) {
      throw std::runtime_error("no target lidar ips resolved");
    }

    validateConfig(base_cfg_);
    ensureDir(output_cloud_dir_);

    RCLCPP_INFO(get_logger(), "config_path: %s", config_path_.c_str());
    RCLCPP_INFO(get_logger(), "accumulation_time_sec: %.3f", accumulation_time_sec_);
    RCLCPP_INFO(get_logger(), "roi_distance_threshold: %.4f", roi_distance_threshold_);
    RCLCPP_INFO(get_logger(), "ransac_distance_threshold: %.4f", ransac_distance_threshold_);
    RCLCPP_INFO(get_logger(), "num_repeats: %d", num_repeats_);
    RCLCPP_INFO(get_logger(), "enable_padding_channel: %s", enable_padding_channel_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "padding_roi_margin_m: %.4f", padding_roi_margin_);

    setupLidarStates();
  }

  int exitCode() const { return exit_code_; }

private:
  struct PerLidarState {
    std::string lidar_ip;
    std::string input_topic;
    Config cfg;
    Figure8VerificationConfig figure8_cfg;

    bool started{false};
    bool finished{false};
    int current_run_index{0};
    builtin_interfaces::msg::Time start_time{};
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_accum_cloud;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub;
    std::vector<CalibrationRunRecord> all_runs;
    BatchCalibrationSummary batch_summary;
    Figure8VerificationResult figure8_result;
  };

  std::vector<std::string> resolveTargetLidarIps() const {
    if (!target_lidar_ips_csv_.empty()) {
      return splitCommaSeparated(target_lidar_ips_csv_);
    }
    if (!target_lidar_ip_.empty()) {
      return {target_lidar_ip_};
    }
    return loadAllLidarIpsFromYaml(config_path_);
  }

  void setupLidarStates() {
    if (!input_topic_.empty() && target_lidar_ips_.size() != 1) {
      throw std::runtime_error("parameter input_topic can only be used when exactly one lidar is selected");
    }

    for (const auto & lidar_ip : target_lidar_ips_) {
      PerLidarState st;
      st.lidar_ip = lidar_ip;
      st.input_topic = (!input_topic_.empty() && target_lidar_ips_.size() == 1)
        ? input_topic_
        : loadTopicFromYaml(config_path_, lidar_ip);
      st.cfg = base_cfg_;
      st.cfg.lidar_ip = lidar_ip;
      st.figure8_cfg = figure8_cfg_template_;
      st.raw_accum_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
      st.raw_accum_cloud->reserve(300000);

      RCLCPP_INFO(get_logger(), "target_lidar_ip: %s", lidar_ip.c_str());
      RCLCPP_INFO(get_logger(), "input_topic[%s]: %s", lidar_ip.c_str(), st.input_topic.c_str());

      auto sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        st.input_topic,
        rclcpp::SensorDataQoS(),
        [this, lidar_ip](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
          this->onCloud(lidar_ip, msg);
        });
      st.sub = sub;
      lidar_states_.emplace(lidar_ip, std::move(st));
    }
  }

  void resetForNextRun(PerLidarState & st) {
    st.started = false;
    st.raw_accum_cloud->clear();
    st.raw_accum_cloud->width = 0;
    st.raw_accum_cloud->height = 1;
    st.raw_accum_cloud->is_dense = false;
  }

  void onCloud(const std::string & lidar_ip, const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_) return;
    auto it = lidar_states_.find(lidar_ip);
    if (it == lidar_states_.end()) return;
    auto & st = it->second;
    if (st.finished) return;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    try {
      pcl::fromROSMsg(*msg, cloud);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "[%s] failed to convert PointCloud2 to PCL: %s", lidar_ip.c_str(), e.what());
      st.finished = true;
      maybeFinalizeAll();
      return;
    }

    if (!st.started) {
      st.start_time = msg->header.stamp;
      st.started = true;
      RCLCPP_INFO(get_logger(), "[%s] run %d/%d: first cloud received, start accumulation", lidar_ip.c_str(), st.current_run_index + 1, num_repeats_);
    }

    std::size_t appended = 0;
    for (const auto & pt : cloud.points) {
      if (!pcl::isFinite(pt)) continue;
      const Vector3d p(pt.x, pt.y, pt.z);
      if (!isReasonablePoint(p)) continue;
      st.raw_accum_cloud->points.push_back(pt);
      ++appended;
    }
    st.raw_accum_cloud->width = static_cast<std::uint32_t>(st.raw_accum_cloud->points.size());
    st.raw_accum_cloud->height = 1;
    st.raw_accum_cloud->is_dense = false;

    const double elapsed = elapsedSec(msg->header.stamp, st.start_time);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
      "[%s] run %d/%d accumulating... latest_appended=%zu total_points=%zu elapsed=%.3f / %.3f sec",
      lidar_ip.c_str(), st.current_run_index + 1, num_repeats_, appended, st.raw_accum_cloud->points.size(), elapsed, accumulation_time_sec_);

    if (elapsed >= accumulation_time_sec_) {
      processCurrentRun(st);
      if (finished_) rclcpp::shutdown();
    }
  }

  double elapsedSec(const builtin_interfaces::msg::Time & now,
                    const builtin_interfaces::msg::Time & then) const {
    const double s_now = static_cast<double>(now.sec) + 1e-9 * static_cast<double>(now.nanosec);
    const double s_then = static_cast<double>(then.sec) + 1e-9 * static_cast<double>(then.nanosec);
    return s_now - s_then;
  }

  void processCurrentRun(PerLidarState & st) {
    try {
      if (!st.raw_accum_cloud || st.raw_accum_cloud->empty()) {
        throw std::runtime_error("no accumulated point cloud available for calibration");
      }

      if (!output_cloud_dir_.empty()) {
        const std::string raw_pcd = (std::filesystem::path(output_cloud_dir_) /
          (std::string("lidar_") + sanitizeIp(st.lidar_ip) + "_run_" + std::to_string(st.current_run_index) + "_accumulated_raw.pcd")).string();
        pcl::io::savePCDFileBinary(raw_pcd, *st.raw_accum_cloud);
        RCLCPP_INFO(get_logger(), "[%s] saved accumulated raw cloud to: %s", st.lidar_ip.c_str(), raw_pcd.c_str());
      }

      YamlInput yin = loadYamlInput(st.cfg.yaml_path, st.cfg.lidar_ip, st.cfg);
      CalibrationRunRecord rec = processOneRun(
        st.current_run_index, st.lidar_ip, st.raw_accum_cloud, yin, st.cfg, enable_padding_channel_, padding_roi_margin_);
      st.all_runs.push_back(rec);

      if (rec.run_success) {
        printResult(rec.dual_result.selected.measurements, rec.dual_result.selected.solver_result, yin);
        RCLCPP_INFO(get_logger(), "[%s] run %d/%d selected_channel=%s strict_accept=%s final_cost=%.6f",
          st.lidar_ip.c_str(), st.current_run_index + 1, num_repeats_, rec.selected_channel.c_str(),
          rec.strict_accept ? "true" : "false", rec.final_cost);
      } else {
        RCLCPP_WARN(get_logger(), "[%s] run %d/%d failed: %s", st.lidar_ip.c_str(), st.current_run_index + 1, num_repeats_, rec.reject_reason.c_str());
      }

      ++st.current_run_index;
      if (st.current_run_index >= num_repeats_) {
        finalizeLidar(st);
        st.finished = true;
        maybeFinalizeAll();
      } else {
        resetForNextRun(st);
      }
    } catch (const std::exception & e) {
      exit_code_ = static_cast<int>(ExitCode::kRuntimeError);
      RCLCPP_ERROR(get_logger(), "[%s] processCurrentRun failed: %s", st.lidar_ip.c_str(), e.what());
      st.finished = true;
      maybeFinalizeAll();
    }
  }

  void finalizeLidar(PerLidarState & st) {
    try {
      st.batch_summary = aggregateBatchResults(st.all_runs);
      st.figure8_result = verifyFigure8(st.figure8_cfg, st.batch_summary);
      if (st.figure8_result.success &&
        st.figure8_result.pass &&
        st.figure8_result.refined_result_available &&
        st.figure8_cfg.replace_final_result_on_success &&
        st.batch_summary.representative_run_index >= 0 &&
        st.batch_summary.representative_run_index < static_cast<int>(st.batch_summary.all_runs.size())) {

        auto & rep = st.batch_summary.all_runs[st.batch_summary.representative_run_index];
        rep.final_rpy_rad = st.figure8_result.refined_rpy_rad;
        rep.final_t_m = st.figure8_result.refined_t_m;
        rep.final_R = rpyToR(rep.final_rpy_rad);

        st.batch_summary.robust_rpy_rad = rep.final_rpy_rad;
        st.batch_summary.robust_t_m = rep.final_t_m;
        st.batch_summary.robust_q = rotationMatrixToQuaternion(rep.final_R);
      }

      if (st.batch_summary.accepted_runs <= 0) {
        RCLCPP_ERROR(get_logger(), "[%s] batch finished but no accepted runs", st.lidar_ip.c_str());
      } else if (!st.batch_summary.batch_stable) {
        RCLCPP_WARN(get_logger(), "[%s] batch finished but stability check failed", st.lidar_ip.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "[%s] batch finished and stability check passed", st.lidar_ip.c_str());
      }
    } catch (const std::exception & e) {
      exit_code_ = static_cast<int>(ExitCode::kRuntimeError);
      RCLCPP_ERROR(get_logger(), "[%s] finalizeLidar failed: %s", st.lidar_ip.c_str(), e.what());
    }
  }

  void maybeFinalizeAll() {
    bool all_done = !lidar_states_.empty();
    for (const auto & kv : lidar_states_) {
      if (!kv.second.finished) {
        all_done = false;
        break;
      }
    }
    if (!all_done || finished_) return;
    finished_ = true;
    finalizeAll();
  }

  void finalizeAll() {
    try {
      std::map<std::string, BatchCalibrationSummary> summaries;
      std::map<std::string, Figure8VerificationResult> fig8_results;
      bool has_runtime_error = (exit_code_ == static_cast<int>(ExitCode::kRuntimeError));
      bool all_strict = true;
      bool any_accepted = false;

      for (const auto & ip : target_lidar_ips_) {
        const auto it = lidar_states_.find(ip);
        if (it == lidar_states_.end()) continue;
        const auto & st = it->second;
        summaries.emplace(ip, st.batch_summary);
        fig8_results.emplace(ip, st.figure8_result);

        if (st.batch_summary.accepted_runs > 0) any_accepted = true;
        if (st.batch_summary.accepted_runs <= 0) {
          all_strict = false;
        } else if (!st.batch_summary.batch_stable) {
          all_strict = false;
        }
      }

      saveMultiBatchCalibrationYaml(output_result_path_, target_lidar_ips_, summaries, fig8_results);
      if (!output_result_path_.empty()) {
        RCLCPP_INFO(get_logger(), "saved multi-lidar batch calibration yaml to: %s", output_result_path_.c_str());
      }

      if (has_runtime_error) {
        exit_code_ = static_cast<int>(ExitCode::kRuntimeError);
      } else if (!any_accepted) {
        exit_code_ = static_cast<int>(ExitCode::kSolveFailed);
        RCLCPP_ERROR(get_logger(), "all lidar batches finished but no accepted runs");
      } else if (!all_strict) {
        exit_code_ = static_cast<int>(ExitCode::kUsableButNotStrict);
        RCLCPP_WARN(get_logger(), "multi-lidar batch finished but at least one lidar failed stability/acceptance");
      } else {
        exit_code_ = static_cast<int>(ExitCode::kStrictAccept);
        RCLCPP_INFO(get_logger(), "multi-lidar batch finished and all lidar stability checks passed");
      }
    } catch (const std::exception & e) {
      exit_code_ = static_cast<int>(ExitCode::kRuntimeError);
      RCLCPP_ERROR(get_logger(), "finalizeAll failed: %s", e.what());
    }
  }

  std::string sanitizeIp(const std::string & ip) const {
    std::string out = ip;
    std::replace(out.begin(), out.end(), '.', '_');
    std::replace(out.begin(), out.end(), ':', '_');
    return out;
  }

private:
  mutable std::mutex mutex_;
  Config base_cfg_;
  std::string config_path_;
  std::string target_lidar_ip_;
  std::string target_lidar_ips_csv_;
  std::vector<std::string> target_lidar_ips_;
  std::string input_topic_;
  double accumulation_time_sec_{3.0};
  double roi_distance_threshold_{0.02};
  double ransac_distance_threshold_{0.01};
  std::string output_result_path_;
  std::string output_cloud_dir_;
  int num_repeats_{5};
  bool enable_padding_channel_{true};
  double padding_roi_margin_{0.02};
  Figure8VerificationConfig figure8_cfg_template_;

  bool finished_{false};
  int exit_code_{static_cast<int>(ExitCode::kRuntimeError)};
  std::map<std::string, PerLidarState> lidar_states_;
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
