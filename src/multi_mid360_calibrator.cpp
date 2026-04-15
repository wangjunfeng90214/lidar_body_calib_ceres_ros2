#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/point_tests.h>

#include <yaml-cpp/yaml.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

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

Matrix3d rodrigues(const Vector3d & w) {
  const double theta = w.norm();
  if (theta < 1e-12) {
    return Matrix3d::Identity();
  }
  const Vector3d k = w / theta;
  const Matrix3d K = skew(k);
  return Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
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

string sanitizeIpToFrame(const string & ip) {
  string out = "lidar_";
  for (char c : ip) {
    out.push_back(c == '.' ? '_' : c);
  }
  return out;
}

struct PlaneMeasurement {
  Vector3d n_lidar;
  double d_lidar{0.0};
  Vector3d N0_body;
  double D0_body{0.0};  // plane prior in form N^T p = D
  std::size_t plane_index{0};
  std::size_t inlier_count{0};
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

    // Small rotation on body prior plane normal.
    const Eigen::Matrix<T, 3, 3> R_delta = rodriguesTemplate<T>(dtheta_v);
    Eigen::Matrix<T, 3, 1> Ni = R_delta * N0;
    const T Ni_norm = ceres::sqrt(Ni.squaredNorm() + T(1e-12));
    Ni /= Ni_norm;
    const T Di = D0 + dD[0];

    Eigen::Matrix<T, 3, 1> nB_hat = R * nL;
    const T nB_hat_norm = ceres::sqrt(nB_hat.squaredNorm() + T(1e-12));
    nB_hat /= nB_hat_norm;
    const T dB_hat = dL - nB_hat.dot(t_v);

    residuals[0] = T(wn_) * (nB_hat[0] - Ni[0]);
    residuals[1] = T(wn_) * (nB_hat[1] - Ni[1]);
    residuals[2] = T(wn_) * (nB_hat[2] - Ni[2]);
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
    declare_parameter<double>("wn", 10.0);
    declare_parameter<double>("wd", 20.0);
    declare_parameter<double>("wp", 2.0);
    declare_parameter<double>("wD", 5.0);
    declare_parameter<bool>("publish_tf", true);
    declare_parameter<string>("output_result_path", "/tmp/lidar_body_calib_result.yaml");

    config_path_ = get_parameter("config_path").as_string();
    target_lidar_ip_ = get_parameter("target_lidar_ip").as_string();
    accumulation_time_sec_ = get_parameter("accumulation_time_sec").as_double();
    roi_distance_threshold_ = get_parameter("roi_distance_threshold").as_double();
    ransac_distance_threshold_ = get_parameter("ransac_distance_threshold").as_double();
    min_plane_points_ = get_parameter("min_plane_points").as_int();
    voxel_leaf_size_ = get_parameter("voxel_leaf_size").as_double();
    wn_ = get_parameter("wn").as_double();
    wd_ = get_parameter("wd").as_double();
    wp_ = get_parameter("wp").as_double();
    wD_ = get_parameter("wD").as_double();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    output_result_path_ = get_parameter("output_result_path").as_string();

    if (!loadConfig(config_path_)) {
      throw std::runtime_error("Failed to load config: " + config_path_);
    }

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
      cfg.roll = node["roll"].as<double>();
      cfg.pitch = node["pitch"].as<double>();
      cfg.yaw = node["yaw"].as<double>();

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

      if (cfg.planes.size() < 3) {
        RCLCPP_WARN(get_logger(), "Lidar %s has only %zu priors. Calibration is best with >=3 planes.",
          cfg.ip.c_str(), cfg.planes.size());
      }

      lidar_configs_[cfg.ip] = cfg;
      lidar_states_[cfg.ip] = LidarState{};
      RCLCPP_INFO(get_logger(), "Loaded lidar %s from topic %s with %zu planes.",
        cfg.ip.c_str(), cfg.topic.c_str(), cfg.planes.size());
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    voxelDownsample(state.accumulated, filtered, voxel_leaf_size_);
    RCLCPP_INFO(get_logger(), "Lidar %s accumulated %.2fs, raw=%zu, voxel=%zu",
      ip.c_str(), elapsed, state.accumulated->size(), filtered->size());

    const auto result = calibrateSingleLidar(lidar_configs_.at(ip), filtered, msg->header.frame_id);
    if (result.has_value()) {
      writeResultYaml(ip, *result);
      if (publish_tf_) {
        publishTransform(ip, *result, msg->header.frame_id);
      }
      publishMarkers(ip, *result, msg->header.frame_id);
      state.calibrated = true;
      RCLCPP_INFO(get_logger(), "Calibration complete for %s", ip.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "Calibration failed for %s. Restarting accumulation window.", ip.c_str());
      state.started = false;
      state.accumulated->clear();
    }
  }

  struct CalibrationResult {
    Vector3d rpy_opt;
    Vector3d t_opt;
    Matrix3d R_opt;
    double initial_cost{0.0};
    double final_cost{0.0};
    bool success{false};
    string summary;
    vector<PlaneMeasurement> used_measurements;
  };

  std::optional<CalibrationResult> calibrateSingleLidar(
    const LidarConfig & cfg,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const string & input_frame)
  {
    (void)input_frame;
    const auto measurements = extractAndFitPlanes(cfg, cloud);
    if (measurements.size() < 3) {
      RCLCPP_ERROR(get_logger(), "Need at least 3 valid planes, got %zu for %s",
        measurements.size(), cfg.ip.c_str());
      return std::nullopt;
    }

    double rpy[3] = {cfg.roll, cfg.pitch, cfg.yaw};
    double t[3] = {cfg.x, cfg.y, cfg.z};
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

    CalibrationResult result;
    result.rpy_opt = Vector3d(rpy[0], rpy[1], rpy[2]);
    result.t_opt = Vector3d(t[0], t[1], t[2]);
    result.R_opt = rpyToR(result.rpy_opt);
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.success = summary.IsSolutionUsable();
    result.summary = summary.BriefReport();
    result.used_measurements = measurements;

    RCLCPP_INFO(get_logger(), "=== Calibration result for %s ===", cfg.ip.c_str());
    RCLCPP_INFO(get_logger(), "Success: %s", result.success ? "true" : "false");
    RCLCPP_INFO(get_logger(), "Summary: %s", result.summary.c_str());
    RCLCPP_INFO(get_logger(), "Cost: %.8f -> %.8f", result.initial_cost, result.final_cost);
    RCLCPP_INFO(get_logger(), "RPY(deg): [%.6f, %.6f, %.6f]", rad2deg(result.rpy_opt.x()),
      rad2deg(result.rpy_opt.y()), rad2deg(result.rpy_opt.z()));
    RCLCPP_INFO(get_logger(), "t(m): [%.6f, %.6f, %.6f]", result.t_opt.x(), result.t_opt.y(), result.t_opt.z());

    if (!result.success) {
      return std::nullopt;
    }
    return result;
  }

  vector<PlaneMeasurement> extractAndFitPlanes(
    const LidarConfig & cfg,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
  {
    vector<PlaneMeasurement> out;
    const Matrix3d R0 = rpyToR(Vector3d(cfg.roll, cfg.pitch, cfg.yaw));
    const Vector3d t0(cfg.x, cfg.y, cfg.z);

    for (std::size_t i = 0; i < cfg.planes.size(); ++i) {
      const auto & p = cfg.planes[i];
      Vector3d N0 = Vector3d(p[0], p[1], p[2]);
      const double norm = N0.norm();
      if (norm < 1e-9) {
        RCLCPP_WARN(get_logger(), "Plane %zu of %s has invalid normal.", i, cfg.ip.c_str());
        continue;
      }
      N0 /= norm;
      const double d_body_std = p[3] / norm;    // body standard form n^T p + d = 0
      const double D0 = -d_body_std;            // convert to N^T p = D

      const auto pred_lidar_plane = transformPlaneBodyToLidar(R0, t0, N0, d_body_std);
      const auto roi_cloud = extractPlanePoints(cloud, pred_lidar_plane.first, pred_lidar_plane.second, roi_distance_threshold_);
      RCLCPP_INFO(get_logger(), "Plane %zu for %s ROI points=%zu", i, cfg.ip.c_str(), roi_cloud->size());
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

      out.push_back(PlaneMeasurement{n_fit, d_fit, N0, D0, i, inlier_count});
      RCLCPP_INFO(get_logger(), "Plane %zu used: nL=[%.5f %.5f %.5f], dL=%.5f, prior D0=%.5f",
        i, n_fit.x(), n_fit.y(), n_fit.z(), d_fit, D0);
    }
    return out;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanePoints(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const Vector3d & n_lidar,
    double d_lidar,
    double threshold) const
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto & pt : cloud->points) {
      if (!pcl::isFinite(pt)) {
        continue;
      }
      const Vector3d p(pt.x, pt.y, pt.z);
      const double dist = std::abs(n_lidar.dot(p) + d_lidar);
      if (dist < threshold) {
        out->points.push_back(pt);
      }
    }
    out->width = static_cast<uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = false;
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

  void writeResultYaml(const string & ip, const CalibrationResult & result) const {
    YAML::Node root;
    root["ip"] = ip;
    root["success"] = result.success;
    root["summary"] = result.summary;
    root["initial_cost"] = result.initial_cost;
    root["final_cost"] = result.final_cost;

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

    YAML::Node planes;
    for (const auto & m : result.used_measurements) {
      YAML::Node item;
      item["plane_index"] = static_cast<int>(m.plane_index);
      item["inlier_count"] = static_cast<int>(m.inlier_count);
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
      planes.push_back(item);
    }
    root["used_planes"] = planes;

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

  void publishTransform(const string & ip, const CalibrationResult & result, const string & input_frame) {
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

  void publishMarkers(const string & ip, const CalibrationResult & result, const string & input_frame) {
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
  double accumulation_time_sec_{2.0};
  double roi_distance_threshold_{0.08};
  double ransac_distance_threshold_{0.02};
  int min_plane_points_{300};
  double voxel_leaf_size_{0.01};
  double wn_{10.0};
  double wd_{20.0};
  double wp_{2.0};
  double wD_{5.0};
  bool publish_tf_{true};

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
