//rewrite in mrs style

#include <memory>
#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include "image_transport/image_transport.hpp"

/* ROS includes for working with OpenCV and images */
#include <image_transport/camera_subscriber.hpp>

#include <opencv2/imgproc.hpp>

#include "SuperPoint.h"

//mrs
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>

/* custom helper functions from our library */
#include <mrs_lib/param_loader.h>
#include <mrs_lib/node.h>
//#include <mrs_lib/timer_handler.h>



namespace superpoint_test{ 


class SuperPointDetectorNode : public mrs_lib::Node {
public:
  SuperPointDetectorNode(rclcpp::NodeOptions options);
  
private:
  void initialize();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  
  rclcpp::Node::SharedPtr node_;
  rclcpp::Clock::SharedPtr clock_;

  std::string weights_path_;
  std::string image_topic_;
  double threshold_;
  bool do_nms_;
  bool use_cuda_;
  //std::vector<int64_t> img_size_;
  
  torch::Device device_{torch::kCPU};

  std::shared_ptr<ORB_SLAM2::SuperPoint> model_;
  std::unique_ptr<ORB_SLAM2::SPDetector> detector_;

  // image transport
  image_transport::Subscriber sub_image_;
  image_transport::Publisher pub_debug_;

  /* flags */
  std::atomic<bool> is_initialized_ = false;
};


//class constuctor
SuperPointDetectorNode::SuperPointDetectorNode(rclcpp::NodeOptions options) : Node("superpoint_test", options) {
  initialize();
}

void SuperPointDetectorNode::initialize() {
  //this- point to current SuperPoint node instance, then we ge get node SharedPtr
  node_  = this->this_node_ptr();
  clock_ = node_->get_clock();

  // -------- params (mrs_lib ) --------
  mrs_lib::ParamLoader pl(node_);
  pl.addYamlFileFromParam("config");        
  pl.loadParam("weights_path", weights_path_);
  pl.loadParam("image_topic", image_topic_);
  pl.loadParam("threshold", threshold_);
  pl.loadParam("do_nms", do_nms_);
  pl.loadParam("use_cuda", use_cuda_);
  

  if (!pl.loadedSuccessfully()) {
    RCLCPP_ERROR(node_->get_logger(), "failed to load parameters");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_INFO(node_->get_logger(), "========== SuperPoint configuration ==========");
  RCLCPP_INFO(node_->get_logger(), "weights_path: %s", weights_path_.c_str());
  RCLCPP_INFO(node_->get_logger(), "image_topic:  %s", image_topic_.c_str());
  RCLCPP_INFO(node_->get_logger(), "threshold:    %.6f", threshold_);
  RCLCPP_INFO(node_->get_logger(), "do_nms:       %s", do_nms_ ? "true" : "false");
  RCLCPP_INFO(node_->get_logger(), "use_cuda:     %s", use_cuda_ ? "true" : "false");
  RCLCPP_INFO(node_->get_logger(), "device:       %s", device_.is_cuda() ? "CUDA" : "CPU");
  RCLCPP_INFO(node_->get_logger(), "==============================================");

  
  // -------- init model --------
  model_ = std::make_shared<ORB_SLAM2::SuperPoint>();

  // Load weights into libtorch module
  try {
    model_->load_weights(weights_path_);
  } catch (const c10::Error &e) {
    RCLCPP_ERROR(node_->get_logger(),
                  "torch::load failed for '%s'.\n"
                  "This usually means the file is a Python state_dict (.pth) not directly loadable in C++.\n"
                  "Error: %s",
                  weights_path_.c_str(), e.what());
    throw;
  }
  RCLCPP_INFO(node_->get_logger(), "Weights loaded successfully");


  //set device
  bool cuda_available_ = torch::cuda::is_available();
  RCLCPP_INFO(node_->get_logger(), "CUDA available: %s", cuda_available_ ? "true" : "false");
  
  device_ = torch::kCPU;
  if (use_cuda_ && cuda_available_) {
    device_ = torch::kCUDA;
    RCLCPP_INFO(node_->get_logger(), "Using CUDA");
  } else {
    RCLCPP_INFO(node_->get_logger(), "Using CPU");
  }
  
  model_->to(device_);
  model_->eval();

  // -------- init detector --------
  detector_ = std::make_unique<ORB_SLAM2::SPDetector>(model_);

  // -------- image_transport (mrs style) --------
  image_transport::ImageTransport it(node_);

  // Subscribe: if you already have a topic name, use it directly.
  // (In the MRS example they use "~/<name>" convention for remapping) :contentReference[oaicite:2]{index=2}
  sub_image_ = it.subscribe(
    image_topic_, 10,
    std::bind(&SuperPointDetectorNode::onImage, this, std::placeholders::_1)
  );
  
  // Publish debug image
  pub_debug_ = it.advertise("superpoint/debug", 1);

  RCLCPP_INFO(node_->get_logger(), "initialized, subscribing to: %s", image_topic_.c_str());
  is_initialized_ = true;

}


void SuperPointDetectorNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  if (!is_initialized_) return;
  

  // toCvShare avoids copying the image data and instead copies only the (smart) constpointer
  // to the data. Then, the data cannot be changed (it is potentially shared between multiple nodes) and
  // it is automatically freed when all pointers to it are released. If you want to modify the image data,
  // use toCvCopy (see https://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages),
  // or copy the image data using cv::Mat::copyTo() method.
  // Adittionally, toCvShare and toCvCopy will convert the input image to the specified encoding
  // if it differs from the one in the message. Try to be consistent in what encodings you use throughout the code.
  const std::string color_encoding = "bgr8";
  auto cv_ptr = cv_bridge::toCvShare(msg, color_encoding);
  const cv::Mat& img_bgr = cv_ptr->image;

  // ----Superpoint interference ----
  cv::Mat gray;
  if (cv_ptr->image.channels() == 1) {
    gray = cv_ptr->image;
  } else {
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
  }

  /*
  // Resize image if needed
  if (img_size_.size() == 2 && (gray.rows != img_size_[1] || gray.cols != img_size_[0])) {
    cv::resize(gray, gray, cv::Size(img_size_[0], img_size_[1]));
  }
  */
  // SPDetector expects CV_8U grayscale.
  if (gray.type() != CV_8UC1) {
    gray.convertTo(gray, CV_8U);
  }

  // IMPORTANT: Your detect() moves model->to(device) every frame currently.
  // If your SPDetector::detect does model->to(device) internally, it's slower but still works.
  // (Recommended: remove model->to(device) from detect() and keep it only here.)
  try {
    const auto start = std::chrono::high_resolution_clock::now();
    detector_->detect(gray, use_cuda_);

    std::vector<cv::KeyPoint> kpts;
    detector_->getKeyPoints(
      static_cast<float>(threshold_),
      0, gray.cols, 0, gray.rows,
      kpts,
      do_nms_
    );
    const auto end = std::chrono::high_resolution_clock::now();

    RCLCPP_INFO(node_->get_logger(), "detect and getkeypoint -  %zu keypoints in %.1f ms",
      kpts.size(),
      std::chrono::duration<double, std::milli>(end - start).count()
    );

    //Compute descriptors -- not used in this test node
    cv::Mat desc;
    detector_->computeDescriptors(kpts, desc);

    // Draw keypoints
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto &kp : kpts) {
      cv::circle(vis, kp.pt, 2, cv::Scalar(0,255,0), -1);
    }

    // Publish OpenCV image through image_transport 
    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = color_encoding;
    out.image = vis;

    pub_debug_.publish(out.toImageMsg());

    /*
    RCLCPP_INFO_THROTTLE(node_->get_logger(), clock_, 1000,
      "kpts=%zu desc=%dx%d (type=%d) img=%dx%d",
      kpts.size(), desc.rows, desc.cols, desc.type(), gray.cols, gray.rows);
    */
  } catch (const c10::Error &e) {
    RCLCPP_ERROR(node_->get_logger(), "SuperPoint inference error: %s", e.what());
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node_->get_logger(), "Error: %s", e.what());
  }
}

} // namespace superpoint_test
RCLCPP_COMPONENTS_REGISTER_NODE(superpoint_test::SuperPointDetectorNode)