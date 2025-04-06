#include "inference.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <chrono>  // 添加时间库

int main() {
    const std::string model_path = "/home/fyk/fyk/yolo_gpu_detect_openvino/model/rm_buff.onnx";
    const std::string image_path = "/home/fyk/fyk/yolo_gpu_detect_openvino/images/1.png";
    
    cv::Mat image = cv::imread(image_path);
    
    if (image.empty()) {
        std::cerr << "ERROR: image is empty" << std::endl;
        return 1;
    }
    
    const float confidence_threshold = 0.5;
    const float NMS_threshold = 0.5;
    
    yolo::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

    // 添加时间测量
    auto start = std::chrono::high_resolution_clock::now();  // 记录开始时间
    inference.RunInference(image);
    auto end = std::chrono::high_resolution_clock::now();    // 记录结束时间
    
    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "推理时间: " << duration.count() << " 毫秒" << std::endl;
	// std::cout << inference.final_class_id << std::endl;
	// std::cout << inference.final_confidence << std::endl;
	// std::cout << "keypoints: " << inference.final_keypoints.size() << std::endl;

    cv::imshow("image", image);
    cv::waitKey(0);

    return 0;
}