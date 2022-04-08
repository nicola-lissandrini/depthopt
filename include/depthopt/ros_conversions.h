#ifndef ROS_CONVERSIONS_H
#define ROS_CONVERSIONS_H

#include <nlib/nl_multiarray_ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float32MultiArray.h>

#include <torch/all.h>

#include "depthopt/definitions.h"

void headerToTime (Time &timeOut, const std_msgs::Header &headerMsg);

void pointcloudMsgToTensor (torch::Tensor &tensorOut,
					   const sensor_msgs::PointCloud2 &pointcloudMsg,
					   bool artificialPointcloud);

void transformToTensor (torch::Tensor &out, const geometry_msgs::Transform &transformMsg);

void tensorToPointcloud (sensor_msgs::PointCloud2 &outputMsg, const torch::Tensor &tensor);

void tensorToMsg (std_msgs::Float32MultiArray &outputMsg, const torch::Tensor &tensor, const std::vector<float> &extraData);

#endif // ROS_CONVERSIONS_H
