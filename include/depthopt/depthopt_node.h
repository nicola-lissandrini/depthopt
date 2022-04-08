#ifndef DEPTHOPT_NODE_H
#define DEPTHOPT_NODE_H

#include <nlib/nl_node.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float32MultiArray.h>

#include <torch/all.h>

#include "depthopt_modflow.h"
#include "ros_conversions.h"

class DepthOptNode : public nlib::NlNode<DepthOptNode>
{
	NL_NODE(DepthOptNode)

	using ModFlow = DepthOptModFlow;

	struct Params {
		bool artificialPointcloud;
	};

public:
	DepthOptNode (int &argc, char **argv, const std::string &name, uint32_t options = 0);

	void initROS ();
	void initParams ();

	void pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloudMsg);
	void groundTruthCallback (const geometry_msgs::TransformStamped &groundTruthMsg);

	void publishTensor (const torch::Tensor &tensor, ProcessOutputs::OutputType outputType);
	void publishPointcloud (const torch::Tensor &tensor, ProcessOutputs::OutputType outputType);

	DEF_SHARED (DepthOptNode)

protected:
	void onSynchronousClock (const ros::TimerEvent &timeEvent);

private:
	nlib::Channel _pointcloudChannel;
	nlib::Channel _groundTruthChannel;
	Params _params;
};

#endif // DEPTHOPT_NODE_H
