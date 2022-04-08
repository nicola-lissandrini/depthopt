#include "depthopt/depthopt_node.h"

using namespace std;
using namespace nlib;
using namespace torch;


DepthOptNode::DepthOptNode (int &argc, char **argv, const string &name, uint32_t options):
	 Base (argc, argv, name, options)
{
	init<ModFlow> ();

	_pointcloudChannel = sources()->declareSource<Timed<Tensor>> ("pointcloud_source");
	_groundTruthChannel = sources()->declareSource<Timed<Tensor>> ("ground_truth_source");

	sinks()->declareSink ("publish_tensor", &DepthOptNode::publishTensor, this);
	sinks()->declareSink ("publish_pointcloud", &DepthOptNode::publishPointcloud, this);

	finalizeModFlow ();
}

void DepthOptNode::initParams()
{
	_params = {
		.artificialPointcloud = _nlParams.get<bool> ("artificial_pointcloud")
	};
}

vector<const char *> outputStrings = {"estimate",
							   "history",
							   "error_history",
							   "final_error",
							   "relative_ground_truth",
							   "pointcloud_old",
							   "pointcloud_predicted",
							   "pointcloud_next",
							   "misc"};

void DepthOptNode::initROS ()
{
	addSub ("pointcloud", _nlParams.get<int> ("topics/queue_size", 1), &DepthOptNode::pointcloudCallback);
	addSub ("ground_truth", _nlParams.get<int> ("topics/queue_size", 1), &DepthOptNode::groundTruthCallback);

	// Automatically generate publishers of interest
	vector<ProcessOutputs::OutputType> outputTypes = _nlParams.get<ProcessOutputs::OutputType, vector> ("topics/outputs", outputStrings);
	string prefix = _nlParams.get<string> ("topics/output_prefix");

	for (auto currType : outputTypes) {
		string topicName = prefix + "/"  + outputStrings[currType];
		switch (currType) {
		case ProcessOutputs::OUTPUT_POINTCLOUD_OLD:
		case ProcessOutputs::OUTPUT_POINTCLOUD_PREDICTED:
		case ProcessOutputs::OUTPUT_POINTCLOUD_NEXT:
			addPub<sensor_msgs::PointCloud2> (outputStrings[currType], topicName, 1);
			break;
		default:
			addPub<std_msgs::Float32MultiArray> (outputStrings[currType], topicName, 1);
			break;
		}
	}
}

void DepthOptNode::pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloudMsg)
{
	Timed<Tensor> timedPointcloud;

	pointcloudMsgToTensor (timedPointcloud.obj (), pointcloudMsg, _params.artificialPointcloud);
	headerToTime (timedPointcloud.time (), pointcloudMsg.header);

	sources()->callSource (_pointcloudChannel, timedPointcloud);
}

void DepthOptNode::groundTruthCallback (const geometry_msgs::TransformStamped &groundTruthMsg)
{
	Timed<Tensor> timedGroundTruth;

	transformToTensor (timedGroundTruth.obj (), groundTruthMsg.transform);
	headerToTime (timedGroundTruth.time (), groundTruthMsg.header);

	sources()->callSource (_groundTruthChannel, timedGroundTruth);
}

void DepthOptNode::publishTensor (const torch::Tensor &tensor, ProcessOutputs::OutputType outputType)
{
	std_msgs::Float32MultiArray tensorMsg;

	tensorToMsg (tensorMsg, tensor, {static_cast<float> (outputType)});
	try {
		publish (outputStrings[outputType], tensorMsg);
	} catch (const std::out_of_range &e) { } // ignore non declared outputs
}

void DepthOptNode::publishPointcloud (const at::Tensor &tensor, ProcessOutputs::OutputType outputType)
{
	sensor_msgs::PointCloud2 pointcloudMsg;
	tensorToPointcloud (pointcloudMsg, tensor);
	try {
		publish (outputStrings[outputType], pointcloudMsg);
	} catch (const std::out_of_range &e) { } // ignore non declared outputs
}

void DepthOptNode::onSynchronousClock (const ros::TimerEvent &timeEvent) {

}

int main (int argc, char *argv[])
{
	//WAIT_GDB;
	DepthOptNode don(argc, argv, "depthopt");

	return don.spin ();
}
