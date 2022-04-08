#include "ros_conversions.h"

using namespace std;
using namespace nlib;
using namespace torch;

void headerToTime (Time &timeOut, const std_msgs::Header &headerMsg) {
	timeOut = Time () + chrono::duration_cast<Duration> (chrono::nanoseconds(headerMsg.stamp.toNSec ()));
}

void pointcloudMsgToTensor (Tensor &tensorOut,
					   const sensor_msgs::PointCloud2 &pointcloudMsg,
					   bool artificialPointcloud)
{
	const int pointcloudSize = pointcloudMsg.height * pointcloudMsg.width;
	if (artificialPointcloud)
		tensorOut = torch::from_blob ((void *) pointcloudMsg.data.data(),
								{pointcloudSize, 3},
								torch::TensorOptions().dtype(torch::kFloat32));
	else
		tensorOut =  torch::from_blob ((void *) pointcloudMsg.data.data(),
								{pointcloudSize, 4},
								torch::TensorOptions().dtype (torch::kFloat32))
					 .index ({indexing::Ellipsis, indexing::Slice(0,3)});
}

void transformToTensor (Tensor &out, const geometry_msgs::Transform &transformMsg)
{
	out = torch::tensor ({transformMsg.translation.x,
					  transformMsg.translation.y,
					  transformMsg.translation.z,
					  transformMsg.rotation.x,
					  transformMsg.rotation.y,
					  transformMsg.rotation.z,
					  transformMsg.rotation.w}, kFloat);
}

void tensorToPointcloud (sensor_msgs::PointCloud2 &outputMsg, const torch::Tensor &tensor)
{
	sensor_msgs::PointField pointFieldProto;
	pointFieldProto.datatype = 7;
	pointFieldProto.count = 1;

	pointFieldProto.name = "x";
	pointFieldProto.offset = 0;
	outputMsg.fields.push_back (pointFieldProto);
	pointFieldProto.name = "y";
	pointFieldProto.offset = 4;
	outputMsg.fields.push_back (pointFieldProto);
	pointFieldProto.name = "z";
	pointFieldProto.offset = 8;
	outputMsg.fields.push_back (pointFieldProto);

	outputMsg.height = 1;
	outputMsg.width = tensor.size(0);

	outputMsg.data = vector<uint8_t> ((uint8_t*)tensor.data_ptr (), (uint8_t*)tensor.data_ptr () + tensor.numel () * tensor.element_size ());
	outputMsg.header.frame_id = "map";
	outputMsg.point_step = D_3D * tensor.element_size ();
}

void tensorToMsg (std_msgs::Float32MultiArray &outputMsg, const torch::Tensor &tensor, const std::vector<float> &extraData)
{
	MultiArray32Manager array(vector<int> (tensor.sizes().begin(), tensor.sizes().end()), extraData.size ());

	memcpy (array.data (), extraData.data (), extraData.size () * sizeof (float));
	memcpy (array.data () + extraData.size (), tensor.data_ptr(), tensor.element_size() * tensor.numel ());

	outputMsg = array.msg();
}
