#include "depthopt/depthopt_modflow.h"

using namespace std;
using namespace nlib;
using namespace torch;
using namespace lietorch;

void DepthOptModFlow::loadModules ()
{
	loadModule<WindowModule> ();
	loadModule<FrequencyEstimatorModule> ();
	loadModule<GroundTruthSyncModule> ();
	loadModule<OptimizerModule> ();
	loadModule<ProcessOutputs> ();
}

void WindowModule::setupNetwork ()
{
	_windowedChannel = createChannel<Timed<Tensor>> ("windowed_pointcloud");
	requestConnection ("pointcloud_source", &WindowModule::rawPointcloudSlot);
}

void FrequencyEstimatorModule::setupNetwork ()
{
	_frequencyPeriodChannel = createChannel<double> ("pointcloud_last_period");
	_frequencyEstimateChannel = createChannel<double> ("pointcloud_estimate_period");

	requestConnection ("windowed_pointcloud", &FrequencyEstimatorModule::windowedPointcloudSlot);
}

void GroundTruthSyncModule::setupNetwork ()
{
	_readyChannel = createChannel ("ground_truth_ready");
	_groundTruthChannel = createChannel<TargetGroup> ("relative_ground_truth");
	requestConnection ("windowed_pointcloud", &GroundTruthSyncModule::windowedPointcloudSlot);
	requestConnection ("ground_truth_source", &GroundTruthSyncModule::groundTruthSlot);
}

void OptimizerModule::setupNetwork ()
{
	_resultsChannel = createChannel<Optimizer::Results> ("results");
	requestConnection ("windowed_pointcloud", &OptimizerModule::pointcloudSlot);
	requestEnablingChannel ("ground_truth_ready");
}

void ProcessOutputs::setupNetwork ()
{
	requestConnection ("results", &ProcessOutputs::resultsSlot);
	requestConnection ("relative_ground_truth", &ProcessOutputs::groundTruthSlot);
	requestConnection ("pointcloud_last_period", &ProcessOutputs::updateFrequencySlot);
	
	requestEnablingChannel ("ground_truth_ready");
	
	_tensorSink = requireSink<Tensor, OutputType> ("publish_tensor");
	_pointcloudSink = requireSink<Tensor, OutputType> ("publish_pointcloud");
}


/************************
 * Callbacks
 * **********************/

void WindowModule::rawPointcloudSlot (const Timed<Tensor> &timedPointcloud) {
	_pointcloudWindow->add (timedPointcloud);

	if (_pointcloudWindow->isReady ())
		// Skip according to mode:
		// sliding: until queue full
		// decimate: every N packets
		emit (_windowedChannel, _pointcloudWindow->get ());
}

void FrequencyEstimatorModule::windowedPointcloudSlot (const Timed<torch::Tensor> &timedPointcloud)
{
	_frequencyEstimator.tick (timedPointcloud.time ());

	emit ("pointcloud_last_period", _frequencyEstimator.lastPeriodSeconds ());
	emit ("pointcloud_estimate_period", _frequencyEstimator.estimateSeconds ());
}

void GroundTruthSyncModule::windowedPointcloudSlot (const Timed<torch::Tensor> &timedPointcloud) {
	boost::optional<float> expiredMs, futureMs;
	_groundTruthSync->addSynchronizationMarker (timedPointcloud.time (), expiredMs, futureMs);

	if (expiredMs.has_value ()) {
		ROS_WARN_STREAM ("Ground truth matching the supplied timestamp has expired by " << *expiredMs << "ms.\n"
																					  "Using last ground truth stored, probabily outdated.\n"
																					  "Consider increasing 'ground_truth_queue_length' to avoid this issue");

	}

	if (futureMs.has_value ()) {
		ROS_WARN_STREAM("Last received pointcloud is " << *futureMs << "ms in the future of the last ground truth received.\n"
														    "Extrapolating prediction");
	}

	if (_groundTruthSync->markersReady ()) {
		emit (_readyChannel);
		emit (_groundTruthChannel, _groundTruthSync->getLastRelativeGroundTruth ());
	}
}

void GroundTruthSyncModule::groundTruthSlot (const Timed<torch::Tensor> &timedGroundTruth)
{
	GroundTruthSync::GroundTruth updatedGroundTruth;

	updatedGroundTruth.obj () = lietorch::Pose (timedGroundTruth.obj ());
	updatedGroundTruth.time () = timedGroundTruth.time ();

	_groundTruthSync->updateGroundTruth (updatedGroundTruth);
}

void OptimizerModule::pointcloudSlot (const Timed<Tensor> &pointcloud)
{
	_costFunction->updatePointcloud (pointcloud.obj());

	if (!_costFunction->isReady ())
		return;

	if (_params.enableLocalMinHeuristics)
		_optimizer->localMinHeuristics ();
	else
		_optimizer->optimize ();

	emit (_resultsChannel, _optimizer->results ());
}

void ProcessOutputs::resultsSlot (const Optimizer::Results &results) {
	_results = results;

	emit (_tensorSink, processEstimate (), OUTPUT_ESTIMATE);
	emit (_tensorSink, processHistory (), OUTPUT_HISTORY);
	//emit (_tensorSink, processErrorHistory (), OUTPUT_ERROR_HISTORY);
	//emit (_tensorSink, processFinalError (), OUTPUT_FINAL_ERROR);
	emit (_tensorSink, processRelativeGroundTruth (), OUTPUT_RELATIVE_GROUND_TRUTH);
	for (OutputType currType : {OUTPUT_POINTCLOUD_OLD,
						   OUTPUT_POINTCLOUD_PREDICTED,
						   OUTPUT_POINTCLOUD_NEXT})
		emit (_pointcloudSink, processPointcloud (currType), currType);
	emit (_tensorSink, processMisc (), OUTPUT_MISC);
}

TargetGroup sampleTimeNormalize (const TargetGroup &estimate, double periodSec) {
	return (estimate.log () * (1 / periodSec)).exp ();
}

Tensor ProcessOutputs::processEstimate() const {
	TargetGroup estimateCameraFrame;

	if (typeid(TargetGroup) == typeid(Pose))
		estimateCameraFrame = pose_cast<TargetGroup> (_params.opticalToCameraFrame * lietorch::Pose(_results.estimate.coeffs.clone ()));
	else
		estimateCameraFrame = pose_cast<TargetGroup> (_params.opticalToCameraFrame * _results.estimate.coeffs.clone ());

	if (_params.sampleTimeNormalize)
		return sampleTimeNormalize (estimateCameraFrame,
							   _lastPeriodSec).coeffs;
	else
		return estimateCameraFrame.coeffs;
}

Tensor ProcessOutputs::processHistory () const {
	Tensor historiesTensor = torch::empty ({static_cast<long>(_results.histories.size ()),
									static_cast<long>(_results.history.size ()),
									TargetGroup::Dim}, kFloat);

	int i = 0;
	for (auto currHistory : _results.histories) {
		int j = 0;

		for (auto currEstimate : currHistory) {
			historiesTensor[i][j] = currEstimate.coeffs;
			j++;
		}
		i++;
	}

	return historiesTensor;
}

Tensor ProcessOutputs::processRelativeGroundTruth () const {
	if (_params.sampleTimeNormalize)
		return sampleTimeNormalize (_lastRelativeGroundTruth,
							   _lastPeriodSec).coeffs;
	else
		return _lastRelativeGroundTruth.coeffs;
}

Tensor ProcessOutputs::processPointcloud (OutputType type) const {
	PointcloudMatch::Pointclouds::Ptr pointcloudsPtr = dynamic_pointer_cast<PointcloudMatch::Pointclouds> (_results.costInfoPtr);
	Tensor choosenPointcloud;

	switch (type) {
	case OUTPUT_POINTCLOUD_OLD:
		choosenPointcloud = pointcloudsPtr->old;
		break;
	case OUTPUT_POINTCLOUD_PREDICTED:
		choosenPointcloud = pointcloudsPtr->predicted;
		break;
	case OUTPUT_POINTCLOUD_NEXT:
		choosenPointcloud = pointcloudsPtr->next;
		break;
	default:
		break;
	}

	return choosenPointcloud;
}

Tensor ProcessOutputs::processMisc() const
{
	return torch::tensor ({_lastPeriodSec}, kFloat);
}

void ProcessOutputs::groundTruthSlot(const TargetGroup &relativeGroundTruth) {
	_lastRelativeGroundTruth = relativeGroundTruth;
}

void ProcessOutputs::updateFrequencySlot (double lastPeriodSec) {
	_lastPeriodSec = lastPeriodSec;
}

/********************
 * Parameters
 * ******************/

void GroundTruthSyncModule::initParams (const nlib::NlParams &nlParams)
{
	GroundTruthSync::Params groundTruthSyncParams = {
	    .queueLength = nlParams.get<int> ("queue_length"),
	    .msOffset = nlParams.get<float> ("ms_offset")
	};
	
	_groundTruthSync = std::make_shared<GroundTruthSync> (groundTruthSyncParams);
}

void WindowModule::initParams (const nlib::NlParams &nlParams) {
	PointcloudWindow::Params windowParams = {
	    .mode = nlParams.get<PointcloudWindow::Mode> ("mode", {"sliding", "downsample"}),
	    .size = static_cast<uint32_t> (nlParams.get<int> ("size"))
	};
	
	_pointcloudWindow = std::make_shared<PointcloudWindow> (windowParams);
}

void OptimizerModule::initParams(const nlib::NlParams &nlParams)
{
	_params = {
	    .enableLocalMinHeuristics = nlParams.get<bool> ("local_min_heuristics/enable")
	};
	
	Optimizer::Params params = {
	    .stepSizes = nlParams.get<Tensor> ("step_sizes"),
	    .normWeights = nlParams.get<Tensor> ("norm_weights"),
	    .initializationType = nlParams.get<Optimizer::InitializationType> ("initialization_type", {"identity", "last"}),
	    .threshold = nlParams.get<float> ("threshold"),
	    .maxIterations = nlParams.get<int> ("max_iterations"),
	    .localMinHeuristics = {
		   .count = nlParams.get<int> ("local_min_heuristics/count"),
		   .scatter = nlParams.get<float> ("local_min_heuristics/scatter")
	    },
	    .recordHistory = nlParams.get<bool> ("record_history"),
	    .disable = nlParams.get<bool> ("disable")
	};
	
	Landscape::Params landscapeParams = {
	    .measureRadius = nlParams.get<float> ("landscape/measure_radius"),
	    .smoothRadius = nlParams.get<float> ("landscape/smooth_radius"),
	    .clipArea = nlParams.get<nlib::Range> ("landscape/clip_area"),
	    .precision = nlParams.get<int> ("landscape/precision"),
	    .batchSize = nlParams.get<int> ("landscape/batch_size"),
	    .decimation = nlParams.get<int> ("landscape/decimation"),
	    .stochastic = nlParams.get<bool> ("landscape/stochastic")
	};
	
	PointcloudMatch::Params pointcloudMatchParams = {
	    .batchSize = nlParams.get<int> ("cost/batch_size"),
	    .stochastic = nlParams.get<bool> ("cost/stochastic"),
	    .reshuffleBatchIndexes = nlParams.get<bool> ("cost/reshuffle_batch_indexes")
	};

	_costFunction = make_shared<PointcloudMatch> (landscapeParams, pointcloudMatchParams);
	_optimizer = make_shared<Optimizer> (params, _costFunction);
}

void ProcessOutputs::initParams (const NlParams &nlParams) {
	_params = {
	    .sampleTimeNormalize = nlParams.get<bool> ("sample_time_normalize"),
	    .opticalToCameraFrame = lietorch::Pose (Position::Identity (),
									    Quaternion (nlParams.get<Tensor> ("optical_to_camera_frame")))
	};
}






















