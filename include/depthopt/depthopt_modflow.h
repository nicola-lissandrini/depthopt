#ifndef DEPTHOPT_MODFLOW_H
#define DEPTHOPT_MODFLOW_H

#include "depthopt/definitions.h"
#include <nlib/nl_modflow.h>
#include "optimizer.h"
#include "synchronization.h"

class DepthOptModFlow : public nlib::NlModFlow
{
public:
	DepthOptModFlow ():
		 nlib::NlModFlow ()
	{}

	void loadModules () override;

	DEF_SHARED (DepthOptModFlow)
};

class WindowModule : public nlib::NlModule
{
public:
	WindowModule (nlib::NlModFlow *modFlow):
		 nlib::NlModule (modFlow, "window")
	{}

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void rawPointcloudSlot (const Timed<torch::Tensor> &timedPointcloud);

private:
	using PointcloudWindow = ReadingWindow<Timed<torch::Tensor>>;
	nlib::Channel _windowedChannel;
	PointcloudWindow::Ptr _pointcloudWindow;
};

class FrequencyEstimatorModule : public nlib::NlModule
{
public:
	FrequencyEstimatorModule (nlib::NlModFlow *modFlow):
		 nlib::NlModule (modFlow, "frequency_estimator")
	{}

	void setupNetwork () override;

	void windowedPointcloudSlot (const Timed<torch::Tensor> &timedPointcloud);

private:
	nlib::Channel _frequencyPeriodChannel;
	nlib::Channel _frequencyEstimateChannel;
	FrequencyEstimator<Clock, Duration> _frequencyEstimator;
};

class GroundTruthSyncModule : public nlib::NlModule
{
public:
	GroundTruthSyncModule (nlib::NlModFlow *modFlow):
		 nlib::NlModule (modFlow, "ground_truth_sync")
	{}

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void windowedPointcloudSlot (const Timed<torch::Tensor> &timedPointcloud);
	void groundTruthSlot (const Timed<torch::Tensor> &timedGroundTruth);

	DEF_SHARED (GroundTruthSyncModule)

private:
	nlib::Channel _readyChannel;
	nlib::Channel _groundTruthChannel;
	GroundTruthSync::Ptr _groundTruthSync;
};

class OptimizerModule : public nlib::NlModule
{
public:
	struct Params {
		bool enableLocalMinHeuristics;
	};

	OptimizerModule (nlib::NlModFlow *modFlow):
		 nlib::NlModule (modFlow, "optimizer")
	{}

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void pointcloudSlot (const Timed<torch::Tensor> &pointcloud);

	DEF_SHARED (OptimizerModule)

private:
	Params _params;
	nlib::Channel _resultsChannel;
	nlib::Channel _usedPointcloudChannel;

	PointcloudMatch::Ptr _costFunction;
	Optimizer::Ptr _optimizer;
};

class ProcessOutputs : public nlib::NlModule
{
public:
	struct Params {
		bool sampleTimeNormalize;
		lietorch::Pose opticalToCameraFrame;
	};

	enum OutputType {
		OUTPUT_ESTIMATE,
		OUTPUT_HISTORY,
		OUTPUT_ERROR_HISTORY,
		OUTPUT_FINAL_ERROR,
		OUTPUT_RELATIVE_GROUND_TRUTH,
		OUTPUT_POINTCLOUD_OLD,
		OUTPUT_POINTCLOUD_PREDICTED,
		OUTPUT_POINTCLOUD_NEXT,
		OUTPUT_MISC
	};

	ProcessOutputs (nlib::NlModFlow *modFlow):
		 nlib::NlModule (modFlow, "outputs")
	{}

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void resultsSlot (const Optimizer::Results &results);
	void groundTruthSlot (const TargetGroup &relativeGroundTruth);
	void updateFrequencySlot (double lastPeriodSec);

	torch::Tensor processEstimate () const;
	torch::Tensor processHistory () const;
	torch::Tensor processErrorHistory () const;
	torch::Tensor processFinalError () const;
	torch::Tensor processRelativeGroundTruth () const;
	torch::Tensor processPointcloud (OutputType type) const;
	torch::Tensor processMisc () const;

	DEF_SHARED (ProcessOutputs)

private:
	double _lastPeriodSec;
	Params _params;
	TargetGroup _lastRelativeGroundTruth;
	Optimizer::Results _results;
	torch::Tensor _pointcloud;

	nlib::Channel _tensorSink;
	nlib::Channel _pointcloudSink;
};


#endif // DEPTHOPT_MODFLOW_H
