#include "depthopt/optimizer.h"

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace lietorch;

PointcloudMatch::PointcloudMatch (const Landscape::Params &landscapeParams,
						    const Params &params):
	 _landscape (landscapeParams),
	 _params(params)
{
	_flags.addFlag ("old_pointcloud");
	_flags.addFlag ("new_pointcloud");

	_sumOut = [] (const Tensor &t) { return t.sum(0); };
}

void PointcloudMatch::updatePointcloud (const Tensor &pointcloud)
{
	if (_flags["new_pointcloud"]) {
		_flags.set ("old_pointcloud");
		_oldPointcloud = _landscape.pointcloud ();
	}

	_flags.set ("new_pointcloud");
	_landscape.setPointcloud (pointcloud);
}

Tensor PointcloudMatch::pointcloud () const {
	return _landscape.pointcloud ();
}

Tensor PointcloudMatch::oldPointcloudBatch (bool clipUninformative) const
{
	if (!_params.stochastic)
		return _oldPointcloud;

	if (clipUninformative) {
		Tensor oldBatchValidIdxes = _landscape.selectInformativeIndexes (_landscape.batchIndexes (), _oldPointcloud);

		return _oldPointcloud.index ({oldBatchValidIdxes, Ellipsis});
	} else
		return _oldPointcloud.index ({_landscape.batchIndexes (), Ellipsis});
}

PointcloudMatch::Vector
    PointcloudMatch::value (const TargetGroup &x)
{
	_landscape.shuffleBatchIndexes ();

	Tensor predicted = x * oldPointcloudBatch ();
	Tensor totalValue = torch::zeros ({1}, kFloat);

	for (int i = 0; i < predicted.size (0); i++) {
		const Tensor &curr = predicted[i];
		totalValue += _landscape.value (curr);
	}

	return totalValue;
}

PointcloudMatch::Tangent
    PointcloudMatch::gradient (const TargetGroup &x)
{
	Tangent totalGradient;
	Tensor landscapeGradient, jacobian;

	_landscape.shuffleBatchIndexes ();

	_lastPredicted = x * oldPointcloudBatch ();

	if (_params.reshuffleBatchIndexes)
		_landscape.shuffleBatchIndexes ();

	landscapeGradient = _landscape.gradient (_lastPredicted);
	totalGradient = x.differentiate (landscapeGradient, _lastPredicted,
							   _sumOut, jacobian);

	return totalGradient;
}

bool PointcloudMatch::isReady() const {
	return _flags.all ();
}

CostFunction::Info::Ptr PointcloudMatch::getInfo() const
{
	Pointclouds::Ptr infoPtr = make_shared<Pointclouds> ();

	infoPtr->old = oldPointcloudBatch ();
	infoPtr->predicted = _lastPredicted;
	infoPtr->next = _landscape.pointcloudBatch ();

	return infoPtr;
}

Optimizer::Optimizer (const Params &params,
				  const CostFunction::Ptr &costFunction):
	 _params(params),
	 _costFunction(costFunction),
	 _seq(0)
{}

TargetGroup Optimizer::getInitialValue () {
	switch (_params.initializationType) {
	case INITIALIZATION_IDENTITY:
		return TargetGroup::Identity ();
	case INITIALIZATION_LAST:
		return _results.estimate;
	}
}

void Optimizer::optimize () {
	_results.histories.clear ();
	optimize (getInitialValue ());
}

void Optimizer::optimize (const TargetGroup &initialization)
{
	TargetGroup state = initialization;
	TargetGroup nextState;
	bool terminationCondition = false;
	int iterations = 0;

	_results.history.clear ();

	while (!terminationCondition) {
		if (_params.recordHistory)
			_results.history.push_back (state);
		
		nextState = state - _costFunction->gradient (state) * _params.stepSizes;

		terminationCondition =// (nextState.dist (state, _params.normWeights).item ().toFloat () < _params.threshold)
						    (iterations >= _params.maxIterations);
		state = nextState;
		iterations++;
	}

	_results.costInfoPtr = _costFunction->getInfo ();
	_results.estimate = state;
	_results.histories.push_back (_results.history);
	_seq++;
}

void Optimizer::localMinHeuristics ()
{
	// Get first estimate
	optimize ();
	TargetGroup firstEstimate = _results.estimate;

	    // Draw new initializations
	Tensor initializationsNoiseTangent = torch::normal (0., _params.localMinHeuristics.scatter,
											  {_params.localMinHeuristics.count,
											   TargetGroup::Tangent::Dim});
	Tensor initializationsTangent = _results.estimate.log ().coeffs + initializationsNoiseTangent;

	for (int i = 0; i < _params.localMinHeuristics.count; i++) {
		TargetGroup currentInitialization = TargetGroup::Tangent (initializationsTangent[i]).exp ();

		optimize (currentInitialization);
	}

	_results.estimate = firstEstimate;
}

Optimizer::Results Optimizer::results() const {
	return _results;
}











