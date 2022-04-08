#include "depthopt/landscape.h"

using namespace std;
using namespace torch;
using namespace torch::indexing;

Smoother::Smoother (int dim, int samplesCount, float variance):
	 _params{dim, samplesCount, variance}
{}

Tensor MontecarloSmoother::evaluate (const Fcn &f, const Tensor &x)
{
	Tensor xVar = torch::normal (0., _params.radius,
						    {_params.samplesCount, _params.dim});

	Tensor xEval = xVar + x.unsqueeze (1);

	return f(xEval).mean (1);
}

Landscape::Landscape (const Params &params):
	 _params(params)
{
	assert (_params.measureRadius > _params.smoothRadius && "Smooth radius must be smaller than measure radius");

	_smoothGain = computeSmoothGain ();
	_flags.addFlag ("pointcloud_set", true);

	_smoother = make_shared<MontecarloSmoother> (+Dim, _params.precision, _params.smoothRadius);

	_valueLambda = [this] (const torch::Tensor &p) {
		return this->preSmoothValue (p);
	};
	_gradientLambda = [this] (const torch::Tensor &p) {
		return this->preSmoothGradient (p);
	};

	// Init indexing grid
	auto xyGrid = torch::meshgrid ({torch::arange (0, params.batchSize),
							  torch::arange (0, params.precision)});

	_xGrid = xyGrid[0].reshape({1,-1});
	_yGrid = xyGrid[1].reshape({1,-1});
}

void Landscape::setPointcloud (const torch::Tensor &pointcloud) {
	_pointcloud = pointcloud.slice (0, 0, torch::nullopt, _params.decimation).clone ();

	_flags.set ("pointcloud_set");
}

Tensor Landscape::pointcloud () const {
	return _pointcloud;
}

void Landscape::shuffleBatchIndexes ()
{
	int left = _params.batchSize;
	int used = 0;
	bool validLeft = true;
	Tensor permutation = torch::randperm (_pointcloud.size (0));

	_batchIndexes = torch::empty ({0}, kLong);

	while (left > 0 && validLeft) {
		Tensor selectedPermutation = permutation.slice (0, used, used + left);

		Tensor currentValidIdxes = selectInformativeIndexes (selectedPermutation, _pointcloud);
		if (currentValidIdxes.size(0) == 0)
			validLeft = false;

		_batchIndexes = torch::cat ({_batchIndexes, currentValidIdxes});

		used += left;
		left = _params.batchSize - _batchIndexes.size (0);
	}
}

Tensor Landscape::selectInformativeIndexes (const Tensor &indexes, const Tensor &pointcloud) const {
	Tensor selectedPointcloud = pointcloud.index ({indexes, Ellipsis});
	return indexes.index ({(selectedPointcloud.isfinite ().sum(1) > 0)
						  .logical_and (selectedPointcloud.norm(2,1) > _params.clipArea.min)
						  .logical_and (selectedPointcloud.norm(2,1) < _params.clipArea.max)});
}

Tensor Landscape::batchIndexes () const {
	return _batchIndexes;
}

Tensor Landscape::pointcloudBatch () const
{
	if (!_params.stochastic)
		return _pointcloud;

	return _pointcloud.index ({_batchIndexes, Ellipsis});
}

Tensor Landscape::peak (const Tensor &v) const {
	return (- v * 0.5 / (_params.measureRadius * _params.measureRadius));
}

Tensor Landscape::preSmoothValue (const Tensor &p) const {
	Tensor distToMeasures = (p - _pointcloud).pow(2).sum(2);

	return peak (distToMeasures.index ({distToMeasures.argmin(0)[0],
								 Ellipsis})) * _smoothGain;
}

Tensor Landscape::value (const Tensor &p)
{
	assert (_flags.all ());
	assert (_pointcloud.size (0) > 0);

	return _smoother->evaluate (_valueLambda, p);
}

Tensor Landscape::preSmoothGradient (const Tensor &p) const
{
	Tensor pointcloudCurrent = pointcloudBatch ();
	Tensor pointcloudDiff = p - pointcloudCurrent.unsqueeze (1).unsqueeze (2);
	Tensor distToPointcloud = pointcloudDiff.pow (2).sum (3);
	Tensor collapsedDist, idxes;

	tie (collapsedDist, idxes) = distToPointcloud.min (0);

	Tensor collapsedDiff = pointcloudDiff.permute ({1,2,0,3})
						  .index ({_xGrid.slice (1, 0, idxes.numel ()),
								 _yGrid.slice (1, 0, idxes.numel ()),
								 idxes.reshape ({1,-1}),
								 Ellipsis})
						  .reshape ({-1, _params.precision, 3});

	return collapsedDiff / (_params.measureRadius * _params.measureRadius) * _smoothGain;
}

Tensor Landscape::gradient (const Tensor &p)
{
	assert (_flags.all ());
	assert (_pointcloud.size (0) > 0);

	return _smoother->evaluate (_gradientLambda, p);
}

float Landscape::computeNoAmplificationGain() const {
	return 0.5 * M_SQRT2 * (_params.measureRadius * _params.measureRadius)
		  / (pow(M_PI, 1.5) * pow (_params.smoothRadius, 3))
		  * (2 * pow (_params.measureRadius, 2)
			- 3 * pow (_params.smoothRadius, 2));
}

float Landscape::computeSmoothGain() const {
	return pow (2 * M_PI * pow (_params.smoothRadius, 2), 1.5)
		  * computeNoAmplificationGain ();
}


















