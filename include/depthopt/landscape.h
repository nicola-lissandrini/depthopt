#ifndef LANDSCAPE_H
#define LANDSCAPE_H

#include <functional>
#include <torch/all.h>

#include <nlib/nl_utils.h>

class Smoother
{
	struct Params {
		int dim;
		int samplesCount;
		float radius;
	};

public:
	using Fcn = std::function<torch::Tensor(torch::Tensor)>;

	Smoother (int dim, int samplesCount, float radius);

	virtual torch::Tensor evaluate (const Fcn &f, const torch::Tensor &x) = 0;

	DEF_SHARED (Smoother)
	
protected:
	Params _params;
};

class MontecarloSmoother : public Smoother
{
public:
	MontecarloSmoother (int dim, int samplesCount, float radius):
		 Smoother (dim, samplesCount, radius)
	{}
	
	torch::Tensor evaluate(const Fcn &f, const torch::Tensor &x) override;
	
	DEF_SHARED (MontecarloSmoother)
};

class RiemannSmoother : public Smoother
{
public:
	RiemannSmoother (int dim, int samplesCount, float radius):
		 Smoother (dim, samplesCount, radius)
	{}
	// TODO
	torch::Tensor evaluate(const Fcn &f, const torch::Tensor &x) override;
	
	DEF_SHARED (RiemannSmoother)
};

class Landscape
{
public:
	static constexpr int Dim = D_3D;
	
	struct Params {
		float measureRadius;
		float smoothRadius;
		nlib::Range clipArea;
		int precision;
		int batchSize; // number of simultaneous landscape points in evaluation
		int decimation;
		bool stochastic;
	};
	
	Landscape (const Params &params);
	
	void setPointcloud (const torch::Tensor &pointcloud);
	void shuffleBatchIndexes ();
	torch::Tensor batchIndexes () const;
	torch::Tensor pointcloud () const;
	torch::Tensor pointcloudBatch () const;
	
	torch::Tensor value (const torch::Tensor &p);
	torch::Tensor gradient (const torch::Tensor &p);
	
	torch::Tensor selectInformativeIndexes (const torch::Tensor &indexes,
									const torch::Tensor &pointcloud) const;
	
	DEF_SHARED (Landscape)
	
private:
	torch::Tensor peak (const torch::Tensor &v) const;
	torch::Tensor preSmoothValue (const torch::Tensor &p) const;
	torch::Tensor preSmoothGradient (const torch::Tensor &p) const;
	
	float computeNoAmplificationGain () const;
	float computeSmoothGain () const;
	
private:
	Params _params;
	Smoother::Fcn _valueLambda;
	Smoother::Fcn _gradientLambda;
	Smoother::Ptr _smoother;
	torch::Tensor _pointcloud;
	torch::Tensor _batchIndexes;
	torch::Tensor _xGrid, _yGrid;
	nlib::ReadyFlagsStr _flags;
	float _smoothGain;
};


#endif // LANDSCAPE_H
