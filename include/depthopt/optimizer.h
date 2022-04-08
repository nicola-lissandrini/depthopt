#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <torch/all.h>
#include <lietorch/pose.h>
#include "depthopt/definitions.h"

#include "landscape.h"

class CostFunction
{
public:
	using Tangent = typename TargetGroup::Tangent;
	using Coeffs = typename TargetGroup::DataType;
	using Vector = typename TargetGroup::Vector;

	struct Info {
		virtual ~Info () = default;

		DEF_SHARED (Info);
	};

	CostFunction () = default;

	virtual Vector value (const TargetGroup &x) = 0;
	virtual Tangent gradient (const TargetGroup &x) = 0;
	virtual bool isReady () const = 0;
	virtual Info::Ptr getInfo () const { return nullptr; }

	DEF_SHARED (CostFunction)
};

#define COST_FUNCTION_INHERIT_TRAITS(Base) \
    using Tangent = typename Base::Tangent;\
    using Coeffs = typename Base::Coeffs;\
    using Vector = typename Base::Vector;

class PointcloudMatch : public CostFunction
{
	COST_FUNCTION_INHERIT_TRAITS (CostFunction)

public:
	struct Params {
		int batchSize;
		bool stochastic;
		bool reshuffleBatchIndexes;
	};

	struct Pointclouds : public CostFunction::Info {
		torch::Tensor old;
		torch::Tensor predicted;
		torch::Tensor next;

		//Pointclouds (const Pointclouds &) = default;
		DEF_SHARED (Pointclouds)
	};

	PointcloudMatch (const Landscape::Params &landscapeParams,
				  const Params &pointcloudMatchParams);

	void updatePointcloud (const torch::Tensor &pointcloud);
	torch::Tensor pointcloud () const;
	torch::Tensor oldPointcloudBatch (bool clipUninformative = true) const;
	Vector value (const TargetGroup &x) override;
	Tangent gradient (const TargetGroup &x) override;
	bool isReady () const override;
	CostFunction::Info::Ptr getInfo () const override;

	DEF_SHARED(PointcloudMatch)

private:
	Params _params;
	Landscape _landscape;
	torch::Tensor _lastPredicted;
	torch::Tensor _oldPointcloud;
	nlib::ReadyFlagsStr _flags;
	lietorch::OpFcn _sumOut;
};

class Optimizer
{
	using Vector = TargetGroup::Vector;
	using Coeffs = TargetGroup::DataType;

public:
	using History = std::vector<TargetGroup>;
	using Histories = std::vector<History>;

	struct Results {
		TargetGroup estimate;
		History history;
		Histories histories;
		CostFunction::Info::Ptr costInfoPtr;

		Results &operator = (const Results &) = default;
	};

	enum InitializationType {
		INITIALIZATION_IDENTITY = 0,
		INITIALIZATION_LAST
	};

	struct Params {
		torch::Tensor stepSizes;
		torch::Tensor normWeights;
		InitializationType initializationType;
		float threshold;
		int maxIterations;
		struct {
			int count;
			float scatter;
		} localMinHeuristics;
		bool recordHistory;
		bool disable;
	};

	Optimizer (const Params &params,
			 const CostFunction::Ptr &costFunction);

	void optimize ();
	Results results () const;
	void localMinHeuristics ();

	DEF_SHARED(Optimizer)

private:
	void optimize (const TargetGroup &initialization);
	TargetGroup getInitialValue ();

private:
	int _seq;
	Params _params;
	CostFunction::Ptr _costFunction;
	Results _results;
};


#endif // OPTIMIZER_H
