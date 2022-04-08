#include "depthopt/synchronization.h"

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace lietorch;

GroundTruthSync::GroundTruthSync (const Params &_params):
	 _params(_params)
{
	_offset = chrono::milliseconds((long) _params.msOffset);
}

bool GroundTruthSync::groundTruthReady() {
	return _groundTruthSignal.size () > 0;
}

bool GroundTruthSync::markersReady() {
	return _markerQueue.size () == 2;
}

void GroundTruthSync::updateGroundTruth(const GroundTruth &newGroundTruth)
{
	_groundTruthSignal.addBack (newGroundTruth);
	
	if (_groundTruthSignal.size () > _params.queueLength)
		_groundTruthSignal.removeFront ();
}

void GroundTruthSync::addSynchronizationMarker (const Time &markerTime, boost::optional<float> &expiredMs, boost::optional<float> &futureMs)
{
	Time adjusted = markerTime + _offset;
	_markerQueue.push (adjusted);
	
	if (_groundTruthSignal.size () > 0) {
		if (adjusted < _groundTruthSignal.timeStart ())
			expiredMs = chrono::duration_cast<chrono::duration<float, std::milli>> (_groundTruthSignal.timeStart () - adjusted).count();
		
		if (adjusted > _groundTruthSignal.timeEnd ())
			futureMs = chrono::duration_cast<chrono::duration<float, std::milli>> (adjusted - _groundTruthSignal.timeEnd ()).count ();
		
	}

	if (_markerQueue.size() > 2)
		_markerQueue.pop ();
}

TargetGroup GroundTruthSync::getLastRelativeGroundTruth() const
{
	lietorch::Pose first = _groundTruthSignal(_markerQueue.front ());
	lietorch::Pose second = _groundTruthSignal(_markerQueue.back ());
	
	return pose_cast<TargetGroup> (first.inverse () * second);
}

const GroundTruthSync::MarkerQueue &GroundTruthSync::markers() const {
	return _markerQueue;
}
