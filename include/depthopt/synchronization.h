#ifndef SYNCHRONIZATION_H
#define SYNCHRONIZATION_H

#include <chrono>
#include <deque>
#include <type_traits>
#include <vector>
#include <nlib/nl_utils.h>
#include "lietorch/algorithms.h"
#include <type_traits>

#include "depthopt/definitions.h"

template<typename T, template<typename ...> class container = std::vector, typename precision = std::chrono::microseconds>
class Signal
{
public:
	using Duration = precision;
	using Clock = std::chrono::system_clock;
	using Time = std::chrono::time_point<Clock, Duration>;
	using Sample = nlib::TimedObject<T, Clock, Duration>;
	using DataType = container<Sample>;
	using iterator = typename DataType::iterator;
	using Neighbors = std::pair<boost::optional<Sample>, boost::optional<Sample>>;

	struct Params {
		uint delay;
	};

private:
	Neighbors neighbors (const Time &t) const {
		auto closest = std::lower_bound (_timedData.begin(), _timedData.end(), t);

		if (closest == _timedData.begin ())
			return {boost::none, *closest};
		if (closest == _timedData.end ())
			return {*std::prev (closest), boost::none};

		return {*std::prev(closest), *closest};
	}

	T extrapolation (const Sample &first, const Sample &second, const Time &t) const {
		using namespace std::chrono;
		return first.obj() + (second.obj() - first.obj()) * (duration_cast<duration<float>> (t - first.time()).count() /
													 duration_cast<duration<float>>(second.time() - first.time()).count ());
	}

public:
	Signal (const Params &_params = {.delay = 1}):
		 _params(_params)
	{}

	void addBack (const Sample &x) {
		_timedData.push_back (x);
	}

	template<typename T1 = T, typename = std::enable_if_t<
							 std::is_same<DataType,
									    std::deque<
										   nlib::TimedObject<T1,
														 std::chrono::system_clock,
														 Duration>>>::value>>
	void removeFront () {
		_timedData.pop_front ();
	}

	Sample operator[] (int i) const {
		return i >= 0 ? _timedData[i] :
				 *prev(_timedData.end(), -i);
	}

	Sample &operator[] (int i) {
		return i >= 0 ? _timedData[i] :
				 *prev(_timedData.end(), -i);
	}

	T at (const Time &t) const
	{
		if (inBounds (t))
			return interpolate (t);
		else
			return extrapolate (t, _params.delay);
	}

	T operator() (const Time &t) const {
		return at (t);
	}

	T operator() (const Duration &afterBegin) {
		return at(_timedData.front().time () + afterBegin);
	}

	uint indexAt (const Time &t) const {
		return std::distance (_timedData.begin(), prev(std::lower_bound(_timedData.begin(), _timedData.end(), t)));
	}

	T interpolate (const Time &t) const
	{
		assert (inBounds (t) && "Supplied time out of signal bounds");

		boost::optional<Sample> before, after;
		std::tie (before, after) = neighbors (t);

		return extrapolation (*before, *after, t);
	}

	T extrapolate (const Time &t, uint delay) const
	{
		assert (!inBounds (t) && "Supplied time in signal bounds. Use interpolate instead");
		Sample first, second;

		if (!inLowerBound (t)) {
			first = _timedData.front ();
			second = *std::next (_timedData.begin (), delay);
		} else {
			first = *std::prev (_timedData.end (), delay + 1);
			second = *std::prev (_timedData.end ());
		}

		return extrapolation (first, second, t);
	}

	Sample before (const Time &t) const {
		assert (inLowerBound (t) && "Supplied time before signal begin");

		return *neighbors (t).first;
	}

	Sample after(const Time &t) const {
		assert (inUpperBound (t) && "Supplied time after signal end");

		return *neighbors (t).second;
	}

	bool inBounds (const Time &t) const {
		return inLowerBound (t) && inUpperBound (t);
	}

	bool inLowerBound (const Time &t) const {
		return t > _timedData.front ().time ();
	}

	bool inUpperBound (const Time &t) const {
		return t < _timedData.back().time ();
	}

	size_t size () const {
		return _timedData.size ();
	}

	Time timeStart () const {
		return _timedData.front().time ();
	}

	Time timeEnd () const {
		return _timedData.back().time ();
	}

	typename DataType::const_iterator begin () const {
		return _timedData.begin ();
	}

	typename DataType::const_iterator end () const {
		return _timedData.end ();
	}

	DEF_SHARED(Signal)
private:
	DataType _timedData;
	Params _params;
};

template<typename T, template<typename ...> class container>
std::ostream &operator << (std::ostream &os, const Signal<T, container> &sig) {
	for (const typename Signal<T, container>::Sample &curr : sig) {
		os << curr << "\n";
	}

	os << "[ Signal " << TYPE(T) << " {" << sig.size () << "} ]\n";
	return os;
}


template<class Reading>
class ReadingWindow
{
public:
	enum Mode {
		MODE_SLIDING,
		MODE_DOWNSAMPLE
	};

	struct Params {
		Mode mode;
		uint size;

		DEF_SHARED(Params)
	};

private:
	void addDownsample (const Reading &newReading);
	void addSliding (const Reading &newReading);

public:
	ReadingWindow (const Params &params);

	void add (const Reading &newReading);
	Reading get ();
	bool isReady () const;
	size_t size () const {
		return _readingQueue.size ();
	}
	void reset();

	DEF_SHARED(ReadingWindow)

private:
	std::queue<Reading> _readingQueue;
	uint _skipped;
	Params _params;
};

template<class Reading>
ReadingWindow<Reading>::ReadingWindow(const ReadingWindow::Params &params):
	 _params(params),
	 _skipped(params.size)
{
}

template<class Reading>
void ReadingWindow<Reading>::reset ()
{
	_readingQueue = {};
	_skipped = _params.size;
}

template<class Reading>
void ReadingWindow<Reading>::addDownsample (const Reading &newReading)
{
	if (_skipped == _params.size) {
		_skipped = 0;
		if (_readingQueue.size () > 0)
			_readingQueue.pop ();
		_readingQueue.push (newReading);
	} else
		_skipped++;
}

template<class Reading>
void ReadingWindow<Reading>::addSliding (const Reading &newReading)
{
	_readingQueue.push (newReading);

	if (_readingQueue.size () == _params.size)
		_readingQueue.pop ();
}

template<class Reading>
void ReadingWindow<Reading>::add(const Reading &newReading)
{
	switch (_params.mode) {
	case MODE_DOWNSAMPLE:
		addDownsample (newReading);
		break;
	case MODE_SLIDING:
		addSliding (newReading);
		break;
	}
}

template<class Reading>
Reading ReadingWindow<Reading>::get () {
	return _readingQueue.front ();
}

template<class Reading>
bool ReadingWindow<Reading>::isReady() const
{
	switch (_params.mode) {
	case MODE_DOWNSAMPLE:
		return _skipped == 0;
	case MODE_SLIDING:
		return _readingQueue.size () == _params.size;
	}
}

template<typename Clock, typename Duration>
class FrequencyEstimator
{
	using Time = std::chrono::time_point<Clock, Duration>;

public:
	FrequencyEstimator ();

	void tick ();
	void tick (const Time &now);

	double estimateSeconds () const;
	double estimateHz () const;
	double lastPeriodSeconds () const;
	void reset();


private:
	Time last;
	Duration lastPeriod;
	Duration averagePeriod;
	uint seq;

};

template<typename Clock, typename Duration>
FrequencyEstimator<Clock, Duration>::FrequencyEstimator():
	 seq(0)
{}

template<typename Clock, typename Duration>
void FrequencyEstimator<Clock, Duration>::reset () {
	seq = 0;
}

template<typename Clock, typename Duration>
void FrequencyEstimator<Clock, Duration>::tick () {
	tick (Clock::now ());
}

template<typename Clock, typename Duration>
void FrequencyEstimator<Clock, Duration>::tick (const Time &now)
{
	if (seq == 0) {
		last = now;
	} else {
		Duration currentPeriod = now - last;
		lastPeriod = currentPeriod;
		last = now;

		averagePeriod = averagePeriod + std::chrono::duration_cast<Duration> (1. / double (seq + 1) * (currentPeriod - averagePeriod));
	}

	seq++;
}

template<typename Clock, typename Duration>
double FrequencyEstimator<Clock, Duration>::estimateHz() const {
	return 1 / estimateSeconds ();
}

template<typename Clock, typename Duration>
double FrequencyEstimator<Clock, Duration>::lastPeriodSeconds() const {
	return std::chrono::duration_cast<std::chrono::duration<float>> (lastPeriod).count ();
}

template<typename Clock, typename Duration>
double FrequencyEstimator<Clock, Duration>::estimateSeconds () const {
	return std::chrono::duration_cast<std::chrono::duration<float>> (averagePeriod).count ();
}

class GroundTruthSync
{
public:
	struct Params {
		int queueLength;
		float msOffset;

		DEF_SHARED(Params)
	};

	using GroundTruthSignal = Signal<lietorch::Pose, std::deque>;
	using GroundTruth = typename GroundTruthSignal::Sample;
	using MarkerQueue = std::queue<Time>;

public:
	GroundTruthSync (const Params &_params);

	bool groundTruthReady ();
	bool markersReady ();
	void updateGroundTruth (const GroundTruth &newGroundTruth);
	void addSynchronizationMarker (const Time &markerTime, boost::optional<float> &expiredMs, boost::optional<float> &futureMs);
	TargetGroup getLastRelativeGroundTruth () const;
	void reset ();
	const MarkerQueue &markers () const;

	DEF_SHARED(GroundTruthSync)

private:
	Params _params;
	Duration _offset;
	GroundTruthSignal _groundTruthSignal;
	MarkerQueue _markerQueue;

};


#endif // SYNCHRONIZATION_H
