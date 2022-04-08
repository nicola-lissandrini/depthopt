#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <lietorch/pose.h>
#include <nlib/nl_utils.h>

/*
 * Define global time units
 */
using Duration = std::chrono::microseconds;
using Clock = std::chrono::system_clock;
using Time = std::chrono::time_point<Clock, Duration>;
template<typename T>
using Timed = nlib::TimedObject<T, Clock, Duration>;

using TargetGroup = lietorch::Pose;

#endif // DEFINITIONS_H
