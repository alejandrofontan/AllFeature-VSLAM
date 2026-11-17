#pragma once

// Debug report: AF_REPORT(key, feature, value) records one per-frame diagnostic, written to a
// CSV at shutdown. Compiled in only with -DALLFEATURE_REPORT=ON (CMakeLists.txt). When it is
// off the macro expands to nothing, so its arguments are never evaluated and a call costs nothing.
//
// A report call must never change what the code does: no call inside a condition the algorithm
// depends on, and nothing reads the report back.

#ifdef ALLFEATURE_REPORT

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace AF_VSLAM
{

class Report
{
public:
    static Report& instance();

    // Records one entry, stamped with the calling thread's current context (set_context).
    // `function` is the reporting function's name; AF_REPORT fills it in from __func__.
    void add(const char* function, const std::string& key, const std::string& feature, double value);

    // Names the calling thread and the frame it is working on. Each thread calls this once per
    // frame or keyframe; every add() from that thread is stamped with it until the next call.
    static void set_context(const std::string& thread, long unsigned int frame_id);

    // Writes every entry to `path` as CSV: thread,frame_id,call,function,key,feature,value. Call once, after
    // System::Shutdown() has joined the threads, so nothing is still adding entries.
    void write(const std::string& path);

private:
    Report() = default;

    struct Entry
    {
        std::string thread;
        long unsigned int frame_id;
        long unsigned int call;
        std::string function;
        std::string key;
        std::string feature;
        double value;
    };

    std::mutex mutex_;
    std::vector<Entry> entries_;
};

// Groups the entries of one function call: while a ReportScope is alive, every add() from its
// thread -- from that function or anything it calls -- carries the same call id. The ids count
// up per thread (1, 2, 3, ...), not globally, so they do not depend on how the threads
// interleave; an entry is identified by (thread, call). Scopes nest: on exit the enclosing
// call's id is restored. Entries reported outside any scope have call id 0.
//
// On exit it also records the call's wall-clock duration as a `time_ms` entry. That is
// inclusive time: everything the function calls, and any wait on a lock, counts. `time_ms` is
// the one entry that differs between otherwise identical runs -- filter it out before comparing
// two reports.
class ReportScope
{
public:
    explicit ReportScope(const char* function);
    ~ReportScope();
    ReportScope(const ReportScope&) = delete;
    ReportScope& operator=(const ReportScope&) = delete;

private:
    const char* function_;
    long unsigned int enclosing_call_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace AF_VSLAM

#define AF_REPORT(key, feature, value) \
    ::AF_VSLAM::Report::instance().add(__func__, (key), (feature), static_cast<double>(value))

// Opens a call scope for the rest of the enclosing block; see ReportScope. Put it first in a
// function whose entries should be groupable per call.
#define AF_REPORT_SCOPE() ::AF_VSLAM::ReportScope af_report_scope_{__func__}

// Sets the calling thread's context: its name and the frame (or keyframe's frame) it is working
// on. Not inherited by OpenMP workers -- report outside parallel regions.
#define AF_REPORT_CONTEXT(thread, frame_id) \
    ::AF_VSLAM::Report::set_context((thread), (frame_id))

// Writes the report as CSV. Call once, after System::Shutdown().
#define AF_REPORT_WRITE(path) ::AF_VSLAM::Report::instance().write(path)

#else

#define AF_REPORT(key, feature, value) do {} while(0)
#define AF_REPORT_SCOPE() do {} while(0)
#define AF_REPORT_CONTEXT(thread, frame_id) do {} while(0)
#define AF_REPORT_WRITE(path) do {} while(0)

#endif
