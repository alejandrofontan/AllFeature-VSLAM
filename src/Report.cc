#include "Report.h"
#include <fstream>
#include <iomanip>
#include <limits>

#ifdef ALLFEATURE_REPORT

namespace AF_VSLAM
{

namespace
{
// Per-thread context: who is reporting, and about which frame. thread_local, so each of the
// four threads has its own copy and setting it needs no lock.
thread_local std::string context_thread{"unknown"};
thread_local long unsigned int context_frame_id{0};

// Per-thread call scope (ReportScope): the id entries are stamped with now, and the last id
// this thread handed out.
thread_local long unsigned int current_call{0};
thread_local long unsigned int last_call{0};
}

Report& Report::instance()
{
    static Report report;
    return report;
}

void Report::set_context(const std::string& thread, const long unsigned int frame_id)
{
    context_thread = thread;
    context_frame_id = frame_id;
}

void Report::add(const char* function, const std::string& key, const std::string& feature, const double value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back(Entry{context_thread, context_frame_id, current_call, function, key, feature, value});
}

ReportScope::ReportScope(const char* function):
    function_(function), enclosing_call_(current_call), start_(std::chrono::steady_clock::now())
{
    current_call = ++last_call;
}

ReportScope::~ReportScope()
{
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();

    // Recorded before the enclosing call's id is restored, so the row belongs to this call.
    Report::instance().add(function_, "time_ms", "", elapsed_ms);
    current_call = enclosing_call_;
}

void Report::write(const std::string& path)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(path);
    if(!file)
        return;

    // Enough digits to write a double and read the same double back.
    file << std::setprecision(std::numeric_limits<double>::max_digits10);

    file << "thread,frame_id,call,function,key,feature,value\n";
    for(const Entry& entry : entries_)
        file << entry.thread << ',' << entry.frame_id << ',' << entry.call << ',' << entry.function << ','
             << entry.key << ','
             << entry.feature << ',' << entry.value << '\n';
}

} // namespace AF_VSLAM

#endif
