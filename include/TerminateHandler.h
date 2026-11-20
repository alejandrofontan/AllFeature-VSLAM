/**
 * Module: AllFeature-VSLAM - TerminateHandler.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.0
 * - Created: 2026-11-21
 * - License: GPLv3 License
 *
 * What the entry points do with an exception nobody caught, on any thread. std::terminate is reached without
 * unwinding, so this runs while the System and its worker threads still exist: nothing is destroyed under them.
 *
 *   std::runtime_error   a failure of the environment or the inputs (a missing model file, a sidecar that does not
 *                        match, a provider that cannot run a graph): one [ERROR] line and exit code 1.
 *   anything else        a bug (std::logic_error such as an out-of-range index, std::bad_alloc, a non-standard
 *                        exception, or std::terminate called with no exception at all): the message, the throwing
 *                        thread's stack and std::abort(), which leaves the core dump.
 *
 * AF_FATAL (afvslam_log.hpp) stays the way a function ends the process itself; this is for what escapes.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

#include <execinfo.h>

#include "afvslam_log.hpp"

namespace AF_VSLAM
{

[[noreturn]] inline void on_terminate()
{
    if (const std::exception_ptr current = std::current_exception()) {
        try {
            std::rethrow_exception(current);
        } catch (const std::runtime_error& e) {
            AF_ERROR(e.what());
            std::cout.flush();
            std::fflush(nullptr);
            std::_Exit(1);   // no destructors: the worker threads are still running
        } catch (const std::exception& e) {
            AF_ERROR("uncaught exception: " << e.what());
        } catch (...) {
            AF_ERROR("uncaught non-standard exception");
        }
    } else {
        AF_ERROR("std::terminate called without an exception");
    }
    // The stack of the thread that got here (symbolised by the dynamic linker; `addr2line -e <binary> <addr>` or
    // `atos -o <binary> <addr>` resolves what it leaves as offsets)
    void* frames[64];
    const int depth = backtrace(frames, 64);
    backtrace_symbols_fd(frames, depth, 2);
    std::abort();
}

/// Call first in main().
inline void install_terminate_handler()
{
    std::set_terminate(on_terminate);
}

} // namespace AF_VSLAM
