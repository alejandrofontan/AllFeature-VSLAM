#pragma once
#include <iostream>
#include <string_view>
#include <iomanip>

// ── ANSI color codes ─────────────────────────────────────────────────────────
#ifndef AFVSLAM_NO_COLOR
  #define _AF_RST  "\033[0m"
  #define _AF_DIM  "\033[2m"
  #define _AF_DBG  "\033[34m"      // blue
  #define _AF_INF  "\033[32m"      // green
  #define _AF_WRN  "\033[33m"      // yellow
  #define _AF_ERR  "\033[31m"      // red
  #define _AF_FTL  "\033[1;31m"    // bold red
#else
  #define _AF_RST  ""
  #define _AF_DIM  ""
  #define _AF_DBG  ""
  #define _AF_INF  ""
  #define _AF_WRN  ""
  #define _AF_ERR  ""
  #define _AF_FTL  ""
#endif

// ── Library tag ──────────────────────────────────────────────────────────────
#define _AF_TAG  _AF_DIM "[" _AF_RST "AFVSLAM" _AF_DIM "]" _AF_RST " "

// ── Path trimmer (compile-time) ───────────────────────────────────────────────
namespace afvslam::detail {
  constexpr std::string_view short_path(std::string_view path) {
      for (std::string_view marker : {"src/", "include/"}) {
          auto pos = path.rfind(marker);
          if (pos != std::string_view::npos) return path.substr(pos);
      }
      return path;
  }
} // namespace afvslam::detail

#define _AF_FILENAME afvslam::detail::short_path(__FILE__)

// ── Log level gate (define AFVSLAM_LOG_LEVEL=0..4, default=1) ────────────────
#ifndef AFVSLAM_LOG_LEVEL
  #define AFVSLAM_LOG_LEVEL 1
#endif

// ── Core macro ────────────────────────────────────────────────────────────────
#define _AF_LOG(color, label, stream, msg)                                  \
  do {                                                                       \
    (stream) << _AF_TAG color "[" label "]" _AF_RST " "                     \
             << _AF_DIM << _AF_FILENAME << ":" << __LINE__ << _AF_RST       \
             << "  " << msg << "\n";                                         \
  } while(0)

// ── Public macros ─────────────────────────────────────────────────────────────
#if AFVSLAM_LOG_LEVEL <= 0
  #define AF_DEBUG(msg) _AF_LOG(_AF_DBG, "DEBUG", std::cout, msg)
#else
  #define AF_DEBUG(msg) do {} while(0)
#endif

#if AFVSLAM_LOG_LEVEL <= 1
  #define AF_INFO(msg)  _AF_LOG(_AF_INF, "INFO", std::cout, msg)
#else
  #define AF_INFO(msg)  do {} while(0)
#endif

#if AFVSLAM_LOG_LEVEL <= 2
  #define AF_WARN(msg)  _AF_LOG(_AF_WRN, "WARN", std::cerr, msg)
#else
  #define AF_WARN(msg)  do {} while(0)
#endif

#define AF_CONFIG_BEGIN(title) \
  std::cout << _AF_TAG _AF_WRN "[" title "]" _AF_RST "\n"

#define AF_CONFIG_FIELD(key, value)                              \
  do {                                                           \
    std::ostringstream _af_ss;                                   \
    _af_ss << value;                                             \
    std::cout << "    " _AF_DIM "|" _AF_RST "  "                \
              << key << "  " << _af_ss.str() << "\n";           \
  } while(0)

#define AF_CONFIG_END() \
  std::cout << "    " _AF_DIM "|___\n" _AF_RST

// ── Profiling macros ──────────────────────────────────────────────────────────
#define AF_PROFILE_BEGIN(title) \
  std::cout << _AF_TAG _AF_WRN "[" title "]" _AF_RST "\n"

#define AF_PROFILE_FIELD(times, label)                               \
  do {                                                               \
    std::ostringstream _af_ss;                                       \
    _af_ss << std::fixed << std::setprecision(2)                     \
           << AF_VSLAM::map_median(times);                           \
    std::cout << "    " _AF_DIM "|" _AF_RST "  "                    \
              << std::left << std::setw(30) << (label)              \
              << _af_ss.str() << " ms\n";                            \
  } while(0)

#define AF_PROFILE_END() \
  std::cout << "    " _AF_DIM "|___\n" _AF_RST

// ERROR and FATAL are never silenced
#define AF_ERROR(msg) _AF_LOG(_AF_ERR, "ERROR", std::cerr, msg)
#define AF_FATAL(msg) do { _AF_LOG(_AF_FTL, "FATAL", std::cerr, msg); std::abort(); } while(0)

// ── ASCII banner ──────────────────────────────────────────────────────────────
inline void af_print_banner() {
    std::cout
    << _AF_DIM
    << "\n"
    << "  █████╗ ███████╗    ██╗   ██╗███████╗██╗      █████╗ ███╗   ███╗\n"
    << " ██╔══██╗██╔════╝    ██║   ██║██╔════╝██║     ██╔══██╗████╗ ████║\n"
    << " ███████║█████╗█████╗██║   ██║███████╗██║     ███████║██╔████╔██║\n"
    << " ██╔══██║██╔══╝╚════╝╚██╗ ██╔╝╚════██║██║     ██╔══██║██║╚██╔╝██║\n"
    << " ██║  ██║██║          ╚████╔╝ ███████║███████╗██║  ██║██║ ╚═╝ ██║\n"
    << " ╚═╝  ╚═╝╚═╝           ╚═══╝  ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝\n"
    << "                 AllFeature Visual SLAM  |  v1.0                 \n"
    << _AF_RST << "\n";
}

