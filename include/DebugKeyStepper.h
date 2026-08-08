#ifndef AF_VSLAM_DEBUG_KEY_STEPPER_H
#define AF_VSLAM_DEBUG_KEY_STEPPER_H

#include <atomic>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <cstdio>

// RAII: put terminal in raw-ish mode so we can read single key presses
struct TerminalRawMode {
    termios oldt{};
    bool active{false};

    TerminalRawMode() {
        if (tcgetattr(STDIN_FILENO, &oldt) == 0) {
            termios newt = oldt;
            newt.c_lflag &= ~(ICANON | ECHO); // no line buffering, no echo
            newt.c_cc[VMIN]  = 0;
            newt.c_cc[VTIME] = 0;
            if (tcsetattr(STDIN_FILENO, TCSANOW, &newt) == 0) {
                active = true;
            }
        }
    }

    ~TerminalRawMode() {
        if (active) tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    }
};

inline bool stdin_has_data() {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);
    timeval tv{0, 0}; // no wait
    int rv = select(STDIN_FILENO + 1, &set, nullptr, nullptr, &tv);
    return (rv > 0) && FD_ISSET(STDIN_FILENO, &set);
}

// Starts a thread that increments `allowance` each time any key is pressed.
// If 'q' is pressed, sets `quit=true`.
inline std::thread startKeyAllowanceThread(std::atomic<int>& allowance,
                                          std::atomic<bool>& quit)
{
    return std::thread([&]() {
        TerminalRawMode raw; // applies to this process; RAII restores on exit
        while (!quit.load(std::memory_order_relaxed)) {
            if (stdin_has_data()) {
                unsigned char c = 0;
                ssize_t n = ::read(STDIN_FILENO, &c, 1);
                if (n == 1) {
                    if (c == 'q' || c == 'Q') {
                        quit.store(true, std::memory_order_relaxed);
                        break;
                    }
                    // Any other key press allows one image to pass
                    allowance.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                usleep(1000); // 1 ms, avoid busy spinning
            }
        }
    });
}

#endif //AF_VSLAM_DEBUG_KEY_STEPPER_H
