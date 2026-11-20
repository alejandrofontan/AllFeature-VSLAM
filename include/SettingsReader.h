#pragma once

#include <string>
#include <type_traits>

#include <opencv2/core/persistence.hpp>

namespace AF_VSLAM
{

// cv::FileStorage hands back an UNQUOTED string value together with any trailing "# comment" on its
// line (`Scope: map   # all keyframes` reads as "map   # all keyframes"). This drops everything from
// the first '#' that starts the value or follows whitespace, and trims the whitespace around what is
// left. A quoted value that itself contains " #" is cut there too: no setting has one.
inline void strip_inline_comment(std::string& value)
{
    for(size_t i = 0; i < value.size(); i++)
    {
        if(value[i] == '#' && (i == 0 || value[i - 1] == ' ' || value[i - 1] == '\t'))
        {
            value.erase(i);
            break;
        }
    }
    const size_t first = value.find_first_not_of(" \t\r\n");
    const size_t last = value.find_last_not_of(" \t\r\n");
    value = (first == std::string::npos) ? std::string{} : value.substr(first, last - first + 1);
}

// `settings[key]` into `field` when the key is present; `field` keeps its value otherwise. A string
// value is cleaned with strip_inline_comment, so a string key may carry an inline comment whether or
// not it is quoted. The one reader behind every LoadParameters.
template <typename T>
void read_setting_if_present(const cv::FileStorage& settings, const char* key, T& field)
{
    const cv::FileNode node = settings[key];
    if(node.empty())
        return;
    node >> field;
    if constexpr (std::is_same_v<T, std::string>)
        strip_inline_comment(field);
}

}
