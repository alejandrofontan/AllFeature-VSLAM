/**
 * Module: AllFeature-VSLAM - PlaceCellSettings.cc
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-05
 * - License: GPLv3 License
 */

#include "PlaceCellSettings.h"

namespace AF_VSLAM
{

PlaceCellSettings PlaceCellSettings::Load(const cv::FileStorage& fSettings)
{
    PlaceCellSettings settings{};

    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };
    // cv::FileStorage has no bool reader: 0/1 integers
    auto read_flag_if_present = [&read_if_present](const char* key, bool& field)
    {
        int value = field ? 1 : 0;
        read_if_present(key, value);
        field = (value != 0);
    };

    read_if_present("PlaceCell.Verbosity", settings.verbosity);
    // cv::FileStorage hands back an UNQUOTED scalar together with any trailing
    // "# comment" on its line (seen on LocalMapping.KeyframeCullingScope in run logs):
    // keep the first token only, so `Verbosity: warn   # ...` reads as "warn".
    {
        const size_t hash = settings.verbosity.find('#');
        if(hash != std::string::npos)
            settings.verbosity.erase(hash);
        const size_t first = settings.verbosity.find_first_not_of(" \t\r\n");
        const size_t last = settings.verbosity.find_last_not_of(" \t\r\n");
        settings.verbosity = (first == std::string::npos) ? std::string{} : settings.verbosity.substr(first, last - first + 1);
    }
    read_flag_if_present("PlaceCell.LogElapsed", settings.log_elapsed);

    read_flag_if_present("PlaceCell.Profile", settings.profile);
    read_flag_if_present("PlaceCell.PrintProfile", settings.print_profile);

    read_flag_if_present("PlaceCell.Record", settings.record);
    read_flag_if_present("PlaceCell.Dump", settings.dump);

    read_flag_if_present("PlaceCell.Visualize", settings.visualize);
    read_if_present("PlaceCell.VisualizeMaxHz", settings.visualize_max_hz);
    read_flag_if_present("PlaceCell.VisualizeCentred", settings.visualize_centred);
    read_if_present("PlaceCell.VisualizeHistoryLastN", settings.visualize_history_last_n);

    return settings;
}

} // namespace AF_VSLAM
