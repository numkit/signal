/// @file library.hpp
/// @ingroup group_signal
#pragma once

namespace numkit {

/// @addtogroup group_signal
class PluginContext;

class SignalLibrary
{
public:
    static void install(PluginContext &engine);
};


/// @}
} // namespace numkit
