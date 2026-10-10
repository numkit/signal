// src/module.cpp
//
// Plugin entry point for nk_signal dynamic module.
// Exports nk_plugin_init conforming to NUMKIT_PLUGIN_ABI_VERSION.

#include <numkit/plugin.hpp>
#include <numkit/signal/library.hpp>

extern "C" NUMKIT_PLUGIN_EXPORT bool numkit_plugin_init(numkit::PluginContext &ctx)
{
    if (ctx.abiVersion() != NUMKIT_PLUGIN_ABI_VERSION) {
        return false;
    }
    numkit::SignalLibrary::install(ctx);
    return true;
}

extern "C" NUMKIT_PLUGIN_EXPORT bool nk_plugin_init(numkit::PluginContext &ctx)
{
    return numkit_plugin_init(ctx);
}
