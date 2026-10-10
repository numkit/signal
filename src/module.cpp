// src/module.cpp
//
// Plugin entry point for nk_signal dynamic module.
// Exports nk_plugin_init conforming to NK_PLUGIN_ABI_VERSION.

#include <nk/plugin.hpp>
#include <nk/signal/library.hpp>

extern "C" NK_PLUGIN_EXPORT bool nk_plugin_init(nk::PluginContext &ctx)
{
    if (ctx.abiVersion() != NK_PLUGIN_ABI_VERSION) {
        return false;
    }
    nk::SignalLibrary::install(ctx);
    return true;
}
