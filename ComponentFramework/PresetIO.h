#pragma once
#include <map>
#include <string>
#include <vector>

// Save/load "look" presets as plain key=value text files. Deliberately free
// of GL/SDL/ImGui so it unit-tests under any compiler. Unknown keys are
// ignored on load and missing keys keep their current values, so preset
// files stay forward- and backward-compatible across app versions.
namespace PresetIO {

    using KV = std::map<std::string, std::string>;

    // "# SPH Fluid Preset v1\nversion=1\n" + sorted key=value lines
    std::string Serialize(const KV& kv);
    // '#' comments, blank lines, and garbage lines are ignored; first value
    // wins on duplicate keys.
    KV Parse(const std::string& text);

    bool SaveFile(const std::string& path, const KV& kv);
    bool LoadFile(const std::string& path, KV& out);

    // Basenames (no extension) of *.txt files in dir, sorted; empty on error.
    std::vector<std::string> ListPresets(const std::string& dir);

    // Keep [A-Za-z0-9 _-], trim; "preset" if nothing survives.
    std::string SanitizeName(const std::string& raw);

    // Typed accessors. Floats use "%.9g" so every float round-trips exactly.
    void  PutF (KV& kv, const char* key, float v);
    void  PutI (KV& kv, const char* key, int v);
    void  PutB (KV& kv, const char* key, bool v);
    void  PutF3(KV& kv, const char* key, const float v[3]);   // "r,g,b"
    float GetF (const KV& kv, const char* key, float def);
    int   GetI (const KV& kv, const char* key, int def);
    bool  GetB (const KV& kv, const char* key, bool def);
    void  GetF3(const KV& kv, const char* key, float out[3]); // unchanged if missing/bad

} // namespace PresetIO
