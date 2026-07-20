#include "PresetIO.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace PresetIO {

std::string Serialize(const KV& kv) {
    std::string out = "# SPH Fluid Preset v1\nversion=1\n";
    for (const auto& [k, v] : kv) {
        if (k == "version") continue;
        out += k; out += '='; out += v; out += '\n';
    }
    return out;
}

KV Parse(const std::string& text) {
    KV kv;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // CRLF-safe
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos || eq == 0) continue;            // garbage line
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        kv.emplace(key, val);                                        // first value wins
    }
    return kv;
}

bool SaveFile(const std::string& path, const KV& kv) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << Serialize(kv);
    return f.good();
}

bool LoadFile(const std::string& path, KV& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = Parse(ss.str());
    return true;
}

std::vector<std::string> ListPresets(const std::string& dir) {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const auto& p = e.path();
        if (p.extension() == ".txt") names.push_back(p.stem().string());
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::string SanitizeName(const std::string& raw) {
    std::string out;
    for (char c : raw) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == ' ' || c == '_' || c == '-';
        if (ok) out += c;
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out.empty() ? "preset" : out;
}

void PutF(KV& kv, const char* key, float v) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.9g", double(v));
    kv[key] = buf;
}
void PutI(KV& kv, const char* key, int v)  { kv[key] = std::to_string(v); }
void PutB(KV& kv, const char* key, bool v) { kv[key] = v ? "1" : "0"; }
void PutF3(KV& kv, const char* key, const float v[3]) {
    char buf[144];
    std::snprintf(buf, sizeof(buf), "%.9g,%.9g,%.9g", double(v[0]), double(v[1]), double(v[2]));
    kv[key] = buf;
}

float GetF(const KV& kv, const char* key, float def) {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    char* end = nullptr;
    const float v = std::strtof(it->second.c_str(), &end);
    return (end && end != it->second.c_str()) ? v : def;
}
int GetI(const KV& kv, const char* key, int def) {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    char* end = nullptr;
    const long v = std::strtol(it->second.c_str(), &end, 10);
    return (end && end != it->second.c_str()) ? int(v) : def;
}
bool GetB(const KV& kv, const char* key, bool def) {
    return GetI(kv, key, def ? 1 : 0) != 0;
}
void GetF3(const KV& kv, const char* key, float out[3]) {
    auto it = kv.find(key);
    if (it == kv.end()) return;
    float r, g, b;
    if (std::sscanf(it->second.c_str(), "%f,%f,%f", &r, &g, &b) == 3) {
        out[0] = r; out[1] = g; out[2] = b;
    }
}

} // namespace PresetIO
