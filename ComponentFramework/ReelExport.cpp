#include "ReelExport.h"
#include "AudioBands.h"

#include "dr_wav.h"
#include "dr_mp3.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

namespace {

bool HasExtension(const std::string& path, const char* ext) {
    size_t n = std::strlen(ext);
    if (path.size() < n) return false;
    std::string tail = path.substr(path.size() - n);
    for (char& c : tail) c = char(std::tolower((unsigned char)c));
    return tail == ext;
}

// Decode an entire file to interleaved float32 mono is not what dr_libs gives;
// we get interleaved float per channel, then downmix. Returns false on failure.
bool DecodeToMono(const char* path, std::vector<float>& mono, unsigned& sampleRate,
                  std::string& error) {
    std::string p(path ? path : "");
    if (p.empty()) { error = "no audio file given"; return false; }

    if (HasExtension(p, ".wav")) {
        unsigned channels = 0;
        drwav_uint64 totalFrames = 0;
        float* interleaved = drwav_open_file_and_read_pcm_frames_f32(
            p.c_str(), &channels, &sampleRate, &totalFrames, nullptr);
        if (!interleaved) { error = "could not open/decode WAV"; return false; }
        if (channels == 0) { drwav_free(interleaved, nullptr); error = "WAV has no channels"; return false; }
        mono.resize((size_t)totalFrames);
        for (drwav_uint64 i = 0; i < totalFrames; ++i) {
            float acc = 0.0f;
            for (unsigned c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
            mono[(size_t)i] = acc / float(channels);
        }
        drwav_free(interleaved, nullptr);
        return true;
    }

    if (HasExtension(p, ".mp3")) {
        drmp3_config cfg{};
        drmp3_uint64 totalFrames = 0;
        float* interleaved = drmp3_open_file_and_read_pcm_frames_f32(
            p.c_str(), &cfg, &totalFrames, nullptr);
        if (!interleaved) { error = "could not open/decode MP3"; return false; }
        if (cfg.channels == 0) { drmp3_free(interleaved, nullptr); error = "MP3 has no channels"; return false; }
        sampleRate = cfg.sampleRate;
        unsigned channels = cfg.channels;
        mono.resize((size_t)totalFrames);
        for (drmp3_uint64 i = 0; i < totalFrames; ++i) {
            float acc = 0.0f;
            for (unsigned c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
            mono[(size_t)i] = acc / float(channels);
        }
        drmp3_free(interleaved, nullptr);
        return true;
    }

    error = "unsupported file type (use .wav or .mp3)";
    return false;
}

} // namespace

ReelAnalysis AnalyzeTrack(const char* path, int fps, float maxSeconds) {
    ReelAnalysis out;
    if (fps <= 0) fps = 30;

    std::vector<float> mono;
    unsigned sampleRate = 0;
    if (!DecodeToMono(path, mono, sampleRate, out.error)) return out;
    if (sampleRate == 0 || mono.empty()) { out.error = "empty audio"; return out; }

    const float fs = float(sampleRate);
    out.sampleRate = int(sampleRate);

    // Optionally trim to maxSeconds
    size_t totalSamples = mono.size();
    if (maxSeconds > 0.0f) {
        size_t cap = size_t(maxSeconds * fs);
        if (cap > 0 && cap < totalSamples) totalSamples = cap;
    }

    const double samplesPerFrame = fs / double(fps);
    const int frameCount = int(std::floor(double(totalSamples) / samplesPerFrame));
    if (frameCount <= 0) { out.error = "track too short for one frame"; return out; }

    out.frameCount = frameCount;
    out.bass.resize(frameCount);
    out.mid.resize(frameCount);
    out.treble.resize(frameCount);

    // Match the live reactor's default attack/release (see AudioReactive.h).
    const float attackCoeff  = EnvelopeCoeff(15.0f,  fs);
    const float releaseCoeff = EnvelopeCoeff(250.0f, fs);

    BandState s;
    size_t sampleIdx = 0;
    for (int f = 0; f < frameCount; ++f) {
        // Process all samples up to this frame's boundary, then sample the
        // envelope. Envelope state carries across frames (continuous, like live).
        size_t frameEnd = size_t(std::llround(double(f + 1) * samplesPerFrame));
        if (frameEnd > totalSamples) frameEnd = totalSamples;
        for (; sampleIdx < frameEnd; ++sampleIdx) {
            ProcessSample(mono[sampleIdx], s, fs, attackCoeff, releaseCoeff);
        }
        out.bass[f]   = std::min(4.0f, s.envBass);
        out.mid[f]    = std::min(4.0f, s.envMid);
        out.treble[f] = std::min(4.0f, s.envTreble);
    }

    return out;
}
