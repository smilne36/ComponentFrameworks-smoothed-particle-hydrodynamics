#pragma once
#include <vector>
#include <string>

// Per-frame Bass/Mid/Treble envelopes for an audio track, used to drive the
// offline (frame-accurate) Reels render. Pure CPU: no OpenGL, no Windows.
struct ReelAnalysis {
    int sampleRate = 0;
    int frameCount = 0;                    // number of video frames
    std::vector<float> bass, mid, treble;  // one value per frame (size == frameCount)
    std::string error;                     // empty on success
};

// Decodes a .wav or .mp3 file (by extension), downmixes to mono, runs it
// through the shared AudioBands DSP, and samples the three band envelopes once
// per video frame (every sampleRate/fps input samples). frameCount is
// ceil(seconds * fps), optionally capped by maxSeconds (0 = whole track).
ReelAnalysis AnalyzeTrack(const char* path, int fps, float maxSeconds);

// Finds bass drops in a per-frame envelope: rising crossings of an adaptive
// threshold (1.6x the rolling 4s average, floored at 0.25), at least
// minGapSec apart, capped at 16. Returns drop times in seconds.
std::vector<float> DetectDrops(const std::vector<float>& bass, int fps, float minGapSec);
