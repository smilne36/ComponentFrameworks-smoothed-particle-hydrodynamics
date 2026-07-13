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
