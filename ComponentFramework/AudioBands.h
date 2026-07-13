#pragma once
#include <cmath>

// Shared Bass/Mid/Treble band-split + envelope DSP, used by BOTH the live
// WASAPI reactor (AudioReactive.cpp) and the offline Reels analyzer
// (ReelExport.cpp) so the two produce identical band values.
//
// One-pole filters, chosen over biquads because they are correct by
// construction: the lowpass is the exact digital form of an RC filter and the
// highpass is literally input minus lowpass. For driving splashes and color
// pulses (not audio playback), the gentler rolloff is irrelevant.

// Per-sample smoothing coefficient for a one-pole lowpass at cutoff fcHz.
inline float FilterAlpha(float fcHz, float fsHz) {
    return 1.0f - std::exp(-6.2831853f * fcHz / fsHz);
}

// Per-sample coefficient for an envelope with time constant tcMs.
inline float EnvelopeCoeff(float tcMs, float fsHz) {
    return 1.0f - std::exp(-1000.0f / (std::fmax(tcMs, 0.1f) * fsHz));
}

inline float OnePoleLowpass(float x, float& state, float alpha) {
    state += alpha * (x - state);
    return state;
}

inline float OnePoleHighpass(float x, float& lpState, float alpha) {
    return x - OnePoleLowpass(x, lpState, alpha);
}

// Attack/release peak-envelope follower.
inline void UpdateEnvelope(float rectified, float& env, float attackCoeff, float releaseCoeff) {
    float c = (rectified > env) ? attackCoeff : releaseCoeff;
    env += c * (rectified - env);
}

struct BandState {
    float lpBass = 0.0f, lpMidHi = 0.0f, lpMidLo = 0.0f, lpTreble = 0.0f;
    float envBass = 0.0f, envMid = 0.0f, envTreble = 0.0f;
};

// Feeds one mono sample through the three band filters + envelope followers.
inline void ProcessSample(float mono, BandState& s, float fs,
                          float attackCoeff, float releaseCoeff) {
    const float aBass   = FilterAlpha(150.0f,  fs);
    const float aMidHi  = FilterAlpha(2000.0f, fs);
    const float aMidLo  = FilterAlpha(250.0f,  fs);
    const float aTreble = FilterAlpha(2800.0f, fs);

    float bass   = OnePoleLowpass(mono, s.lpBass, aBass);
    float midHi  = OnePoleLowpass(mono, s.lpMidHi, aMidHi);          // strip highs
    float mid    = OnePoleHighpass(midHi, s.lpMidLo, aMidLo);        // then strip lows
    float treble = OnePoleHighpass(mono, s.lpTreble, aTreble);

    UpdateEnvelope(std::fabs(bass),   s.envBass,   attackCoeff, releaseCoeff);
    UpdateEnvelope(std::fabs(mid),    s.envMid,    attackCoeff, releaseCoeff);
    UpdateEnvelope(std::fabs(treble), s.envTreble, attackCoeff, releaseCoeff);
}
