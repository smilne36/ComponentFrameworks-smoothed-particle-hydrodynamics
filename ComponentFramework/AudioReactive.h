#pragma once
#include <thread>
#include <atomic>
#include <mutex>
#include <string>

/// Captures the computer's audio OUTPUT (whatever is playing through the
/// speakers) via WASAPI loopback on a background thread, splits it into
/// Bass / Mid / Treble envelopes, and publishes them as atomics for the
/// main thread to read once per frame. No microphone, no extra libraries —
/// just the Windows Core Audio API (needs ole32.lib).
class AudioReactive {
public:
    AudioReactive() = default;
    ~AudioReactive();                    // stops the capture thread if still running

    // --- Main-thread API ---
    void Start();                        // launches the capture thread; no-op if running
    void Stop();                         // signals + joins the thread; safe if not running
    bool IsRunning()   const { return running.load(std::memory_order_relaxed); }
    bool IsCapturing() const { return capturing.load(std::memory_order_relaxed); }
    std::string GetStatusText() const;   // "Idle" / "Capturing (48000 Hz)" / "Error: ..."

    float GetBass()   const { return bassLevel.load(std::memory_order_relaxed); }
    float GetMid()    const { return midLevel.load(std::memory_order_relaxed); }
    float GetTreble() const { return trebleLevel.load(std::memory_order_relaxed); }

    // Tunable from the UI while capturing; read by the audio thread per packet.
    std::atomic<float> attackMs { 15.0f };
    std::atomic<float> releaseMs{ 250.0f };
    std::atomic<float> gain     { 1.0f };

private:
    void CaptureThreadMain();            // owns all COM objects as locals; never touches GL

    std::thread        captureThread;
    std::atomic<bool>  running      { false };
    std::atomic<bool>  capturing    { false };
    std::atomic<bool>  stopRequested{ false };
    std::atomic<float> bassLevel  { 0.0f };
    std::atomic<float> midLevel  { 0.0f };
    std::atomic<float> trebleLevel{ 0.0f };

    mutable std::mutex statusMutex;      // guards statusText only
    std::string        statusText{ "Idle" };
    void SetStatus(const std::string& s);
};
