#include "AudioReactive.h"
#include "AudioBands.h"   // shared band-split + envelope DSP (also used by ReelExport)
#define NOMINMAX          // windows.h defines min/max macros that break std::min/std::max
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Main-thread API
// ---------------------------------------------------------------------------

AudioReactive::~AudioReactive() {
    Stop();
}

void AudioReactive::Start() {
    if (running.load()) return;
    stopRequested.store(false);
    running.store(true);
    captureThread = std::thread(&AudioReactive::CaptureThreadMain, this);
}

void AudioReactive::Stop() {
    stopRequested.store(true);
    if (captureThread.joinable()) captureThread.join();
    running.store(false);
    capturing.store(false);
    bassLevel.store(0.0f);
    midLevel.store(0.0f);
    trebleLevel.store(0.0f);
}

std::string AudioReactive::GetStatusText() const {
    std::lock_guard<std::mutex> lock(statusMutex);
    return statusText;
}

void AudioReactive::SetStatus(const std::string& s) {
    std::lock_guard<std::mutex> lock(statusMutex);
    statusText = s;
}

// ---------------------------------------------------------------------------
// Section B — WASAPI loopback capture thread (Windows Core Audio, COM)
//
// Loopback on the default RENDER endpoint captures whatever the computer is
// playing, regardless of source app — no microphone, no "Stereo Mix" device.
// Modeled on Microsoft's loopback-recording reference sample. A polling loop
// is used instead of event-driven capture: loopback events are driven by the
// render endpoint's activity and stop firing when nothing is playing, whereas
// polling just sees empty packets and keeps running; shutdown is also simpler
// (the stop flag is checked every wake).
// ---------------------------------------------------------------------------

#define NOMINMAX        // windows.h defines min/max macros that break std::min/std::max
#include <initguid.h>   // define the ksmedia GUIDs in this TU (no ksguid.lib needed)
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <mmreg.h>
#include <ksmedia.h>

void AudioReactive::CaptureThreadMain() {
    // MTA: this thread has no message pump, which STA COM would require.
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) { SetStatus("Error: CoInitializeEx failed"); running.store(false); return; }
    struct ComGuard { ~ComGuard() { CoUninitialize(); } } comGuard;

    IMMDeviceEnumerator* enumerator = nullptr;
    IMMDevice*           device     = nullptr;
    IAudioClient*        client     = nullptr;
    IAudioCaptureClient* capture    = nullptr;
    WAVEFORMATEX*        mixFmt     = nullptr;

    auto cleanup = [&]() {
        if (capture)    capture->Release();
        if (client)     client->Release();
        if (device)     device->Release();
        if (enumerator) enumerator->Release();
        if (mixFmt)     CoTaskMemFree(mixFmt);
        capturing.store(false);
    };

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator), (void**)&enumerator);
    if (FAILED(hr)) { SetStatus("Error: audio subsystem unavailable"); cleanup(); return; }

    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(hr)) { SetStatus("Error: no default playback device"); cleanup(); return; }

    hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&client);
    if (FAILED(hr)) { SetStatus("Error: could not open audio device"); cleanup(); return; }

    hr = client->GetMixFormat(&mixFmt);
    if (FAILED(hr) || !mixFmt) { SetStatus("Error: no mix format"); cleanup(); return; }

    const bool isFloat =
        (mixFmt->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) ||
        (mixFmt->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
         IsEqualGUID(reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mixFmt)->SubFormat,
                     KSDATAFORMAT_SUBTYPE_IEEE_FLOAT));
    if (!isFloat) { SetStatus("Error: unsupported mix format"); cleanup(); return; }

    const REFERENCE_TIME bufferDuration = 200 * 10000; // 200 ms in 100ns units
    hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
                            bufferDuration, 0, mixFmt, nullptr);
    if (FAILED(hr)) { SetStatus("Error: loopback init failed"); cleanup(); return; }

    hr = client->GetService(__uuidof(IAudioCaptureClient), (void**)&capture);
    if (FAILED(hr)) { SetStatus("Error: capture service failed"); cleanup(); return; }

    hr = client->Start();
    if (FAILED(hr)) { SetStatus("Error: capture start failed"); cleanup(); return; }

    const float fs  = float(mixFmt->nSamplesPerSec);
    const int   nCh = int(mixFmt->nChannels);
    capturing.store(true);
    SetStatus("Capturing (" + std::to_string(mixFmt->nSamplesPerSec) + " Hz)");

    BandState state;

    while (!stopRequested.load(std::memory_order_relaxed)) {
        Sleep(10);

        // Recompute per-packet so the UI sliders take effect live.
        const float attackCoeff  = EnvelopeCoeff(attackMs.load(std::memory_order_relaxed),  fs);
        const float releaseCoeff = EnvelopeCoeff(releaseMs.load(std::memory_order_relaxed), fs);

        UINT32 packetLen = 0;
        hr = capture->GetNextPacketSize(&packetLen);
        while (SUCCEEDED(hr) && packetLen > 0) {
            BYTE*  data = nullptr;
            UINT32 numFrames = 0;
            DWORD  flags = 0;
            hr = capture->GetBuffer(&data, &numFrames, &flags, nullptr, nullptr);
            if (FAILED(hr)) break;

            const bool  silent  = (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;
            const float* samples = reinterpret_cast<const float*>(data);

            for (UINT32 f = 0; f < numFrames; ++f) {
                float mono = 0.0f;
                if (!silent && samples) {
                    for (int c = 0; c < nCh; ++c) mono += samples[f * nCh + c];
                    mono /= float(nCh);
                }
                ProcessSample(mono, state, fs, attackCoeff, releaseCoeff);
            }

            // Clamped before publishing: a runaway amplitude fed into the SPH
            // impulse pass could destabilize the solver.
            const float g = gain.load(std::memory_order_relaxed);
            bassLevel.store(std::min(4.0f, state.envBass * g),     std::memory_order_relaxed);
            midLevel.store(std::min(4.0f, state.envMid * g),       std::memory_order_relaxed);
            trebleLevel.store(std::min(4.0f, state.envTreble * g), std::memory_order_relaxed);
             
            capture->ReleaseBuffer(numFrames);
            hr = capture->GetNextPacketSize(&packetLen);
        }
    }

    client->Stop();
    cleanup();
    SetStatus("Idle");
}
