// ─────────────────────────────────────────────────────────────────────────────
// ChatKit — shared hands-free voice layer (TTS + swappable STT).
// Shared between the Control Station and Life Advice Coach iOS apps via the
// standalone `chat-kit` Swift package. Diagnostics are surfaced through the
// injectable `SpeechService.logHook` closure so the package stays free of any
// app-specific logging dependency.
// ─────────────────────────────────────────────────────────────────────────────

import Foundation
import AVFoundation
import Speech
import Combine

/// Which speech-to-text backend dictation uses.
///
/// `apple` is the built-in `SFSpeechRecognizer` path implemented in this file.
/// `fluid` delegates to an injected `AltDictationEngine` (e.g. the FluidAudio /
/// Nemotron Core ML engine). The shared file deliberately does NOT depend on
/// FluidAudio; the host app registers `SpeechService.altEngineFactory` so apps
/// without that dependency (e.g. Life Advice Coach) still compile.
public enum STTProvider: String, CaseIterable, Identifiable {
    case apple
    case fluid
    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .apple: return "Apple (built-in)"
        case .fluid: return "Nemotron (on-device)"
        }
    }
}

/// A pluggable dictation backend. Implementations own their own audio capture
/// and recognition, and report progress back through the supplied closures.
@MainActor
public protocol AltDictationEngine: AnyObject {
    /// True once the model is downloaded and loaded and ready to recognize.
    var isModelReady: Bool { get }
    /// Whether the model files exist on disk (downloaded), independent of
    /// whether they're currently loaded into memory.
    var isModelDownloaded: Bool { get }
    /// Human-readable status (e.g. "Downloading model… 42%", "Ready").
    var statusText: String { get }
    /// Downloads/loads the model if needed. `onStatus` receives progress text.
    func prepare(onStatus: @escaping (String) -> Void) async
    /// Deletes the downloaded model files from disk, freeing space. The engine
    /// unloads any in-memory model first. Returns once removal completes.
    func deleteModel() async throws
    /// Begins continuous dictation. `onPartial` fires with the live transcript,
    /// `onFinal` with the transcript when `stop()` is called, `onError` on
    /// failure. `onListening` reflects capture state.
    func start(seed: String,
               onPartial: @escaping (String) -> Void,
               onFinal: @escaping (String) -> Void,
               onError: @escaping (String) -> Void,
               onListening: @escaping (Bool) -> Void)
    /// Stops capture and delivers the final transcript via the start `onFinal`.
    func stop()
}

/// Hands-free voice layer: text-to-speech for assistant replies and
/// speech-to-text dictation for composing messages.
///
/// TTS uses `AVSpeechSynthesizer`. STT uses the `Speech` framework's
/// on-device/`SFSpeechRecognizer` recognition fed by an `AVAudioEngine` tap.
/// Both share the app's `AVAudioSession`; we deliberately do NOT keep the
/// session active when idle so the app doesn't hold the audio route.
@MainActor
public final class SpeechService: NSObject, ObservableObject {
    public static let shared = SpeechService()

    // MARK: Diagnostics

    /// Optional logging hook the host app installs to capture diagnostic lines
    /// (ChatKit deliberately doesn't depend on any app-specific logger). No-op
    /// until set.
    public static var logHook: ((String) -> Void)?

    public static func log(_ text: String) { logHook?(text) }

    // MARK: Published state (drives composer/settings UI)

    /// True while the synthesizer is actively speaking.
    @Published public private(set) var isSpeaking = false
    /// True while dictation is capturing microphone audio.
    @Published public private(set) var isListening = false
    /// Live partial transcript while dictating (empty when idle).
    @Published public private(set) var partialTranscript = ""
    /// Last user-facing error (permission denied, recognizer unavailable, …).
    @Published public var errorMessage: String?

    // MARK: STT provider selection

    /// Factory the host app registers to supply the alternate (FluidAudio /
    /// Nemotron) dictation engine. Left nil in apps without that dependency, in
    /// which case selecting `.fluid` falls back to Apple.
    public static var altEngineFactory: (() -> AltDictationEngine)?

    private static let providerDefaultsKey = "stt.provider"

    /// The currently selected STT backend, persisted across launches. Selecting
    /// `.fluid` when no factory is registered silently falls back to `.apple`.
    @Published public var sttProvider: STTProvider {
        didSet {
            UserDefaults.standard.set(sttProvider.rawValue, forKey: Self.providerDefaultsKey)
        }
    }

    /// Whether an alternate engine is available in this build.
    public var altEngineAvailable: Bool { Self.altEngineFactory != nil }

    /// Live status string from the alternate engine (download/load progress).
    @Published public private(set) var altStatusText = ""

    /// Lazily-created alternate engine instance.
    private var _altEngine: AltDictationEngine?
    private var altEngine: AltDictationEngine? {
        if _altEngine == nil { _altEngine = Self.altEngineFactory?() }
        return _altEngine
    }

    // MARK: TTS

    private let synthesizer = AVSpeechSynthesizer()

    // MARK: STT

    private let audioEngine = AVAudioEngine()
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    /// Called with the final (or best-effort) transcript when dictation stops.
    private var onFinalTranscript: ((String) -> Void)?
    private var onPartialTranscript: ((String) -> Void)?
    /// Incremented every time a recognition segment starts. Callbacks from a
    /// previous segment's task carry an older generation and are ignored, so a
    /// stale final/error can't clobber the freshly restarted segment.
    private var recognitionGeneration = 0
    private var tapInstalled = false
    /// Fires after a short silence to proactively commit the current segment and
    /// restart recognition, because iOS on-device recognition frequently drops
    /// earlier words and starts over after a pause WITHOUT ever firing isFinal.
    private var silenceTimer: Timer?
    /// The last segment text we saw; used to detect "no new speech" for the
    /// silence timer and to detect iOS silently truncating the running segment.
    private var lastSegmentSnapshot = ""
    private let silenceCommitInterval: TimeInterval = 0.8
    /// Text committed from earlier recognition segments in this listening
    /// session. iOS finalizes a segment after a pause; we fold that segment's
    /// text into here and restart the recognizer, so a pause never erases the
    /// running transcript. `partialTranscript` = committed + current segment.
    private var committedText = ""
    /// The live text of the segment currently being recognized.
    private var segmentText = ""
    /// True while we intend to keep listening; distinguishes a user Stop from an
    /// automatic segment finalization that should transparently restart.
    private var wantsListening = false

    override init() {
        let raw = UserDefaults.standard.string(forKey: Self.providerDefaultsKey)
        self.sttProvider = STTProvider(rawValue: raw ?? "") ?? .apple
        super.init()
        synthesizer.delegate = self
    }

    /// Prepares the alternate engine (downloads/loads model). Safe to call
    /// repeatedly; updates `altStatusText`. No-op if no alternate engine.
    public func prepareAltEngine() async {
        guard let engine = altEngine else { return }
        await engine.prepare { [weak self] status in
            Task { @MainActor in self?.altStatusText = status }
        }
        altStatusText = engine.statusText
    }

    /// Whether the alternate engine's model is downloaded to disk.
    public var altModelDownloaded: Bool { altEngine?.isModelDownloaded ?? false }

    /// Deletes the alternate engine's downloaded model from disk. Updates
    /// `altStatusText`. No-op if no alternate engine.
    public func deleteAltModel() async {
        guard let engine = altEngine else { return }
        do {
            try await engine.deleteModel()
            altStatusText = engine.statusText
        } catch {
            altStatusText = "Delete failed: \(error.localizedDescription)"
        }
    }

    /// True when the selected provider is the alternate engine AND it's wired up.
    private var usingAltEngine: Bool {
        sttProvider == .fluid && Self.altEngineFactory != nil
    }

    // MARK: - Text to speech

    /// Speaks `text` aloud, cancelling any in-flight utterance first.
    /// `rate` is 0...1, mapped onto the platform's min/max speech rate.
    public func speak(_ text: String, rate: Double = 0.5) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        // Don't talk over active dictation.
        guard !isListening else { return }

        stopSpeaking()
        configureSession(for: .playback)

        let utterance = AVSpeechUtterance(string: trimmed)
        let min = AVSpeechUtteranceMinimumSpeechRate
        let max = AVSpeechUtteranceMaximumSpeechRate
        let defaultRate = AVSpeechUtteranceDefaultSpeechRate
        // Map 0.5 -> default, 0 -> min, 1 -> max so the slider feels natural.
        if rate <= 0.5 {
            utterance.rate = min + (defaultRate - min) * Float(rate / 0.5)
        } else {
            utterance.rate = defaultRate + (max - defaultRate) * Float((rate - 0.5) / 0.5)
        }
        utterance.voice = Self.bestVoice()
        SpeechService.log("TTS voice: \(utterance.voice?.name ?? "system") q=\(utterance.voice.map { String(describing: $0.quality) } ?? "?")")
        synthesizer.speak(utterance)
    }

    /// Picks the highest-quality installed voice for the current locale so the
    /// output sounds natural rather than robotic. Premium > Enhanced > Default.
    /// The user downloads Enhanced/Premium voices in
    /// Settings → Accessibility → Spoken Content → Voices.
    private static func bestVoice() -> AVSpeechSynthesisVoice? {
        let preferred = AVSpeechSynthesisVoice.currentLanguageCode()
        let langCode = String(preferred.prefix(2))
        let voices = AVSpeechSynthesisVoice.speechVoices()

        func rank(_ q: AVSpeechSynthesisVoiceQuality) -> Int {
            switch q {
            case .premium: return 3
            case .enhanced: return 2
            default: return 1
            }
        }
        // Prefer voices matching the full locale, then the language, and among
        // those the highest quality. Skip novelty voices.
        let candidates = voices.filter {
            !$0.identifier.contains("eloquence") &&
            ($0.language == preferred || $0.language.hasPrefix(langCode))
        }
        let exact = candidates.filter { $0.language == preferred }
        let pool = exact.isEmpty ? candidates : exact
        return pool.max { rank($0.quality) < rank($1.quality) }
            ?? AVSpeechSynthesisVoice(language: preferred)
    }

    public func stopSpeaking() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
    }

    // MARK: - Speech to text (dictation)

    /// Requests mic + speech permission then begins dictation. The live text is
    /// published on `partialTranscript`; the final result is delivered to
    /// `onResult` (also called if dictation is stopped early).
    public func startListening(seed: String = "",
                        onPartial: ((String) -> Void)? = nil,
                        onResult: @escaping (String) -> Void) {
        guard !isListening else { return }
        stopSpeaking()

        // Route to the alternate (FluidAudio / Nemotron) engine when selected.
        if usingAltEngine, let engine = altEngine {
            onFinalTranscript = nil
            onPartialTranscript = nil
            engine.start(
                seed: seed.trimmingCharacters(in: .whitespacesAndNewlines),
                onPartial: { [weak self] text in
                    self?.partialTranscript = text
                    onPartial?(text)
                },
                onFinal: { [weak self] text in
                    self?.isListening = false
                    self?.partialTranscript = ""
                    onResult(text)
                },
                onError: { [weak self] message in
                    self?.isListening = false
                    self?.errorMessage = message
                },
                onListening: { [weak self] listening in
                    self?.isListening = listening
                }
            )
            return
        }

        onFinalTranscript = onResult
        onPartialTranscript = onPartial
        committedText = seed.trimmingCharacters(in: .whitespacesAndNewlines)
        segmentText = ""
        wantsListening = true

        // Start every session from a clean engine/tap state. The audio engine
        // and mic tap may still be up from a previous session, or the session
        // may be configured for TTS playback — in which case beginRecognition
        // would skip reconfiguring for .record and the first turn would capture
        // no audio. Reset here so beginRecognition always reinstalls the record
        // tap on an active record session.
        resetEngine()

        requestAuthorization { [weak self] granted in
            guard let self else { return }
            SpeechService.log("STT startListening: seed=\(self.committedText.count)ch granted=\(granted)")
            guard granted else {
                self.wantsListening = false
                self.errorMessage = "Microphone or speech recognition permission was denied. Enable it in Settings."
                return
            }
            do {
                try self.beginRecognition()
            } catch {
                self.errorMessage = "Couldn't start dictation: \(error.localizedDescription)"
                self.teardownEngine()
            }
        }
    }

    /// Stops the audio capture and finalizes the transcript. Only an explicit
    /// user Stop ends the session — pauses are handled by restarting internally.
    public func stopListening() {
        if usingAltEngine, let engine = altEngine {
            engine.stop()
            return
        }
        guard isListening || wantsListening else { return }
        SpeechService.log("STT stopListening (user): committed=\(committedText.count)ch segment=\(segmentText.count)ch")
        wantsListening = false
        let finalText = currentTranscript
        teardownEngine()
        onFinalTranscript?(finalText)
        onFinalTranscript = nil
        onPartialTranscript = nil
    }

    /// committed text plus the in-progress segment, normalized.
    private var currentTranscript: String {
        let joined = (committedText + " " + segmentText)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.replacingOccurrences(of: "  ", with: " ")
    }

    // MARK: - Private

    private func beginRecognition() throws {
        let recognizer = SFSpeechRecognizer(locale: Locale.current) ?? SFSpeechRecognizer()
        guard let recognizer, recognizer.isAvailable else {
            errorMessage = "Speech recognition isn't available right now."
            return
        }
        self.recognizer = recognizer

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        self.request = request

        let inputNode = audioEngine.inputNode

        // Install the mic tap only on the first segment; on a restart after a
        // pause the engine and tap stay alive and the tap already feeds
        // self.request (the current request). Activate the audio session BEFORE
        // reading the input format: reading it while the session is inactive can
        // return an invalid (0 Hz / 0-channel) format, and installTapOnBus then
        // throws an uncaught Obj-C exception that aborts the app.
        if !tapInstalled {
            configureSession(for: .record)
            // Defensively clear any tap left over from a prior session.
            inputNode.removeTap(onBus: 0)
            let format = inputNode.inputFormat(forBus: 0)
            guard format.sampleRate > 0, format.channelCount > 0 else {
                errorMessage = "Microphone isn't ready. Please try again."
                teardownEngine()
                return
            }
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
                self?.request?.append(buffer)
            }
            tapInstalled = true
        }
        if !audioEngine.isRunning {
            audioEngine.prepare()
            try audioEngine.start()
        }

        segmentText = ""
        lastSegmentSnapshot = ""
        isListening = true

        recognitionGeneration += 1
        let generation = recognitionGeneration
        SpeechService.log("STT beginRecognition: gen=\(generation) onDevice=\(recognizer.supportsOnDeviceRecognition) engineRunning=\(audioEngine.isRunning) tapInstalled=\(tapInstalled)")
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if !self.wantsListening {
                    SpeechService.log("STT cb gen=\(generation) ignored (wantsListening=false)")
                    return
                }
                // Ignore callbacks from a superseded segment.
                guard generation == self.recognitionGeneration else {
                    SpeechService.log("STT cb gen=\(generation) ignored (stale, current=\(self.recognitionGeneration))")
                    return
                }
                if let result {
                    let text = result.bestTranscription.formattedString
                    SpeechService.log("STT cb gen=\(generation) result final=\(result.isFinal) text=\"\(text.prefix(40))\"(\(text.count)ch)")

                    // Detect iOS silently truncating/restarting the running
                    // segment: a new partial that is shorter than what we
                    // already have AND isn't a prefix-extension means iOS threw
                    // away earlier words (this is the pause bug). Commit what we
                    // had, then treat this callback's text as the new segment.
                    if !text.isEmpty,
                       !self.lastSegmentSnapshot.isEmpty,
                       text.count < self.lastSegmentSnapshot.count,
                       !self.lastSegmentSnapshot.hasPrefix(text) {
                        SpeechService.log("STT truncation detected: was \(self.lastSegmentSnapshot.count)ch now \(text.count)ch — committing")
                        self.commitSegment()
                    }

                    if !text.isEmpty || !result.isFinal {
                        self.segmentText = text
                        self.lastSegmentSnapshot = text
                        self.partialTranscript = self.currentTranscript
                        self.onPartialTranscript?(self.partialTranscript)
                    }
                    if result.isFinal {
                        self.commitSegmentAndRestart()
                    } else {
                        // iOS on-device recognition often drops earlier words and
                        // restarts after a pause WITHOUT firing isFinal. So arm a
                        // short silence timer: if no new speech arrives, commit
                        // the current words and restart a fresh segment before
                        // iOS can discard them.
                        self.armSilenceTimer(generation: generation)
                    }
                } else if let error {
                    SpeechService.log("STT cb gen=\(generation) error=\(error.localizedDescription) segment=\(self.segmentText.count)ch")
                    // On error, if the user still wants to listen, restart;
                    // otherwise finish with what we have.
                    if self.wantsListening {
                        self.commitSegmentAndRestart()
                    } else {
                        self.finishRecognition()
                    }
                }
            }
        }
    }

    /// Folds the current segment text into the committed transcript (no restart).
    private func commitSegment() {
        if !segmentText.isEmpty {
            committedText = (committedText + " " + segmentText)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        segmentText = ""
        lastSegmentSnapshot = ""
        partialTranscript = currentTranscript
        onPartialTranscript?(partialTranscript)
    }

    /// (Re)arms the silence timer. If no new speech arrives within
    /// `silenceCommitInterval`, commit the current words and restart a fresh
    /// recognition segment so iOS can't discard them on the next utterance.
    private func armSilenceTimer(generation: Int) {
        silenceTimer?.invalidate()
        let snapshot = segmentText
        silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceCommitInterval, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                guard self.wantsListening, generation == self.recognitionGeneration else { return }
                // Only act if speech stalled (segment unchanged) and we have text.
                guard !self.segmentText.isEmpty, self.segmentText == snapshot else { return }
                SpeechService.log("STT silence commit: \(self.segmentText.count)ch after pause")
                self.commitSegmentAndRestart()
            }
        }
    }

    /// Folds the just-finalized segment into the committed transcript and spins
    /// up a fresh recognition segment on the still-running audio engine.
    private func commitSegmentAndRestart() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        commitSegment()
        SpeechService.log("STT commitSegmentAndRestart: committed now=\(committedText.count)ch wantsListening=\(wantsListening)")

        // Bump the generation so the outgoing task's cancel-triggered callback
        // is ignored, then tear down just the recognition task/request. Keep the
        // audio engine and mic tap alive so capture is continuous.
        recognitionGeneration += 1
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil

        guard wantsListening else { finishRecognition(); return }
        do {
            try beginRecognition()
        } catch {
            errorMessage = "Dictation restart failed: \(error.localizedDescription)"
            finishRecognition()
        }
    }

    /// Recognition ended and the user did not ask to continue.
    private func finishRecognition() {
        guard isListening || wantsListening else { return }
        SpeechService.log("STT finishRecognition: delivering \(currentTranscript.count)ch")
        wantsListening = false
        let finalText = currentTranscript
        teardownEngine()
        onFinalTranscript?(finalText)
        onFinalTranscript = nil
        onPartialTranscript = nil
    }

    /// Tears down any lingering audio engine/tap/recognition state WITHOUT
    /// clearing the transcript or listening intent. Used at the start of a new
    /// session so beginRecognition reconfigures cleanly for recording.
    private func resetEngine() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        tapInstalled = false
        recognitionGeneration += 1
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        lastSegmentSnapshot = ""
    }

    private func teardownEngine() {
        SpeechService.log("STT teardownEngine: wiping committed=\(committedText.count)ch segment=\(segmentText.count)ch")
        silenceTimer?.invalidate()
        silenceTimer = nil
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        tapInstalled = false
        recognitionGeneration += 1
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        isListening = false
        partialTranscript = ""
        committedText = ""
        segmentText = ""
        deactivateSession()
    }

    private enum SessionMode { case playback, record }

    private func configureSession(for mode: SessionMode) {
        let session = AVAudioSession.sharedInstance()
        do {
            switch mode {
            case .playback:
                try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            case .record:
                try session.setCategory(.playAndRecord, mode: .measurement,
                                        options: [.duckOthers, .defaultToSpeaker, .allowBluetooth])
            }
            try session.setActive(true, options: [])
        } catch {
            errorMessage = "Audio session error: \(error.localizedDescription)"
        }
    }

    private func deactivateSession() {
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    private func requestAuthorization(_ completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { speechStatus in
            let speechOK = speechStatus == .authorized
            AVAudioApplication.requestRecordPermission { micOK in
                Task { @MainActor in completion(speechOK && micOK) }
            }
        }
    }
}

extension SpeechService: AVSpeechSynthesizerDelegate {
    public nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = true }
    }

    public nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            self.deactivateSession()
        }
    }

    public nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = false }
    }
}
