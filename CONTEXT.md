# Context — MyJarvisAIBot

Shared vocabulary for this project. Glossary only: no implementation detail, no
decisions. Decisions live in `MASTER-PLAN.md`, code lives in the code.

## The persona

**DmillsGPT** — the character the bot plays: Darrick Miller, host of the
basketball podcast *Numbers on the Board*. Not one thing but three independent
layers, built and evaluated separately.

**Voiceprint** — the acoustic timbre. What he *sounds* like. A TTS artifact,
trained from clean isolated audio. Private and undistributed: the weights and
training audio never leave this machine and are never committed.

**Register** — diction, cadence, catchphrases, comedic timing, and how he builds
an argument. What he *sounds like saying*. A prompt-and-retrieval artifact.
Most of the perceived character lives here rather than in Voiceprint.

**Canon** — his takes, running bits, and show history. What he *knows and
believes*. A retrieval artifact over the transcript corpus.

## The system

**Brain** — the agent core: reasoning, tools, memory. Runs as a headless service.

**Body** — the audio loop, LEDs, and Arduino peripherals. A client of the Brain.
The seam between them is a WebSocket, which is why the Body can later be a phone.

**Turn** — one contiguous stretch of a single speaker talking. The atom of the
corpus pipeline.

**Crosstalk** — a turn overlapped in time by another speaker. Fatal for
Voiceprint training, harmless for Canon.

## Words we don't use

**Train** — ambiguous across three unrelated pipelines. Say **fine-tune**
(adjusting model weights), **retrieve** (fetching context at inference time), or
**clone** (building a Voiceprint from reference audio) instead.

**Account** — say **Google account** or **Microsoft tenant**. They behave
differently and only one of them is in scope.
