const int repsPerSet = 6;
const unsigned long sampleIntervalMs = 10;
const unsigned long repIntervalMs = 2500;
const unsigned long repActiveMs = 1500;
const unsigned long preSetRestMs = 500;
const unsigned long postSetRestMs = 2500;
const unsigned long setDurationMs =
    preSetRestMs + (repsPerSet * repIntervalMs) + postSetRestMs;

const int baseline = 300;
const int noiseRange = 10;
const int repAmplitudes[repsPerSet] = {570, 540, 500, 455, 395, 355};

bool setRunning = false;
bool setCompleted = false;
unsigned long setStartMs = 0;
unsigned long lastSampleMs = 0;

void startFakeSet() {
  setStartMs = millis();
  lastSampleMs = 0;
  setRunning = true;
  setCompleted = false;
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'S') {
      startFakeSet();
    }
  }
}

int fakeSignalForElapsed(unsigned long elapsedMs) {
  int signal = baseline + random(-noiseRange, noiseRange + 1);

  if (elapsedMs < preSetRestMs) {
    return signal;
  }

  unsigned long setMs = elapsedMs - preSetRestMs;
  int repIndex = setMs / repIntervalMs;
  unsigned long repMs = setMs % repIntervalMs;

  if (repIndex >= repsPerSet || repMs > repActiveMs) {
    return signal;
  }

  float progress = (float)repMs / (float)repActiveMs;
  float roundedPulse = sin(progress * PI);
  roundedPulse = roundedPulse * roundedPulse;

  signal += (int)(repAmplitudes[repIndex] * roundedPulse);

  return signal;
}

void outputSample(unsigned long elapsedMs, int signal) {
  Serial.print(elapsedMs);
  Serial.print(",");
  Serial.println(signal);
}

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));
}

void loop() {
  readSerialCommands();

  unsigned long now = millis();

  if (now - lastSampleMs < sampleIntervalMs) {
    return;
  }

  lastSampleMs = now;

  if (!setRunning) {
    if (setCompleted) {
      outputSample(setDurationMs, baseline + random(-noiseRange, noiseRange + 1));
    }

    return;
  }

  unsigned long elapsedMs = now - setStartMs;

  if (elapsedMs > setDurationMs) {
    setRunning = false;
    setCompleted = true;
    outputSample(setDurationMs, baseline + random(-noiseRange, noiseRange + 1));
    return;
  }

  outputSample(elapsedMs, fakeSignalForElapsed(elapsedMs));
}
