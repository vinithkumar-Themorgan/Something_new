#include <Arduino.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

// FireBeetle ESP32-S3: D2/D3 are active-LOW end limits.
// D6 is STEP and D7 is DIR.
#define LIMIT_REVERSE 3      // D2 (GPIO3)
#define LIMIT_FORWARD 38     // D3 (GPIO38)
#define STEP_PIN 18          // D6 (GPIO18)
#define DIR_PIN 9            // D7 (GPIO9)
#define FORWARD HIGH
#define REVERSE LOW

constexpr uint32_t PULSES_PER_REV = 400;
constexpr uint32_t MAX_SPEED = 15000;
constexpr float START_SPEED = 300.0f;
constexpr float ACCELERATION = 15000.0f;  // pulses/s^2; user's current setting
constexpr uint32_t PULSE_WIDTH_US = 5;   // check driver STEP minimum high time
constexpr uint32_t DIR_SETUP_US = 20;

volatile bool reverseEdgeSeen = false;
volatile bool forwardEdgeSeen = false;

void IRAM_ATTR reverseLimitISR() { reverseEdgeSeen = true; }
void IRAM_ATTR forwardLimitISR() { forwardEdgeSeen = true; }

uint32_t targetSpeed = 3000;
float currentSpeed = 0.0f;
uint32_t stepInterval = 0;
uint32_t remainingSteps = 0;
uint32_t completedSteps = 0;
bool motorRunning = false;
bool stepHigh = false;
bool directionForward = true;
uint32_t lastStepTime = 0;
uint32_t pulseStartTime = 0;
uint32_t directionSetTime = 0;
uint32_t moveStartMicros = 0;
uint32_t lastMoveDurationUs = 0;
bool hasMoveResult = false;
char lastStopReason[48] = "NONE";

char commandBuffer[64];
uint8_t commandIndex = 0;

inline bool reverseLimitActive() { return digitalRead(LIMIT_REVERSE) == LOW; }
inline bool forwardLimitActive() { return digitalRead(LIMIT_FORWARD) == LOW; }

// Never clear a high STEP pulse merely to change direction.
// Stops *commanded pulses*; actual motor/load can continue moving.
void stopMotor(const char* reason) {
  if (!motorRunning) return;

  // Capture stop time before serial printing or other slow operations.
  const uint32_t stopTimeMicros = micros();
  lastMoveDurationUs = stopTimeMicros - moveStartMicros;
  hasMoveResult = true;
  strncpy(lastStopReason, reason, sizeof(lastStopReason) - 1);
  lastStopReason[sizeof(lastStopReason) - 1] = '\0';

  digitalWrite(STEP_PIN, LOW);
  motorRunning = false;
  stepHigh = false;
  currentSpeed = 0.0f;
  remainingSteps = 0;

  Serial.println();
  Serial.println("==============================");
  Serial.println(reason);
  Serial.print("Issued pulses: ");
  Serial.println(completedSteps);
  Serial.print("Travel time (seconds): ");
  Serial.println(lastMoveDurationUs / 1000000.0f, 3);
  Serial.print("Travel time (ms): ");
  Serial.println(lastMoveDurationUs / 1000.0f, 2);
  Serial.println("Waiting for next command.");
  Serial.println("==============================");
}

void updateStepInterval() {
  if (currentSpeed < 1.0f) currentSpeed = 1.0f;
  stepInterval = static_cast<uint32_t>(1000000.0f / currentSpeed);
  if (stepInterval < 2 * PULSE_WIDTH_US) stepInterval = 2 * PULSE_WIDTH_US;
}

void moveMotor(int32_t steps) {
  if (motorRunning) {
    Serial.println("Motor busy. Send STOP first.");
    return;
  }
  if (steps == 0) {
    Serial.println("Invalid step count.");
    return;
  }

  bool requestedForward = steps > 0;
  // Discard historical edge events from the previous move.
  noInterrupts();
  reverseEdgeSeen = false;
  forwardEdgeSeen = false;
  interrupts();

  if (reverseLimitActive() && forwardLimitActive()) {
    Serial.println("FAULT: Both limits active. Movement blocked.");
    return;
  }
  if (requestedForward && (forwardLimitActive() || forwardEdgeSeen)) {
    Serial.println("Forward limit active. Only reverse allowed.");
    return;
  }
  if (!requestedForward && (reverseLimitActive() || reverseEdgeSeen)) {
    Serial.println("Reverse limit active. Only forward allowed.");
    return;
  }

  directionForward = requestedForward;
  remainingSteps = steps > 0 ? static_cast<uint32_t>(steps)
                             : static_cast<uint32_t>(-static_cast<int64_t>(steps));
  // Each accepted M command starts a fresh count and timing result.
  completedSteps = 0;
  lastMoveDurationUs = 0;
  hasMoveResult = false;
  strncpy(lastStopReason, "RUNNING", sizeof(lastStopReason));
  lastStopReason[sizeof(lastStopReason) - 1] = '\0';
  currentSpeed = (targetSpeed < START_SPEED) ? float(targetSpeed) : START_SPEED;
  updateStepInterval();
  digitalWrite(STEP_PIN, LOW);
  stepHigh = false;
  digitalWrite(DIR_PIN, directionForward ? FORWARD : REVERSE);
  directionSetTime = micros();
  lastStepTime = directionSetTime;

  Serial.println();
  Serial.println("========== MOVE START ==========");
  Serial.print("Direction: ");
  Serial.println(directionForward ? "FORWARD" : "REVERSE");
  Serial.print("Requested pulses: ");
  Serial.println(remainingSteps);
  Serial.print("Target frequency: ");
  Serial.print(targetSpeed);
  Serial.println(" pulses/sec");
  Serial.print("Target RPM: ");
  Serial.println(targetSpeed * 60.0f / PULSES_PER_REV, 1);
  Serial.println("Timer reset: starts with this movement.");

  // Start measurement AFTER printing to avoid counting serial output time.
  const uint32_t startNow = micros();
  directionSetTime = startNow;
  lastStepTime = startNow;
  moveStartMicros = startNow;
  motorRunning = true;
}

void updateMotionSpeed() {
  if (remainingSteps == 0) return;

  const float stoppingSteps = currentSpeed > START_SPEED
      ? (currentSpeed * currentSpeed - START_SPEED * START_SPEED) / (2.0f * ACCELERATION)
      : 0.0f;

  if (remainingSteps <= stoppingSteps) {
    currentSpeed -= ACCELERATION / currentSpeed;
    if (currentSpeed < START_SPEED) currentSpeed = START_SPEED;
  } else if (currentSpeed < targetSpeed) {
    currentSpeed += ACCELERATION / currentSpeed;
    if (currentSpeed > targetSpeed) currentSpeed = targetSpeed;
  } else if (currentSpeed > targetSpeed) {
    currentSpeed = float(targetSpeed);
  }
  updateStepInterval();
}

void updateStepper() {
  if (!motorRunning) return;

  // Edge flag catches brief LOW transitions that polling could miss.
  // Level check detects switches already held LOW.
  // The actual stop is made here in the main loop, NOT inside the ISR.
  if (reverseLimitActive() && forwardLimitActive()) {
    stopMotor("FAULT: BOTH LIMITS ACTIVE!");
    return;
  }
  if (directionForward && (forwardEdgeSeen || forwardLimitActive())) {
    stopMotor("FORWARD LIMIT TRIGGERED!");
    return;
  }
  if (!directionForward && (reverseEdgeSeen || reverseLimitActive())) {
    stopMotor("REVERSE LIMIT TRIGGERED!");
    return;
  }

  uint32_t now = micros();
  if (stepHigh) {
    if (static_cast<uint32_t>(now - pulseStartTime) >= PULSE_WIDTH_US) {
      digitalWrite(STEP_PIN, LOW);
      stepHigh = false;
      if (remainingSteps == 0) {
        stopMotor("MOVE COMPLETED!");
        return;
      }
      updateMotionSpeed();
    }
    return;
  }

  if (static_cast<uint32_t>(now - directionSetTime) < DIR_SETUP_US) return;
  if (remainingSteps > 0 &&
      static_cast<uint32_t>(now - lastStepTime) >= stepInterval) {
    // Check the intended limit again immediately before raising STEP.
    if ((directionForward && (forwardEdgeSeen || forwardLimitActive())) ||
        (!directionForward && (reverseEdgeSeen || reverseLimitActive()))) {
      stopMotor("LIMIT TRIGGERED BEFORE NEXT PULSE!");
      return;
    }
    digitalWrite(STEP_PIN, HIGH);
    pulseStartTime = micros();
    lastStepTime = pulseStartTime;
    stepHigh = true;
    --remainingSteps;
    ++completedSteps;
  }
}

void showStatus() {
  Serial.println("\n========== SYSTEM STATUS ==========");
  Serial.print("D2 reverse limit: ");
  Serial.println(reverseLimitActive() ? "ACTIVE" : "NORMAL");
  Serial.print("D3 forward limit: ");
  Serial.println(forwardLimitActive() ? "ACTIVE" : "NORMAL");
  Serial.print("Motor: ");
  Serial.println(motorRunning ? "RUNNING" : "STOPPED");
  Serial.print("Direction: ");
  Serial.println(directionForward ? "FORWARD" : "REVERSE");
  Serial.print("Target frequency: ");
  Serial.println(targetSpeed);
  Serial.print("Current commanded frequency: ");
  Serial.println(currentSpeed, 1);
  Serial.print("Issued pulses: ");
  Serial.println(completedSteps);
  Serial.print("Remaining pulses: ");
  Serial.println(remainingSteps);
  if (motorRunning) {
    Serial.print("Current movement time (s): ");
    Serial.println((uint32_t)(micros() - moveStartMicros) / 1000000.0f, 3);
  } else if (hasMoveResult) {
    Serial.print("Last movement time (s): ");
    Serial.println(lastMoveDurationUs / 1000000.0f, 3);
    Serial.print("Last stop reason: ");
    Serial.println(lastStopReason);
  } else {
    Serial.println("Last movement time: none yet");
  }
  Serial.println("===================================");
}

void processCommand(char* cmd) {
  if (strcmp(cmd, "STOP") == 0) {
    if (motorRunning) stopMotor("MANUAL STOP!");
    else Serial.println("Motor already stopped.");
    return;
  }
  if (strcmp(cmd, "STATUS") == 0) {
    showStatus();
    return;
  }
  if (cmd[0] == 'F' && cmd[1] == ' ') {
    if (motorRunning) {
      Serial.println("Stop motor before changing speed.");
      return;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long speed = strtoul(cmd + 2, &end, 10);
    if (errno == ERANGE || end == cmd + 2 || *end != '\0' ||
        speed < 1 || speed > MAX_SPEED || cmd[2] == '-') {
      Serial.println("Use F 1 through F 15000");
      return;
    }
    targetSpeed = static_cast<uint32_t>(speed);
    Serial.print("Target frequency: ");
    Serial.print(targetSpeed);
    Serial.println(" pulses/sec");
    Serial.print("Target RPM: ");
    Serial.println(targetSpeed * 60.0f / PULSES_PER_REV, 1);
    return;
  }
  if (cmd[0] == 'M' && cmd[1] == ' ') {
    errno = 0;
    char* end = nullptr;
    long steps = strtol(cmd + 2, &end, 10);
    if (errno == ERANGE || end == cmd + 2 || *end != '\0' ||
        steps == 0 || steps > 1000000L || steps < -1000000L) {
      Serial.println("Use M -1000000 through M 1000000 (excluding 0)");
      return;
    }
    moveMotor(static_cast<int32_t>(steps));
    return;
  }
  Serial.println("Unknown command. Use F n, M n, STOP, STATUS.");
}

void readSerial() {
  // Cap processing per loop pass so command input cannot starve limit polling.
  for (uint8_t n = 0; n < 16 && Serial.available(); ++n) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (commandIndex > 0) {
        commandBuffer[commandIndex] = '\0';
        commandIndex = 0;
        processCommand(commandBuffer);
      }
    } else if (commandIndex < sizeof(commandBuffer) - 1) {
      if (c >= 'a' && c <= 'z') c -= 32;
      commandBuffer[commandIndex++] = c;
    } else {
      commandIndex = 0;
      Serial.println("Command too long.");
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  pinMode(LIMIT_REVERSE, INPUT_PULLUP);
  pinMode(LIMIT_FORWARD, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(LIMIT_REVERSE), reverseLimitISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(LIMIT_FORWARD), forwardLimitISR, FALLING);

  delay(1000);
  Serial.println("\nFireBeetle ESP32-S3 linear actuator controller");
  Serial.println("400 pulses/rev | acceleration 15000 pulses/sec^2");
  Serial.println("D2 reverse / D3 forward: LOW = limit active");
  Serial.println("F 3000 | M 4000 | M -4000 | STOP | STATUS");
  Serial.println("Timer resets on every accepted M command.");
  Serial.println("Timer ends at limit, completed pulses, or STOP.");
  showStatus();
}

void loop() {
  updateStepper();
  readSerial();
}
