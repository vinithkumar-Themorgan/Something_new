// Catcher X - ESP32-S3 motion/safety controller (NO ENCODER)
// WARNING: Software E-stop / limit inputs are not safety-rated. Use an
// independent hardwired E-stop and overtravel protection.
#include <Arduino.h>
#include <U8g2lib.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// D3 = HOME; D2 = END (user's latest assignment)
constexpr uint8_t HOME_PIN = 38;  // D3, active LOW
constexpr uint8_t END_PIN = 3;    // D2, active LOW
constexpr uint8_t ESTOP_PIN = 5;  // A1, active LOW, external pull-up to 3.3 V
constexpr uint8_t STEP_PIN = 18;  // D6
constexpr uint8_t DIR_PIN = 9;    // D7
constexpr uint8_t LCD_RST = 2;   // SCL
constexpr uint8_t LCD_CS = 16;   // MISO
constexpr uint8_t LCD_SID = 15;  // MOSI
constexpr uint8_t LCD_CLK = 17;  // SCK
U8G2_ST7920_128X64_F_SW_SPI lcd(U8G2_R0, LCD_CLK, LCD_SID, LCD_CS, LCD_RST);

constexpr int HOME_DIR = -1;        // negative position direction drives toward D3
constexpr int END_DIR = +1;         // positive position direction drives toward D2
constexpr uint32_t MAX_FREQ = 15000;
constexpr float START_FREQ = 500.0f; // matched to proven 15 kHz firmware
// Runtime-adjustable ramp rates, in STEP pulses/s^2.
// Tune independently; higher numbers may stall the motor.
constexpr float DEFAULT_ACCEL = 20000.0f;
constexpr float DEFAULT_DECEL = 20000.0f;
constexpr float MIN_RAMP = 100.0f;
constexpr float MAX_RAMP = 100000.0f;
constexpr uint32_t PULSE_HIGH_US = 5;
constexpr uint32_t DIR_SETUP_US = 20;
constexpr uint32_t ESTOP_RELEASE_MS = 150;
constexpr uint32_t JOG_TIMEOUT_MS = 350;
constexpr uint32_t HOMING_TIMEOUT_MS = 45000;
constexpr uint32_t CAL_MAX_PULSES = 250000;
constexpr float TRAVEL_MM = 432.0f;
constexpr uint32_t TELEMETRY_MS = 150;

volatile bool homeEdge = false, endEdge = false;
void IRAM_ATTR homeISR() { homeEdge = true; }
void IRAM_ATTR endISR() { endEdge = true; }

bool atHome() { return digitalRead(HOME_PIN) == LOW; }
bool atEnd() { return digitalRead(END_PIN) == LOW; }
bool estopLow() { return digitalRead(ESTOP_PIN) == LOW; }

enum Mode { IDLE, MOVE, JOG, HOMING, MODE_SEEK_END, CAL_HOME, CAL_HOLD_HOME, CAL_END, CAL_HOLD_END, CAL_CENTER };
Mode mode = IDLE;
const char* modeName() {
  switch (mode) {
    case MOVE: return "MOVE"; case JOG: return "JOG";
    case HOMING: return "HOME"; case MODE_SEEK_END: return "SEEK_END"; case CAL_HOME: return "CAL_HOME";
    case CAL_HOLD_HOME: return "CAL_HOLD_HOME"; case CAL_HOLD_END: return "CAL_HOLD_END";
    case CAL_END: return "CAL_END"; case CAL_CENTER: return "CAL_CENTER";
    default: return "IDLE";
  }
}

int64_t positionSteps = 0, targetSteps = 0;
uint32_t strokeSteps = 0;    // counts between physical sensors, measured during CAL
bool positionValid = false, calibrated = false;
bool softEstop = false, physicalEstop = false;
uint32_t highSinceMs = 0;
uint32_t targetFreq = 3000;
float acceleration = DEFAULT_ACCEL;
float deceleration = DEFAULT_DECEL;
float speedNow = 0.0f;
uint32_t pulsePeriodUs = 2000; // cached period; avoid float division on each idle loop
int motionDir = 0, jogDir = 0;
uint32_t lastJogMs = 0, modeStartedMs = 0;
uint32_t lastPulseUs = 0, riseUs = 0, dirSetUs = 0;
bool pulseHigh = false, stopRequested = false;
uint32_t motionStartUs = 0, elapsedUs = 0, lastIssued = 0;
char lastEvent[32] = "READY";
char lineBuffer[72]; uint8_t lineLen = 0;
bool screenDirty = true;

void emit(const char* event) {
  strncpy(lastEvent, event, sizeof(lastEvent)-1);
  lastEvent[sizeof(lastEvent)-1] = '\0';
  Serial.print("EVENT,"); Serial.println(event);
  screenDirty = true;
}

void finishMotion(const char* why) {
  // Disable STEP generation before printing anything.
  digitalWrite(STEP_PIN, LOW);
  pulseHigh = false;
  if (mode != IDLE) elapsedUs = (uint32_t)(micros() - motionStartUs);
  speedNow = 0; motionDir = 0; stopRequested = false;
  mode = IDLE;
  emit(why);
  Serial.printf("RESULT,%.3f,%lu,%lld\n", elapsedUs/1000000.0f,
                (unsigned long)lastIssued, (long long)positionSteps);
}

void safetyPoll() {
  if (estopLow()) {
    highSinceMs = 0;
    if (!physicalEstop) {
      physicalEstop = true;
      // Position is not necessarily reliable after an unplanned motor stop.
      positionValid = false;
      finishMotion("PHYSICAL_ESTOP");
    }
  } else if (physicalEstop) {
    uint32_t ms = millis();
    if (!highSinceMs) highSinceMs = ms ? ms : 1;
    if ((uint32_t)(ms - highSinceMs) >= ESTOP_RELEASE_MS && !estopLow()) {
      physicalEstop = false; highSinceMs = 0;
      emit("ESTOP_RELEASED_REHOME");
    }
  }
  if (softEstop || physicalEstop) {
    if (mode != IDLE) finishMotion(softEstop ? "SOFTWARE_ESTOP" : "PHYSICAL_ESTOP");
    return;
  }
  if (atHome() && atEnd()) {
    if (mode != IDLE) finishMotion("BOTH_LIMITS_FAULT");
    positionValid = false;
  }
}

bool safeToMove(int dir) {
  if (physicalEstop || softEstop || estopLow()) return false;
  if (atHome() && atEnd()) return false;
  if (dir < 0 && atHome()) return false;
  if (dir > 0 && atEnd()) return false;
  return true;
}

void beginMotion(Mode next) {
  mode = next;
  stopRequested = false;
  speedNow = 0; motionDir = 0; pulseHigh = false;
  pulsePeriodUs = 2000;
  digitalWrite(STEP_PIN, LOW);
  motionStartUs = micros(); lastPulseUs = motionStartUs;
  modeStartedMs = millis(); lastIssued = 0;
  elapsedUs = 0; screenDirty = true;
}

void beginHome(bool forCalibration) {
  if (physicalEstop || softEstop || estopLow() || (atHome() && atEnd())) {
    emit("HOME_BLOCKED"); return;
  }
  if (mode != IDLE) { emit("BUSY"); return; }
  positionValid = false;
  if (atHome()) {
    positionSteps = 0; positionValid = true;
    if (!forCalibration) { emit("AT_HOME"); return; }
    // Pause so Python can capture ArUco at stationary HOME.
    mode = CAL_HOLD_HOME; emit("CAL_AT_HOME"); return;
  }
  beginMotion(forCalibration ? CAL_HOME : HOMING);
  emit(forCalibration ? "CAL_HOMING" : "HOMING");
}

void beginEndSeek() {
  if (mode != IDLE) { emit("BUSY"); return; }
  if (physicalEstop || softEstop || estopLow() || (atHome() && atEnd())) {
    emit("END_BLOCKED"); return;
  }
  if (atEnd()) {
    if (calibrated) { positionSteps = strokeSteps; positionValid = true; }
    emit("AT_END"); return;
  }
  if (!calibrated) positionValid = false;
  beginMotion(MODE_SEEK_END); emit("SEEKING_END");
}

void beginCalibration() {
  if (mode != IDLE) { emit("BUSY"); return; }
  calibrated = false;
  strokeSteps = 0;
  beginHome(true);
}

void beginJog(int dir) {
  if (dir == 0 || !safeToMove(dir)) { emit("JOG_BLOCKED"); return; }
  if (mode != IDLE && mode != JOG) { emit("BUSY"); return; }
  // Existing jog in opposite direction must brake to zero first.
  if (mode == JOG && jogDir != dir) {
    stopRequested = true; emit("JOG_BRAKING"); return;
  }
  jogDir = dir; lastJogMs = millis();
  if (mode == IDLE) { beginMotion(JOG); emit("JOGGING"); }
}

void gotoMM(float mm) {
  if (!calibrated || !positionValid || !strokeSteps || !isfinite(mm)) {
    emit("POSITION_NOT_CALIBRATED"); return;
  }
  if (physicalEstop || softEstop || estopLow()) { emit("ESTOP_BLOCKED"); return; }
  if (mm < 0.0f || mm > TRAVEL_MM) { emit("OUT_OF_RANGE"); return; }
  if (mode != IDLE && mode != MOVE && mode != CAL_CENTER) { emit("BUSY"); return; }
  targetSteps = llroundf(mm * ((float)strokeSteps / TRAVEL_MM));
  if (mode == IDLE) beginMotion(MOVE);
  else mode = MOVE;
  stopRequested = false;
  emit("TARGET_ACCEPTED");
}

// Immediately stop the oscillator on a limit. Only expected homing/calibration
// sensor activations permit the next motion stage.
void handleEndStops() {
  if (mode == IDLE) return;
  if (atHome() && atEnd()) { positionValid = false; finishMotion("BOTH_LIMITS_FAULT"); return; }
  if ((motionDir < 0 || mode == HOMING || mode == CAL_HOME) &&
      (homeEdge || atHome()) && atHome()) {
    digitalWrite(STEP_PIN, LOW); pulseHigh = false;
    if (mode == HOMING || mode == CAL_HOME) {
      bool cal = mode == CAL_HOME;
      finishMotion("AT_HOME");
      positionSteps = 0; positionValid = true;
      if (cal && !physicalEstop && !softEstop) { mode = CAL_HOLD_HOME; emit("CAL_AT_HOME"); }
    } else { positionSteps = 0; positionValid = true; finishMotion("HOME_LIMIT"); }
    return;
  }
  if (motionDir > 0 && (endEdge || atEnd()) && atEnd()) {
    if (mode == CAL_END) {
      uint32_t count = (uint32_t)positionSteps;
      finishMotion("AT_END");
      if (count < 100 || count > CAL_MAX_PULSES) { positionValid = false; emit("CAL_INVALID_STROKE"); return; }
      strokeSteps = count; calibrated = true;
      positionSteps = (int64_t)strokeSteps; positionValid = true;
      Serial.printf("CAL,STROKE_STEPS,%lu,MM,%.1f\n",(unsigned long)strokeSteps,TRAVEL_MM);
      mode = CAL_HOLD_END; emit("CAL_AT_END");
    } else {
      if (calibrated) { positionSteps = strokeSteps; positionValid = true; }
      else positionValid = false;
      finishMotion("AT_END");
    }
  }
}

void updateMotion() {
  if (mode == IDLE || mode == CAL_HOLD_HOME || mode == CAL_HOLD_END) return;
  safetyPoll();
  if (mode == IDLE) return;
  handleEndStops();
  if (mode == IDLE) return;
  if ((mode == HOMING || mode == CAL_HOME || mode == CAL_END || mode == MODE_SEEK_END) &&
       (millis() - modeStartedMs > HOMING_TIMEOUT_MS || lastIssued > CAL_MAX_PULSES)) {
    positionValid = false; finishMotion("HOMING_TIMEOUT"); return;
  }
  if (mode == JOG && (uint32_t)(millis() - lastJogMs) > JOG_TIMEOUT_MS)
    stopRequested = true;

  // FAST PATH: While STEP is HIGH, only check the 5 us pulse width.
  // Safety and limit checks have already run above. Avoid all motion
  // planning and floating point work until STEP is LOW.
  uint32_t now = micros();
  if (pulseHigh) {
    if ((uint32_t)(now - riseUs) >= PULSE_HIGH_US) {
      digitalWrite(STEP_PIN, LOW);
      pulseHigh = false;
    }
    return;
  }

  // A quick deadline check keeps the motion planner OUT of the waiting loop.
  // Do not suppress limit/E-stop checks above; those run every loop iteration.
  // If no direction has been selected, process the command immediately.
  if (motionDir != 0 && (uint32_t)(now - lastPulseUs) < pulsePeriodUs)
    return;

  int desiredDir = 0;
  float desiredSpeed = 0.0f;
  if (mode == HOMING || mode == CAL_HOME) { desiredDir = HOME_DIR; desiredSpeed = 900.0f; }
  else if (mode == CAL_END) { desiredDir = END_DIR; desiredSpeed = 1800.0f; }
  else if (mode == MODE_SEEK_END) { desiredDir = END_DIR; desiredSpeed = 900.0f; }
  else if (mode == JOG) { desiredDir = jogDir; desiredSpeed = (float)targetFreq; }
  else { // MOVE or CAL_CENTER: approach target with braking-distance logic
    int64_t error = targetSteps - positionSteps;
    // Never issue additional pulses past the requested position.
    // The braking profile should already have reduced speed on approach.
    if (!error) {
      bool wasCalCenter = mode == CAL_CENTER;
      finishMotion(wasCalCenter ? "CAL_COMPLETE" : "TARGET_REACHED"); return;
    }
    desiredDir = error > 0 ? 1 : (error < 0 ? -1 : 0);
    float remaining = (float)llabs(error);
    // Same per-step braking-distance approach as the proven old firmware.
    // Avoid sqrtf() in the high-frequency motion loop.
    const float stopDistance = speedNow > START_FREQ
      ? (speedNow * speedNow - START_FREQ * START_FREQ) / (2.0f * deceleration)
      : 0.0f;
    desiredSpeed = remaining <= stopDistance ? START_FREQ : (float)targetFreq;
    if (desiredSpeed < START_FREQ) desiredSpeed = START_FREQ;
  }
  if (stopRequested) desiredSpeed = 0;
  if (desiredDir == 0) desiredSpeed = 0;
  if (motionDir && desiredDir && desiredDir != motionDir) desiredSpeed = 0;

  // For a moving rotor, deceleration is bounded by the configured DEC value. The next pulse
  // interval is derived from the current commanded speed.
  if (speedNow <= 0.0f && desiredSpeed <= 0.0f) {
    if (stopRequested || mode == JOG) { finishMotion("STOPPED"); return; }
  }
  if (motionDir != 0 && desiredDir != motionDir && speedNow <= START_FREQ) {
    speedNow = 0; motionDir = 0;
    if (stopRequested) { finishMotion("STOPPED"); return; }
    lastPulseUs = now;
    return;
  }
  if (motionDir == 0 && desiredDir != 0) {
    if (!safeToMove(desiredDir)) { finishMotion("LIMIT_BLOCKED"); return; }
    digitalWrite(DIR_PIN, desiredDir > 0 ? LOW : HIGH); // Inverted driver direction
    dirSetUs = now; motionDir = desiredDir;
    speedNow = fminf(START_FREQ, desiredSpeed);
    pulsePeriodUs = (uint32_t)(1000000.0f / fmaxf(speedNow, 1.0f));
    lastPulseUs = now;
    return;
  }
  if (motionDir == 0) return;

  // Acceleration is updated ONCE PER STEP below, not once per loop() cycle.
  if (desiredSpeed == 0 && speedNow <= START_FREQ) {
    speedNow = 0; motionDir = 0;
    finishMotion("STOPPED"); return;
  }
  if ((uint32_t)(now - dirSetUs) < DIR_SETUP_US) return;
  if (estopLow() || physicalEstop || softEstop || !safeToMove(motionDir)) {
    positionValid = false; finishMotion("SAFETY_INHIBIT"); return;
  }
  digitalWrite(STEP_PIN, HIGH);
  riseUs = micros(); lastPulseUs = riseUs; pulseHigh = true;
  positionSteps += motionDir;
  ++lastIssued;
  // Per-step acceleration/deceleration approximation dv = a / v.
  // Trapezoidal/triangular speed profile: update once per generated STEP.
  // Speed increases by a/v for acceleration, and decreases by d/v for braking.
  const float denom = fmaxf(speedNow, START_FREQ);
  if (speedNow < desiredSpeed)
    speedNow = fminf(desiredSpeed, speedNow + acceleration / denom);
  else if (speedNow > desiredSpeed)
    speedNow = fmaxf(START_FREQ, speedNow - deceleration / denom);
  // Calculate next pulse period ONCE per emitted STEP, never on idle spins.
  pulsePeriodUs = (uint32_t)(1000000.0f / fmaxf(speedNow, START_FREQ));
  if (pulsePeriodUs < PULSE_HIGH_US + 4) pulsePeriodUs = PULSE_HIGH_US + 4;
}

void emitTelemetry() {
  // T,mode,physical_estop,software_estop,home,end,position_mm,target_mm,
  // speed_pulses_s,configured_frequency,position_valid,calibrated,stroke_steps,
  // last_time_seconds,issued_steps
  float mm = (calibrated && strokeSteps) ? (float)positionSteps * TRAVEL_MM/strokeSteps : 0;
  float targetMM = (calibrated && strokeSteps) ? (float)targetSteps * TRAVEL_MM/strokeSteps : 0;
  Serial.printf("T,%s,%d,%d,%d,%d,%.2f,%.2f,%.0f,%lu,%d,%d,%lu,%.3f,%lu\n",
    modeName(),(int)physicalEstop,(int)softEstop,(int)atHome(),(int)atEnd(),mm,targetMM,
    speedNow,(unsigned long)targetFreq,(int)positionValid,(int)calibrated,
    (unsigned long)strokeSteps,elapsedUs/1000000.0f,(unsigned long)lastIssued);
}

void drawScreen() {
  if (mode != IDLE) return; // slow software SPI must not disturb STEP timing
  lcd.clearBuffer(); lcd.setFont(u8g2_font_6x10_tf);
  lcd.drawStr(0,9,"CATCHER X"); lcd.drawHLine(0,12,128);
  lcd.setCursor(0,23); lcd.print("STATE:");
  lcd.print(physicalEstop||softEstop ? "E-STOP" : (positionValid ? "READY" : "HOME NEEDED"));
  lcd.setCursor(0,33); lcd.print("H:");lcd.print(atHome()?"ON":"--");
  lcd.print(" E:");lcd.print(atEnd()?"ON":"--");
  lcd.setCursor(0,43); lcd.print("POS:");
  if (positionValid && calibrated && strokeSteps) lcd.print((float)positionSteps*TRAVEL_MM/strokeSteps,1);
  else lcd.print("UNCAL");
  lcd.setCursor(0,53); lcd.print("F:"); lcd.print(targetFreq);
  lcd.print(" A:"); lcd.print((int)acceleration);
  lcd.setCursor(0,63); lcd.print("TIME:");lcd.print(elapsedUs/1000000.0f,2);lcd.print("s");
  lcd.sendBuffer(); screenDirty = false;
}

void processCommand(char* cmd) {
  if (!strcmp(cmd,"STATUS")) {
    emitTelemetry();
    Serial.printf("RAMP,ACC,%.0f,DEC,%.0f\n", acceleration, deceleration);
    return;
  }
  if (!strcmp(cmd,"ESTOP")) {
    softEstop = true; positionValid = false; finishMotion("SOFTWARE_ESTOP"); return;
  }
  if (!strcmp(cmd,"CLEAR")) {
    if (estopLow() || physicalEstop) { emit("PHYSICAL_ESTOP_ACTIVE"); return; }
    softEstop = false; emit("SOFTWARE_ESTOP_CLEARED_REHOME"); return;
  }
  if (!strcmp(cmd,"STOP") || !strcmp(cmd,"JOGSTOP")) {
    if (mode == IDLE) { emit("ALREADY_STOPPED"); return; }
    if (mode == CAL_HOLD_HOME || mode == CAL_HOLD_END) {
      positionValid = false; calibrated = false; mode = IDLE; emit("CAL_ABORTED"); return;
    }
    if (mode == CAL_END || mode == CAL_HOME || mode == HOMING || mode == MODE_SEEK_END) positionValid = false;
    stopRequested = true; emit("BRAKING"); return;
  }
  if (!strcmp(cmd,"CALNEXT")) {
    if (physicalEstop || softEstop || estopLow()) { emit("ESTOP_BLOCKED"); return; }
    if (mode == CAL_HOLD_HOME) { beginMotion(CAL_END); emit("CAL_TO_END"); }
    else if (mode == CAL_HOLD_END) {
      targetSteps = (int64_t)strokeSteps / 2;
      beginMotion(CAL_CENTER); emit("CAL_RETURN_CENTER");
    } else emit("CALNEXT_NOT_EXPECTED");
    return;
  }
  if (!strcmp(cmd,"HOME")) { beginHome(false); return; }
  if (!strcmp(cmd,"CAL")) { beginCalibration(); return; }
  if (!strcmp(cmd,"END")) { beginEndSeek(); return; }
  if (!strcmp(cmd,"CENTER")) { gotoMM(TRAVEL_MM/2); return; }
  if (!strcmp(cmd,"JOG L")) { beginJog(-1); return; }
  if (!strcmp(cmd,"JOG R")) { beginJog(+1); return; }
  if (!strncmp(cmd,"ACC ",4) || !strncmp(cmd,"DEC ",4)) {
    const bool isAccel = cmd[0] == 'A';
    char* end = nullptr;
    const float value = strtof(cmd + 4, &end);
    if (end == cmd + 4 || *end != '\0' || !isfinite(value) ||
        value < MIN_RAMP || value > MAX_RAMP) {
      emit("BAD_RAMP_USE_100_TO_100000"); return;
    }
    if (isAccel) acceleration = value;
    else deceleration = value;
    Serial.printf("RAMP,ACC,%.0f,DEC,%.0f\n", acceleration, deceleration);
    emit(isAccel ? "ACC_SET" : "DEC_SET");
    return;
  }
  if (!strcmp(cmd, "RAMP")) {
    Serial.printf("RAMP,ACC,%.0f,DEC,%.0f\n", acceleration, deceleration);
    return;
  }
  if (!strncmp(cmd,"F ",2)) {
    char* e = nullptr; unsigned long v = strtoul(cmd+2,&e,10);
    if (!e || *e || v < 100 || v > MAX_FREQ || cmd[2]=='-') { emit("BAD_FREQ"); return; }
    targetFreq = v; emit("FREQ_SET"); return;
  }
  if (!strncmp(cmd,"GOTO ",5)) {
    char* e = nullptr; float mm = strtof(cmd+5,&e);
    if (e == cmd+5 || *e || !isfinite(mm)) { emit("BAD_GOTO"); return; }
    gotoMM(mm); return;
  }
  if (!strncmp(cmd,"STEP ",5)) {
    char* e = nullptr; long n = strtol(cmd+5,&e,10);
    if (e == cmd+5 || *e || n == 0 || labs(n)>1000000 || !calibrated || !positionValid) {
      emit("BAD_STEP_OR_UNCALIBRATED"); return;
    }
    if (mode != IDLE && mode != MOVE) { emit("BUSY"); return; }
    targetSteps = positionSteps + n;
    if (targetSteps < 0 || targetSteps > strokeSteps) { emit("OUT_OF_RANGE"); return; }
    if (mode == IDLE) beginMotion(MOVE);
    else mode = MOVE;
    emit("TARGET_ACCEPTED"); return;
  }
  emit("UNKNOWN_COMMAND");
}

void pollSerial() {
  // Limit command parsing work per loop pass.
  for (uint8_t i=0; i<24 && Serial.available(); ++i) {
    char ch = (char)Serial.read();
    if (ch=='\r'||ch=='\n') {
      if (lineLen) { lineBuffer[lineLen]=0; lineLen=0; processCommand(lineBuffer); }
    } else if (lineLen < sizeof(lineBuffer)-1) {
      if (ch>='a' && ch<='z') ch -= 32;
      lineBuffer[lineLen++] = ch;
    } else lineLen=0;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(STEP_PIN, OUTPUT); digitalWrite(STEP_PIN, LOW);
  pinMode(DIR_PIN, OUTPUT); digitalWrite(DIR_PIN, LOW);
  pinMode(HOME_PIN, INPUT_PULLUP); pinMode(END_PIN, INPUT_PULLUP);
  pinMode(ESTOP_PIN, INPUT); // external 3.3V pull-up
  attachInterrupt(digitalPinToInterrupt(HOME_PIN), homeISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(END_PIN), endISR, FALLING);
  physicalEstop = estopLow();
  lcd.begin(); lcd.clearBuffer(); lcd.setFont(u8g2_font_6x10_tf);
  lcd.drawStr(15,26,"CATCHER X"); lcd.drawStr(10,46,"CONTROLLER"); lcd.sendBuffer();
  Serial.println("EVENT,BOOT_CATCHER_X");
  Serial.printf("RAMP,ACC,%.0f,DEC,%.0f\n", acceleration, deceleration);
  emitTelemetry();
}

void loop() {
  safetyPoll();
  // Clear stale limit-edge flags only when the associated sensor is released.
  if (!atHome()) homeEdge = false;
  if (!atEnd()) endEdge = false;
  updateMotion();
  pollSerial();
  static uint32_t lastT=0, lastLCD=0;
  uint32_t ms=millis();
  // No periodic blocking Serial.printf() while STEP pulses are running.
  // In this diagnostic build, UI telemetry updates resume when stopped.
  if (mode == IDLE && (uint32_t)(ms-lastT)>=TELEMETRY_MS) {
    emitTelemetry(); lastT=ms;
  }
  if (mode==IDLE && (screenDirty || (uint32_t)(ms-lastLCD)>500)) {
    drawScreen(); lastLCD=ms;
  }
}
