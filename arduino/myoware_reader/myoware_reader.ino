const int emgPin = A0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int emgValue = analogRead(emgPin);

  Serial.print(millis());
  Serial.print(",");
  Serial.println(emgValue);

  delay(10);
}