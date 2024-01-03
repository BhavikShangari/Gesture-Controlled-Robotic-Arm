#include<cvzone.h>
#include<Servo.h>

SerialData serialData(4,3);
int valsRec[4];
Servo r1;
Servo r2;
Servo r3;
Servo r4;
Servo r5;
void setup() {
  serialData.begin();
  r1.attach(3);
  r2.attach(5);
  r3.attach(6);
  r4.attach(9);
  r5.attach(10);
}

void loop() {

  serialData.Get(valsRec);
  r1.write(valsRec[0]);
  Serial.println(valsRec[0]);
  r2.write(valsRec[1]);
  r3.write(valsRec[2]);
  r4.write(valsRec[3]);
  r5.write(valsRec[3]);

}