import cv2
import mediapipe as mp 
import time

"""
For the stable use of mediapipe.solutions we need Python3.12
In the sample repository attribute .tasks is being used, but
for now for more simple start we have .solution which we can change later
"""

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)



def main():
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,

    ) as hands:
        while True:
            attempt = 0
            success, img = cap.read()
            while not success and attempt < 5:
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1
            if not success:
                print("Failed to read the frame")
                break

            img = cv2.flip(img, 1)

            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,


                    )

                    finger_tips = {
                        "Thumb":hand_landmarks.landmark[4],
                        "Index":hand_landmarks.landmark[8],
                        "Middle":hand_landmarks.landmark[12],
                        "Ring":hand_landmarks.landmark[16],
                        "Pinky":hand_landmarks.landmark[20],

                    }

                    for name, landmark in finger_tips.items():
                        x,y = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(
                            img,
                            name,
                            (x, y -10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,255,255),
                            1   
                        )

                        cv2.circle(
                            img,
                            (x,y),
                            5,
                            (0,255,0),
                            -1
                        )

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap .release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()