import face_recognition
import cv2

# Load known faces and their names (contoh: Anda memiliki daftar gambar wajah dengan nama masing-masing)
known_face_encodings = []
known_face_names = []

# Contoh: Anda memiliki dua gambar wajah dengan nama 'John' dan 'Jane'
john_image = face_recognition.load_image_file("john.jpg")
john_face_encoding = face_recognition.face_encodings(john_image)[0]
known_face_encodings.append(john_face_encoding)
known_face_names.append("John")

bryan_image = face_recognition.load_image_file("bryan.jpg")
bryan_face_encoding = face_recognition.face_encodings(bryan_image)[0]
known_face_encodings.append(bryan_face_encoding)
known_face_names.append("Bryan")

jane_image = face_recognition.load_image_file("jane.jpg")
jane_face_encoding = face_recognition.face_encodings(jane_image)[0]
known_face_encodings.append(jane_face_encoding)
known_face_names.append("Jane")

# Inisialisasi webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Temukan semua wajah dalam frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop melalui setiap wajah yang terdeteksi dalam frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Bandingkan wajah yang terdeteksi dengan wajah yang dikenal
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default name jika wajah tidak dikenal

        # Jika ada wajah yang cocok dengan yang dikenal, pilih nama yang cocok
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Gambar kotak dan nama pada wajah yang terdeteksi
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Tampilkan frame yang telah diperbarui
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bebaskan sumber daya dan tutup jendela
video_capture.release()
cv2.destroyAllWindows()
