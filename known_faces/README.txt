Face Recognition Database
========================

Add photos of people you want to recognize in this directory.

Structure:
----------
known_faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── person3/
    └── photo1.jpg

Instructions:
-------------
1. Create a new folder for each person (use their name)
2. Add 2-5 clear photos of their face
3. Supported formats: JPG, PNG, BMP
4. Photos should be well-lit and show the face clearly
5. The system will automatically load faces on startup

Example:
--------
To add "John Doe":
1. Create folder: known_faces/John_Doe/
2. Add photos: john1.jpg, john2.jpg, john3.jpg
3. Restart the application

The system will create a cache file (face_encodings.pkl) for faster loading.
If you add new faces, delete this cache file to rebuild the database.
