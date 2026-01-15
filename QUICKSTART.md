# Quick Start Guide

## Human Recognition and Activity Understanding System

### Installation (5 minutes)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Setup** (Optional)
   ```bash
   python setup.py
   ```

### Add Known Faces (2 minutes)

1. Create folders in `known_faces/` for each person:
   ```
   known_faces/
   ├── John/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   └── Sarah/
       └── photo1.jpg
   ```

2. Add 2-5 clear face photos per person

### Run the System (1 minute)

**Webcam Mode:**
```bash
python src/main.py
```

**Video File Mode:**
```bash
python src/main.py --video test.mp4 --output result.mp4
```

### Keyboard Controls

- **q** - Quit
- **s** - Save screenshot
- **p** - Pause/resume
- **r** - Reset tracker

### Test the System

```bash
python test_system.py
```

### Troubleshooting

**Camera not opening?**
- Try different camera index: `python src/main.py --camera 1`
- Check camera permissions

**Low FPS?**
- Edit `config.yaml`: Set `detection.device: "cpu"`
- Use lighter model: `detection.model: "yolov8n"`

**Face recognition not working?**
- Add more photos to `known_faces/`
- Delete `face_encodings.pkl` to rebuild database
- Adjust tolerance in `config.yaml`

### What You'll See

- ✅ Bounding boxes around detected persons
- ✅ Person IDs (maintained across frames)
- ✅ Names (if face is recognized)
- ✅ Skeleton overlay showing pose
- ✅ Activity labels (standing, sitting, walking, etc.)
- ✅ Real-time FPS counter
- ✅ Person count

### Supported Activities

1. Standing
2. Sitting
3. Walking
4. Sleeping
5. Using Phone
6. Writing
7. Talking

### Configuration

Edit `config.yaml` to customize:
- Detection model and confidence
- Face recognition tolerance
- Pose estimation complexity
- Activity smoothing
- Visualization options

### Need Help?

See [README.md](README.md) for detailed documentation.

---

**Ready to go!** Just run `python src/main.py` and point your webcam at people to see the system in action.
