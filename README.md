# 2D Floor Plan to 3D Model Converter

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🏆 Built for [EMU Future Minds Hackathon 2025 - Construction AI Challenge](https://hackathon.emu.edu.tr/)

Transform 2D architectural floor plans into interactive 3D models with AI-powered room detection, automatic lighting placement, and door/window identification.

![SpaceViz Demo](https://img.shields.io/badge/Status-Hackathon_Project-success)

## ✨ Features

### Core Capabilities
- 🏗️ **Intelligent Wall Detection** - Advanced computer vision algorithms isolate structural walls from floor plan images
- 🏠 **Automatic Room Detection** - AI identifies and labels rooms (Living Room, Bedroom, Kitchen, etc.)
- 🚪 **Door & Window Recognition** - Detects openings in walls and classifies them as doors or windows
- 💡 **Smart Lighting Placement** - Automatically positions lights based on room type and size
- 📐 **Accurate Measurements** - Calculates room dimensions and areas in square meters
- 🎨 **Color-Coded Rooms** - Each room receives a unique color for easy visualization
- 🌓 **Day/Night Modes** - Toggle between lighting scenarios
- 📦 **3D Model Export** - Download models in industry-standard OBJ format

### Technical Highlights
- ⚡ **Real-time Processing** - Fast conversion with optimized CV algorithms
- 🎯 **High Accuracy** - Multi-strategy wall detection ensures reliable results
- 🔄 **Interactive 3D Viewer** - Rotate, pan, zoom with smooth controls
- 📊 **Detailed Metadata** - JSON export with complete conversion statistics
- 🌐 **RESTful API** - Easy integration with any frontend framework

## 🎓 Learning Journey

This project was developed as part of the **EMU Future Minds Hackathon 2025 - Construction AI Challenge**, where I deepened my knowledge of:

- **Computer Vision Techniques**: Adaptive thresholding, morphological operations, distance transforms, and skeletonization
- **Image Processing Libraries**: OpenCV for image manipulation and scikit-image for advanced morphological analysis
- **3D Geometry**: Mesh generation, vertex/face creation, and spatial coordinate systems
- **AI-Powered Classification**: Room type detection based on geometric properties
- **Full-Stack Development**: Flask backend API design and Three.js 3D visualization

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/spaceviz.git
cd spaceviz
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the server**
```bash
python app.py
```

4. **Open the demo**
```
Navigate to: http://localhost:5000
Open demo.html in your browser
```

## 📖 Usage

### Web Interface

1. **Upload** a floor plan image (PNG, JPG, BMP, TIFF)
2. **Click** "Convert to 3D Model with Room Detection"
3. **Explore** the interactive 3D model
4. **Download** the OBJ file for use in Blender, Unity, or other 3D software

### API Usage

#### 1. Upload Floor Plan
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:5000/api/upload', {
  method: 'POST',
  body: formData
});

const { job_id } = await response.json();
```

#### 2. Convert to 3D
```javascript
const response = await fetch(`http://localhost:5000/api/convert/${job_id}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wall_height: 2.7,      // Wall height in meters
    scale: 0.01,           // Meters per pixel
    wall_thickness_min: 3, // Minimum wall thickness (pixels)
    wall_thickness_max: 25 // Maximum wall thickness (pixels)
  })
});

const result = await response.json();
// result contains: stats, rooms, openings, lights, download links
```

#### 3. Download Files
```javascript
// Download 3D model
window.open(`http://localhost:5000/api/download/${job_id}/obj`);

// Download preview image
window.open(`http://localhost:5000/api/download/${job_id}/preview`);

// Download metadata JSON
window.open(`http://localhost:5000/api/download/${job_id}/json`);
```

## 🔧 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wall_height` | 2.7 | Wall height in meters |
| `floor_thickness` | 0.3 | Floor thickness in meters |
| `scale` | 0.01 | Meters per pixel conversion |
| `wall_thickness_min` | 3 | Minimum wall thickness (pixels) |
| `wall_thickness_max` | 25 | Maximum wall thickness (pixels) |
| `min_wall_length` | 30 | Minimum wall segment length (pixels) |

## 🛠️ How It Works

### 1. Preprocessing
- Converts image to grayscale
- Applies adaptive thresholding to isolate lines
- Removes noise with morphological operations

### 2. Wall Detection (Multi-Strategy Approach)
```
Strategy 1: Morphological Operations
→ Detect thick horizontal/vertical lines

Strategy 2: Distance Transform
→ Filter by wall thickness (3-25 pixels)

Strategy 3: Component Analysis
→ Remove small disconnected elements

Strategy 4: Structure Retention
→ Keep only 3 largest connected components
```

### 3. Room Detection
- Inverts wall mask to find enclosed spaces
- Identifies room contours with flood fill
- Classifies room types based on:
  - Area size
  - Aspect ratio
  - Shape complexity
- Calculates dimensions in real-world units

### 4. Opening Detection
- Finds gaps in wall structures
- Classifies as doors (0.8-1.2m) or windows (0.3-0.8m)
- Records position, orientation, and dimensions

### 5. Lighting Placement
- Analyzes room type and size
- Places ceiling lights at optimal positions
- Adjusts intensity based on room function:
  - Kitchens/Bathrooms: Bright white
  - Bedrooms: Warm, dimmed
  - Living Rooms: Soft ambient

### 6. 3D Model Generation
- Creates floor base with proper thickness
- Extrudes walls vertically to specified height
- Colors rooms with unique identifiers
- Generates vertices and faces for mesh
- Exports as OBJ with vertex colors

## 📁 Project Structure

```
spaceviz/
├── app.py              # Flask API server
├── converter.py        # Core CV algorithms & 3D generation
├── demo.html           # Interactive 3D viewer
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── uploads/           # Temporary upload storage
└── outputs/           # Generated 3D models & previews
```

## 🎨 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/upload` | POST | Upload floor plan image |
| `/api/convert/{job_id}` | POST | Convert to 3D model |
| `/api/status/{job_id}` | GET | Check conversion status |
| `/api/download/{job_id}/obj` | GET | Download 3D model |
| `/api/download/{job_id}/preview` | GET | Download preview image |
| `/api/download/{job_id}/json` | GET | Download metadata |
| `/api/jobs` | GET | List all jobs |
| `/api/delete/{job_id}` | DELETE | Delete job and files |

## 🎯 Use Cases

- **Real Estate**: Quickly visualize property layouts
- **Interior Design**: Plan room arrangements and furniture placement
- **Construction Planning**: Estimate materials and space utilization
- **Property Marketing**: Create engaging 3D tours from 2D plans
- **Architecture Education**: Learn spatial design concepts
- **Home Renovation**: Visualize modifications before construction

## 🔬 Technical Stack

### Backend
- **Flask** - Lightweight web framework
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computations
- **scikit-image** - Advanced morphological operations
- **Pillow** - Image format handling

### Frontend
- **Three.js** - 3D rendering and visualization
- **Vanilla JavaScript** - No heavy frameworks
- **HTML5/CSS3** - Modern, responsive UI

## 📊 Performance

- **Processing Time**: ~2-5 seconds for typical floor plans
- **File Retention**: 2 hours (automatic cleanup)
- **Max Upload Size**: 16MB
- **Supported Formats**: PNG, JPG, BMP, TIFF

## 🤝 Contributing

Contributions are welcome! This project was a learning experience, and there's always room for improvement.

Areas for enhancement:
- [ ] Deep learning models for better room classification
- [ ] Furniture detection and placement
- [ ] Multi-floor support
- [ ] Texture mapping for materials
- [ ] Export to additional formats (FBX, GLTF)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **EMU Future Minds Hackathon 2025** - For providing the opportunity and inspiration
- **OpenCV Community** - For excellent documentation and tutorials
- **Three.js Team** - For making 3D visualization accessible

## 📧 Contact

**Developer**: Amin

**Project Link**: [https://github.com/yourusername/spaceviz](https://github.com/yourusername/spaceviz)

**Hackathon**: [EMU Future Minds Hackathon 2025](https://hackathon.emu.edu.tr/)

---

⭐ If you found this project helpful, please give it a star!

Built with ❤️ for the EMU Future Minds Hackathon 2025 - Construction AI Challenge
