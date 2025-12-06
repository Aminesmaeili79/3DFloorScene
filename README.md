# Floor Plan to 3D Converter - Backend API

Simple Flask backend that converts 2D floor plan images to 3D models (OBJ format).

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Server runs at: `http://localhost:5000`

## API Endpoints

### 1. Upload Image
```
POST /api/upload
```
Upload a floor plan image (PNG, JPG, BMP, TIFF)

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "job_id": "abc123...",
  "filename": "floorplan.png"
}
```

### 2. Convert to 3D
```
POST /api/convert/{job_id}
```
Convert uploaded image to 3D model

**Request Body (optional):**
```json
{
  "wall_height": 2.7,
  "scale": 0.01,
  "wall_thickness_min": 3,
  "wall_thickness_max": 25
}
```

**Response:**
```json
{
  "stats": {
    "vertices": 1234,
    "faces": 856
  },
  "files": {
    "obj": "/api/download/{job_id}/obj",
    "preview": "/api/download/{job_id}/preview"
  }
}
```

### 3. Download Files
```
GET /api/download/{job_id}/obj       - Download 3D model
GET /api/download/{job_id}/preview   - Download preview image
GET /api/download/{job_id}/json      - Download metadata
```

### 4. Check Status
```
GET /api/status/{job_id}
```

### 5. Health Check
```
GET /api/health
```

## How the Converter Works

The converter uses computer vision to transform 2D floor plans into 3D models:

1. **Preprocessing**: Converts image to grayscale and applies adaptive thresholding to isolate lines

2. **Wall Detection**: Uses 4 strategies to detect only main structural walls:
   - Morphological operations to find thick horizontal/vertical lines
   - Distance transform to filter by wall thickness (3-25 pixels)
   - Component analysis to remove small disconnected elements
   - Keeps only the 3 largest connected structures

3. **3D Model Building**:
   - Creates a base floor with proper thickness
   - Extrudes detected walls vertically to specified height (default 2.7m)
   - Generates vertices and faces for the 3D mesh
   - Assigns colors (floor: brown, walls: white)

4. **Export**: Saves as OBJ file compatible with Blender, Unity, etc.

## Example Usage

```javascript
// Upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const res1 = await fetch('http://localhost:5000/api/upload', {
  method: 'POST',
  body: formData
});
const { job_id } = await res1.json();

// Convert
const res2 = await fetch(`http://localhost:5000/api/convert/${job_id}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wall_height: 2.7 })
});
const result = await res2.json();

// Download
window.open(`http://localhost:5000/api/download/${job_id}/obj`);
```

## Configuration Parameters

- `wall_height`: Wall height in meters (default: 2.7)
- `floor_thickness`: Floor thickness in meters (default: 0.3)
- `scale`: Meters per pixel (default: 0.01)
- `wall_thickness_min`: Min wall thickness in pixels (default: 3)
- `wall_thickness_max`: Max wall thickness in pixels (default: 25)
- `min_wall_length`: Min wall segment length in pixels (default: 30)

## Files

- `app.py` - Flask API server
- `converter.py` - Conversion logic
- `requirements.txt` - Dependencies

## Notes

- Files are automatically deleted after 2 hours
- Max upload size: 16MB
- CORS enabled for frontend integration