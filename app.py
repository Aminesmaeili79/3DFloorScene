from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
import numpy as np
from pathlib import Path
import json
import traceback
from datetime import datetime, timedelta
import threading
import time

# Import conversion utilities
from converter import FloorPlanConverter

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
CLEANUP_INTERVAL = 3600  # 1 hour
FILE_RETENTION_TIME = 7200  # 2 hours

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Store conversion jobs
conversion_jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    while True:
        try:
            current_time = time.time()
            cutoff_time = current_time - FILE_RETENTION_TIME
            
            # Cleanup uploads
            for file_path in UPLOAD_FOLDER.glob('*'):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
            
            # Cleanup outputs
            for file_path in OUTPUT_FOLDER.glob('*'):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
            
            # Cleanup old jobs from memory
            jobs_to_remove = []
            for job_id, job in conversion_jobs.items():
                if job.get('timestamp', 0) < cutoff_time:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del conversion_jobs[job_id]
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(CLEANUP_INTERVAL)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(conversion_jobs)
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        upload_path = UPLOAD_FOLDER / f"{job_id}.{file_ext}"
        file.save(upload_path)
        
        # Store job information
        conversion_jobs[job_id] = {
            'status': 'uploaded',
            'filename': filename,
            'upload_path': str(upload_path),
            'timestamp': time.time(),
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Upload failed',
            'details': str(e)
        }), 500

@app.route('/api/convert/<job_id>', methods=['POST'])
def convert_to_3d(job_id):
    try:
        # Check if job exists
        if job_id not in conversion_jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = conversion_jobs[job_id]
        
        if job['status'] == 'processing':
            return jsonify({'error': 'Conversion already in progress'}), 409
        
        # Get configuration from request
        config = request.json or {}
        
        conversion_config = {
            'wall_height': config.get('wall_height', 2.7),
            'floor_thickness': config.get('floor_thickness', 0.3),
            'scale': config.get('scale', 0.01),
            'wall_thickness_min': config.get('wall_thickness_min', 3),
            'wall_thickness_max': config.get('wall_thickness_max', 25),
            'min_wall_length': config.get('min_wall_length', 30)
        }
        
        # Update job status
        job['status'] = 'processing'
        job['config'] = conversion_config
        job['processing_started'] = datetime.now().isoformat()
        
        # Perform conversion
        converter = FloorPlanConverter(conversion_config)
        
        # Load image
        upload_path = job['upload_path']
        image = cv2.imread(upload_path)
        
        if image is None:
            job['status'] = 'failed'
            job['error'] = 'Failed to load image'
            return jsonify({'error': 'Failed to load image'}), 500
        
        # Convert to 3D
        result = converter.convert(image)
        
        if result is None or result['model'] is None:
            job['status'] = 'failed'
            job['error'] = 'Conversion failed - no walls detected'
            return jsonify({'error': 'No walls detected in image'}), 422
        
        # Save output files
        base_name = f"{job_id}"
        obj_path = OUTPUT_FOLDER / f"{base_name}.obj"
        json_path = OUTPUT_FOLDER / f"{base_name}.json"
        preview_path = OUTPUT_FOLDER / f"{base_name}_preview.png"
        
        # Export OBJ file
        converter.export_obj(result['model'], obj_path)
        
        # Export metadata JSON
        metadata = {
            'job_id': job_id,
            'original_filename': job['filename'],
            'converted_at': datetime.now().isoformat(),
            'config': conversion_config,
            'stats': {
                'vertices': len(result['model']['vertices']),
                'faces': len(result['model']['faces']),
                'wall_height': result['model']['wall_height'],
                'scale': result['model']['scale']
            },
            'files': {
                'obj': f"{base_name}.obj",
                'preview': f"{base_name}_preview.png"
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save preview image
        cv2.imwrite(str(preview_path), result['preview'])
        
        # Update job with results
        job['status'] = 'completed'
        job['completed_at'] = datetime.now().isoformat()
        job['output_files'] = {
            'obj': str(obj_path),
            'json': str(json_path),
            'preview': str(preview_path)
        }
        job['stats'] = metadata['stats']
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'completed',
            'stats': metadata['stats'],
            'files': {
                'obj': f"/api/download/{job_id}/obj",
                'metadata': f"/api/download/{job_id}/json",
                'preview': f"/api/download/{job_id}/preview"
            }
        }), 200
        
    except Exception as e:
        # Update job status
        if job_id in conversion_jobs:
            conversion_jobs[job_id]['status'] = 'failed'
            conversion_jobs[job_id]['error'] = str(e)
            conversion_jobs[job_id]['traceback'] = traceback.format_exc()
        
        return jsonify({
            'error': 'Conversion failed',
            'details': str(e)
        }), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    if job_id not in conversion_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = conversion_jobs[job_id]
    
    response = {
        'job_id': job_id,
        'status': job['status'],
        'filename': job.get('filename'),
        'created_at': job.get('created_at')
    }
    
    if job['status'] == 'completed':
        response['stats'] = job.get('stats')
        response['files'] = {
            'obj': f"/api/download/{job_id}/obj",
            'metadata': f"/api/download/{job_id}/json",
            'preview': f"/api/download/{job_id}/preview"
        }
    elif job['status'] == 'failed':
        response['error'] = job.get('error')
    
    return jsonify(response)

@app.route('/api/download/<job_id>/<file_type>', methods=['GET'])
def download_file(job_id, file_type):
    if job_id not in conversion_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = conversion_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Conversion not completed'}), 400
    
    # Get file path based on type
    if file_type == 'obj':
        file_path = job['output_files']['obj']
        mimetype = 'model/obj'
        as_attachment = True
    elif file_type == 'json':
        file_path = job['output_files']['json']
        mimetype = 'application/json'
        as_attachment = False
    elif file_type == 'preview':
        file_path = job['output_files']['preview']
        mimetype = 'image/png'
        as_attachment = False
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        file_path,
        mimetype=mimetype,
        as_attachment=as_attachment,
        download_name=os.path.basename(file_path)
    )

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    jobs_list = []
    for job_id, job in conversion_jobs.items():
        jobs_list.append({
            'job_id': job_id,
            'status': job['status'],
            'filename': job.get('filename'),
            'created_at': job.get('created_at')
        })
    
    return jsonify({
        'total': len(jobs_list),
        'jobs': jobs_list
    })

@app.route('/api/delete/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    if job_id not in conversion_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = conversion_jobs[job_id]
    
    try:
        # Delete uploaded file
        if os.path.exists(job['upload_path']):
            os.remove(job['upload_path'])
        
        # Delete output files if they exist
        if 'output_files' in job:
            for file_path in job['output_files'].values():
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Remove from jobs dict
        del conversion_jobs[job_id]
        
        return jsonify({
            'success': True,
            'message': 'Job deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete job',
            'details': str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large',
        'max_size': f'{MAX_FILE_SIZE / (1024*1024)}MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*80)
    print("🚀 FLOOR PLAN CONVERTER API SERVER")
    print("="*80)
    print(f"📁 Upload folder: {UPLOAD_FOLDER.absolute()}")
    print(f"📁 Output folder: {OUTPUT_FOLDER.absolute()}")
    print(f"🌐 Starting server on http://localhost:5000")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=True)