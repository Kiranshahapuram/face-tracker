import os
import shutil
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Directory where videos will be stored
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Tracker - Upload Video</title>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --accent: #38bdf8;
            --text: #f8fafc;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: var(--card);
            padding: 2.5rem;
            border-radius: 1rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            width: 400px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h1 { font-size: 1.5rem; margin-bottom: 1.5rem; color: var(--accent); }
        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            padding: 2rem;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover { border-color: var(--accent); background: rgba(56, 189, 248, 0.05); }
        input[type="file"] { display: none; }
        .btn {
            background: var(--accent);
            color: var(--bg);
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: bold;
            cursor: pointer;
            margin-top: 1.5rem;
            width: 100%;
        }
        .status { margin-top: 1rem; font-size: 0.9rem; opacity: 0.8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Intelligent Face Tracker</h1>
        <p>Upload a video file to process visitors</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label class="upload-area" for="file-input">
                <span id="file-name">Click to select MP4 video</span>
                <input type="file" name="video" id="file-input" accept="video/mp4" onchange="updateName()">
            </label>
            <button type="submit" class="btn">Start Tracking</button>
        </form>
    </div>
    <script>
        function updateName() {
            const input = document.getElementById('file-input');
            document.getElementById('file-name').innerText = input.files[0].name;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file uploaded", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Update config.json to point to this new video
    import json
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    config_data['video']['source'] = file_path
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    return f"Video uploaded to {file_path}. Config updated. You can now run main.py."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
