import os
import sys
from flask import Flask, request, redirect, url_for, render_template, flash
import numpy as np # For formatting the output

# Add the src directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.audio_analyzer import load_and_extract_pitch

# Define the upload folder at the project root
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'uploads'))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey' # Needed for flashing messages

# Ensure the templates directory is correctly identified by Flask
# This assumes app.py is in src/webapp/ and templates are in src/webapp/templates/
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and processes the audio file."""
    if 'audio_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['audio_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = file.filename # Use the original filename, can be secured later
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Process the audio file
            pitches, voiced_flags, voiced_probs = load_and_extract_pitch(filepath)

            # For display purposes, convert numpy arrays to string representations
            # Limiting the output for brevity in the template
            pitches_str = np.array2string(pitches[:20], precision=2, separator=', ') + "..." if len(pitches) > 20 else np.array2string(pitches, precision=2, separator=', ')
            voiced_flags_str = np.array2string(voiced_flags[:20], separator=', ') + "..." if len(voiced_flags) > 20 else np.array2string(voiced_flags, separator=', ')
            
            return render_template('results.html', 
                                   pitches_str=pitches_str, 
                                   voiced_flags_str=voiced_flags_str)
        
        except FileNotFoundError:
            flash(f"Error: The file {filename} was not found after saving.")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"An error occurred during analysis: {e}")
            return redirect(url_for('index'))
        finally:
            # Ensure the temporary file is deleted
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Note: For development, use `flask run`. 
    # The following is for direct script execution (though not recommended for Flask's dev server in prod).
    app.run(debug=True)
