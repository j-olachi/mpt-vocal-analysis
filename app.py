"""
MPT Vocal Analysis System - Flask Backend
Final Production Version

This is the main Flask server that handles:
- Web interface serving
- Audio processing requests
- MPT calculation and classification
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
from mpt_vocal_analyzer import analyze_mpt_audio

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Serve the MPT testing interface"""
    return render_template('index.html')


@app.route('/api/analyze-mpt', methods=['POST'])
def analyze_mpt():
    """
    API endpoint for MPT analysis
    
    Receives base64-encoded audio from browser
    Returns MPT duration and clinical classification
    """
    try:
        # Get audio data from request
        data = request.json
        audio_base64 = data.get('audio_data')
        
        if not audio_base64:
            return jsonify({
                'success': False,
                'error': 'No audio data provided'
            }), 400
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Analyze with MPT vocal analyzer
        result = analyze_mpt_audio(audio_bytes)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'MPT Vocal Analysis System'
    })


if __name__ == '__main__':
    print('\n' + '='*70)
    print('🎤 MPT VOCAL ANALYSIS SYSTEM - STARTING')
    print('='*70)
    print('\n📱 Open your browser to: http://localhost:5000')
    print('💡 Press Ctrl+C to stop the server\n')
    print('='*70 + '\n')
    
    app.run(debug=True, port=5000, host='0.0.0.0')
