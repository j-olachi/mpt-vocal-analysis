"""
MPT Vocal Analyzer - Core Analysis Module
Final Production Version

This module handles all vocal analysis:
- Audio format conversion
- Voice activity detection
- MPT calculation
- Clinical classification
"""

import webrtcvad
import numpy as np
import subprocess
import tempfile
import os


# ============================================================================
# CLINICAL SETTINGS - Adjust these for your environment
# ============================================================================

class ClinicalSettings:
    """
    MPT Detection Settings
    Tune these based on your clinical environment
    """
    
    # Ambient noise calibration
    CALIBRATION_DURATION = 1.0  # seconds - measure background noise
    
    # Speech detection thresholds
    SILENCE_TIMEOUT = 0.8  # seconds - silence before stopping
                           # Decrease if recording runs too long
                           # Increase if stops during patient pauses
    
    MIN_CONSECUTIVE_SILENCE = 8  # frames (~0.24s) of continuous silence
    
    # MPT validation
    MIN_VALID_MPT = 1.0  # seconds - reject if shorter
    MAX_RECORDING = 45   # seconds - safety cutoff
    
    # VAD sensitivity
    # None = auto-calibrate, or force 0-3:
    # 0 = catch all speech (quiet room)
    # 1 = balanced (normal room)  
    # 2 = moderate filtering (busy ER)
    # 3 = aggressive (noisy ER)
    FORCE_VAD_MODE = None


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_mpt_audio(audio_bytes):
    """
    Main function to analyze MPT from audio
    
    Args:
        audio_bytes: Raw audio data from browser (any format)
        
    Returns:
        dict: {
            'success': bool,
            'mpt': float (seconds),
            'classification': {
                'urgency': str,
                'esi_level': int,
                'category': str,
                'action': str,
                'color': str
            },
            'debug_info': {...}
        }
    """
    
    print("\n" + "="*70)
    print("🎤 STARTING MPT VOCAL ANALYSIS")
    print("="*70)
    
    try:
        # Step 1: Convert audio to WAV format
        print("\n1️⃣ Converting audio to WAV format...")
        wav_data = convert_to_wav(audio_bytes)
        
        if not wav_data:
            return {
                'success': False,
                'error': 'Failed to convert audio format',
                'mpt': 0
            }
        
        print("   ✓ Conversion successful")
        
        # Step 2: Analyze speech with VAD
        print("\n2️⃣ Analyzing speech with Voice Activity Detection...")
        result = detect_and_measure_mpt(wav_data)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70 + "\n")
        
        return result
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return {
            'success': False,
            'error': f'Analysis error: {str(e)}',
            'mpt': 0
        }


# ============================================================================
# AUDIO CONVERSION
# ============================================================================

def convert_to_wav(audio_bytes):
    """
    Convert browser audio (WebM/MP4) to WAV format
    
    Uses FFmpeg to convert to 16kHz mono 16-bit WAV
    (format required by WebRTC VAD)
    """
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as f:
            f.write(audio_bytes)
            input_path = f.name
        
        output_path = input_path.replace('.webm', '.wav')
        
        # Convert with FFmpeg
        result = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000',      # 16kHz sample rate
            '-ac', '1',          # Mono
            '-f', 'wav',         # WAV format
            output_path
        ], capture_output=True, check=True)
        
        # Read converted file
        with open(output_path, 'rb') as f:
            wav_data = f.read()
        
        # Cleanup
        os.remove(input_path)
        os.remove(output_path)
        
        return wav_data
    
    except subprocess.CalledProcessError as e:
        print(f"   FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"   Conversion error: {e}")
        return None


# ============================================================================
# VOICE ACTIVITY DETECTION & MPT MEASUREMENT
# ============================================================================

def detect_and_measure_mpt(wav_data):
    """
    Detect speech and measure MPT duration
    
    Process:
    1. Calibrate to ambient noise
    2. Detect when speech starts
    3. Detect when speech ends
    4. Calculate duration
    5. Classify urgency
    """
    
    # Audio parameters
    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480
    FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit = 2 bytes/sample
    
    # Skip WAV header (44 bytes)
    audio_data = wav_data[44:]
    total_frames = len(audio_data) // FRAME_BYTES
    
    # Step 1: Calibrate to ambient noise
    print("\n   🔊 Calibrating to ambient noise...")
    calibration_frames = int(ClinicalSettings.CALIBRATION_DURATION * SAMPLE_RATE / FRAME_SAMPLES)
    noise_level = measure_noise_level(audio_data, calibration_frames, FRAME_BYTES)
    
    # Step 2: Configure VAD
    if ClinicalSettings.FORCE_VAD_MODE is not None:
        vad_mode = ClinicalSettings.FORCE_VAD_MODE
        print(f"   🔧 Using FORCED VAD mode: {vad_mode}")
    else:
        vad_mode = select_vad_mode(noise_level)
    
    vad = webrtcvad.Vad(vad_mode)
    
    # Step 3: Process audio frames
    print("\n   🎙️ Processing audio frames...")
    
    speech_start = None
    last_speech = None
    is_speaking = False
    consecutive_silence = 0
    
    speech_frame_count = 0
    
    for frame_idx in range(total_frames):
        # Extract frame
        start = frame_idx * FRAME_BYTES
        end = start + FRAME_BYTES
        frame = audio_data[start:end]
        
        if len(frame) < FRAME_BYTES:
            break
        
        time_now = frame_idx * FRAME_MS / 1000.0
        
        # Safety timeout
        if time_now > ClinicalSettings.MAX_RECORDING:
            print(f"   ⏱️ Max recording time reached ({ClinicalSettings.MAX_RECORDING}s)")
            break
        
        # Detect speech in frame
        try:
            has_speech = vad.is_speech(frame, SAMPLE_RATE)
        except:
            has_speech = False
        
        if has_speech:
            # Speech detected
            consecutive_silence = 0
            speech_frame_count += 1
            
            if not is_speaking:
                # Speech START
                is_speaking = True
                speech_start = time_now
                print(f"   ▶️ Speech STARTED at {time_now:.2f}s")
            
            last_speech = time_now
        
        else:
            # Silence detected
            if is_speaking:
                consecutive_silence += 1
                silence_duration = consecutive_silence * FRAME_MS / 1000.0
                
                # Check if silence threshold reached
                if (consecutive_silence >= ClinicalSettings.MIN_CONSECUTIVE_SILENCE and
                    silence_duration >= ClinicalSettings.SILENCE_TIMEOUT):
                    
                    # Speech END
                    mpt_duration = last_speech - speech_start
                    print(f"   ⏹️ Speech ENDED at {last_speech:.2f}s")
                    print(f"   ⏱️ MPT Duration: {mpt_duration:.2f}s")
                    is_speaking = False
                    break
    
    # Calculate final MPT
    if speech_start is None or last_speech is None:
        mpt_duration = 0
    elif is_speaking:
        # Still speaking when recording ended
        mpt_duration = last_speech - speech_start
        print(f"   ℹ️ Recording ended during speech")
        print(f"   ⏱️ MPT Duration: {mpt_duration:.2f}s")
    
    # Calculate statistics
    total_processed = frame_idx if 'frame_idx' in locals() else 0
    speech_pct = (speech_frame_count / total_processed * 100) if total_processed > 0 else 0
    
    print(f"\n   📊 Statistics:")
    print(f"      Frames processed: {total_processed}")
    print(f"      Speech frames: {speech_frame_count} ({speech_pct:.1f}%)")
    print(f"      Noise level: {noise_level:.4f}")
    print(f"      VAD mode: {vad_mode}")
    
    # Validate MPT
    if mpt_duration < ClinicalSettings.MIN_VALID_MPT:
        return {
            'success': False,
            'error': f'MPT too short ({mpt_duration:.1f}s). Minimum {ClinicalSettings.MIN_VALID_MPT}s required.',
            'mpt': round(mpt_duration, 2),
            'debug_info': {
                'noise_level': round(noise_level, 4),
                'vad_mode': vad_mode,
                'speech_percentage': round(speech_pct, 1),
                'total_frames': total_processed
            }
        }
    
    # Classify based on clinical thresholds
    classification = classify_mpt(mpt_duration)
    
    print(f"\n   🏥 Classification: {classification['urgency']} (ESI-{classification['esi_level']})")
    
    return {
        'success': True,
        'mpt': round(mpt_duration, 2),
        'classification': classification,
        'debug_info': {
            'noise_level': round(noise_level, 4),
            'vad_mode': vad_mode,
            'speech_percentage': round(speech_pct, 1),
            'total_frames': total_processed,
            'speech_frames': speech_frame_count
        }
    }


# ============================================================================
# NOISE CALIBRATION
# ============================================================================

def measure_noise_level(audio_data, num_frames, frame_bytes):
    """
    Measure ambient noise level from initial frames
    
    Returns normalized noise level (0.0 - 1.0)
    """
    amplitudes = []
    
    for i in range(min(num_frames, len(audio_data) // frame_bytes)):
        start = i * frame_bytes
        end = start + frame_bytes
        frame = audio_data[start:end]
        
        if len(frame) == frame_bytes:
            samples = np.frombuffer(frame, dtype=np.int16)
            avg_amp = np.abs(samples).mean()
            amplitudes.append(avg_amp)
    
    if amplitudes:
        noise = np.mean(amplitudes) / 32768.0  # Normalize to 0-1
        print(f"      Ambient noise level: {noise:.4f}")
        return noise
    
    return 0.01  # Default low noise


def select_vad_mode(noise_level):
    """
    Select appropriate VAD aggressiveness based on noise
    
    Quiet environment → Less aggressive (mode 1)
    Noisy environment → More aggressive (mode 3)
    """
    
    if noise_level < 0.015:
        mode = 1
        env = "Very Quiet"
    elif noise_level < 0.035:
        mode = 2
        env = "Moderate Noise"
    elif noise_level < 0.055:
        mode = 3
        env = "Noisy"
    else:
        mode = 3
        env = "Very Noisy"
    
    print(f"      Environment: {env} → VAD Mode {mode}")
    return mode


# ============================================================================
# CLINICAL CLASSIFICATION
# ============================================================================

def classify_mpt(mpt_seconds):
    """
    Classify MPT based on clinical thresholds
    
    Based on research:
    - <8s: Severe respiratory compromise (ESI-1)
    - <10s: Urgent evaluation needed (ESI-2)
    - <15s: Below normal reserve (ESI-2)
    - <20s: Borderline normal (ESI-3)
    - ≥20s: Normal respiratory function (ESI-4)
    
    Args:
        mpt_seconds: MPT duration in seconds
        
    Returns:
        dict: Clinical classification
    """
    
    if mpt_seconds < 8:
        return {
            'urgency': 'IMMEDIATE',
            'esi_level': 1,
            'category': 'Severe respiratory compromise',
            'action': 'Immediate medical intervention required',
            'color': 'RED',
            'description': 'Critical MPT indicating severe respiratory impairment. High aspiration risk. Requires immediate physician evaluation.'
        }
    
    elif mpt_seconds < 10:
        return {
            'urgency': 'URGENT',
            'esi_level': 2,
            'category': 'Significant respiratory impairment',
            'action': 'Urgent medical evaluation needed',
            'color': 'ORANGE',
            'description': 'Significantly reduced MPT suggesting important respiratory limitation. Urgent evaluation recommended.'
        }
    
    elif mpt_seconds < 15:
        return {
            'urgency': 'CONCERNING',
            'esi_level': 2,
            'category': 'Below normal respiratory reserve',
            'action': 'Medical evaluation recommended',
            'color': 'YELLOW',
            'description': 'MPT below normal range indicating reduced respiratory reserve. Medical evaluation advisable.'
        }
    
    elif mpt_seconds < 20:
        return {
            'urgency': 'BORDERLINE',
            'esi_level': 3,
            'category': 'Lower end of normal range',
            'action': 'Monitor for changes',
            'color': 'YELLOW',
            'description': 'MPT at lower end of normal. May benefit from monitoring or reassessment if symptomatic.'
        }
    
    else:
        return {
            'urgency': 'NORMAL',
            'esi_level': 4,
            'category': 'Normal respiratory reserve',
            'action': 'No immediate respiratory concerns',
            'color': 'GREEN',
            'description': 'MPT within normal range indicating adequate respiratory function.'
        }
