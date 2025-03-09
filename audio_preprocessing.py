#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
import json
import yaml
from pathlib import Path
import concurrent.futures
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audio-preprocessor")

class AudioPreprocessor:
    def __init__(self, config_path="audio_config.yaml"):
        """Initialize the audio preprocessing pipeline."""
        self.load_config(config_path)
        self.setup()
        
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.config = {
                "output": {
                    "format": "wav",  # Output format (wav, mp3, m4a, etc.)
                    "directory": "processed_audio",
                    "suffix": "_processed"
                },
                "compression": {
                    "enabled": True,
                    "format": "m4a",
                    "codec": "aac",
                    "bitrate": "128k"
                },
                "preprocessing": {
                    "noise_reduction": {
                        "enabled": True,
                        "strength": "medium"  # low, medium, high
                    },
                    "normalization": {
                        "enabled": True,
                        "target_level": -16,  # dB LUFS
                        "true_peak": -1.5  # dB
                    },
                    "compression": {
                        "enabled": False,
                        "threshold": -20,
                        "ratio": 4,
                        "attack": 5,
                        "release": 50
                    },
                    "eq": {
                        "enabled": False,
                        "preset": "speech"  # speech, music, custom
                    }
                }
            }
            
    def setup(self):
        """Set up required directories and check dependencies."""
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output"]["directory"], exist_ok=True)
        
        # Check if FFmpeg is installed
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required tools are installed."""
        # Check FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            logger.info("FFmpeg is installed.")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            sys.exit(1)

    def process_audio_files(self, input_path):
        """Process audio files in the input path (file or directory)."""
        input_path = Path(input_path)
        
        if input_path.is_file():
            self._process_single_file(input_path)
        elif input_path.is_dir():
            # Process all audio files
            audio_extensions = ['.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac']
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(list(input_path.glob(f"*{ext}")))
            
            if not audio_files:
                logger.warning(f"No audio files found in {input_path}")
                return
                
            logger.info(f"Found {len(audio_files)} audio files to process")
            
            # Process files with a thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._process_single_file, file) 
                          for file in audio_files]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(f"File processing failed: {exc}")
        else:
            logger.error(f"Input path {input_path} does not exist")
            
    def _process_single_file(self, file_path):
        """Apply the audio preprocessing chain to a single file."""
        file_path = Path(file_path)
        output_dir = Path(self.config["output"]["directory"])
        
        # Define output filename
        output_filename = f"{file_path.stem}{self.config['output']['suffix']}.{self.config['output']['format']}"
        output_path = output_dir / output_filename
        
        logger.info(f"Processing file: {file_path}")
        
        # Check if output already exists
        if output_path.exists():
            logger.info(f"Output file {output_path} already exists. Skipping.")
            return output_path
            
        # Build the FFmpeg filter chain
        filter_chain = self._build_filter_chain()
            
        # Compress to desired format if requested
        if self.config["compression"]["enabled"]:
            output_format = self.config["compression"]["format"]
            output_codec = self.config["compression"]["codec"]
            output_bitrate = self.config["compression"]["bitrate"]
            
            # Update output path if compressing to a different format
            if output_format != self.config["output"]["format"]:
                output_filename = f"{file_path.stem}{self.config['output']['suffix']}.{output_format}"
                output_path = output_dir / output_filename
                
            # Add encoding options
            encoding_options = [
                "-c:a", output_codec,
                "-b:a", output_bitrate
            ]
        else:
            # Use PCM for WAV or best quality for the chosen format
            if self.config["output"]["format"] == "wav":
                encoding_options = ["-c:a", "pcm_s16le"]
            else:
                # Default to high quality encodings for other formats
                encoding_options = []
        
        # Build the complete FFmpeg command
        cmd = ["ffmpeg", "-i", str(file_path)]
        
        # Add filter complex if we have any filters
        if filter_chain:
            cmd.extend(["-filter_complex", filter_chain])
            
        # Add encoding options
        cmd.extend(encoding_options)
        
        # Add output file and force overwrite
        cmd.extend(["-y", str(output_path)])
        
        # Execute the command
        try:
            logger.info(f"Running FFmpeg: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            logger.info(f"Successfully processed to {output_path}")
            return output_path
        except subprocess.SubprocessError as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _build_filter_chain(self):
        """Build the FFmpeg filter chain based on config."""
        filters = []
        
        # Noise reduction
        if self.config["preprocessing"]["noise_reduction"]["enabled"]:
            strength = self.config["preprocessing"]["noise_reduction"]["strength"]
            
            # Map strength to FFmpeg settings
            strength_map = {
                "low": "0.2",
                "medium": "0.4",
                "high": "0.6"
            }
            
            noise_strength = strength_map.get(strength, "0.4")
            filters.append(f"afftdn=nr={noise_strength}:nf=-25")
        
        # Normalization
        if self.config["preprocessing"]["normalization"]["enabled"]:
            target_level = self.config["preprocessing"]["normalization"]["target_level"]
            true_peak = self.config["preprocessing"]["normalization"]["true_peak"]
            
            # Use EBU R128 loudness normalization
            filters.append(f"loudnorm=I={target_level}:TP={true_peak}:LRA=7")
        
        # Dynamic range compression
        if self.config["preprocessing"]["compression"]["enabled"]:
            threshold = self.config["preprocessing"]["compression"]["threshold"]
            ratio = self.config["preprocessing"]["compression"]["ratio"]
            attack = self.config["preprocessing"]["compression"]["attack"]
            release = self.config["preprocessing"]["compression"]["release"]
            
            # Apply compression
            filters.append(f"acompressor=threshold={threshold}dB:ratio={ratio}:attack={attack}:release={release}")
        
        # Equalization
        if self.config["preprocessing"]["eq"]["enabled"]:
            preset = self.config["preprocessing"]["eq"]["preset"]
            
            if preset == "speech":
                # Speech enhancement EQ - boost mids, cut lows and highs
                filters.append("equalizer=f=100:width_type=h:width=200:g=-3")  # Cut low rumble
                filters.append("equalizer=f=300:width_type=h:width=200:g=2")   # Boost low-mids (for voice body)
                filters.append("equalizer=f=1500:width_type=h:width=1000:g=3") # Boost mids (for voice clarity)
                filters.append("equalizer=f=4000:width_type=h:width=1000:g=1") # Slight boost to highs (for presence)
                filters.append("equalizer=f=8000:width_type=h:width=2000:g=-2") # Cut highs (to reduce sibilance)
            elif preset == "music":
                # Music enhancement EQ - more balanced
                filters.append("equalizer=f=60:width_type=h:width=100:g=1")    # Slight bass boost
                filters.append("equalizer=f=300:width_type=h:width=200:g=0")   # Neutral low-mids
                filters.append("equalizer=f=1500:width_type=h:width=1000:g=1") # Slight mid boost
                filters.append("equalizer=f=6000:width_type=h:width=2000:g=2") # Boost highs (for detail)
            # Custom EQ could be added here
        
        # De-essing (optional, can be added as additional preprocessing)
        # filters.append("adeclick=threshold=20:blocksize=64")  # Remove clicks and pops
        
        # Join all filters with commas for FFmpeg
        if filters:
            return ",".join(filters)
        else:
            return ""

def generate_config_file(output_path):
    """Generate a default configuration file."""
    config = {
        "output": {
            "format": "wav",  # Output format (wav, mp3, m4a, etc.)
            "directory": "processed_audio",
            "suffix": "_processed"
        },
        "compression": {
            "enabled": True,
            "format": "m4a",
            "codec": "aac",
            "bitrate": "128k"
        },
        "preprocessing": {
            "noise_reduction": {
                "enabled": True,
                "strength": "medium"  # low, medium, high
            },
            "normalization": {
                "enabled": True,
                "target_level": -16,  # dB LUFS
                "true_peak": -1.5  # dB
            },
            "compression": {
                "enabled": False,
                "threshold": -20,
                "ratio": 4,
                "attack": 5,
                "release": 50
            },
            "eq": {
                "enabled": False,
                "preset": "speech"  # speech, music, custom
            }
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Generated default config file at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files for better transcription quality")
    parser.add_argument("input", help="Input audio file or directory containing audio files")
    parser.add_argument("--config", "-c", default="audio_config.yaml", help="Path to config file")
    parser.add_argument("--generate-config", action="store_true", help="Generate a default config file")
    args = parser.parse_args()
    
    if args.generate_config:
        generate_config_file(args.config)
        if not os.path.exists(args.input):
            return
    
    processor = AudioPreprocessor(args.config)
    processor.process_audio_files(args.input)
    
if __name__ == "__main__":
    main()