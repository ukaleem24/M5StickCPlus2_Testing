import socket
import time
import os
import struct
import wave
import threading


# Listen on all interfaces
UDP_IP = "0.0.0.0"
IMU_PORT = 1234
AUDIO_PORT = 1235


IMU_OUTFILE = "imu_log_received.csv"
AUDIO_OUTFILE = "audio_received.wav"


# Global state for multi-device audio
audio_files = {}  # {source_ip: wave_file}
audio_sample_rate = 16000
audio_channels = 1
audio_sample_width = 2
audio_frame_counts = {}  # {source_ip: frame_count}


def get_audio_filename(source_ip):
    """Generate filename for each device."""
    return f"audio_{source_ip.replace('.', '_')}.wav"


def init_audio_file(source_ip):
    """Initialize WAV file for a device."""
    if source_ip not in audio_files:
        filename = get_audio_filename(source_ip)
        audio_file = wave.open(filename, 'wb')
        audio_file.setnchannels(audio_channels)
        audio_file.setsampwidth(audio_sample_width)
        audio_file.setframerate(audio_sample_rate)
        audio_files[source_ip] = audio_file
        audio_frame_counts[source_ip] = 0
        print(f"[AUDIO] Initialized WAV file for {source_ip}: {filename}")


def write_audio_samples(source_ip, samples):
    """Write audio samples to the correct device's WAV file."""
    init_audio_file(source_ip)
    if source_ip in audio_files:
        audio_files[source_ip].writeframes(samples)
        audio_frame_counts[source_ip] += len(samples) // audio_sample_width


def imu_receiver():
    """Listen for IMU data on UDP_PORT from multiple devices."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)  # 256 KB buffer
    sock.bind((UDP_IP, IMU_PORT))
    print(f"[IMU] Listening on {UDP_IP}:{IMU_PORT}...")
   
    imu_files = {}  # {source_ip: file_handle}
   
    try:
        while True:
            data, addr = sock.recvfrom(2048)
            source_ip = addr[0]
           
            # Initialize file for new device
            if source_ip not in imu_files:
                filename = f"imu_{source_ip.replace('.', '_')}.csv"
                imu_files[source_ip] = open(filename, "a", buffering=1, encoding="utf-8")
                imu_files[source_ip].write("recv_ts,src,raw\n")
                print(f"[IMU] New device detected: {source_ip} -> {filename}")
           
            now = time.time()
            try:
                text = data.decode('utf-8').strip().replace('\n', ' ')
            except Exception:
                text = repr(data)
            line = f"{now:.6f},{source_ip},{text}\n"
            imu_files[source_ip].write(line)
    except KeyboardInterrupt:
        print('[IMU] Interrupted.')
    finally:
        for source_ip, f in imu_files.items():
            f.close()
            print(f"[IMU] Closed file for {source_ip}")
        sock.close()


def audio_receiver():
    """Listen for audio data on AUDIO_PORT from multiple devices."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)  # 256 KB buffer
    sock.bind((UDP_IP, AUDIO_PORT))
    print(f"[AUDIO] Listening on {UDP_IP}:{AUDIO_PORT}...")
   
    device_state = {}  # {source_ip: {'expected_seq': int, 'missed': list}}
    audio_ts_files = {}  # {source_ip: file_handle for timestamps}
   
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            source_ip = addr[0]
           
            # Initialize state for new device
            if source_ip not in device_state:
                device_state[source_ip] = {'expected_seq': 0, 'missed': []}
                print(f"[AUDIO] New device detected: {source_ip}")
            
            # Initialize timestamp file for new device
            if source_ip not in audio_ts_files:
                ts_filename = f"audio_ts_{source_ip.replace('.', '_')}.csv"
                audio_ts_files[source_ip] = open(ts_filename, "a", buffering=1, encoding="utf-8")
                audio_ts_files[source_ip].write("seq,timestamp_us,sample_count\n")
                print(f"[AUDIO] Initialized timestamp file for {source_ip}: {ts_filename}")
           
            if len(data) >= 14:
                # Parse header: 4 bytes seq + 2 bytes count + 8 bytes timestamp
                seq = struct.unpack('<I', data[0:4])[0]
                sample_count = struct.unpack('<H', data[4:6])[0]
                timestamp_us = struct.unpack('<Q', data[6:14])[0]
                audio_data = data[14:]
                
                # Write to timestamp file
                if source_ip in audio_ts_files:
                    audio_ts_files[source_ip].write(f"{seq},{timestamp_us},{sample_count}\n")
               
                expected_seq = device_state[source_ip]['expected_seq']
               
                # Check for packet loss per device
                if seq != expected_seq:
                    gap = seq - expected_seq
                    device_state[source_ip]['missed'].append((expected_seq, seq - 1))
                    print(f"[AUDIO] {source_ip} - PACKET LOSS: Expected seq {expected_seq}, got {seq} (gap of {gap} packets)")
               
                device_state[source_ip]['expected_seq'] = seq + 1
               
                write_audio_samples(source_ip, audio_data)
                print(f"[AUDIO] {source_ip} - Packet {seq}: {sample_count} samples, {len(audio_data)} bytes")
            else:
                print(f"[AUDIO] Invalid packet from {source_ip} (too short): {len(data)} bytes")
    except KeyboardInterrupt:
        print('[AUDIO] Interrupted.')
    finally:
        # Close all timestamp files
        for source_ip in audio_ts_files:
            audio_ts_files[source_ip].close()
            ts_filename = f"audio_ts_{source_ip.replace('.', '_')}.csv"
            print(f"[AUDIO] Closed timestamp file for {source_ip}: {ts_filename}")
        
        # Close all WAV files and print summary
        for source_ip in audio_files:
            audio_files[source_ip].close()
            frames = audio_frame_counts[source_ip]
            duration = frames / audio_sample_rate
            filename = get_audio_filename(source_ip)
            print(f"\n[AUDIO] {source_ip}: Closed {filename}")
            print(f"  Total frames: {frames}, Duration: {duration:.2f}s")
           
            if source_ip in device_state and device_state[source_ip]['missed']:
                missed = device_state[source_ip]['missed']
                print(f"  ⚠️  Missed {len(missed)} packet groups:")
                for start, end in missed:
                    print(f"    - Packets {start} to {end}")
            elif source_ip in device_state:
                print(f"  ✓ No packet loss detected")
       
        sock.close()


# Run both receivers in parallel threads
if __name__ == "__main__":
    imu_thread = threading.Thread(target=imu_receiver, daemon=True)
    audio_thread = threading.Thread(target=audio_receiver, daemon=True)
   
    imu_thread.start()
    audio_thread.start()
   
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")





