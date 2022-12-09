import argparse
import glob
import os
import tqdm
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="./dataset", help='Dataset path')
    parser.add_argument('-o', '--output', type=str, default="./filelists/48k_audio_filelist.txt", help='File list output path')
    parser.add_argument('-s', '--sr', type=int, default=48000, help='File target sample rate')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    audio_files = list(glob.glob(os.path.join(args.input, "**/*.wav"), recursive=True))

    target_sr = args.sr
    total_time = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, audio_path in enumerate(tqdm.tqdm(audio_files)):
            audio = sf.SoundFile(audio_path)
            sec = audio.frames / audio.samplerate
            if audio.frames / audio.samplerate * target_sr < 16384 * 1.2:
                continue
            audio_path = audio_path.replace("\\", "/")
            f.write(f"{audio_path}\n")
            total_time += sec
    
    print(f"Total time: {total_time//3600}h")